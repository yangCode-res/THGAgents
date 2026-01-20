from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from unittest.case import doModuleCleanups
from tqdm import tqdm
from Core.Agent import Agent
from ExampleText.index import ExampleText
from Memory.index import Memory,Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import (
    ENTITY_DEFINITIONS, EntityDefinition, EntityType, KGEntity,
    format_all_entity_definitions, format_entity_definition)

"""
Entity Extraction Agent.
Extracts biomedical entities from text and performs ontology mapping.
Input: None (retrieves text from subgraph workspace)
Output: None (stores extracted entities in subgraphs in memory)
Entry point: agent.process()
"""
class EntityExtractionAgent(Agent):
    """
    Entity Extraction Agent template.

    Inherits from base Agent class, pre-configured with:
    - template_id = "entity_extractor"
    - Reasonable default name/responsibility

    Override extract_from_text() to integrate actual NER/LLM extraction logic.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        name: str = "Entity Extraction Agent",
        system: str = "You are a careful biomedical classifier. Return STRICT JSON only.",
        responsibility: str = '''You are a specialized Entity Extraction Agent for biomedical literature. 
        Your task is to identify and classify all biomedical entities with high precision and appropriate ontological mapping''',
        entity_focus: Optional[List[Any]] = None,
        relation_focus: Optional[List[Any]] = None,
        priority: int = 1,
        memory: Optional[Memory] = None,
        metadata: Optional[Dict[str, Any]] = None,
        THRESH: float = 0.6
    ) -> None:
        super().__init__(
            client=client,
            model_name=model,
            system_prompt=system,
        )
        self.memory = memory or get_memory()
        self.configure(
            template_id="entity_extractor",
            name=name,
            responsibility=responsibility,
            entity_focus=list(entity_focus or []),
            relation_focus=list(relation_focus or []),
            priority=priority,
            metadata=dict(metadata or {}),
        )
        self.THRESH = THRESH
        self.step1_sys_desc = (
            "You are a rigorous biomedical type detector. Return STRICT JSON only; "
            "no explanations, prefixes/suffixes, or Markdown."
        )
        self.step2_sys_desc = (
            "You are a rigorous biomedical entity extractor. Return STRICT JSON only; "
            "no explanations, prefixes/suffixes, or Markdown.\n\n"
        )
        self.allKGEntities: List[KGEntity] = []
    
    def build_type_detection_prompt(
        self,
        text: str,
        entity_definitions: Dict[EntityType, EntityDefinition] = ENTITY_DEFINITIONS,
        order: Optional[List[EntityType]] = None,
    ) -> str:
        """
        Build production-level prompt for "entity type detection (classify types only, don't extract entities)".
        - Closed set = determined by entity_definitions/order, type keys always use Enum values (lowercase: 'disease','drug',...)
        - Output must be strict JSON:
        {
            "present": ["disease","drug", ...],        # Only items from closed set allowed
            "scores": {"disease": 0.0, "drug": 0.0, ...}  # Score each type in closed set, range [0,1]
        }
        - If no matches: present=[] and all scores = 0.0
        """
        if order is None:
            order = list(EntityType)
        closed_set: List[str] = [et.value for et in order if et in entity_definitions]
        detailed_defs = format_all_entity_definitions(
            entity_definitions=entity_definitions,
            order=order,
        )

        scores_template_items = ", ".join([f'"{t}": 0.0' for t in closed_set])
        scores_template = "{ " + scores_template_items + " }"

        
        task_desc = (
            "Task: Decide which ENTITY TYPES (closed set) appear in the text. "
            "Do NOT list mentions. Do NOT extrapolate or use world knowledge."
        )
        rules = [
            "Base your decision ONLY on the provided text; avoid hallucinations.",
            "If there is explicit, text-grounded evidence (incl. local synonym/abbreviation), mark that type as present.",
            "If no direct evidence, mark as absent.",
            "Output has two parts:",
            "  (1) present: list of type names (subset of the closed set; use lowercase keys, e.g., 'disease').",
            "  (2) scores: confidence in [0,1] for EACH type in the closed set; 0.0 when absent; ≥0.6 when present; ≥0.8 when strongly evident.",
            "If none present: present = [] and all scores = 0.0.",
            "Return STRICT JSON only.",
        ]
        boundary = (
            "Closed-set and detailed definitions (for boundary alignment; do NOT restate in output):\n\n"
            f"{detailed_defs}\n"
        )
        schema = (
            "Output (STRICT JSON):\n"
            "{\n"
            '  "present": ["disease","drug"],\n'
            f'  "scores": {scores_template}\n'
            "}\n"
            f"Closed set (allowed lowercase values only): {closed_set}"
        )
        prompt = (
            f"User:\n{task_desc}\n\n"
            + "\n".join(f"- {r}" for r in rules) + "\n\n"
            f"{boundary}\n"
            f"{schema}\n\n"
            "Text (decide ONLY from this text):\n<<<\n"
            f"{text}\n"
            ">>>\n"
        )

        return prompt
    def validate_and_fix_type_result(self,raw_json_text: str, closed_set: List[str]) -> Dict:
        """
        - Parse JSON; force present to be subset of closed set;
        - Complete scores for each type in closed set; clip scores to [0,1];
        - If present is empty but some scores > 0, don't auto-add to present (preserve "model-determined" semantics).
        """
        text = (raw_json_text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].lstrip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        if text and not text.lstrip().startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                text = text[start:end+1].strip()

        try:
            data = json.loads(text)
        except Exception as e:
            self.logger.warning(
                f"[EntityExtraction] step1 JSON decode failed: {e}, raw={raw_json_text[:200]!r}, cleaned={text[:200]!r}"
            )
            return {
                "present": [],
                "scores": {t: 0.0 for t in closed_set},
            }
        present = data.get("present", [])
        scores = data.get("scores", {})
        present = [t for t in present if t in closed_set]
        fixed_scores = {}
        for t in closed_set:
            v = float(scores.get(t, 0.0))
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            fixed_scores[t] = v

        return {"present": present, "scores": fixed_scores}

    def build_single_type_extraction_prompt(
        self,
        text: str,
        definition: EntityDefinition,
        max_entities: int = 50,
    ) -> str:
        """
         Step-2 prompt: extract ONLY the given entity type from `text`.
        Uses `format_entity_definition(definition, index=1)` to provide the boundary.
        """
        type_key = (definition.name or "").strip().lower()
        boundary = format_entity_definition(definition, index=1) 

        rules = [
            "Use ONLY the provided text; no world knowledge or extrapolation.",
            "Extract entities of this single type ONLY.",
            "Match local synonyms/abbreviations/case variants when evidenced in text.",
            "Deduplicate (case-insensitive): keep one entry per entity, prefer first occurrence; sort by first offset.",
            "Character spans use 0-based, half-open [start, end) over the raw text (including spaces/newlines).",
            f"Return at most {max_entities} entities.",
            "If none found, return an empty array.",
            "For EACH entity, provide a ~20-word single-sentence description grounded ONLY in the document; if insufficient evidence, use \"N/A\".",
            "The description must not rely on external/world knowledge; it must be inferable from the document."
            "Return STRICT JSON only; no explanations or extra text.",
        ]
        ontoloty_mapping=(
            "ONTOLOGY MAPPING:\n"
            "- Use standard identifiers when known (MESH:D001241, NCBI:5743)\n"
            "- Set \"N/A\" when no standard identifier available\n"
            "- Prioritize well-established ontologies (MeSH, NCBI, UniProt)\n"
        )
        example=(
            "EXAMPLES:\n"
            "Text: \"Aspirin inhibits COX-2 enzyme activity\"\n"
            "Output: [{\"mention\": \"Aspirin\", \"type\": \"DRUG\", \"normalized_id\": \"MESH:D001241\", \"aliases\": [\"acetylsalicylic acid\"], \"description\": \"Drug described as inhibiting COX-2 enzyme activity.\"}, {\"mention\": \"COX-2\", \"type\": \"PROTEIN\", \"normalized_id\": \"NCBI:5743\", \"aliases\": [\"cyclooxygenase-2\", \"PTGS2\"], \"description\": \"Enzyme described as the inhibition target of aspirin.\"}]\n"
        )
        schema = (
            "Output (STRICT JSON):\n"
            "{\n"
            f'  "type": "{boundary}",\n'
            '  "entities": [\n'
            "    {\n"
            '      "mention": "verbatim mention from text",\n'
            '      "span": [start, end],\n'
            '      "confidence": 0.0,\n'
            '      "normalized_id": "ontology:identifier or N/A",\n'
            '      "aliases": ["synonym1", "synonym2"]\n'
            '      "description": "20-word single-sentence description grounded in the document; if insufficient evidence, provide a standard biomedical definition based on general knowledge."\n'
            "    }\n"
            "  ]\n"
            "}"
        )


        return (
            "User:\n"
            f"Task: Extract entities of type [{boundary}] only (closed set = this single type). Do not output other types.\n"
            + "\n".join(f"- {r}" for r in rules)
            + "\n\n"
            "Type boundary for disambiguation (DO NOT restate in output):\n"
            f"{ontoloty_mapping}\n"
            f"{example}\n"
            f"{schema}\n\n"
            "Text (extract ONLY from this text):\n<<<\n"
            f"{text}\n"
            ">>>\n"
        )
    
    def _deduplicate_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        """
        Remove duplicate entities based on name similarity.

        Args:
            entities: List of entities to deduplicate

        Returns:
            List of unique entities
        """
        if not entities:
            return []

        unique_entities = {}

        for entity in entities:
            key = entity.name.lower().strip()
            if key in unique_entities:
                existing = unique_entities[key]
                if entity.entity_type != "Unknown" and existing.entity_type == "Unknown":
                    existing.entity_type = entity.entity_type
                if entity.aliases:
                    existing.aliases.extend(entity.aliases)
                    existing.aliases = list(set(existing.aliases))  # Remove duplicates
                if entity.normalized_id != "N/A" and existing.normalized_id == "N/A":
                    existing.normalized_id = entity.normalized_id

            else:
                unique_entities[key] = entity

        return list(unique_entities.values())
    
    def step1(self, text: str,sg_id: Optional[str] = None) -> str:
        """
        Step 1: Check which entity types exist among candidate types
        """
        step1_prompt = self.build_type_detection_prompt(text=text,entity_definitions=ENTITY_DEFINITIONS,order=list(EntityType))
        response = self.call_llm(step1_prompt)
        prefix = f"[EntityExtraction] sg_id={sg_id} " if sg_id is not None else "[EntityExtraction] "
        self.logger.info(
            f"{prefix}step1 raw response (first 400 chars) = {response[:400]!r}"
        )
        closed_set = [et.value for et in EntityType] 
        result = self.validate_and_fix_type_result(raw_json_text=response, closed_set=closed_set)
        selected = [t for t in result["present"] if result["scores"].get(t, 0.0) >= self.THRESH]
        allowed = {e.value: e for e in EntityType}
        def defs_from_selected(selected, defs=ENTITY_DEFINITIONS):
            return [defs[allowed[t]] for t in selected if t in allowed]
        result=defs_from_selected(selected)
        return result
    

    def step2(self, text: str, type_list: List[EntityDefinition]) -> List[KGEntity]:
        """
        Step 2: Entity extraction + ontology mapping (parallel extraction by type)
        Returns list of KGEntity extracted from current text (single document scope)
        """
        kg_entities: List[KGEntity] = []

        def _extract_for_type(defn: EntityDefinition) -> List[KGEntity]:
            """
            Extract entities for a single EntityDefinition, return corresponding KGEntity list
            (for thread pool parallel calls)
            """
            prompt = self.build_single_type_extraction_prompt(text=text, definition=defn)
            response = self.call_llm(prompt)
            try:
                parsed = self.parse_json(response)
            except Exception:
                self.logger.error(f"Failed to parse JSON response for {defn.name}: {response}")
                parsed = {}

            if isinstance(parsed, dict):
                items = parsed.get("entities") or []
            elif isinstance(parsed, list):
                items = parsed
            else:
                items = []
            local_entities: List[KGEntity] = []
            count = 0
            for entity in items:
                if not isinstance(entity, dict):
                    continue
                local_entities.append(KGEntity(
                    entity_id=entity.get("mention", ""),
                    entity_type=defn.name,
                    name=entity.get("mention", ""),
                    normalized_id=entity.get("normalized_id", "N/A"),
                    description=entity.get("description", "N/A"),
                    aliases=entity.get("aliases", []) or []
                ))
                count += 1
            self.logger.info(f"{defn.name} Extracted {count} entities")
            return local_entities
        if not type_list:
            return []
        max_workers = min(2, len(type_list)) 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_extract_for_type, defn): defn
                for defn in type_list
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Entity extraction (per type)"):
                ents = fut.result()
                kg_entities.extend(ents)
        return kg_entities
    def _process_single_subgraph(self, sg_id: str, sg: Subgraph) -> None:
        """
        Process single subgraph:
        - Get text from sg.meta["text"]
        - Call step1/step2 to extract entities
        - Write back to subgraph entities after deduplication
        """
        if hasattr(sg, 'entities') and len(sg.entities.by_id) > 0: 
            self.logger.info(f"[EntityExtraction] sg_id={sg_id} already has entities, skipping.")
            return
        text = sg.meta.get("text", "") or ""
        if not text.strip():
            return
        type_list = self.step1(text, sg_id=sg_id)
        self.logger.info(f"[EntityExtraction] sg_id={sg_id} step1 types={[d.name for d in type_list]}")
        if not text.strip():
            self.logger.warning(
                f"[EntityExtraction] sg_id={sg_id} has EMPTY text in sg.meta['text'], skip."
            )
        kg_entities = self.step2(text, type_list)
        kg_entities = self._deduplicate_entities(kg_entities)
        sg.upsert_many_entities(kg_entities)
        self.memory.register_subgraph(sg)

    def process(self, max_workers: Optional[int] = None) -> None:
        """
        Execute entity extraction main process in parallel (read subgraphs from memory):

        - Iterate through self.memory.subgraphs, treat each subgraph as a document
        - Subgraph meta["text"] contains the original text
        - Use thread pool to process subgraphs in parallel
        - Write extraction results directly back to corresponding subgraph entities
        """
        subgraphs_items = list(self.memory.subgraphs.items())
        self.logger.info(f"[EntityExtraction] total subgraphs={len(subgraphs_items)}")
        self.logger.info(f"[EntityExtraction] sg_ids={list(self.memory.subgraphs.keys())}")
        items_to_process = []
        for sg_id, sg in subgraphs_items:
            if hasattr(sg, 'entities') and len(sg.entities.by_id)>0:
                continue
            items_to_process.append((sg_id, sg))
        if max_workers is None:
            cpu_count = os.cpu_count() or 8
            max_workers = min(8, max(1, cpu_count))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_single_subgraph, sg_id, sg)
                for sg_id, sg in items_to_process
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Entity extraction over subgraphs"):
                try:
                    fut.result()
                except Exception as e:
                    self.logger.exception(f"[EntityExtraction] subgraph worker failed: {e}")
        