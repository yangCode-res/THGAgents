import json
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class ContextSummaryAgent(Agent):
    """
    Based on:
      - User query
      - KG paths (nodes + edges) extracted by PathExtractionAgent

    Calls LLM to generate a batch of:
      - Mechanistically sound
      - Query-relevant
      - Verifiable hypotheses
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        query: str,
        memory: Optional[Memory] = None,
        max_paths: int = 10,
        hypotheses_per_path: int = 3,
        output_path: Optional[str] = None,
    ):
        system_prompt = """
        You are a biomedical AI4Science assistant.
        Given a user query and a mechanistic path extracted from a biomedical knowledge graph,
        Your another task is to according to several given paths and their contexts,
        generate a more comprehensive context in given format with its content sticking to the entity and relations of the paths.

        The input will be a JSON payload containing the user query and a single KG path, along with the context of the path.
        The input format is as follows:
        {
            "task": "generate_entity_granular_context",
            "query": self.query,
            "key_entity": key_entity,
            "contexts": target_contexts 
        }
        You should respond ONLY with valid JSON in the following format:

        {{
        "context": [
            "{synthesized_context_1}",
            "{synthesized_context_2}",
            ...
        ]
        }}

        Do not include any text outside the JSON response.
        And you could NOT use your external knowledge to enhance the quality of the context.
        """


        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.query = query

        self.max_paths = max_paths
        self.hypotheses_per_path = hypotheses_per_path
        self.output_path = output_path 
    def serialize_path(
        self,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
    ) -> str:
        parts = []
        for i, node in enumerate(node_path):
            parts.append(node.name+":"+node.entity_type)
            if i < len(edge_path):
                edge = edge_path[i]
                parts.append(f"-[{edge.relation}]->")
        return "".join(parts)
    
    
    def _generate_context(self, key_entity: str, path_context_payloads: List[Dict[str, str]]):
        """Generate aggregated context for a single entity. Input is multiple paths and their contexts for that entity."""
        target_contexts = path_context_payloads
        if not target_contexts:
            self.logger.warning(f"[ContextGeneration] No path contexts for entity {key_entity}")
            return []

        user_payload_data = {
            "task": "generate_entity_granular_context",
            "query": self.query,
            "key_entity": key_entity,
            "contexts": target_contexts,
        }
        user_content_str = json.dumps(user_payload_data, ensure_ascii=False)
        prompt = (
    "You are a strict Knowledge Graph synthesizer. Your goal is to synthesize a high-density evidence block for the 'key_entity' based ONLY on the provided 'paths', **specifically filtering and prioritizing information that directly helps answer the 'query'**.\n\n"
    
    "STRICT CONSTRAINTS:\n"
    "1. **Query-Driven Selection (CRITICAL):** First, analyze the intent of the 'query' (e.g., looking for predictors, mechanisms, or resistance). Then, **selectively use** the paths that contribute to answering this intent. If a path is irrelevant to the query, ignore it or mention it only briefly.\n"
    "2. **Contextual Alignment:** Do not just list the paths. You must explicitly state **how** the specific path relates to the query (e.g., 'This upregulation of PD-L1 provides a mechanism for the resistance mentioned in the query...').\n"
    "3. **No External Knowledge:** Do NOT add any biological details (e.g., specific drugs, cancer types, gene names) unless they explicitly appear in the 'paths'.\n"
    "4. **Fidelity:** If a path describes a relationship, state it exactly as implied by the path, even if it contradicts general biological knowledge.\n"
    "5. **Mechanistic Precision:** Avoid generic verbs (e.g., 'affects', 'modulates'). You MUST preserve the specific biological actions found in the paths (e.g., 'degrades collagen', 'sequesters antibodies', 'phosphorylates STAT3').\n"
    "6. **Brevity:** Keep the explanation concise, dense, and directly focused on the query.\n\n"
    
    "Input Data:\n"
    f"{user_content_str}\n\n"
    "Respond ONLY with a JSON object containing a single key 'context'."
)
        try:
            raw = self.call_llm(prompt)
            obj = json.loads(raw.replace("```json", "").replace("```", ""))
            context = obj.get("context", [])
            if not isinstance(context, list):
                self.logger.warning("[ContextGeneration] context type error")
                return []
            self.logger.info(f"[ContextGeneration] entity={key_entity} entries={len(context)}")
            return context
        except Exception as e:
            self.logger.warning(
                f"[ContextGeneration] LLM call/parse failed for {key_entity}: {e}"
            )
            return []

    def _build_path_contexts(self, paths: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Organize paths with their source texts into [{path, context}] list."""
        path_contexts: List[Dict[str, str]] = []
        for path in paths[: self.max_paths]:
            contexts: str = ""
            sources: set = set()
            node_path: List[KGEntity] = path.get("nodes", []) or []
            edge_path: List[KGTriple] = path.get("edges", []) or []
            path_str = self.serialize_path(node_path=node_path, edge_path=edge_path)
            for edge in edge_path:
                if getattr(edge, "source", None):
                    sources.add(edge.source)
            for source in sources:
                try:
                    contexts += self.memory.subgraphs[source].meta.get("text", "") + "\n"
                except Exception:
                    continue
            path_contexts.append({"path": path_str, "context": contexts.strip()})
        return path_contexts

    def process(self) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]: # type: ignore
        """
        Main process:
          1) Collect paths and source contexts for each entity to get per-entity path_contexts;
          2) Call context aggregation (_generate_context) once per entity to get entity_context_text;
          3) Merge all entities' path_contexts and all entity_context_text;
          4) Call hypothesis generation (_hypothesis_generation) only once, letting LLM synthesize "all entities' all paths and contexts" to generate exactly 3 hypotheses;
          5) Output results containing global three hypotheses.
        """
        all_paths: Dict[str, List[Any]] = getattr(self.memory, "paths", {}) or {}
        if not all_paths:
            self.logger.warning("[HypothesisGeneration] no extracted paths found in memory (memory.paths missing or empty).")
            return [], {}
        global_path_contexts: List[Dict[str, str]] = []
        all_entities_context_texts: List[str] = []
        entity_context_entries_map: Dict[str, List[Dict[str, Any]]] = {}
        for key_entity, paths in all_paths.items():
            per_entity_path_contexts = self._build_path_contexts(paths)
            if not per_entity_path_contexts:
                continue
            global_path_contexts.extend(per_entity_path_contexts)
            context_entries = self._generate_context(
                key_entity=key_entity, path_context_payloads=per_entity_path_contexts
            )
            entity_context_entries_map[key_entity] = context_entries or []
            entity_context_text = "\n\n".join(
                [
                    "\n".join(
                        [
                            str(item),
                        ]
                    ).strip()
                    for item in (context_entries or [])
                ]
            ).strip()
            if entity_context_text:
                header = f"[Entity] {key_entity}"
                all_entities_context_texts.append((header + "\n" + entity_context_text).strip())
        return all_entities_context_texts,entity_context_entries_map
        