import concurrent.futures
import json
from typing import Dict, List,Optional

from openai import OpenAI
from tqdm import tqdm

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory,Subgraph
from Store.index import get_memory
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
"""
Relationship Extraction Agent.
Extracts relationships between entities from text and forms triples.
Input: None (text retrieved from subgraph workspace meta attribute)
Output: None (stores extracted triples in subgraphs in memory)
Entry point: agent.process()
"""
class RelationshipExtractionAgent(Agent):
    def __init__(self,client:OpenAI,model_name:str,memory:Optional[Memory]=None) -> None:
        self.system_prompt="""You are a specialized Relationship Extraction Agent for biomedical knowledge graphs. Your task is to identify precise relationships between biomedical entities.

OBJECTIVE: 
1. Extract the specific **verb predicate** (e.g., "phosphorylates", "binds to").
2. Map this predicate to one of the following **BioLink Categories**:
   - POSITIVE_REGULATE (activates, increases, stimulates, upregulates)
   - NEGATIVE_REGULATE (inhibits, decreases, blocks, downregulates)
   - CAUSES (leads to, results in, triggers)
   - TREATS (cures, alleviates, prevents)
   - INTERACTS (binds, complexes with)
   - ASSOCIATED (correlated with, linked to)


INPUT: Biomedical text containing entity mentions.

EXTRACTION STRATEGY:
1. Identify biological entities.
2. Identify the specific **verb or verb phrase** connecting them.
3. Normalize the verb to ensure consistency (see NORMALIZATION RULES).

NORMALIZATION RULES (Crucial for Graph Alignment):
1. **Lemmatization:** Convert verbs to their base/infinite form.
   - "inhibited" -> "inhibit"
   - "reduces" -> "reduce"
   - "treating" -> "treat"
2. **Active Voice:** Whenever possible, formulate the relationship in active voice.
   - If text says "A is activated by B", extract: Head=B, Predicate="activate", Tail=A.
3. **Particle Inclusion:** Include necessary prepositions that define the interaction.
   - "binds to" (keep "to")
   - "associated with" (keep "with")
4. **Remove Adverbs/Modifiers:** Strip away words that describe intensity or certainty to facilitate matching.
   - "significantly inhibits" -> "inhibit"
   - "potentially causes" -> "cause"
   - "strongly downregulates" -> "downregulate"
5. **Atomic Predicates:** If multiple verbs are used, split them into separate records.
   - "binds and inhibits" -> Create two entries: one for "bind", one for "inhibit".

QUALITY CONTROLS:
- **Explicit Only:** Do not infer relationships not stated in the text.
- **No Negation:** Ignore relationships that are explicitly negated (e.g., "does not cause").
- **Precision:** The predicate must scientifically describe the interaction mechanism (e.g., prefer "phosphorylate" over "affect").

OUTPUT FORMAT (JSON):
[
  {
    "head": "Entity A",
    "relation": "verb_from_text (normalized)",
    "relation_type": "BIOLINK_CATEGORY",
    "tail": "Entity B",
    "evidence": "..."
  }
]

EXAMPLE:
Text: "Metformin significantly phosphorylates AMPK, thereby activating the pathway."
Output:
[
  {
    "head": "Metformin", 
    "relation": "phosphorylate", 
    "relation_type": "POSITIVE_REGULATE", 
    "tail": "AMPK"
  }
]
"""
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()

    def process(self) -> None:
        subgraphs = self.memory.subgraphs

        tasks = [sg for _, sg in subgraphs.items() if sg]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_subgraph, sg) for sg in tasks]

            with tqdm(total=len(futures), desc="Relationship extraction", unit="subgraph") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger = get_global_logger()
                        logger.info(f"Relationship extraction failed in concurrent processing: {str(e)}")
                    finally:
                        pbar.update(1)
    def process_subgraph(self,subgraph:Subgraph):
        if len(subgraph.relations.all())>0:
            return
        subgraph_id=subgraph.id # type: ignore
        paragraph=subgraph.meta.get("text","")
        causal_types=self.extract_existing_relation(paragraph)
        extracted_triples=[]
        for causal_type in causal_types:
            triples=self.extract_relationships(paragraph,subgraph_id,causal_type)
            for triple in triples:
                extracted_triples.append(triple)
        extracted_triples=self.remove_duplicate_triples(extracted_triples)
        subgraph=self.memory.get_subgraph(subgraph_id) # type: ignore
        if not subgraph:
            subgraph=Subgraph(subgraph_id,subgraph_id,{"text":paragraph})
        subgraph.add_relations(extracted_triples)
        self.memory.register_subgraph(subgraph)
    def extract_existing_relation(self,text:str)->List[str]:
        prompt=f"""return relationship types from provided text and return them as a list (strings only).
        Text to analyze:
        {text}
   - POSITIVE_REGULATE (activates, increases, stimulates, upregulates)
   - NEGATIVE_REGULATE (inhibits, decreases, blocks, downregulates)
   - CAUSES (leads to, results in, triggers)
   - TREATS (cures, alleviates, prevents)
   - INTERACTS (binds, complexes with)
   - ASSOCIATED (correlated with, linked to)
Example output:
[
  "positive_regulate", "negative_regulate", "causes", "treats", "interacts", "associated"
]  
   NOTE:only return the relationship types that are explicitly mentioned in the text.
   And DO NOT return any other information other than the list of relationship types.
"""
        try:
            response=self.call_llm(prompt)
            response=self._extract_json_from_markdown(response)
            relationships=self.parse_json(response)
            if not isinstance(relationships,list):
                return []
            seen=set()
            deduped: List[str]=[]
            for rel in relationships:
                if isinstance(rel,str):
                    norm=self._normalize_relation_type(rel)
                    if norm and norm not in seen:
                        seen.add(norm)
                        deduped.append(norm)
            return deduped
        except Exception as e:
            print("Error extracting causal relationships:", e)
            return []
    def extract_relationships(self,text:str,subgraph_id,causal_type:str) -> List[KGTriple]:
        """
        Relationship extraction
        Parameters:
        text: the paragraph to be extracted (function processes one paragraph at a time, may be called multiple times)
        causal_type: the causal relationship recognized from the causal_extraction agent
        Output:
        list filled with elements defined as KGTriple data structure (definition can be found in KGTriple file)
        """
        relations=causal_type
        prompt = f"""

        From the text below, identify direct relationships between entities.
        Only extract relationships that are explicitly stated or clearly implied in the text.
        Please make sure that the relationships you extract are not in conflict with the provided causal types.
        Text to analyze:
        {text}
        Existing relationship type:
        {relations}
        Return only a JSON array of relationships
        """
        try:
            response=self.call_llm(prompt)
            response=self._extract_json_from_markdown(response)
            relations_data=self.parse_json(response)
            triples: List[KGTriple]=[]
            if not isinstance(relations_data,list):
                relations_data=[]
            for rel_data in relations_data:
                if not isinstance(rel_data,dict):
                    continue
                head=(rel_data.get("head") or "").strip()
                tail=(rel_data.get("tail") or "").strip()
                relation=(rel_data.get("relation") or "").strip()
                if not head or not tail or not relation:
                    continue
                relation_type=(rel_data.get("relation_type") or "").strip()
                if relation_type:
                    relation_type=self._normalize_relation_type(relation_type)
                # fallback to provided causal_type if model omitted/unknown
                if not relation_type:
                    relation_type=self._normalize_relation_type(causal_type)
                triple=KGTriple(
                    head=head,
                    relation=relation,
                    relation_type=relation_type,
                    tail=tail,
                    confidence=None,
                    evidence=["unknown"],
                    mechanism="unknown",
                    source=subgraph_id
                )
                triples.append(triple)
        except Exception as e:
            logger=get_global_logger()
            logger.info(f"Relationship extraction failed{str(e)}")
            return []
        return triples
    

    def entities_exist(self,entity_name:str,entities:List[str])->bool:
         """
         Check if entity exists in entity recognition results
         """
         entity_lower=entity_name.lower()
         return any(entity.lower()==entity_lower for entity in entities)

    def remove_duplicate_triples(self,triples:List[KGTriple])->List[KGTriple]:
        """
        Remove duplicate triples (may have different information like confidence).
        Remove triples with lower confidence and preserve those with higher confidence.
        """
        unique_triple={}
        for triple in triples:
            triple_key=(triple.head, triple.relation, triple.tail, triple.relation_type)
            best=unique_triple.get(triple_key)
            if best is None:
                unique_triple[triple_key]=triple
            else:
                def max_conf(t: KGTriple) -> float:
                    if t.confidence and isinstance(t.confidence, list) and len(t.confidence)>0:
                        try:
                            return float(max(t.confidence))
                        except Exception:
                            return -1.0
                    return -1.0
                if max_conf(triple) > max_conf(best):
                    unique_triple[triple_key]=triple
        return list(unique_triple.values())

    def _normalize_relation_type(self, rel: str) -> str:
        """Normalize relation type strings to canonical set.
        Supported: POSITIVE_REGULATE, NEGATIVE_REGULATE, CAUSES, TREATS, INTERACTS, ASSOCIATED
        Accepts common lowercase or synonym variants.
        """
        if not isinstance(rel, str):
            return ""
        r=rel.strip().lower()
        mapping = {
            "positive_regulate":"POSITIVE_REGULATE",
            "pos_regulate":"POSITIVE_REGULATE",
            "activate":"POSITIVE_REGULATE",
            "activates":"POSITIVE_REGULATE",
            "increase":"POSITIVE_REGULATE",
            "increases":"POSITIVE_REGULATE",
            "upregulate":"POSITIVE_REGULATE",
            "upregulates":"POSITIVE_REGULATE",

            "negative_regulate":"NEGATIVE_REGULATE",
            "neg_regulate":"NEGATIVE_REGULATE",
            "inhibit":"NEGATIVE_REGULATE",
            "inhibits":"NEGATIVE_REGULATE",
            "decrease":"NEGATIVE_REGULATE",
            "decreases":"NEGATIVE_REGULATE",
            "downregulate":"NEGATIVE_REGULATE",
            "downregulates":"NEGATIVE_REGULATE",

            "causes":"CAUSES",
            "cause":"CAUSES",
            "result in":"CAUSES",
            "results in":"CAUSES",
            "lead to":"CAUSES",
            "leads to":"CAUSES",
            "trigger":"CAUSES",
            "triggers":"CAUSES",

            "treats":"TREATS",
            "treat":"TREATS",
            "cures":"TREATS",
            "cure":"TREATS",
            "alleviates":"TREATS",
            "alleviate":"TREATS",
            "prevents":"TREATS",
            "prevent":"TREATS",

            "interacts":"INTERACTS",
            "interact":"INTERACTS",
            "bind":"INTERACTS",
            "binds":"INTERACTS",
            "complexes with":"INTERACTS",

            "associated":"ASSOCIATED",
            "associate":"ASSOCIATED",
            "associated with":"ASSOCIATED",
            "correlated":"ASSOCIATED",
            "correlated with":"ASSOCIATED",
            "linked":"ASSOCIATED",
            "linked to":"ASSOCIATED",
        }
        if r in mapping:
            return mapping[r]
        if r in {"positive_regulate","negative_regulate","causes","treats","interacts","associated"}:
            return mapping.get(r, r.upper())
        if r.upper() in {"POSITIVE_REGULATE","NEGATIVE_REGULATE","CAUSES","TREATS","INTERACTS","ASSOCIATED"}:
            return r.upper()
        return r.upper()

    def _extract_json_from_markdown(self, text: str) -> str:
        """
        Additional processing: if LLM returns Markdown code block format (like ```json\n...\n```),
        extract the JSON content from it.
        """
        if not isinstance(text, str):
            return text
        
        text = text.strip()

        import re

        pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
        match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return text
