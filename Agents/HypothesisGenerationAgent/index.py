import datetime
import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from Core.Agent import Agent
from Agents.Context_Summary.index import ContextSummaryAgent
from Logger.index import get_global_logger
from Memory.index import Memory, load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class HypothesisGenerationAgent(Agent):
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
        your task is to propose plausible, mechanistic, and experimentally testable hypotheses
        that leverage the entities and relations along this path.
        Your another task is to according to several given paths and their contexts,
        generate a more comprehensive context in given format.

        The input will be a JSON payload containing the user query and a single KG path, along with the context of the path.
        The input format is as follows:
        Task 1:
        {
        "task": "generate mechanistic, testable biomedical hypotheses based on a KG path",
        "query": "user query string",
        "paths": "string",it will be given in the format of entity1:EntityType1-[relation1]->entity2:EntityType2-[relation2]->entity3:EntityType3...
        "contexts": "string"
        }
        And you could use your external knowledge to enhance the quality of the hypothesis.
       
        """
        self.contextSummaryAgent = ContextSummaryAgent(client, model_name, query, memory, max_paths, hypotheses_per_path, output_path)
        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.query = query

        self.max_paths = max_paths
        self.hypotheses_per_path = hypotheses_per_path
        self.output_path = output_path


    def _hypothesis_generation(
        self,
        *,
        query: str,
        path_contexts: List[Dict[str, str]],
        entity_context_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate hypothesis list by calling LLM once based on aggregated contexts and paths.

        path_contexts: List[{"path": str, "context": str}]
        entity_context_text: Text formed by concatenating contexts synthesized in step one
        """
        if not path_contexts:
            return []
        contexts_joined = (entity_context_text).strip()

        paths_indexed = [
            {"index": i, "path": pc.get("path", "")} for i, pc in enumerate(path_contexts)
        ]

        payload = {
            "task": "generate mechanistic, testable biomedical hypotheses using ALL provided paths and contexts",
            "query": query,
            "paths": paths_indexed,
            "contexts": contexts_joined,
            "required_hypotheses": self.hypotheses_per_path,
        }

        prompt = (
    "You are a **Distinguished Biomedical Scientist**. "
    "Your expertise spans molecular biology, physiology, and pathology. "
    "Your goal is to synthesize **novel, rigorous, and testable hypotheses** that directly **ANSWER the specific research question** provided in the `query`.\n\n"

    "## INPUT DATA EXPLANATION\n"
    "1. **'contexts' (Background Knowledge):** General scientific summaries of the entities involved. Use this to understand the consensus.\n"
    "2. **'paths' (Evidence Chain):** Specific, indexed logical facts (e.g., Entity A -> Interaction -> Entity B). These are your **immutable evidence points**.\n\n"

    "## CORE MISSION: QUERY-DRIVEN SYNTHESIS (CRITICAL)\n"
    "You are not just summarizing data; you are **solving a problem**. \n"
    "1. **Analyze the Query's Intent:** Determine what the user is really asking (e.g., Is it asking for a *mechanism*? A *biomarker*? A *therapeutic strategy*? A *cause of resistance*?).\n"
    "2. **Filter & Focus:** Select ONLY the paths and contexts that contribute to answering this specific intent. Discard valid but irrelevant scientific facts.\n"
    "3. **Synthesize the Answer:** Construct your hypotheses such that they serve as direct, evidence-based answers to the query.\n"
    "   - *Example:* If the query asks for 'Predictors', do not just describe a 'Mechanism'. You must hypothesize that 'Mechanism X makes Molecule Y a strong Predictor'.\n\n"

    "## LOGICAL REASONING TASKS\n"
    "   - **Connect the Dots:** If Path 1 shows `[A] -> causes -> [B]` and Path 2 shows `[B] -> inhibits -> [C]`, you must hypothesize that `[A] negatively regulates [C] via [B]`.\n"
    "   - **Contextualize:** Use the 'contexts' to explain *why* these interactions happen relevant to the query.\n\n"

    "## HYPOTHESIS DIVERSITY (The 'Scientist's Lens')\n"
    "Generate exactly "f"{self.hypotheses_per_path} hypotheses. Address the query from different scientific angles:\n"
    "   - **Angle 1: The Direct Mechanism:** How does the phenomenon described in the query physically happen? (Focus on binding, catalysis, structures).\n"
    "   - **Angle 2: The Regulatory/Systemic View:** What controls or modulates the phenomenon in the query? (Focus on feedback loops, upstream drivers, environment).\n"
    "   - **Angle 3: The Translational/Functional Outcome:** What is the clinical or phenotypic consequence relative to the query? (Focus on biomarkers, therapy response, disease progression).\n\n"

    "## STRICT SCIENTIFIC STANDARDS\n"
    "1. **Relevance is King:** Every hypothesis must be a direct response to the `query`. If a hypothesis is scientifically true but irrelevant to the user's question, it is a FAILURE.\n"
    "2. **Precision is Mandatory:** Ban vague terms like 'is associated with'. You MUST specify the action (e.g., 'induces apoptosis', 'allosterically hinders').\n"
    "3. **Evidence-Grounded:** Every claim must be traceable to the provided `paths`. Do not hallucinate external genes/drugs.\n\n"

    "## OUTPUT FORMAT\n"
    "Return ONLY a valid JSON object with a key 'hypotheses' containing a list of objects. Each object must have:\n"
    "- `hypothesis`: A concise sentence stating the mechanism **as an answer to the query**.\n"
    "- `mechanistic_basis`: A detailed step-by-step biological explanation (Subject -> Action -> Intermediate -> Outcome).\n"
    "- `experimental_test`: A specific method to validate this answer (e.g., 'CRISPR-Cas9 knockout', 'Clinical cohort stratification').\n"
    "- `predicted_outcome`: The expected result confirming the hypothesis.\n"
    "- `confidence`: float (0.0 to 1.0).\n"
    "- `supporting_paths`: List[str] (Verbatim copies of the path strings used).\n"
    "- `supporting_path_indices`: List[int] (The exact indices from the 'paths' input list).\n\n"

    f"USER PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}"
)

        try:
            raw = self.call_llm(prompt)
            obj = json.loads(raw.replace("```json", "").replace("```", ""))
            hyps = obj.get("hypotheses", [])
            if not isinstance(hyps, list):
                self.logger.warning(
                    f"[HypothesisGeneration] LLM returned invalid 'hypotheses' type: {type(hyps)}"
                )
                return []
            self.logger.info(
                f"[HypothesisGeneration] got {len(hyps)} hypotheses from LLM."
            )
            return hyps[: self.hypotheses_per_path]
        except Exception as e:
            self.logger.warning(
                f"[HypothesisGeneration] LLM call/parse failed: {e}"
            )
            return []
    
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
    
    def serialize_hypothesis(
        self,
        hypothesis: List[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []
        for hypo in hypothesis:
            parts.append(
                f"Title: {hypo.get('title', '')}\n"
                f"Hypothesis: {hypo.get('hypothesis', '')}\n"
                f"Mechanism Explanation: {hypo.get('mechanism_explanation', '')}\n"
                f"Experimental Suggestion: {hypo.get('experimental_suggestion', '')}\n"
                f"Relevance to Query: {hypo.get('relevance_to_query', '')}\n"
                f"Confidence: {hypo.get('confidence', 0.0)}\n"
            )
        return "\n".join(parts)
    
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

    def process(self) -> List[Dict[str, Any]]:
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
            return []
        global_path_contexts: List[Dict[str, str]] = []
        all_entities_context_texts: List[str] = []
        entity_context_entries_map: Dict[str, List[Dict[str, Any]]] = {}
        for key_entity, paths in all_paths.items():
            per_entity_path_contexts = self._build_path_contexts(paths)
            if not per_entity_path_contexts:
                continue
            global_path_contexts.extend(per_entity_path_contexts)
        all_entities_context_texts,entity_context_entries_map=self.contextSummaryAgent.process()
        combined_entity_context_text = "\n\n".join(all_entities_context_texts).strip() if all_entities_context_texts else None
        global_hypotheses = self._hypothesis_generation(
            query=self.query,
            path_contexts=global_path_contexts,
            entity_context_text=combined_entity_context_text,
        )

        results: List[Dict[str, Any]] = [
            {
                "entity": "GLOBAL",
                "paths": [pc["path"] for pc in global_path_contexts],
                "context_by_entity": entity_context_entries_map,
                "hypotheses": global_hypotheses,
            }
        ]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        default_output_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
        )

        if self.output_path:
            output_path = self.output_path
            out_dir = os.path.dirname(output_path) or "."
        else:
            out_dir = default_output_dir
            output_path = os.path.join(out_dir, f"output_hypothesis{timestamp}.json")

        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=4,
                default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o)
            )

        if self.memory:
            self.memory.add_hypothesesDir(output_path)
        return results
        