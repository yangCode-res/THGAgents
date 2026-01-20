import datetime
import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

from openai import OpenAI

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class HypothesisEditAgent(Agent):
    """
    Based on:
      - User query
      - Hypothesis feedback
      - KG paths (nodes + edges) extracted by PathExtractionAgent
      - Context information (path-related literature fragments, etc.)

    Calls LLM to modify and generate new hypotheses.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        query: str,
        memory: Optional[Memory] = None,
        original_hypotheses: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str] = None,
    ):
        system_prompt = system_prompt = """
    You are an expert biomedical AI4Science assistant specializing in **Hypothesis Refinement and Scientific Critique**.
    
    Your task is to REVISE and IMPROVE a previously generated scientific hypothesis based on specific **Feedback** provided, while ensuring strict **Alignment with the User's Original Query**.

    ## CORE OBJECTIVES
    1. **Address the Feedback:** You must explicitly fix the flaws identified in the critique (e.g., add missing mechanistic steps, fix logical gaps, suggest better controls).
    2. **Re-align with Query Intent (CRITICAL):** 
       - Review the `query` carefully. Does the `original_hypothesis` actually answer it?
       - If the query asks for a **Predictor/Biomarker**, but the hypothesis describes a **Treatment Mechanism**, you MUST reframe the revised hypothesis to focus on the predictive aspect (e.g., "High levels of X predict resistance..." instead of "Drug Y reduces X...").
       - If the query asks for a **Mechanism**, ensure the revised hypothesis provides a causal chain, not just a correlation.

    ## INPUT DATA
    The input is a JSON payload containing:
    1. `query`: The research question to be answered.
    2. `original_hypothesis_data`: The draft hypothesis.
    3. `feedback`: Specific critique to address.
    4. `contexts`: Supporting knowledge (optional).

    ## OUTPUT REQUIREMENTS
    Respond ONLY with valid JSON in the exact schema below. The content must be:
    - **Scientifically Rigorous:** Use precise terminology.
    - **Self-Contained:** The revised hypothesis should stand alone without needing the original.
    - **Query-Centric:** The `hypothesis` statement should read like a direct answer to the `query`.

    Output Schema:
    {{
      "hypotheses": [
        {{
          "hypothesis": "REVISED full hypothesis statement. It must directly answer the 'query' while incorporating the 'feedback'.",
          "mechanistic_basis": "REVISED, high-density mechanism explanation (Subject -> Action -> Intermediate -> Outcome).",
          "experimental_test": "REVISED experimental design. Make it concrete (e.g., 'Use Cre-LoxP mice', 'Analyze TCGA cohort'). Address any design flaws mentioned in feedback.",
          "predicted_outcome": "Expected qualitative/quantitative result.",
          "confidence": 0.0,
          "supporting_paths": ["List of path strings used (keep original if valid, remove if irrelevant)"],
          "supporting_path_indices": [0, 1],
          "modification_note": "Briefly explain: 1) How you addressed the feedback, and 2) How you ensured alignment with the query."
        }}
      ]
    }}

    Do not include any text outside the JSON response.
    """

        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.query = query
        self.original_hypotheses = None
        self.output_path = output_path
        
    def serialize_hypothesis(self, hypothesis: List[Dict[str, Any]]) -> str:
        lines = []
        for hypo in hypothesis:
            title = hypo.get('title', '')
            hypothesis_text = hypo.get('hypothesis', '') or hypo.get('modified_hypothesis', '')
            mechanistic_basis = hypo.get('mechanistic_basis', '') or hypo.get('mechanism_explanation', '')
            experimental_test = hypo.get('experimental_test', '') or hypo.get('experimental_suggestion', '')
            predicted_outcome = hypo.get('predicted_outcome', '')
            relevance = hypo.get('relevance_to_query', '')
            confidence = hypo.get('confidence', 0.0)
            supporting_paths = hypo.get('supporting_paths', [])
            supporting_indices = hypo.get('supporting_path_indices', [])

            lines.append(
                f"Title: {title}\n"
                f"Hypothesis: {hypothesis_text}\n"
                f"Mechanistic Basis: {mechanistic_basis}\n"
                f"Experimental Test: {experimental_test}\n"
                f"Predicted Outcome: {predicted_outcome}\n"
                f"Relevance to Query: {relevance}\n"
                f"Confidence: {confidence}\n"
                f"Supporting Paths: {', '.join(supporting_paths) if isinstance(supporting_paths, list) else str(supporting_paths)}\n"
                f"Supporting Path Indices: {', '.join(map(str, supporting_indices)) if isinstance(supporting_indices, list) else str(supporting_indices)}\n"
            )
        return "\n".join(lines)
    

    def process(self) -> List[Dict[str, Any]]:
        self.original_hypotheses = self.load_hypotheses_from_file(self.memory)
        for original_hypothesis in self.original_hypotheses:
            self.logger.info(
                f"[HypothesisGeneration] Processing original hypothesis: {original_hypothesis.get('hypothesis', '')}"
            )
        combined: List[Dict[str, Any]] = []

        def _flatten_memory_path_pmids() -> List[List[str]]:
            flat: List[List[str]] = []
            try:
                all_paths: Dict[str, List[Any]] = getattr(self.memory, "paths", {}) or {}
                for _, path_list in all_paths.items():
                    for pd in path_list or []:
                        edges = pd.get("edges", []) or []
                        seen: set = set()
                        pmids: List[str] = []
                        for e in edges:
                            try:
                                src = getattr(e, "source", None) or (e.get("source") if isinstance(e, dict) else None)
                            except Exception:
                                src = None
                            if not src:
                                continue
                            pmid = str(src).split("_")[0]
                            if pmid and pmid not in seen:
                                seen.add(pmid)
                                pmids.append(pmid)
                        flat.append(pmids)
            except Exception:
                return []
            return flat

        global_path_pmids: List[List[str]] = _flatten_memory_path_pmids()

        for hypothesis in self.original_hypotheses:
            modified_hyps = hypothesis.get("hypotheses") or []
            feedback_list = hypothesis.get("feedback") or []
            paths = hypothesis.get("paths") or []
            context_by_entity = hypothesis.get("context_by_entity") or {}

            for idx, h in enumerate(modified_hyps):
                feedback_item = feedback_list[idx] if idx < len(feedback_list) else ""

                title = h.get("title") or (h.get("hypothesis", "").split(".")[0][:80])
                original_hypothesis_data = {
                    "title": title or "",
                    "hypothesis": h.get("hypothesis", ""),
                    "mechanism_explanation": h.get("mechanistic_basis", ""),
                    "experimental_suggestion": h.get("experimental_test", ""),
                    "predicted_outcome": h.get("predicted_outcome", ""),
                    "confidence": h.get("confidence", 0.0),
                    "supporting_paths": h.get("supporting_paths", []),
                    "supporting_path_indices": h.get("supporting_path_indices", []),
                }

                payload = {
                    "task": "refine a hypothesis based on feedback",
                    "query": self.query,
                    "original_hypothesis_data": original_hypothesis_data,
                    "feedback": feedback_item,
                    "contexts": {
                        "paths": paths,
                        "context_by_entity": context_by_entity,
                    },
                }

                prompt = (
                    "Now you need to refine the following hypothesis based on the feedback provided.\n"
                    "Here is the input JSON payload:\n"
                    f"{json.dumps(payload, ensure_ascii=False)}\n"
                    "Please provide the revised hypothesis in the specified JSON format."
                )

                raw = self.call_llm(prompt).replace("```json", "").replace("```", "")
                try:
                    response = json.loads(raw)
                    refined_hyps = response.get("hypotheses", [])
                    for rh in refined_hyps:
                        combined.append(rh)
                    self.logger.info(
                        f"[HypothesisGeneration] Refined hypotheses generated for: {hypothesis.get('hypothesis', '')}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"[HypothesisGeneration][LLM process] JSON parse failed for hypothesis: {hypothesis.get('hypothesis', '')}, error: {e}"
                    )

        filtered: List[Dict[str, Any]] = []
        for rh in combined:
            idxs = rh.get("supporting_path_indices") or []
            pmid_set: List[str] = []
            seen_pmids: set = set()
            for i in idxs:
                try:
                    pmids = global_path_pmids[i]
                except Exception:
                    pmids = []
                for p in pmids:
                    if p and p not in seen_pmids:
                        seen_pmids.add(p)
                        pmid_set.append(p)
            filtered.append({
                "hypothesis": rh.get("hypothesis", ""),
                "mechanistic_basis": rh.get("mechanistic_basis", "") or rh.get("mechanism_explanation", ""),
                "experimental_test": rh.get("experimental_test", "") or rh.get("experimental_suggestion", ""),
                "predicted_outcome": rh.get("predicted_outcome", ""),
                "confidence": rh.get("confidence", 0.0),
                "supporting passage": pmid_set,
            })

        self.original_hypotheses = [
            {
                "query": self.query,
                "groundtruth": "",
                "candidate_hypotheses": filtered,
            }
        ]
        
        output_dir = self.output_path
        if not output_dir:
            output_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
            )
        output_path=os.path.join(output_dir,"output.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                self.original_hypotheses, 
                f, 
                ensure_ascii=False, 
                indent=4, 
                default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o)
            )
        if self.memory:
            self.memory.add_hypothesesDir(output_path)
        return self.original_hypotheses
    @staticmethod
    def load_hypotheses_from_file(memory: Memory) -> List[Dict[str, Any]]:
        print("memory.hypothesesdir=>",memory.hypothesesdir)
        """Read output.json file from memory.hypothesesDir"""
        with open(memory.hypothesesdir, 'r', encoding='utf-8') as f:
            return json.load(f)