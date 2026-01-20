import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from Core.Agent import Agent
from Memory.index import Memory
from Store.index import get_memory


class ReflectionAgent(Agent):
    """
    Input: hypothesis (dict)
    Output: reflection report (Strict Structured JSON Dictionary)

    Key Features:
    - Enforces PascalCase keys (e.g., SafetyEthics) for code stability.
    - Validates JSON schema before returning.
    - Returns a Python Dictionary, not a string.
    """

    @staticmethod
    def load_hypotheses_from_file(memory: Memory) -> list:
        """Read output.json file from memory.hypothesesDir"""
        with open(memory.hypothesesdir, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def extract_hypotheses(hypotheses_data: list) -> Dict[str, list]:
        """
        Extract entity -> hypotheses mapping from output.json data.
        """
        return {item.get("entity"): item.get("hypotheses", []) for item in hypotheses_data}

    _RUBRIC_ANCHORS = r"""
    You must evaluate the hypothesis using the following 6 criteria.

    For EACH criterion, give:
    - Score: integer in [1, 5]
    - Rationale: concise, evidence-grounded explanation
    - Concerns: list of concrete issues (if any)
    - Suggestions: list of actionable fixes (must be specific)

    Scoring anchors (1–5):

    (1) Novelty (newness / non-triviality)
    1. Almost no novelty: common trope; trivial recombination; no new mechanism or prediction.
    2. Weak novelty: minor variant; straightforward extension of known ideas.
    3. Moderate novelty: identifiable new linkage, but close to existing work or not clearly differentiated.
    4. Clear novelty: under-discussed mechanism/structure; differences from prior work are explicit.
    5. High novelty: non-obvious mechanism/problem framing with testable predictions and potential impact.

    (2) Plausibility (mechanistic coherence / self-consistency)
    1. Implausible: contradicts basic biomedical/clinical knowledge; broken causal chain.
    2. Low plausibility: relies on many unstated strong assumptions; causal direction confused.
    3. Partly plausible: mostly coherent but key steps missing; boundary conditions unclear.
    4. Plausible: coherent chain; assumptions stated; minor weak links remain.
    5. Highly plausible: complete, coherent mechanism; handles alternatives/counterexamples.

    (3) Grounding (evidence traceability / support)
    1. No evidence: mostly speculation; references missing or irrelevant.
    2. Weak evidence: scattered hints; key claims not supported; overgeneralization risk.
    3. Moderate: some claims supported; gaps remain; claim↔evidence mapping imprecise.
    4. Strong: most key claims supported by cited evidence/data/path reasoning.
    5. Very strong: complete traceable chain (claim↔evidence↔path/data) with uncertainty calibration.

    (4) Testability (operational verification)
    1. Not testable: too abstract; depends on unavailable data/conditions; no measurable endpoints.
    2. Hard to test: high cost/uncontrollable; endpoints/metrics unclear.
    3. Partly testable: directions exist but controls/metrics/expected outcomes are vague.
    4. Testable: executable experiments/analyses with clear controls and discriminative outcomes.
    5. Highly testable: multiple independent validation routes with clear metrics, expected outcomes, reproducibility.

    (5) Specificity (precision / boundary conditions)
    1. Too vague: only “may relate/affect”; lacks actor/condition/direction/scope.
    2. Somewhat vague: partial objects; missing constraints (context, timing, magnitude).
    3. Moderate: actors and direction stated; parameters/scenarios/exclusions incomplete.
    4. Specific: “who under what conditions via what mechanism affects what”; scope bounded.
    5. Very specific: directly translatable to an experimental hypothesis with constraints and non-applicable cases.

    (6) Safety & Ethics (risk, compliance, responsible framing)
    1. High risk / non-compliant: unsafe procedures, harmful advice, or unethical guidance.
    2. Significant risk: may enable unsafe actions; ignores IRB/privacy/biosafety constraints.
    3. Manageable but needs additions: add compliance prerequisites (IRB, de-identification, biosafety).
    4. Safe: risks identified and mitigations provided; feasible under compliance framework.
    5. Very safe: explicit compliance path + clear limitations + safer alternatives.

    General requirements:
    - Be strict: do not inflate scores.
    - Suggestions must be executable (e.g., “Add boundary condition X”, “Define metric Y”).
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        memory: Optional[Memory] = None,
        context: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = 5,
    ):

        json_template = json.dumps({
            "Novelty": {
                "score": "X/5",
                "rationale": "...",
                "concerns": ["concern 1", "concern 2"],
                "suggestions": ["suggestion 1", "suggestion 2"]
            },
            "Plausibility": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Grounding": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Testability": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Specificity": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "SafetyEthics": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "OverallSummary": {
                "Strengths": ["..."],
                "Weaknesses": ["..."],
                "PriorityMustFix": ["..."],
                "NiceToFix": ["..."],
                "RiskFlags": ["..."],
                "EditInstructions": ["..."]
            }
        }, indent=2)
        system_prompt = f"""You are a rigorous scientific reviewer.

        {self._RUBRIC_ANCHORS}

        You will receive:
        - A hypothesis (as JSON text)
        - Optional context (as JSON text)

        IMPORTANT OUTPUT RULES:
        1. Output MUST be a valid JSON object.
        2. Do NOT output Markdown code fences (like ```json).
        3. The JSON structure MUST exactly match the following example:

        {json_template}
        """
        super().__init__(client, model_name, system_prompt)

        self.memory = memory or get_memory()
        self.context = context or {}
        self.hypotheses_data = None
        self.max_workers: int = 5
    def process(self) -> list:
        """
        Evaluate modified_hypotheses for each entity (multi-threaded) and save back to output.json

        Args:
            max_workers: Maximum number of threads, default 5
        """
        self.hypotheses_data=self.load_hypotheses_from_file(self.memory)
        max_workers=self.max_workers
        tasks: List[Tuple[int, int, Dict[str, Any]]] = []
        for item_idx, item in enumerate(self.hypotheses_data):
            base_hypotheses = item.get("hypotheses", [])
            item["feedback"] = [None] * len(base_hypotheses)
            for hyp_idx, hypothesis in enumerate(base_hypotheses):
                tasks.append((item_idx, hyp_idx, hypothesis))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.call_for_each_hypothesis, task[2]): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                item_idx, hyp_idx, hypothesis = future_to_task[future]
                try:
                    result = future.result()
                    self.hypotheses_data[item_idx]["feedback"][hyp_idx] = result
                    print(f"✓ Completed: {hypothesis.get('hypothesis', 'N/A')[:50]}...")
                except Exception as e:
                    print(f"✗ Error for hypothesis: {hypothesis.get('hypothesis', 'N/A')}: {e}")
                    self.hypotheses_data[item_idx]["feedback"][hyp_idx] = {"error": str(e)}

        self._save_to_file()
        
        return self.hypotheses_data

    def _save_to_file(self) -> None:
        """Save updated data back to output.json"""
        with open(self.memory.hypothesesdir, 'w', encoding='utf-8') as f:
            json.dump(self.hypotheses_data, f, ensure_ascii=False, indent=4)

    def call_for_each_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent logic:
        1. Calls LLM.
        2. Cleans and extracts JSON from response.
        3. Validates schema.
        4. Returns structured Dict.
        """
        hypothesis_text = json.dumps(hypothesis, ensure_ascii=False, indent=2)

        user_message = (
            "Hypothesis (JSON):\n"
            f"{hypothesis_text}\n\n"
            "Context (JSON):\n"
            "Review Task:\n"
            "Provide a critique in strict JSON format based on the rubric.\n"
            "Output JSON ONLY. No preamble. No markdown."
        )

        raw_response = self.call_llm(user_message)

        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError("Empty reflection output from model.")

        cleaned_json_str = self._clean_and_extract_json(raw_response)
        try:
            data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON Parse Error: {e}\nRaw Output: {raw_response}")

        self._validate_schema(data)
        return data

    def get_scores_only(self) -> Dict[str, str]:
        if self.hypotheses_data is None:
            try:
                self.hypotheses_data = self.load_hypotheses_from_file(self.memory)
            except Exception as e:
                raise ValueError(f"hypotheses_data is not initialized and failed to load: {e}")

        scores_dict={}
        score_keys = ["Novelty", "Plausibility", "Grounding", "Testability", "Specificity", "SafetyEthics"]
        for item_idx, item in enumerate(self.hypotheses_data):
            base_hypotheses = item.get("hypotheses", [])
            for hyp_idx, hypothesis in enumerate(base_hypotheses):
                full_result = self.call_for_each_hypothesis(hypothesis)
                scores = {key: full_result[key]["score"] for key in score_keys}
                scores["hypothesis"] = hypothesis
                print(f"Scores for {item.get('entity')}: {scores}")
                scores_dict[item.get("entity")] = scores
        return scores_dict

    def _clean_and_extract_json(self, text: str) -> str:
        """
        Helper to strip markdown code blocks from LLM output.
        """
        text = text.strip()
        text = re.sub(r"^```(json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text
    def _validate_schema(self, data: Dict[str, Any]) -> None:
        """
        Check if the returned Dict contains all necessary Keys and correct types.
        This prevents RevisionAgent from crashing due to missing fields.
        """
        required_criteria = [
            "Novelty", "Plausibility", "Grounding", 
            "Testability", "Specificity", "SafetyEthics", 
            "OverallSummary"
        ]

        for key in required_criteria:
            if key not in data:
                raise ValueError(f"ReflectionAgent Output Missing required top-level key: '{key}'")

        criteria_subfields = ["score", "rationale", "concerns", "suggestions"]
        
        for key in required_criteria[:-1]: # Exclude OverallSummary from this loop
            item = data[key]
            if not isinstance(item, dict):
                raise ValueError(f"Key '{key}' must be a dictionary.")
            
            for sub in criteria_subfields:
                if sub not in item:
                    raise ValueError(f"Key '{key}' missing subfield '{sub}'.")
            
