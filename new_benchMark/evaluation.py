from __future__ import annotations

import json
from dataclasses import dataclass
import os
import statistics
from typing import Dict, List, Optional

from numpy import average, float32
from openai import OpenAI

# ============================================================
# 0) Constants
# ============================================================

METRIC_KEYS: List[str] = [
    "Novelty",
    "Plausibility",
    "Grounding",
    "Testability",
    "Specificity",
    "SafetyEthics",
]

# ============================================================
# 1) LLM Configuration
# ============================================================

@dataclass
class LLMConfig:
    client: OpenAI
    model: str
    base_url: Optional[str] = None



open_ai_api=""
open_ai_url=""
model_name=""
_API_KEY = open_ai_api
_BASE_URL = open_ai_url
_MODEL_NAME = model_name
_LLM: Optional[LLMConfig] = None


def _get_llm() -> LLMConfig:
    """
    懒加载 LLM：
    - 第一次调用时，用 _API_KEY 创建 OpenAI client，并缓存到 _LLM
    - 后面所有调用直接复用同一个 client 和 model
    """
    global _LLM
    if _LLM is None:
        client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
        _LLM = LLMConfig(client=client, model=_MODEL_NAME)
    return _LLM


def configure_llm(client: OpenAI, model: str) -> None:
    global _LLM
    _LLM = LLMConfig(client=client, model=model)


# ============================================================
# 2) 判定是否 HIT（方向一致即可）
# ============================================================

def _hit_system_prompt() -> str:
    return (
            '''You are a biomedical "mechanistic match" judge.

    Your job is to decide whether a CANDIDATE_HYPOTHESIS is a HIT with respect to a GOLD_STATEMENT.

    You MUST base your judgment ONLY on the information in the GOLD_STATEMENT and simple logical implications.
    Do NOT use external knowledge or your own domain facts.

    --------------------------------
    1. What counts as a HIT (hit = true)
    --------------------------------
    First, read the GOLD_STATEMENT and extract its CORE MECHANISTIC CLAIM, which typically includes:

    - (A) the main exposure / intervention / risk factor / biological change
    - (B) the central mechanistic pathway (key mediator(s) or process), including direction of effect
    - (C) the primary outcome / phenotype / disease effect
    - (D) any key specificity that is central to the claim
    (e.g., a specific bacterial genus/species, a specific gene, a specific brain region, a specific cell type)

    Then judge the CANDIDATE_HYPOTHESIS:

    Set hit = true ONLY IF all of the following are satisfied:

    1) It describes the SAME main exposure/intervention (A) as the gold,
    not just a vaguely related or broader/different factor.

    2) It targets the SAME primary outcome (C) as the gold,
    with the same overall direction of effect (e.g., increase vs decrease, risk vs protection).

    3) It follows the SAME core mechanistic story (B),
    i.e., the main causal chain from exposure to outcome matches the gold,
    possibly with extra intermediate steps that are logically consistent.

    4) If the GOLD_STATEMENT emphasizes a specific subtype / entity / location as CENTRAL (D)
    (e.g., a particular bacterial genus/species, a specific brain region, a named receptor or gene),
    then the candidate MUST also mention that same specific entity/location as part of the core mechanism.
    A much more generic description (e.g., "butyrate-producing bacteria" when the gold is about a specific genus)
    should be treated as incomplete and NOT a hit.
    ---------------------------------------
        2. What is NOT a HIT (hit = false)
        ---------------------------------------
        Set hit = false if ANY of the following is true:

        - The main exposure / intervention (A) is changed, missing, or broadened so much that the specific gold factor is lost.
        - The primary outcome / phenotype (C) is changed, missing, or substantially different.
        - The direction of effect between exposure and outcome is different or unclear.
        - The mechanistic pathway (B) is substantially different or only vaguely related.
        - The candidate is much more generic and does not preserve a specific key entity/location (D) that is central in the gold.
        - The overlap is only superficial (e.g., overlaps in topic or a few words, but not in the core mechanism).

        Partial matches, generic restatements, or hypotheses that only capture part of the causal chain
        must be treated as NOT a hit.
    '''
    )


def _hit_user_prompt(hypothesis: str, gold: str) -> str:
    return f"""
GOLD STATEMENT:
{gold}
HYPOTHESIS:
{hypothesis}

Decide whether the HYPOTHESIS is roughly in the SAME DIRECTION as the GOLD STATEMENT
(broadly the same relation/phenomenon, not opposite or unrelated).

Return ONLY JSON with this exact shape:

{{
  "hit": true,
  "hit_possibility_score": float, a number between 0 and 1 (0 means not hit, 1 means fully hit),
  ""explanation": string (a brief explanation of your judgment)
}}
""".strip()


def _judge_hit(hypothesis: str, gold: str, query: str) -> float:
    llm = _get_llm()
    client = llm.client
    model = llm.model

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _hit_system_prompt()},
            {"role": "user", "content": _hit_user_prompt(hypothesis, gold)},
        ],
    )

    obj = json.loads(resp.choices[0].message.content)
    return float(obj.get("hit_possibility_score", 0.0))


# ============================================================
# 3) 判定是否幻觉（和 gold 明显无关 / 相反）
# ============================================================

def _hallucination_system_prompt() -> str:
    return """You are a specialized Scientific Mechanism Auditor.

Your goal is to quantify the hallucination rate of mechanistic claims while being FAIR across hypotheses of different lengths and granularity.
Long, detailed hypotheses must NOT be penalized simply for containing more atomic links.
You must distinguish between:
- Supported (consistent with established biomedical knowledge)
- Plausible/Innovative (reasonable but not directly confirmed; NOT hallucination)
- Hallucinated (directionally wrong, contradicts consensus, or mechanistically incoherent)

[QUERY]: {query}
[TARGET HYPOTHESIS OBJECT]: {hypothesis_json_str}

FAIR EVALUATION PROTOCOL

Step 0) Scope
- Evaluate ONLY biological/mechanistic causal links (A -> B).
- Do NOT count purely methodological statements (e.g., "use ML to integrate data", "cross-validation improves robustness") as mechanisms.
- If the text mixes "predicted outcomes" or "study design" with mechanisms, only extract mechanisms.

Step 1) Mechanism Extraction (FAIR Top-K)
1. Extract all candidate atomic causal links (A -> B) with direction.
2. Then select the TOP-K MOST CENTRAL mechanistic links for scoring, where K = 8.
   - "Central" means: the link is essential to the hypothesis' core explanatory chain and most relevant to predicting response to neoadjuvant PD-1/PD-L1 blockade in early-stage lung cancer.
   - Prefer non-redundant links; merge synonyms; avoid counting the same idea twice.
3. If fewer than K links exist, score all of them.

Step 2) Knowledge Verification (3-way classification)
For each of the selected Top-K links, label it as exactly ONE of:
- "supported": consistent with widely accepted biomedical knowledge OR supported by multiple credible lines of evidence.
- "plausible": not clearly established but biologically reasonable; could be innovative; no strong contradiction.
- "hallucinated": direction is wrong OR contradicts strong consensus OR the causal claim is mechanistically incoherent (e.g., reverses causality, conflates distinct concepts, asserts impossible direct control).

Important:
- "plausible" MUST NOT be counted as hallucination.
- Only "hallucinated" links are counted toward hallucination rate.

Step 3) Evidence & Traceability (brief)
For each Top-K link, provide:
- a short reason for its label
- optional evidence pointer types (no need to fetch links): {review, clinical, preclinical, in_vitro, database, none}
If unsure, prefer "plausible" over "hallucinated" unless there is a clear contradiction.

OUTPUT FORMAT
Return ONLY a valid JSON object:

{
  "statistics": {
    "total_mechanisms": <int>,              // MUST equal the number of Top-K links scored (<=K)
    "hallucinated_count": <int>,            // number of links labeled "hallucinated"
    "supported_count": <int>,               // number of links labeled "supported"
    "plausible_count": <int>                // number of links labeled "plausible"
  }
}

Constraints:
- total_mechanisms must be <= 8.
- hallucinated_count + supported_count + plausible_count must equal total_mechanisms.
- The "hallucinations" list must include EXACTLY the items whose label is "hallucinated".
- Do not output any non-JSON text."""

def _hallucination_user_prompt(hypothesis: str) -> str:
    return f"""
HYPOTHESIS:
{hypothesis}

Assess the hypothesis for hallucinations based on the ground truth.

Return ONLY JSON format.
""".strip()


def _judge_hallucination(hypothesis: str) -> Dict[str, float]:
    llm = _get_llm()
    client = llm.client
    model = llm.model

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _hallucination_system_prompt()},
            {"role": "user", "content": _hallucination_user_prompt(hypothesis)},
        ],
    )

    obj = json.loads(resp.choices[0].message.content)
    statistics=obj.get("statistics",{})
    hallu_count=int(statistics.get("hallucinated_count",0))
    total_count=int(statistics.get("total_mechanisms",1))
    hallu_ratio=float(hallu_count)/float(total_count)
    plausible_count=int(statistics.get("plausible_count",0))
    # mechanisms=obj.get("mechanisms",[])
    # hallu_desc=obj.get("hallucinations",[])
    # summary=obj.get("summary","")
    # print("Mechanisms:",mechanisms)
    # print("Hallucinations:",hallu_desc)
    # print("Summary:",summary)
    return {"hallu_count":hallu_count,"total_count":total_count,"hallu_ratio":hallu_ratio,"plausible_count":plausible_count}


# ============================================================
# 4) Metrics Scorer (hypothesis only; NO gold)
# ============================================================

def _metrics_system_prompt() -> str:
    return (
        "You are a strict scientific hypothesis reviewer with conservative standards.\n"
        "You will evaluate a few hypotheses based on your internal scientific knowledge (The hypotheses to evaluate whose data and generative models are up to late 2023).\n"
        "Please rate for each hypothesis separately.\n"
        "Your Goal: Provide a critical, quantitative assessment across 6 metrics.\n"
        "IMPORTANT: Be critical. High scores (5) should be rare and reserved for exceptional quality.\n"
        "\n"
        "Evaluation Criteria & Rubrics:\n"
        "1. Novelty: Is this idea surprising or non-trivial?\n"
        "   - 1: Common knowledge or trivial tautology.\n"
        "   - 5: Highly original connection previously unexplored.\n"
        "2. Plausibility: Is it scientifically sound based on established mechanisms?\n"
        "   - 1: Contradicts basic laws of physics/biology.\n"
        "   - 5: Strong mechanistic logic perfectly consistent with known science.\n"
        "3. Grounding: (Since no context is provided, evaluate 'Self-Containment')\n"
        "   - 1: Pure speculation with no referenced logic.\n"
        "   - 5: Cites specific mechanisms/papers/data implicitly within the hypothesis.\n"
        "4. Testability: Can it be falsified experimentally?\n"
        "   - 1: Vague, subjective, or metaphysical (cannot be tested).\n"
        "   - 5: Proposes specific, measurable variables and experimental conditions.\n"
        "5. Specificity: Precision of entities and relationships.\n"
        "   - 1: Broad terms (e.g., 'drugs affect cells').\n"
        "   - 5: Exact entities (e.g., 'Drug X inhibits Protein Y in Z cells at N concentration').\n"
        "6. SafetyEthics: Risk assessment.\n"
        "   - 1: Dangerous, unethical, or promotes harm.\n"
        "   - 5: Safe, ethical, and socially responsible.\n"
        "\n"
        "Output Format:\n"
        "Return ONLY valid JSON. Structure:\n"
        "{\n"
        "  \"metrics\": {\n"
        "    \"Novelty\":      { \"Score\": int},\n"
        "    \"Plausibility\": { \"Score\": int},\n"
        "    \"Grounding\":    { \"Score\": int},\n"
        "    \"Testability\":  { \"Score\": int},\n"
        "    \"Specificity\":  { \"Score\": int},\n"
        "    \"SafetyEthics\": { \"Score\": int}\n"
        "  }\n"
        "}\n"
    )


def _metrics_user_prompt(hypothesis: str) -> str:
    return f"""
HYPOTHESIS:
{hypothesis}

Return ONLY JSON:
{{
  "metrics": {{
    "Novelty":      {{ "Score": 3 }},
    "Plausibility": {{ "Score": 3 }},
    "Grounding":    {{ "Score": 3 }},
    "Testability":  {{ "Score": 3 }},
    "Specificity":  {{ "Score": 3 }},
    "SafetyEthics": {{ "Score": 3 }}
  }}
}}
""".strip()


def _score_metrics(hypothesis: str) -> Dict[str, int]:
    llm = _get_llm()
    client = llm.client
    model = llm.model

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _metrics_system_prompt()},
            {"role": "user", "content": _metrics_user_prompt(hypothesis)},
        ],
    )

    obj = json.loads(resp.choices[0].message.content)
    if  isinstance(obj,list):
        obj = obj[0]
    
    raw = obj["metrics"]
    return {k: int(raw[k].get("Score")) for k in METRIC_KEYS}


# ============================================================
# 5) Aggregation Utilities
# ============================================================

def _normalize_k(k: Optional[int], n: int) -> int:
    kk = 3 if k is None else int(k)
    if kk < 1:
        kk = 1
    if kk > 3:
        kk = 3
    if kk > n:
        kk = n
    return kk


def _compute_overall_hit(hit_flags: List[bool]) -> bool:
    return any(hit_flags)


def _compute_rank(hit_flags: List[bool]) -> int:
    """
    返回第一个 hit 出现的位置（1-based）。
    如果完全没有 hit，则返回 4（表示 >3）。
    """
    for i, h in enumerate(hit_flags):
        if h:
            return i + 1
    return 4


def _compute_hallucination_count(hallu_flags: List[bool]) -> int:
    return sum(1 for f in hallu_flags if f)


def _avg_metrics(metrics_list: List[Dict[str, int]]) -> Dict[str, float]:
    if not metrics_list:
        return {k: 0 for k in METRIC_KEYS}
    k = len(metrics_list)
    out: Dict[str, float] = {}
    for key in METRIC_KEYS:
        s = sum(int(m[key]) for m in metrics_list)
        v = s / k
        if v < 0:
            v = 0
        if v > 5:
            v = 5
        out[key] = v
    return out


# ============================================================
# 6) Public API: evaluate
# ============================================================

def evaluate(
    candidate_hypotheses: List[str],
    gold_hypothesis: str,
    query:str,
    k: Optional[int] = 3,
) -> Dict[str, object]:
    """
    对 top-k 个候选假设同时做三件事：
    1) 质量打分：返回 6 个维度的平均分 Avg Metrics（1-5）
    2) 命中 + 排名：是否有 hit（方向大致一致），以及第一个 hit 的 rank（1,2,3；没有则为 4）
    3) 幻觉数量：top-k 中被判为 hallucination 的个数（hallucination_count）

    返回 JSON dict:
      {
        "hit": bool,                 # 是否至少有一个 hit
        "rank": int,                 # 第一个 hit 的位置（1-based，无则 4）
        "hallucination_count": int,  # 幻觉数量
        "Avg Metrics": { ... }       # 六个维度的平均分
      }
    """
    kk = _normalize_k(k, len(candidate_hypotheses))
    top_hypotheses = candidate_hypotheses[:kk]

    hit_flags: List[float] = []
    hallu_rates: List[float] = []
    metrics_list: List[Dict[str, int]] = []
    hallu_count=0
    mechanism_count=0
    plausible_count=0
    for hyp in top_hypotheses:
        # hit_flags.append(_judge_hit(hyp, gold_hypothesis,query))
        hallu_res=_judge_hallucination(hyp)
        hallu_rates.append(hallu_res.get("hallu_ratio",0.0))
        hallu_count += hallu_res.get("hallu_count", 0)
        mechanism_count += hallu_res.get("total_count", 0)
        plausible_count += hallu_res.get("plausible_count",0)
        # metrics_list.append(_score_metrics(hyp))

    # 任务 1：Avg Metrics
    avg_metrics = _avg_metrics(metrics_list) if metrics_list else {k: 0.0 for k in METRIC_KEYS}

    # 任务 2：hit + rank
    overall_hit_rate=max(hit_flags) if hit_flags else 0.0

    # 任务 3：幻觉数量
    hallucination_rate = average(hallu_rates) if hallu_rates else 0.0
    hallu_count=int(hallu_count)
    mechanism_count=int(mechanism_count)
    return {
        "hit": overall_hit_rate,
        "hallucination_rate": hallucination_rate,
        "Avg Metrics": avg_metrics,
        "hallu_count": hallu_count,
        "mechanism_count": mechanism_count,
        "plausible_count":plausible_count
    }

# ============================================================