import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from evaluation import evaluate

# -----------------------------------------------
# Helpers: load gold statements from benchmark
# -----------------------------------------------
def _discover_latest_benchmark(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    cands = list(root.glob("benchmark_*.json"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _build_gold_map(benchmark_path: Path) -> Dict[int, Dict[str, str]]:
    """Return mapping: {group_idx: {hypothesis_id: statement}}"""
    with benchmark_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    groups = data.get("review_groups") or data.get("groups") or []
    out: Dict[int, Dict[str, str]] = {}
    for gi, g in enumerate(groups):
        # Support both layouts: top-level 'hypotheses' or nested under 'query'
        hyps = g.get("hypotheses")
        if hyps is None:
            q = g.get("query") or {}
            hyps = q.get("hypotheses", [])
        m: Dict[str, str] = {}
        for h in hyps or []:
            hid = h.get("hypothesis_id") or h.get("id")
            stmt = h.get("statement") or h.get("text")
            if hid and isinstance(stmt, str) and stmt.strip():
                m[str(hid)] = stmt.strip()
        out[gi] = m
    return out


@dataclass
class EvalCase:
    query: str
    groundtruth: str
    candidate_hypotheses: List[Dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: Any) -> "EvalCase":
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, dict):
            return cls("", "", [])

        return cls(
            query=payload.get("query", ""),
            groundtruth=payload.get("groundtruth", ""),
            candidate_hypotheses=payload.get("candidate_hypotheses") or payload.get("hypotheses", []),
        )

HypExtractor = Callable[[List[Any]], List[str]]
Evaluator = Callable[[List[str], str, int], Dict[str, Any]]


def load_cases(
    case_root: Path,
    group_idx: int,
    max_cases: int = 30,
    benchmark_path: Optional[Path] = Path("./benchmark_3.0.json"),
) -> List[EvalCase]:
    """Load cases, and if benchmark_path provided, override groundtruth with the
    statement from the benchmark (matched by hypothesis_id per group)."""
    gold_map_by_group: Dict[int, Dict[str, str]] = {}
    if benchmark_path is None:
        benchmark_path = _discover_latest_benchmark(case_root.parent / "benchmark_output")
    if benchmark_path and benchmark_path.exists():
        try:
            gold_map_by_group = _build_gold_map(benchmark_path)
        except Exception:
            gold_map_by_group = {}

    gold_map = gold_map_by_group.get(group_idx, {})

    cases: List[EvalCase] = []
    for i in range(max_cases):
        path = case_root.joinpath(f"{i}", f"output.json")
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            payload=payload[0]
        # Try to locate hypothesis_id in multiple plausible locations
        hypothesis_id = f"H{str(i+1).zfill(3)}" 
        if hypothesis_id and str(hypothesis_id) in gold_map:
            payload["groundtruth"] = gold_map[str(hypothesis_id)]

        cases.append(EvalCase.from_payload(payload))
    return cases
    return cases


def default_hyp_extractor(candidates: List[Any]) -> List[str]:
    texts: List[str] = []
    for c in candidates:
        if isinstance(c, str):
            t = c.strip()
            if t:
                texts.append(t)
        elif isinstance(c, dict):
            filtered = {k: v for k, v in c.items() if k != "confidence"}
            if filtered:
                try:
                    t = json.dumps(filtered, ensure_ascii=False).strip()
                except Exception:
                    t = str(filtered).strip()
                if t:
                    texts.append(t)
    return texts


def aggregate_scores(score_lists: Iterable[List[float]]) -> Dict[str, float]:
    keys: List[str] = ["Novelty", "Plausibility", "Grounding", "Testability", "Specificity", "SafetyEthics"]
    totals: Dict[str, float] = {k: 0.0 for k in keys}
    count = 0
    for scores in score_lists:
        count += 1
        for idx, key in enumerate(keys):
            if idx < len(scores):
                totals[key] += scores[idx]
    if count == 0:
        return totals
    return {k: v / count for k, v in totals.items()}


def evaluate_group(
    group_idx: int,
    case_root: Path,
    top_k: int,
    extractor: HypExtractor = default_hyp_extractor,
    evaluator: Evaluator = evaluate,
    max_cases: int = 30,
    max_workers: int = 10,
) -> Dict[str, Any]:
    cases = load_cases(case_root, group_idx, max_cases=max_cases)

    hit_rate_list: List[float] = []
    score_list: List[List[float]] = []
    hallucination_list: List[float] = []
    hallu_count_list: List[int] = []
    mechanism_count_list: List[int] = []
    plausible_count_list: List[int] = []
    def _process_one(case: EvalCase) -> Tuple[float,List[float], float,int,int,int]:
        hyps = extractor(case.candidate_hypotheses)
        result = evaluator(candidate_hypotheses=hyps, gold_hypothesis=case.groundtruth, query=case.query, k=top_k)
        hit = result.get("hit", False)
        metrics_dict = result.get("Avg Metrics") or {}
        scores = [
            float(metrics_dict.get("Novelty", 0)),
            float(metrics_dict.get("Plausibility", 0)),
            float(metrics_dict.get("Grounding", 0)),
            float(metrics_dict.get("Testability", 0)),
            float(metrics_dict.get("Specificity", 0)),
            float(metrics_dict.get("SafetyEthics", 0)),
        ]
        hallucination_rate = result.get("hallucination_rate", 0)
        hallu_count = result.get("hallu_count", 0)
        mechanism_count = result.get("mechanism_count", 0)
        plausible_count= result.get("plausible_count",0)
        return hit, scores, hallucination_rate,hallu_count,mechanism_count,plausible_count

    # Parallel over cases
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers ) as ex:
        for hit, scores, hallucination_rate,hallu_count,mechanism_count,plausible_count in tqdm(
            ex.map(_process_one, cases), total=len(cases), desc=f"group {group_idx}", unit="case"
        ):
            hit_rate_list.append(hit)
            # print("hit_rate_list:", hit_rate_list) 
            score_list.append(scores)
            # print("score_list:", score_list)
            hallucination_list.append(hallucination_rate)
            print("hallucination_list:", hallucination_list)
            hallu_count_list.append(hallu_count)
            print("hallu_count_list:", hallu_count_list)
            mechanism_count_list.append(mechanism_count)
            print("mechanism_count_list:", mechanism_count_list)
            plausible_count_list.append(plausible_count)
            print("plausible_count_list:", plausible_count_list)

    def avg(vals: List[float]) -> float:
        if not vals:
            return 0.0
        return sum(vals) / len(vals) if vals else 0.0

    llm_scores = aggregate_scores(score_list)
    avg_score = sum(llm_scores.values()) / len(llm_scores) if llm_scores else 0.0
    print("hallucination_list:", hallucination_list)
    print("hallu_count_list:", hallu_count_list)
    print("mechanism_count_list:", mechanism_count_list)
    print("plausible_count_list:", plausible_count_list)
    return {
        "group_idx": group_idx,
        "num_cases": len(cases),
        "hit_rate": avg(hit_rate_list),
        "hallucination_rate": avg(hallucination_list),
        "avg_score": avg_score,
        "novelty_score": llm_scores["Novelty"],
        "plausibility_score": llm_scores["Plausibility"],
        "grounding_score": llm_scores["Grounding"],
        "testability_score": llm_scores["Testability"],
        "specificity_score": llm_scores["Specificity"],
        "safety_ethics_score": llm_scores["SafetyEthics"],
        "plausible_count": sum(plausible_count_list),
    }


def _discover_groups(case_root: Path) -> List[int]:
    groups: List[int] = []
    if not case_root.exists():
        return groups
    for d in case_root.iterdir():
        if d.is_dir() and d.name.isdigit():
            groups.append(int(d.name))
    return sorted(groups)


def run_eval(
    case_root: Path,
    output_csv: Path,
    groups: Optional[List[int]] = None,
    top_k: int = 3,
    extractor: HypExtractor = default_hyp_extractor,
    evaluator: Evaluator = evaluate,
    max_cases: int = 30,
) -> None:
    if groups is None:
        groups = _discover_groups(case_root)

    rows: List[Dict[str, Any]] = []
    for gid in groups:
        stats = evaluate_group(
            group_idx=gid,
            case_root=case_root,
            top_k=top_k,
            extractor=extractor,
            evaluator=evaluator,
            max_cases=max_cases,
        )
        if stats["num_cases"] == 0:
            continue
        rows.append(stats)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_idx",
                "num_cases",
                "hit_rate",
                "hallucination_rate",
                "avg_score",
                "novelty_score",
                "plausibility_score",
                "grounding_score",
                "testability_score",
                "specificity_score",
                "safety_ethics_score",
                "plausible_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_batch(
    case_root: Path,
    output_csv: Path,
    groups: Optional[List[int]] = None,
    top_ks: Iterable[int] = (3,),
    extractor: HypExtractor = default_hyp_extractor,
    evaluator: Evaluator = evaluate,
    max_cases: int = 30,
) -> None:
    """Run multiple top-k settings; each top-k writes its own CSV with suffix."""
    top_k_list = list(top_ks)
    for top_k in top_k_list:
        target = output_csv if len(top_k_list) == 1 else output_csv.with_stem(f"{output_csv.stem}_top{top_k}")
        run_eval(
            case_root=case_root,
            output_csv=target,
            groups=groups,
            top_k=top_k,
            extractor=extractor,
            evaluator=evaluator,
            max_cases=max_cases,
        )


# Example usage: adjust paths and groups then execute the functions directly.
if __name__ == "__main__":
    CASE_ROOT = Path("")
    OUTPUT=Path("")
    GROUPS = [0]
    TOP_KS = [3]
    run_batch(case_root=CASE_ROOT, output_csv=OUTPUT, groups=GROUPS, top_ks=TOP_KS)
    print("ROOT:", CASE_ROOT)
        