from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from Config.index import BioBertPath
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity

logger = get_global_logger()
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
"""
Entity Normalization Agent.
Performs entity normalization merging based on existing subgraph entities to reduce redundant nodes.
Input: None (retrieves subgraph entities from memory)
Output: None (updates normalized merged entities back to subgraphs in memory)
Entry point: agent.process()
Normalization process includes three steps:
1. Rule-based normalization (same subgraph + same type, deterministic string matching)
2. BioBERT similarity candidates (same subgraph, no longer restricted by type)
3. LLM adjudication merging (candidate batches parallel requests, merge operations serial execution)
"""


class EntityNormalizationAgent(Agent):
    """
    Subgraph-level Entity Normalization Agent

    Three-step process:

    1) Rule-based normalization (same subgraph + same type, deterministic string matching)
    2) BioBERT similarity candidates (same subgraph, no longer restricted by type)
    3) LLM adjudication merging (candidate batches parallel requests, merge operations serial execution)
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        sim_threshold: float = 0.9,
        max_workers: int = 5,
        llm_batch_size: int = 40,
        llm_max_workers: int = 2,
        memory: Optional[Memory] = None,
    ):
        system_prompt = """
You are a specialized Entity Normalization Agent for biomedical literature.

You are given:
- Local subgraphs built from biomedical texts.
- Entities with:
    - id, type, name, normalized_id (may be N/A),
    - aliases,
    - description: a short definition-like summary.
- High-similarity candidate pairs proposed by a BioBERT encoder.

Your job in the final step:
For each candidate pair, decide whether they refer to the SAME underlying biomedical entity
(should be merged as one node in a knowledge graph) or are DISTINCT but related entities.

Guidelines:
1. Only answer based on the given fields (name, aliases, type, description, normalized_id).
2. Prefer MERGE when:
   - Names and descriptions clearly refer to the same concept / synonym / abbreviation / variant.
   - Or one is a more specific surface form of the other, but not meaningfully distinct in KG granularity.
   - This can happen EVEN IF their types differ slightly due to annotation or schema differences
     (e.g., one tagged as "Gene" and the other as "Protein" but both clearly refer to the same molecule).
3. Prefer NO_MERGE when:
   - One is a cause, regulator, or downstream effect of the other.
   - One is a disease and the other is a biomarker, pathway, phenotype, mechanism, or drug,
     and the text clearly treats them as different roles or levels.
   - They occupy clearly different roles in a clinical or mechanistic chain (e.g., hyperglycemia vs atherosclerosis).
4. Type differences alone DO NOT automatically forbid merging, but they are a strong signal.
   Use them together with the textual evidence to decide.
5. Be precise and conservative. Do NOT merge just because text is similar or they co-occur.
6. Output STRICT JSON ONLY, no comments or extra text.
"""
        super().__init__(client, model_name, system_prompt)
        self.logger = get_global_logger()

        self.sim_threshold = float(sim_threshold)

        self.biobert_dir = BioBertPath
        self.biobert_tokenizer = None
        self.biobert_model = None
        self._load_biobert()

        self.memory = memory or get_memory()
        self.max_workers = max_workers
        self.llm_batch_size = llm_batch_size
        self.llm_max_workers = llm_max_workers

    def process(self) -> None:
        """
        Scheme B: Subgraph-level multi-threading.

        - Different subgraphs run in parallel (ThreadPoolExecutor)
        - Within each subgraph:
            1) Rule-based normalization
            2) BioBERT similarity candidates
            3) LLM batch decision-making (with its own small-scale parallelism internally)

        Parameters:
            max_workers: Number of parallel threads at subgraph level
            llm_batch_size: Number of candidate pairs per LLM request within each subgraph
            llm_max_workers: Upper limit of parallel LLM batches within each subgraph
        """

        if not self.memory.subgraphs:
            logger.info("[EntityNormalize] no subgraphs found in memory, skip.")
            return

        subgraph_items = list(self.memory.subgraphs.items())

        total_before = 0
        total_after = 0
        total_rule_merged = 0
        total_llm_merged = 0

        per_sg_stats: List[Dict[str, Any]] = []

        def worker(item):
            sg_id, sg = item
            try:
                return self._process_one_subgraph(
                    sg_id,
                    sg,
                    llm_batch_size=self.llm_batch_size,
                    llm_max_workers=self.llm_max_workers,
                )
            except Exception as e:
                logger.exception(
                    f"[EntityNormalize] subgraph={sg_id} failed in worker: {e}"
                )
                return 0, 0, 0, 0, 0, sg_id

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(worker, item)
                for item in subgraph_items
            ]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="EntityNormalize | subgraphs (parallel)",
                unit="sg",
            ):
                (
                    before,
                    after_all,
                    rule_merged,
                    llm_merged,
                    num_candidates,
                    sg_id,
                ) = fut.result()

                total_before += before
                total_after += after_all
                total_rule_merged += rule_merged
                total_llm_merged += llm_merged

                per_sg_stats.append(
                    {
                        "sg_id": sg_id,
                        "before": before,
                        "after": after_all,
                        "num_candidates": num_candidates,
                        "rule_merged": rule_merged,
                        "llm_merged": llm_merged,
                        "total_merged": rule_merged + llm_merged,
                    }
                )

        logger.info(
            f"{ANSI_GREEN}[EntityNormalize] done (parallel). "
            f"total_before={total_before}, "
            f"total_after={total_after}, "
            f"total_rule_merged={total_rule_merged}, "
            f"total_llm_merged={total_llm_merged}, "
            f"total_delta={total_before - total_after}{ANSI_RESET}"
        )

        if per_sg_stats:
            logger.info("=" * 100)
            logger.info("[EntityNormalize] Per-subgraph summary:")

            header = (
                f"{'Subgraph':<32}"
                f"{'before':>8}"
                f"{'after':>8}"
                f"{'candidates':>12}"
                f"{'rule_merge':>12}"
                f"{'llm_merge':>12}"
                f"{'total_merge':>12}"
            )
            logger.info(header)
            logger.info("-" * 100)

            for s in sorted(per_sg_stats, key=lambda x: x["sg_id"]):
                logger.info(
                    f"{s['sg_id']:<32}"
                    f"{s['before']:>8}"
                    f"{s['after']:>8}"
                    f"{s['num_candidates']:>12}"
                    f"{s['rule_merged']:>12}"
                    f"{s['llm_merged']:>12}"
                    f"{s['total_merged']:>12}"
                )

            logger.info("=" * 100)

    def _normalize_subgraph_entities(self, sg: Subgraph) -> int:
        entities: List[KGEntity] = sg.entities.all()
        if len(entities) <= 1:
            return 0

        idx_to_ent: Dict[int, KGEntity] = {i: e for i, e in enumerate(entities)}

        type_to_indices: Dict[str, List[int]] = {}
        for idx, ent in idx_to_ent.items():
            etype = ent.entity_type or "Unknown"
            type_to_indices.setdefault(etype, []).append(idx)

        merged_count = 0
        for etype, indices in type_to_indices.items():
            if len(indices) <= 1:
                continue
            merged_count += self._merge_by_string_keys_within_type(sg, idx_to_ent, indices)

        return merged_count

    def _merge_by_string_keys_within_type(
        self,
        sg: Subgraph,
        idx_to_ent: Dict[int, KGEntity],
        indices: List[int],
    ) -> int:
        if len(indices) <= 1:
            return 0

        parent = {i: i for i in indices}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        norm_map: Dict[str, int] = {}

        for idx in indices:
            ent = idx_to_ent[idx]

            surfaces: List[str] = []
            if ent.name:
                surfaces.append(ent.name)
            if ent.aliases:
                surfaces.extend(ent.aliases)

            for s in surfaces:
                norm = _normalize_str(s)
                if not norm:
                    continue

                if norm in norm_map:
                    union(idx, norm_map[norm])
                else:
                    norm_map[norm] = idx

        comp: Dict[int, List[int]] = {}
        for idx in indices:
            root = find(idx)
            comp.setdefault(root, []).append(idx)

        merged_count = 0

        for _, group in comp.items():
            if len(group) <= 1:
                continue

            leader_idx = self._choose_leader(idx_to_ent, group)
            leader = idx_to_ent[leader_idx]

            alias_set = set(_safe_list(leader.aliases))
            leader_norm = _normalize_str(leader.name)

            for idx in group:
                if idx == leader_idx:
                    continue

                ent = idx_to_ent[idx]

                if (not _has_valid_norm_id(leader)) and _has_valid_norm_id(ent):
                    leader.normalized_id = ent.normalized_id

                if ent.name:
                    alias_set.add(ent.name.strip())
                for a in _safe_list(ent.aliases):
                    if a:
                        alias_set.add(a.strip())

                if ent.entity_id in sg.entities.by_id:
                    del sg.entities.by_id[ent.entity_id]

                merged_count += 1

            cleaned_aliases: List[str] = []
            for a in alias_set:
                if not a:
                    continue
                if _normalize_str(a) == leader_norm:
                    continue
                cleaned_aliases.append(a)
            leader.aliases = sorted(set(cleaned_aliases), key=lambda x: x.lower())

        return merged_count

    def _choose_leader(
        self,
        idx_to_ent: Dict[int, KGEntity],
        group: List[int],
    ) -> int:
        """
        Select representative from entity cluster:
        1) Prefer entities with normalized_id;
        2) Prefer longer names;
        3) Prefer more aliases.
        """
        best_idx = group[0]
        best_ent = idx_to_ent[best_idx]

        def score(e: KGEntity) -> Tuple[int, int, int]:
            has_id = 1 if _has_valid_norm_id(e) else 0
            name_len = len(e.name or "")
            alias_count = len(e.aliases or [])
            return has_id, name_len, alias_count

        best_score = score(best_ent)
        for idx in group[1:]:
            ent = idx_to_ent[idx]
            sc = score(ent)
            if sc > best_score:
                best_idx = idx
                best_ent = ent
                best_score = sc

        return best_idx

    def _load_biobert(self) -> None:
        try:
            self.biobert_tokenizer = AutoTokenizer.from_pretrained(
                self.biobert_dir,
                local_files_only=True,
            )
            self.biobert_model = AutoModel.from_pretrained(
                self.biobert_dir,
                local_files_only=True,
            )
            self.biobert_model.eval()
            logger.info(f"[EntityNormalize][BioBERT] loaded from {self.biobert_dir}")
        except Exception as e:
            self.biobert_tokenizer = None
            self.biobert_model = None
            logger.info(
                f"[EntityNormalize][BioBERT] load failed ({e}), skip similarity-based suggestions."
            )

    def _encode_text(self, text: str):
        if not self.biobert_model or not self.biobert_tokenizer:
            return None
        if not text:
            return None

        with torch.no_grad():
            inputs = self.biobert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            outputs = self.biobert_model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return vec

    def _get_ent_text(self, ent: KGEntity) -> str:
        """
        Return text for BioBERT encoding:
        1) Prefer to use name
        2) If name is empty, use description instead
        """
        name = getattr(ent, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()

        desc = getattr(ent, "description", None)
        if isinstance(desc, str) and desc.strip():
            return desc.strip()

        return ""

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _collect_biobert_candidate_pairs(self, sg: Subgraph) -> List[Dict[str, Any]]:
        """
        Use BioBERT to generate candidate pairs:
        - No longer group by type, allow different types of entities to become candidate pairs
        - Similarity = maximum cosine similarity of all combinations between two entities (name + aliases)
        - Add to candidates as long as maximum similarity >= sim_threshold
        """
        entities: List[KGEntity] = sg.entities.all()
        if len(entities) <= 1 or not self.biobert_model:
            return []

        emb_cache: Dict[str, Optional[np.ndarray]] = {}

        def get_vec(text: str) -> Optional[np.ndarray]:
            if text in emb_cache:
                return emb_cache[text]
            vec = self._encode_text(text) if text else None
            emb_cache[text] = vec
            return vec

        ent_surfaces: List[List[str]] = []
        ent_vecs: List[List[Optional[np.ndarray]]] = []

        for ent in entities:
            surfs = self._get_surfaces(ent)
            ent_surfaces.append(surfs)

            vec_list: List[Optional[np.ndarray]] = []
            for s in surfs:
                vec_list.append(get_vec(s))
            ent_vecs.append(vec_list)

        all_candidates: List[Dict[str, Any]] = []
        n = len(entities)

        for i in range(n):
            surfs_i = ent_surfaces[i]
            vecs_i = ent_vecs[i]
            if not surfs_i or not vecs_i:
                continue

            for j in range(i + 1, n):
                surfs_j = ent_surfaces[j]
                vecs_j = ent_vecs[j]
                if not surfs_j or not vecs_j:
                    continue

                best_sim = -1.0
                for vi in vecs_i:
                    if vi is None:
                        continue
                    for vj in vecs_j:
                        if vj is None:
                            continue
                        sim_ij = self._cosine(vi, vj)
                        if sim_ij > best_sim:
                            best_sim = sim_ij

                if best_sim < self.sim_threshold:
                    continue

                ea, eb = entities[i], entities[j]
                type_a = getattr(ea, "entity_type", None) or "Unknown"
                type_b = getattr(eb, "entity_type", None) or "Unknown"

                all_candidates.append(
                    {
                        "subgraph_id": sg.id,
                        "entity_type": f"{type_a}|{type_b}",
                        "ent_a_type": type_a,
                        "ent_b_type": type_b,
                        "similarity": float(best_sim),   # now it's max(name+aliases)
                        "ent_a_id": ea.entity_id,
                        "ent_b_id": eb.entity_id,
                        "ent_a_name": ea.name,
                        "ent_b_name": eb.name,
                        "ent_a_normalized_id": getattr(ea, "normalized_id", "N/A"),
                        "ent_b_normalized_id": getattr(eb, "normalized_id", "N/A"),
                        "ent_a_aliases": list(getattr(ea, "aliases", []) or []),
                        "ent_b_aliases": list(getattr(eb, "aliases", []) or []),
                        "ent_a_description": self._get_ent_text(ea),
                        "ent_b_description": self._get_ent_text(eb),
                    }
                )

        return all_candidates
    def _get_surfaces(self, ent: KGEntity) -> List[str]:
        """
        Return all surface forms of an entity:
        - name
        - aliases list
        Perform simple deduplication (case-insensitive)
        """
        surfaces: List[str] = []

        name = getattr(ent, "name", None)
        if isinstance(name, str) and name.strip():
            surfaces.append(name.strip())

        aliases = getattr(ent, "aliases", None) or []
        for a in aliases:
            if isinstance(a, str) and a.strip():
                surfaces.append(a.strip())

        seen = set()
        uniq = []
        for s in surfaces:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    def _llm_decide_and_merge(
        self,
        sg: Subgraph,
        candidates: List[Dict[str, Any]],
        batch_size: int = 40,
        max_workers: int = 8,
    ) -> int:
        if not candidates:
            return 0

        batches: List[List[Dict[str, Any]]] = []
        prompts: List[str] = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i: i + batch_size]
            batches.append(batch)

            prompt_obj = {
                "subgraph_id": sg.id,
                "instructions": (
                    "For each candidate pair, decide if they are the SAME entity (MERGE) "
                    "or DIFFERENT entities (NO_MERGE). "
                    "Follow the guidelines in the system prompt. "
                    "Return ONLY a JSON array. "
                    "Each item must be:\n"
                    "{\n"
                    '  \"ent_a_id\": \"...\",\n'
                    '  \"ent_b_id\": \"...\",\n'
                    '  \"decision\": \"merge\" or \"no_merge\",\n'
                    '  \"reason\": \"short explanation\"\n'
                    "}\n"
                ),
                "candidates": batch,
            }
            prompts.append(str(prompt_obj))

        num_batches = len(prompts)
        if num_batches == 0:
            return 0

        results: List[Any] = [None] * num_batches
        max_workers = min(max_workers, num_batches)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.call_llm_json_safe, prompts[idx]): idx
                for idx in range(num_batches)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    data = fut.result()
                except Exception as e:
                    logger.info(
                        f"[EntityNormalize][LLM] subgraph={sg.id} batch_{idx} call failed: {e}"
                    )
                    data = []
                results[idx] = data

        merged_count = 0

        for batch_idx, raw in enumerate(results):
            if not raw:
                continue
            if not isinstance(raw, List):
                logger.info(
                    f"[EntityNormalize][LLM] subgraph={sg.id} "
                    f"batch_{batch_idx} invalid LLM output type: {type(raw)}"
                )
                continue

            for item in raw:
                try:
                    ent_a_id = str(item.get("ent_a_id", "")).strip()
                    ent_b_id = str(item.get("ent_b_id", "")).strip()
                    decision = str(item.get("decision", "")).strip().lower()
                except Exception:
                    continue

                if decision != "merge":
                    continue
                if not ent_a_id or not ent_b_id or ent_a_id == ent_b_id:
                    continue

                ea = sg.entities.by_id.get(ent_a_id)
                eb = sg.entities.by_id.get(ent_b_id)
                if ea is None or eb is None:
                    continue  

                leader, removed = self._merge_two_entities(sg, ea, eb)
                if removed:
                    merged_count += 1

        return merged_count

    def _process_one_subgraph(
        self,
        sg_id: str,
        sg: Subgraph,
        llm_batch_size: int = 40,
        llm_max_workers: int = 3,
    ) -> Tuple[int, int, int, int, int, str]:
        """
        Complete normalization process for a single subgraph:
        1) Rule-based normalization
        2) BioBERT similarity candidates
        3) LLM adjudication merging

        Returns: (before, after_all, rule_merged, llm_merged, num_candidates, sg_id)
        """
        entities = sg.entities.all()
        before = len(entities)
        if before == 0:
            return 0, 0, 0, 0, 0, sg_id

        merged_rule = self._normalize_subgraph_entities(sg)

        candidates: List[Dict[str, Any]] = []
        if self.biobert_model is not None:
            candidates = self._collect_biobert_candidate_pairs(sg)
        else:
            logger.info(
                f"[EntityNormalize][BioBERT] subgraph={sg_id} skipped: BioBERT model not loaded."
            )
        num_candidates = len(candidates)

        llm_merged = 0
        if candidates:

            llm_merged = self._llm_decide_and_merge(
                sg,
                candidates,
                batch_size=llm_batch_size,
                max_workers=llm_max_workers,
            )
        else:
            logger.info(
                f"[EntityNormalize][LLM] subgraph={sg_id} no candidates passed to LLM."
            )

        after_all = len(sg.entities.all())

        return before, after_all, merged_rule, llm_merged, num_candidates, sg_id

    def _merge_two_entities(
        self,
        sg: Subgraph,
        ea: KGEntity,
        eb: KGEntity,
    ) -> Tuple[KGEntity, KGEntity | None]:
        idx_to_ent = {0: ea, 1: eb}
        leader_idx = self._choose_leader(idx_to_ent, [0, 1])
        leader = idx_to_ent[leader_idx]
        follower = eb if leader is ea else ea

        if follower.entity_id == leader.entity_id:
            return leader, None

        if (not _has_valid_norm_id(leader)) and _has_valid_norm_id(follower):
            leader.normalized_id = follower.normalized_id

        alias_set = set(_safe_list(leader.aliases))
        if follower.name:
            alias_set.add(follower.name.strip())
        for a in _safe_list(follower.aliases):
            if a:
                alias_set.add(a.strip())

        leader_norm = _normalize_str(leader.name)
        cleaned_aliases = []
        for a in alias_set:
            if not a:
                continue
            if _normalize_str(a) == leader_norm:
                continue
            cleaned_aliases.append(a)
        leader.aliases = sorted(set(cleaned_aliases), key=lambda x: x.lower())

        if follower.entity_id in sg.entities.by_id:
            del sg.entities.by_id[follower.entity_id]

        return leader, follower

    def call_llm_json_safe(self, content: Any) -> Any:
        """
        Call LLM and parse as JSON array. Used in multi-threaded environment, only reads, does not modify shared state.
        """
        if isinstance(content, (dict, list)):
            prompt = (
                "You will receive a JSON-like object describing candidate entity pairs.\n"
                "Respond ONLY with a JSON array as specified.\n\n"
                f"{content}"
            )
        else:
            prompt = str(content)

        raw = self.call_llm(prompt)
        try:
            data = self.parse_json(raw)
        except Exception as e:
            logger.info(f"[EntityNormalize][LLM] parse_json failed: {e}")
            return []
        return data

def _normalize_str(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _safe_list(x):
    return x if isinstance(x, list) else []


def _has_valid_norm_id(e: KGEntity) -> bool:
    nid = getattr(e, "normalized_id", None)
    return bool(nid) and str(nid).strip().upper() != "N/A"