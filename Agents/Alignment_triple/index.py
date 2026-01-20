import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

from Config.index import BioBertPath
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

Embedding = List[float]
"""
Entity Alignment Triple Agent.
Identifies entity alignment triples representing the same real-world entities across different subgraphs
based on entity embeddings and text descriptions within subgraphs.
Input: None (retrieves subgraph and entity information from memory)
Output: None (stores identified entity alignment triples in memory's alignment storage)
Entry point: agent.process()
"""
class AlignmentTripleAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt = """
You are an expert in biomedical knowledge graph entity alignment.

You will receive a single JSON string as user input, with fields such as:
- "source_subgraph", "source_entity_id", "source_entity_name", "source_entity_text"
- "target_subgraph", "target_subgraph_text"
- "candidates": a list of objects { "id": ..., "name": ... }
- "instruction": a natural language description of the task

Your task:
1. Parse the JSON input.
2. Compare the source entity with all candidate entities from the target subgraph.
3. Decide which candidates refer to the SAME real-world biomedical entity as the source entity.

Output format (VERY IMPORTANT):
- You MUST respond with STRICT JSON only.
- The JSON must have exactly one top-level key "keep".
- "keep" must be a list of candidate ids (strings) that should be kept.
- Example: {"keep": ["cand1", "cand3"]}

Rules:
- If no candidate should be aligned with the source entity, return {"keep": []}.
- Do NOT add any other keys, text, comments, or explanations.
- Do NOT change, rename, or invent candidate ids.
- The response must be valid JSON and parseable by a standard JSON parser.
"""
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        self.subgraph_entity_embeddings: Dict[str, Dict[str, Embedding]] = {}
        self.biobert_dir = BioBertPath
        self.biobert_model=None
        self.biobert_tokenizer=None
        self._load_biobert()
        self.subgraph_adj: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}
        self.subgraph_hypergraphs: Dict[str, Dict[str, Any]] = {}
        self.entity_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.subgraph_entities: Dict[str, Dict[str, KGEntity]] = {}
        self.entity2subgraph: Dict[str, str] = {}
    def process(self) -> None:
        for sg_id, sg in self.memory.subgraphs.items():
            if sg.entities.all()==[]:
                self.logger.info(f"AlignmentTripleAgent: Subgraph {sg_id} has no entities, skipping.")
                continue
            if sg.get_relations()==[]:
                self.logger.info(f"AlignmentTripleAgent: Subgraph {sg_id} has no relationships, skipping.")
                continue
            ent_embeds: Dict[str, Embedding] = {}
            ent_map: Dict[str, KGEntity] = {}
            for ent in sg.entities.all():
                text = ent.name or ent.description or ent.normalized_id
                embedding = self._encode_text(text)  
                ent_embeds[ent.entity_id] = embedding
                ent_map[ent.entity_id] = ent 
                self.entity2subgraph[ent.entity_id] = sg_id
           
            self.subgraph_entity_embeddings[sg_id] = ent_embeds
            self.subgraph_entities[sg_id] = ent_map
            id2idx, adj = self.build_adj_for_subgraph(sg)
            self.subgraph_adj[sg_id] = (adj, id2idx)
            H, center_ids, hyperedge_embeds = self.build_hypergraph_for_subgraph(
                sg, id2idx, adj, ent_embeds
            )
            self.subgraph_hypergraphs[sg_id] = {
                "H": H,                      # n × m incidence matrix
                "center_ids": center_ids,    # len = m
                "hyperedge_embeddings": hyperedge_embeds,  # m × d 
            }
            # Quick check logging
            self.logger.info(
                f"[Adjacency] subgraph={sg_id}, |V|={adj.shape[0]}, |E|={int(adj.sum())}"
            )
            self.logger.info(
                f"[Hypergraph] subgraph={sg_id}, |V|={H.shape[0]}, |E_h|={H.shape[1]}"
            )
        self.propagate_embeddings_with_hypergraph(alpha=0.5)
        self.build_entity_alignment(sim_threshold=0.5, top_k=5)

    def _llm_filter_for_one_pair(
        self,
        src_sg_id: str,
        src_eid: str,
        tgt_sg_id: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        src_entity = self._find_entity_in_subgraph(src_sg_id, src_eid)
        src_text =self.memory.subgraphs.get(src_sg_id).get_meta().get("text", "")

        tgt_text = self.memory.subgraphs.get(tgt_sg_id).get_meta().get("text", "")
        tgt_items = []
        for cand in candidates:
            tgt_eid = cand["target_entity"]
            tgt_entity = self._find_entity_in_subgraph(tgt_sg_id, tgt_eid)
            tgt_items.append(
                {
                    "id": tgt_eid,
                    "name":tgt_entity.get_name(),
                }
            )

        user_payload = {
            "source_subgraph": src_sg_id,
            "source_entity_id": src_eid,
            "source_entity_text": src_text,
            "source_entity_name": src_entity.get_name() if src_entity else "",
            "target_subgraph": tgt_sg_id,
            "target_subgraph_text": tgt_text,
            "candidates": tgt_items,
            "instruction": (
                "Read the source entity and the candidate entities. "
                "Decide which candidates refer to the SAME real-world biomedical entity "
                "as the source. Return a JSON object with a single key 'keep', "
                "whose value is a list of candidate ids to keep. "
                "Example: {\"keep\": [\"cand1\", \"cand3\"]}."
            ),
        }

        response = self.call_llm(prompt=json.dumps(user_payload, ensure_ascii=False))
        self.logger.debug(
            f"[EntityAlign-LLM] raw response for ({src_sg_id}, {src_eid}, {tgt_sg_id}) = {response[:400]!r}"
        )

        keep_ids: List[str] = []
        try:
            parsed = self.parse_json_alignment(response)
            if isinstance(parsed, dict):
                keep_ids = [str(x) for x in parsed.get("keep", [])]
            elif isinstance(parsed, list):
                keep_ids = [str(x) for x in parsed]
            else:
                keep_ids = []
        except Exception as e:
            self.logger.warning(
                f"[EntityAlign-LLM] parse JSON failed for ({src_sg_id}, {src_eid}, {tgt_sg_id}): "
                f"raw_content={response!r}, error={e}"
            )
            return []

        id_set = set(keep_ids)
        kept: List[Dict[str, Any]] = []
        for cand in candidates:
            if cand["target_entity"] in id_set:
                kept.append(
                    {
                        "target_subgraph": tgt_sg_id,
                        "target_entity": cand["target_entity"],
                        "similarity": cand["similarity"],
                        "llm_agree": True,
                    }
                )

        return kept
    def _run_llm_alignment_parallel(
        self,
        candidate_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]],
        max_workers: int = 8,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Input: candidate_alignment (filtered by cosine similarity only)
        Output: refined_alignment (filtered by LLM)
        Parallel granularity: each (src_sg_id, src_eid, tgt_sg_id) pair calls LLM separately.
        """
        jobs: List[Tuple[str, str, str, List[Dict[str, Any]]]] = []

        for src_sg_id, ent_map in candidate_alignment.items():
            for src_eid, matches in ent_map.items():
                by_tgt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for m in matches:
                    tgt_sg_id = m["target_subgraph"]
                    by_tgt[tgt_sg_id].append(m)
                for tgt_sg_id, cand_list in by_tgt.items():
                    if not cand_list:
                        continue
                    jobs.append((src_sg_id, src_eid, tgt_sg_id, cand_list))

        self.logger.info(f"[EntityAlign-LLM] total jobs={len(jobs)}")
        merged_results: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        if not jobs:
            return {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    self._llm_filter_for_one_pair,
                    src_sg_id,
                    src_eid,
                    tgt_sg_id,
                    cand_list,
                ): (src_sg_id, src_eid, tgt_sg_id)
                for (src_sg_id, src_eid, tgt_sg_id, cand_list) in jobs
            }

            for future in as_completed(future_to_job):
                src_sg_id, src_eid, tgt_sg_id = future_to_job[future]
                try:
                    kept = future.result()  # List[Dict]
                except Exception as e:
                    self.logger.warning(
                        f"[EntityAlign-LLM] job ({src_sg_id}, {src_eid}, {tgt_sg_id}) "
                        f"failed: {e}"
                    )
                    kept = []

                if kept:
                    merged_results[(src_sg_id, src_eid)].extend(kept)

        # {src_sg_id: {src_eid: [..]}} 
        refined_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for (src_sg_id, src_eid), lst in merged_results.items():
            refined_alignment.setdefault(src_sg_id, {})[src_eid] = lst

        return refined_alignment
    def build_entity_alignment(
        self,
        sim_threshold: float = 0.7,
        top_k: int = 10,
        max_workers: int = 6,
    ) -> None:
        """
        Sequential fusion alignment (no longer updates embedding values, only expands canonical entity set):

        1) Select the "first subgraph" as anchor/canonical;
        2) Treat it as "fused subgraph F", with its entities as the initial canonical set;
        3) Iterate through remaining subgraphs S_i sequentially:
            - Dynamically check sizes of |F| and |S_i|:
              * If |F| <= |S_i|: use F's entities as source;
              * Otherwise: use S_i's entities as source;
            - Perform cosine similarity candidate filtering once;
            - Call _run_llm_alignment_parallel for refined filtering on this (F, S_i) pair;
            - Map alignment results uniformly to anchor_sg_id's canonical entities;
            - **Add unaligned entities from S_i to canonical_raw_embeds (entity set grows)**.
        4) Final self.entity_alignment uses only anchor_sg_id as key, structure:
           {anchor_sg_id: {canonical_eid: [{target_subgraph, target_entity, similarity, llm_agree}, ...]}}

        Note: embeddings are used only for similarity calculation, not updated numerically in this function.
        """

        def _normalize_embeds(embeds: Dict[str, Embedding]) -> Tuple[List[str], np.ndarray]:
            """Convert {entity_id: vector} to (id_list, L2 normalized matrix)."""
            ids: List[str] = []
            vecs: List[np.ndarray] = []
            for eid, emb in embeds.items():
                if emb is None:
                    continue
                v = np.asarray(emb, dtype=np.float32)
                if v.size == 0:
                    continue
                norm = np.linalg.norm(v)
                if norm == 0.0:
                    continue
                v = v / norm
                ids.append(eid)
                vecs.append(v)
            if not ids:
                return [], np.zeros((0, 1), dtype=np.float32)
            mat = np.stack(vecs, axis=0)
            return ids, mat

        if not self.subgraph_entity_embeddings:
            return
        eid_to_sg: Dict[str, str] = {}
        for sg_id, ent_map in self.subgraph_entity_embeddings.items():
            for eid in ent_map.keys():
                eid_to_sg[eid] = sg_id
        all_sg_ids = list(self.subgraph_entity_embeddings.keys())
        anchor_sg_id = all_sg_ids[0]
        self.logger.info(f"[EntityAlign] anchor_subgraph={anchor_sg_id}")

        canonical_raw_embeds: Dict[str, Embedding] = dict(
            self.subgraph_entity_embeddings[anchor_sg_id]
        )
        global_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for tgt_sg_id in all_sg_ids[1:]:
            if tgt_sg_id not in self.subgraph_entity_embeddings:
                continue
            tgt_raw_embeds = self.subgraph_entity_embeddings[tgt_sg_id]
            if not tgt_raw_embeds:
                continue
            canon_ids, canon_mat = _normalize_embeds(canonical_raw_embeds)
            tgt_ids, tgt_mat = _normalize_embeds(tgt_raw_embeds)

            if len(canon_ids) == 0 or len(tgt_ids) == 0:
                continue
            self.logger.info(
                f"[EntityAlign] step align: F(anchor={anchor_sg_id}, |F|={len(canon_ids)}) "
                f"<-> S({tgt_sg_id}, |S|={len(tgt_ids)})"
            )
            candidate_alignment_step: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
            source_is_canonical = len(canon_ids) <= len(tgt_ids)
            if source_is_canonical:
                for i, src_eid in enumerate(canon_ids):
                    src_vec = canon_mat[i : i + 1, :]  # [1, d]
                    sims = (tgt_mat @ src_vec.T).reshape(-1)  # [n_tgt]
                    idx_sorted = np.argsort(-sims)

                    matches_for_entity: List[Dict[str, Any]] = []
                    kept_count = 0
                    for idx in idx_sorted:
                        sim = float(sims[idx])
                        if sim < sim_threshold:
                            break
                        matches_for_entity.append(
                            {
                                "target_subgraph": tgt_sg_id,
                                "target_entity": tgt_ids[idx],
                                "similarity": sim,
                            }
                        )
                        kept_count += 1
                        if kept_count >= top_k:
                            break
                    if matches_for_entity:
                        src_sg_id = eid_to_sg.get(src_eid, anchor_sg_id)
                        candidate_alignment_step[src_sg_id][src_eid] = matches_for_entity

            else:
                src_sg_id = tgt_sg_id
                ent_map: Dict[str, List[Dict[str, Any]]] = {}
                for j, src_eid in enumerate(tgt_ids):
                    src_vec = tgt_mat[j : j + 1, :]  # [1, d]
                    sims = (canon_mat @ src_vec.T).reshape(-1)  # [n_canon]
                    idx_sorted = np.argsort(-sims)
                    matches_for_entity: List[Dict[str, Any]] = []
                    kept_count = 0
                    for idx in idx_sorted:
                        sim = float(sims[idx])
                        if sim < sim_threshold:
                            break
                        matches_for_entity.append(
                            {
                                "target_subgraph": anchor_sg_id,
                                "target_entity": canon_ids[idx],
                                "similarity": sim,
                            }
                        )
                        kept_count += 1
                        if kept_count >= top_k:
                            break
                    if matches_for_entity:
                        ent_map[src_eid] = matches_for_entity
                if ent_map:
                    candidate_alignment_step[src_sg_id] = ent_map
            if not candidate_alignment_step:
                self.logger.info(
                    f"[EntityAlign] step ({anchor_sg_id} vs {tgt_sg_id}) has no cosine candidates above threshold."
                )
                for eid, vec in tgt_raw_embeds.items():
                    canonical_raw_embeds.setdefault(eid, vec)
                self.logger.info(
                    f"[EntityAlign] merge-only step: F size -> {len(canonical_raw_embeds)} after union with {tgt_sg_id}"
                )
                continue
            refined_step = self._run_llm_alignment_parallel(
                candidate_alignment_step,
                max_workers=max_workers,
            )
            if not refined_step:
                self.logger.info(
                    f"[EntityAlign-LLM] step ({anchor_sg_id} vs {tgt_sg_id}) LLM kept nothing."
                )
                anchor_ent_map = self.subgraph_entities.get(anchor_sg_id, {})
                tgt_ent_map = self.subgraph_entities.get(tgt_sg_id, {})
                if tgt_ent_map:
                    for eid, ent in tgt_ent_map.items():
                        if eid not in anchor_ent_map:
                            anchor_ent_map[eid] = ent
                    self.subgraph_entities[anchor_sg_id] = anchor_ent_map
                for eid, vec in tgt_raw_embeds.items():
                    if eid not in canonical_raw_embeds:
                        canonical_raw_embeds[eid] = vec
                self.logger.info(
                    f"[EntityAlign] merge-only step (LLM-none): F size -> {len(canonical_raw_embeds)} after union with {tgt_sg_id}"
                )
                continue
            matched_tgt_ids: set[str] = set()
            for src_sg_id, ent_map in refined_step.items():
                for src_eid, kept_list in ent_map.items():
                    if not kept_list:
                        continue
                    global_alignment.setdefault(src_sg_id, {}).setdefault(src_eid, []).extend(kept_list)
            anchor_ent_map = self.subgraph_entities.get(anchor_sg_id, {})
            tgt_ent_map = self.subgraph_entities.get(tgt_sg_id, {})
            if tgt_ent_map:
                for eid, ent in tgt_ent_map.items():
                    if eid not in anchor_ent_map:
                        anchor_ent_map[eid] = ent
                self.subgraph_entities[anchor_sg_id] = anchor_ent_map
            all_tgt_ids = set(tgt_raw_embeds.keys())
            unmatched_ids = all_tgt_ids - matched_tgt_ids
            for eid in unmatched_ids:
                if eid not in canonical_raw_embeds:
                    canonical_raw_embeds[eid] = tgt_raw_embeds[eid]
            self.logger.info(
                f"[EntityAlign] step ({anchor_sg_id} vs {tgt_sg_id}) merged: "
                f"matched_in_S={len(matched_tgt_ids)}, "
                f"unmatched_added={len(unmatched_ids)}, "
                f"|F| now={len(canonical_raw_embeds)}"
            )
        normalized_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for src_sg_id, ent_map in global_alignment.items():
            for src_eid, match_list in ent_map.items():
                if not match_list:
                    continue
                real_src_sg = self.entity2subgraph.get(src_eid, src_sg_id)
                for m in match_list:
                    tgt_eid = m.get("target_entity")
                    real_tgt_sg = self.entity2subgraph.get(
                        tgt_eid,
                        m.get("target_subgraph"), 
                    )
                    new_m = dict(m)
                    new_m["target_subgraph"] = real_tgt_sg
                    normalized_alignment.setdefault(real_src_sg, {}).setdefault(
                        src_eid, []
                    ).append(new_m)
        self.entity_alignment = normalized_alignment
        self.subgraph_entity_embeddings[anchor_sg_id] = canonical_raw_embeds
        self.memory.alignments.save_from_alignment_dict(normalized_alignment)
        self.logger.info(
            f"[EntityAlign-LLM] Done. anchor_subgraph={anchor_sg_id}, "
            f"#src_subgraphs={len(normalized_alignment)}, "
            f"#total_pairs={sum(len(v) for v in normalized_alignment.values())}"
        )
    def propagate_embeddings_with_hypergraph(self, alpha: float = 0.5):
        """
        Use hypergraph (H) to perform parameter-free convolution propagation on entity embeddings
        in each subgraph, and update self.subgraph_entity_embeddings.

        alpha: residual weight, final X_final = alpha * X + (1 - alpha) * X_propagated
        """
        for sg_id, hyper_info in self.subgraph_hypergraphs.items():
            H = hyper_info["H"]              # [N, M] numpy

            adj, id2idx = self.subgraph_adj.get(sg_id, (None, None))
            ent_embeds = self.subgraph_entity_embeddings.get(sg_id, {})
            n = len(id2idx)
            sample_vec = next(iter(ent_embeds.values()))
            d = sample_vec.shape[0]
            X = np.zeros((n, d), dtype=np.float32)
            for eid, idx in id2idx.items():
                vec = ent_embeds.get(eid, None)
                X[idx] = np.array(vec, dtype=np.float32)
            dv = H.sum(axis=1, keepdims=True)  # [N, 1]
            de = H.sum(axis=0, keepdims=True)  # [1, M]
            dv[dv == 0] = 1.0
            de[de == 0] = 1.0
            X_norm = X / dv                    # [N, d]
            Xe = H.T @ X_norm                  # [M, d]
            Xe = Xe / de.T                     # [M, d]
            X_propagated = H @ Xe              # [N, d]
            X_final = alpha * X + (1.0 - alpha) * X_propagated
            updated_ent_embeds: Dict[str, Embedding] = {}
            for eid, idx in id2idx.items():
                updated_ent_embeds[eid] = X_final[idx]
            self.subgraph_entity_embeddings[sg_id] = updated_ent_embeds
            self.logger.info(
                f"[HypergraphProp] subgraph={sg_id}, alpha={alpha}, "
                f"updated {len(updated_ent_embeds)} entity embeddings."
            )
    def build_hypergraph_for_subgraph(
        self,
        subgraph: Subgraph,
        id2idx: Dict[str, int],
        adj: np.ndarray,
        ent_embeds: Dict[str, Embedding],
    ):
        """
        Build "entity-centered hypergraph" based on entity graph (adj):
        - Each entity i serves as a center, forming a hyperedge e_i
        - e_i contains: i itself + all neighbor nodes connected to i
        Returns:
            H: np.ndarray, shape = [n_nodes, n_hyperedges]
            center_ids: List[str], center entity id for each hyperedge
            hyperedge_embeds: np.ndarray[m, d] (if embeddings exist, otherwise None)
        """
        
        n = adj.shape[0]
        idx2id = {idx: eid for eid, idx in id2idx.items()}
        hyperedges: List[List[int]] = []
        center_ids: List[str] = []

        for center_idx in range(n):
            center_eid = idx2id[center_idx]
            neighbor_idxs = np.nonzero(adj[center_idx])[0].tolist()
            nodes = [center_idx] + neighbor_idxs
            nodes = sorted(set(nodes))
            hyperedges.append(nodes)
            center_ids.append(center_eid)
        m = len(hyperedges)
        H = np.zeros((n, m), dtype=np.float32)
        for e_idx, nodes in enumerate(hyperedges):
            H[nodes, e_idx] = 1.0
        hyperedge_embeds = None
        if ent_embeds:
            sample_vec = next(iter(ent_embeds.values()))
            d = sample_vec.shape[0] if hasattr(sample_vec, "shape") else len(sample_vec)
            hyperedge_embeds = np.zeros((m, d), dtype=np.float32)

            for e_idx, center_eid in enumerate(center_ids):
                vec = ent_embeds.get(center_eid, None)
                hyperedge_embeds[e_idx] = np.array(vec, dtype=np.float32)

        return H, center_ids, hyperedge_embeds
    def build_adj_for_subgraph(self, subgraph: Subgraph):
        """
        For a given subgraph (Subgraph object in Memory), return:
        - id2idx: entity_id -> row/column index
        - adj: adjacency matrix (numpy.ndarray, shape = [n_entities, n_entities])
        """
        entities = subgraph.get_entities()   
        relations = subgraph.get_relations() 
        id2idx: Dict[str, int] = {}
        for idx, ent in enumerate(entities):
            eid = ent.get_id()
            id2idx[eid] = idx
        n = len(id2idx)
        adj = np.zeros((n, n), dtype=int)
        for rel in relations:
            subj = rel.get_subject()   
            obj  = rel.get_object()
            if subj is None or obj is None:
                continue
            subj=KGEntity.from_dict(subj)
            obj=KGEntity.from_dict(obj)
            head_id = subj.get_id()
            tail_id = obj.get_id()
            i = id2idx[head_id]
            j = id2idx[tail_id]
            adj[i, j] += 1
            adj[j, i] += 1
        return id2idx, adj
            
    def _encode_text(self, text: str):
        if not self.biobert_model or not self.biobert_tokenizer:
            self.logger.info(f"[EntityNormalize][BioBERT] model or tokenizer not loaded")
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
        except Exception as e:
            self.logger.info(f"[EntityNormalize][BioBERT] load failed ({e}), skip similarity-based suggestions.")
    def _find_entity_in_subgraph(self, sg_id: str, entity_id: str) -> Optional[KGEntity]:
        """
        First search for entity in self.subgraph_entities,
        if not found, fallback to Memory.subgraphs[sg_id].entities.
        """
        ent_map = self.subgraph_entities.get(sg_id)
        if ent_map is not None:
            ent = ent_map.get(entity_id)
            if ent is not None:
                return ent
        sg: Subgraph = self.memory.subgraphs.get(sg_id)
        if sg is None:
            return None
        for ent in sg.entities.all():
            if getattr(ent, "entity_id", None) == entity_id:
                return ent
        return None