import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.KnowledgeGraphDefinitions.index import KnowledgeGraph
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class PathExtractionAgent(Agent):
    """
    PathExtractionAgent

    Objective:
    - From a local knowledge graph, given a starting entity, search for a "reasoning path" (node sequence + triple sequence)
      with length no more than k, for subsequent scientific hypothesis generation.
    - During search, score all candidate child nodes using LLM, combine with penalty for heuristic search.

    Main public interface:
    - process(): Read keyword_entity_map from Memory, extract one path for each keyword's starting entity,
      write back to Memory. Finally output a summary table showing path length for each starting entity.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        k: int = 5,
        memory: Optional[Memory] = None,
        query: str = "",
    ):
        system_prompt = (
            "You are a biomedical AI4Science assistant.\n"
            "Your job is to score candidate nodes for extending a local knowledge-graph path, "
            "so that the resulting paths are helpful for generating plausible, novel, and testable "
            "scientific hypotheses for the given user query.\n\n"
            "You will ALWAYS respond with a single JSON object mapping candidate_id (string) to an object:\n"
            "{\n"
            '  \"<candidate_id>\": {\n'
            '    \"score\": float in [0, 1],   // higher means better to extend the path\n'
            '    \"reasons\": [str],          // short bullet-style reasons\n'
            '    \"flags\": [str]             // optional tags, e.g., [\"redundant\", \"off_topic\"]\n'
            "  },\n"
            "  ...\n"
            "}\n\n"
            "DO NOT output markdown code fences. DO NOT output any text outside the JSON object."
        )

        super().__init__(client, model_name, system_prompt)

        self.memory: Memory = memory or get_memory()
        self.logger = get_global_logger()
        self.query = query or ""
        self.key_entities: List[KGEntity] = self.memory.get_key_entities()
        self.knowledge_graph: KnowledgeGraph = KnowledgeGraph(
            self.memory.get_allRealationShip()
        )
        self.k = k

        self.node_penalty: Dict[str, float] = {}
        self.penalty_weight: float = 0.5

    def process(self) -> None:
        """
        Main entry point:
        - Iterate through keyword_entity_map in Memory,
        - Search for paths starting from each entity,
        - Write found paths back to Memory.
        - Finally output a concise table log: path length for each starting point.
        (Progress bar / progress logging included)
        """
        keyword_entity_map = self.memory.get_keyword_entity_map()

        tasks: List[Tuple[str, Any]] = []
        for keyword, ent_list in keyword_entity_map.items():
            for ent_data in ent_list:
                tasks.append((keyword, ent_data))

        if not tasks:
            self.logger.info("[PathExtraction] No key entities found to extract paths.")
            return

        total = len(tasks)
        summary_rows: List[Dict[str, Any]] = []

        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(tasks, desc="Extracting KG paths", unit="entity")
            use_tqdm = True
            self.logger.info(
                f"[PathExtraction] Start extracting paths for {total} entities (k={self.k}) with tqdm progress bar."
            )
        except Exception:
            iterator = tasks
            use_tqdm = False
            self.logger.info(
                f"[PathExtraction] Start extracting paths for {total} entities (k={self.k}). "
                f"tqdm not available, fallback to periodic log updates."
            )

        for idx, (keyword, ent_data) in enumerate(iterator, start=1):
            if isinstance(ent_data, KGEntity):
                start_entity = ent_data
            else:
                start_entity = KGEntity(**ent_data)

            node_path, edge_path = self.find_path_with_edges(
                start=start_entity,
                k=self.k,
                adj=self.knowledge_graph.Graph,
            )

            if node_path:
                path_len = len(node_path)
                self.memory.add_extracted_path(keyword, node_path, edge_path)
            else:
                path_len = 0

            summary_rows.append(
                {
                    "keyword": keyword,
                    "entity_name": getattr(start_entity, "name", "") or "",
                    "entity_id": getattr(start_entity, "entity_id", "") or "",
                    "path_len": path_len,
                }
            )

            if not use_tqdm and (idx % 20 == 0 or idx == total):
                pct = idx / total * 100
                self.logger.info(
                    f"[PathExtraction] Progress: {idx}/{total} entities processed ({pct:.1f}%)."
                )

        self._log_summary_table(summary_rows)


    def find_path_with_edges(
        self,
        start: KGEntity,
        k: int,
        adj: Any,
    ) -> Tuple[List[KGEntity], List[KGTriple]]:
        """
        Use DFS for heuristic search:
        - Score all candidate children using LLM uniformly at each step;
        - Use scores plus penalty adjustment as heuristic, expand by score ranking;
        - Maintain global best_path (by accumulated score);
        - When a path reaches a dead end before reaching length k -> consider it a failed path, penalize nodes on the path.

        Returns:
        - best_nodes: List[KGEntity]
        - best_edges: List[KGTriple]
        If no reasonable path with length > 2 is found, return empty list.
        """
        node_path: List[KGEntity] = [start]
        edge_path: List[KGTriple] = []

        best_nodes: List[KGEntity] = node_path.copy()
        best_edges: List[KGTriple] = edge_path.copy()
        best_score: float = 0.0

        def dfs(current: KGEntity, current_score: float) -> None:
            nonlocal best_nodes, best_edges, best_score

            if len(node_path) > 1 and current_score > best_score:
                best_score = current_score
                best_nodes = node_path.copy()
                best_edges = edge_path.copy()

            if len(node_path) >= k:
                return

            neighbors = adj.get(current.entity_id, [])
            candidates: List[KGEntity] = []
            candidate_edges: List[KGTriple] = []

            for child_stub, relation in neighbors[:100]:
                child_data = relation.object
                if isinstance(child_data, KGEntity):
                    child_node = child_data
                else:
                    child_node = KGEntity(**child_data)

                if any(child_node.entity_id == e.entity_id for e in node_path):
                    continue

                candidates.append(child_node)
                candidate_edges.append(relation)

            if not candidates:
                if len(node_path) < k:
                    self._penalize_path(node_path)
                return

            scores_info = self._score_candidates_with_llm(
                node_path=node_path,
                edge_path=edge_path,
                candidates=candidates,
            )

            scored_children: List[Tuple[float, KGEntity, KGTriple]] = []
            for child_node, rel in zip(candidates, candidate_edges):
                info = scores_info.get(child_node.entity_id, {})
                raw_score = float(info.get("score", 0.0))
                penalty = self.node_penalty.get(child_node.entity_id, 0.0)
                effective = raw_score - self.penalty_weight * penalty

                if effective <= 0:
                    continue

                scored_children.append((effective, child_node, rel))

            if not scored_children:
                if len(node_path) < k:
                    self._penalize_path(node_path)
                return

            scored_children.sort(key=lambda x: x[0], reverse=True)

            for effective_score, child_node, rel in scored_children[:30]:
                node_path.append(child_node)
                edge_path.append(rel)

                dfs(child_node, current_score + effective_score)

                node_path.pop()
                edge_path.pop()

        dfs(start, current_score=0.0)

        if len(best_nodes) <= 2:
            self.logger.debug(
                f"[PathExtraction] No sufficiently long path found from start={start.name} "
                f"(best_len={len(best_nodes)} <= 2)."
            )
            return [], []

        self.logger.debug(
            f"[PathExtraction] Found best path from start={start.name} | "
            f"nodes={len(best_nodes)}, edges={len(best_edges)}, score={best_score:.4f}"
        )
        return best_nodes, best_edges

    def _score_candidates_with_llm(
        self,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
        candidates: List[KGEntity],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Call LLM to score all candidate child nodes at the same level at once.

        Returns:
        {
          "<entity_id>": {
            "score": float in [0, 1],
            "reasons": [...],
            "flags": [...]
          },
          ...
        }
        """
        path_entities = [self._serialize_entity(e) for e in node_path]
        path_edges = [self._serialize_triple(tr) for tr in edge_path]
        candidate_payload = [
            {
                "id": c.entity_id,
                "node": self._serialize_entity(c),
            }
            for c in candidates
        ]

        payload = {
            "task": (
                "Score candidate nodes for extending a knowledge-graph path used for "
                "biomedical hypothesis generation."
            ),
            "query": self.query,
            "current_path": {
                "nodes": path_entities,
                "edges": path_edges,
            },
            "candidates": candidate_payload,
            "decision_criterion": {
                "novelty": (
                    "Does this node introduce non-trivial or under-explored connections or mechanisms?"
                ),
                "relevance": (
                    "Is this node relevant to the user query and the existing path, "
                    "especially regarding mechanisms, targets, delivery, safety, or outcomes?"
                ),
                "mechanistic_value": (
                    "Does this node extend a plausible mechanistic / causal chain "
                    "between upstream and downstream biomedical entities?"
                ),
                "verifiability": (
                    "Could hypotheses involving this node be tested or grounded in experiments "
                    "or analyses (e.g., in vitro, in vivo, or clinical studies)?"
                ),
                "do_not_reject_just_because": [
                    "the candidate label already appears in another node's description text",
                    "the candidate is a mechanism/action term rather than a named entity",
                    "the candidate originates from the same sentence as existing nodes "
                    "(local overlap is expected in extracted subgraphs)",
                ],
            },
            "output_format": (
                "Return ONLY a JSON object mapping candidate 'id' to:\n"
                "{\n"
                '  \"score\": float in [0, 1],\n'
                '  \"reasons\": [str],\n'
                '  \"flags\": [str]\n'
                "}"
            ),
        }

        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            raw = self.call_llm(prompt)
            cleaned = self._strip_json_fences(raw)
            obj = json.loads(cleaned)

            if not isinstance(obj, dict):
                raise ValueError("LLM scoring output is not a JSON object.")

            for cid, info in obj.items():
                if not isinstance(info, dict):
                    obj[cid] = {
                        "score": 0.0,
                        "reasons": ["invalid entry"],
                        "flags": ["invalid"],
                    }
                    continue
                if "score" not in info:
                    info["score"] = 0.0
                try:
                    info["score"] = float(info["score"])
                except Exception:
                    info["score"] = 0.0
                if "reasons" not in info or not isinstance(info["reasons"], list):
                    info["reasons"] = []
                if "flags" not in info or not isinstance(info["flags"], list):
                    info["flags"] = []

            return obj

        except Exception as e:
            self.logger.warning(
                f"[PathExtraction][LLM scoring] failed to parse JSON, error={e}"
            )
            fallback = {
                c.entity_id: {
                    "score": 0.0,
                    "reasons": ["fallback 0 score"],
                    "flags": ["llm_error"],
                }
                for c in candidates
            }
            return fallback

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """
        Remove possible ```json / ``` wrapping.
        """
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text

    def _penalize_path(self, nodes: List[KGEntity], amount: float = 1.0) -> None:
        """
        When a path is determined to be a "failed path" (reaches a dead end before reaching length k),
        increase penalty on nodes in the path, reducing their effective scores when encountered later.
        """
        for node in nodes:
            eid = node.entity_id
            self.node_penalty[eid] = self.node_penalty.get(eid, 0.0) + amount

        self.logger.debug(
            "[PathExtraction] Penalized path: "
            + " -> ".join(f"{n.name}({n.entity_id})" for n in nodes)
            )

    @staticmethod
    def _serialize_entity(e: KGEntity) -> Dict[str, Any]:
        return {
            "entity_id": getattr(e, "entity_id", None),
            "name": getattr(e, "name", None),
            "type": getattr(e, "entity_type", None),
            "normalized_id": getattr(e, "normalized_id", None),
            "aliases": getattr(e, "aliases", None),
            "description": getattr(e, "description", None),
        }

    @staticmethod
    def _serialize_triple(t: KGTriple) -> Dict[str, Any]:
        rel_type = getattr(t, "relation", None) or getattr(t, "rel_type", None)
        head = getattr(t, "head", None) or getattr(t, "head_id", None)
        tail = getattr(t, "tail", None) or getattr(t, "tail_id", None)

        subj = getattr(t, "subject", None)
        obj = getattr(t, "object", None)

        return {
            "relation_type": rel_type,
            "head": head,
            "tail": tail,
            "subject": subj,
            "object": obj,
            "source": getattr(t, "source", None),
            "mechanism": getattr(t, "mechanism", None),
            "evidence": getattr(t, "evidence", None),
        }

    def _log_summary_table(self, rows: List[Dict[str, Any]]) -> None:
        """
        Output a simple ASCII table in logs:
        | Keyword | Entity | EntityID | PathLen |
        """
        if not rows:
            self.logger.info("[PathExtraction] No paths extracted for any entity.")
            return

        headers = ["Keyword", "Entity", "EntityID", "PathLen"]

        def _safe_str(x: Any) -> str:
            return str(x) if x is not None else ""
        col_widths = [
            max(len(headers[0]), max(len(_safe_str(r["keyword"])) for r in rows)),
            max(len(headers[1]), max(len(_safe_str(r["entity_name"])) for r in rows)),
            max(len(headers[2]), max(len(_safe_str(r["entity_id"])) for r in rows)),
            max(len(headers[3]), max(len(_safe_str(r["path_len"])) for r in rows)),
        ]

        def _fmt_row(cols: List[str]) -> str:
            return (
                "| "
                + " | ".join(
                    c.ljust(w) for c, w in zip(cols, col_widths)
                )
                + " |"
            )

        sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        lines = [sep_line, _fmt_row(headers), sep_line]

        for r in rows:
            line = _fmt_row(
                [
                    _safe_str(r["keyword"]),
                    _safe_str(r["entity_name"]),
                    _safe_str(r["entity_id"]),
                    _safe_str(r["path_len"]),
                ]
            )
            lines.append(line)
        lines.append(sep_line)

        table_str = "\n".join(lines)
        self.logger.info("[PathExtraction] Path length summary:\n" + table_str)