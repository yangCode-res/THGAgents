import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from Core.Agent import Agent
from Memory.index import Memory
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class SubgraphMerger(Agent):
    """
    Subgraph Merger Agent.

    Functions:
    - Merge all subgraphs from Memory.subgraphs into Memory.entities / Memory.relations
    - Use entity alignment information from Memory.alignments to merge "aligned entities" first, then unaligned entities
    - During merging, maintain:
        * local2global: (subgraph_id, local_entity_id) -> global_entity_id
        * global2local: global_entity_id -> Set[(subgraph_id, local_entity_id)]
      But these mappings **are not directly attached to memory**,
      instead they are written to a .pkl file,
      and the file path is recorded in memory.entity_id_mapping_path.
    """

    def __init__(self, client: OpenAI, model_name: str, memory: Optional[Memory] = None):
        super().__init__(client, model_name, system_prompt="")
        self.local2global: Dict[Tuple[str, str], str] = {}
        self.global2local: Dict[str, Set[Tuple[str, str]]] = {}
        self.memory: Memory = memory or get_memory()
        self.client = client
        self.model_name = model_name

    def _register_mapping(self, sg_id: str, local_eid: str, global_eid: str) -> None:
        """
        Record a mapping of (subgraph, local_entity_id) <-> global_entity_id to:
        - self.local2global
        - self.global2local
        """
        if not sg_id or not local_eid or not global_eid:
            return
        key = (sg_id, local_eid)
        self.local2global[key] = global_eid
        if global_eid not in self.global2local:
            self.global2local[global_eid] = set()
        self.global2local[global_eid].add(key)

    def _get_entity(self, sg_id: str, ent_id: str) -> Optional[KGEntity]:
        sg = self.memory.subgraphs.get(sg_id)
        if sg is None:
            return None
        return sg.entities.by_id.get(ent_id)

    def _ensure_entity(self, x: Any) -> Optional[KGEntity]:
        """Convert subject/object to KGEntity uniformly; return None otherwise."""
        if x is None:
            return None
        if isinstance(x, KGEntity):
            return x
        if isinstance(x, dict):
            from_dict = getattr(KGEntity, "from_dict", None)
            if callable(from_dict):
                return from_dict(x)
            return KGEntity(**x)
        return None

    def _merge_alignments(self):
        """
        Use alignment results from memory.alignments.by_source to merge entities
        from the "same cluster" into the same global entity, and record
        local <-> global mappings.
        """
        for (src_sg, src_eid), aligns in self.memory.alignments.by_source.items():
            src_ent = self._get_entity(src_sg, src_eid)
            if src_ent is None:
                continue
            if (src_sg, src_eid) in self.local2global:
                gid = self.local2global[(src_sg, src_eid)]
                base = self.memory.entities.by_id[gid]
            else:
                base = self.memory.entities.upsert(KGEntity(**src_ent.to_dict()))
                gid = base.entity_id
                self._register_mapping(src_sg, src_eid, gid)
            for al in aligns:
                tgt_key = (al.tgt_subgraph, al.tgt_entity)
                if tgt_key in self.local2global:
                    continue
                tgt_ent = self._get_entity(al.tgt_subgraph, al.tgt_entity)
                if tgt_ent is None:
                    continue
                self.memory.entities._merge(base, KGEntity(**tgt_ent.to_dict()))
                self._register_mapping(al.tgt_subgraph, al.tgt_entity, gid)

    def _merge_unaligned_entities(self):
        """
        Iterate through all subgraph entities:
        - For entities not appearing in local2global, directly upsert to global entity library
        - And record (sg_id, local_id) -> global_id mapping
        """
        for sg_id, sg in self.memory.subgraphs.items():
            for e in sg.entities.all():
                key = (sg_id, e.entity_id)
                if key in self.local2global:
                    continue

                g = self.memory.entities.upsert(KGEntity(**e.to_dict()))
                self._register_mapping(sg_id, e.entity_id, g.entity_id)

    def _merge_relations(self):
        """
        Iterate through all subgraph relations, map their subject/object to global entities,
        then write to memory.relations.
        """
        for sg_id, sg in self.memory.subgraphs.items():
            for r in sg.relations.all():
                subj = self._ensure_entity(r.subject)
                obj = self._ensure_entity(r.object)

                sid = subj.entity_id if subj else None
                oid = obj.entity_id if obj else None

                g_subj = subj
                g_obj = obj

                if sid is not None:
                    gid = self.local2global.get((sg_id, sid))
                    if gid:
                        g_subj = self.memory.entities.by_id.get(gid, g_subj)

                if oid is not None:
                    gid = self.local2global.get((sg_id, oid))
                    if gid:
                        g_obj = self.memory.entities.by_id.get(gid, g_obj)

                new_triple = KGTriple(
                    head=g_subj.name if g_subj else r.head,
                    relation=r.relation,
                    tail=g_obj.name if g_obj else r.tail,
                    confidence=r.confidence,
                    evidence=r.evidence,
                    mechanism=r.mechanism,
                    source=r.source,
                    subject=g_subj,
                    object=g_obj,
                    time_info=r.time_info,
                )
                self.memory.relations.add(new_triple)

    def _dump_mappings_to_pkl(self) -> str:
        """
        Write local2global / global2local to a .pkl file and return the file path.

        Structure is approximately:
        {
          "local2global": {
            (subgraph_id, local_eid): global_eid,
            ...
          },
          "global2local": {
            global_eid: [
              (subgraph_id, local_eid),
              ...
            ],
            ...
          }
        }
        """
        base_dir = Path("cache")
        base_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"entity_id_mapping_{ts}.pkl"
        fpath = base_dir / filename
        data = {
            "local2global": dict(self.local2global),
            "global2local": {
                gid: list(pairs) for gid, pairs in self.global2local.items()
            },
        }
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        return str(fpath)

    def process(self):
        """
        1. Initialize mappings for this merge operation
        2. Merge aligned entities
        3. Merge unaligned entities
        4. Merge relations
        5. Write mappings to pkl file and attach path to memory
        """
        self.local2global = {}
        self.global2local = {}
        self._merge_alignments()
        self._merge_unaligned_entities()
        self._merge_relations()
        mapping_path = self._dump_mappings_to_pkl()
        self.memory.entity_id_mapping_path = mapping_path