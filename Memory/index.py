from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

@dataclass
class SimpleAlignment:
    """
    A simple entity alignment record:
    - src_subgraph / src_entity: Source subgraph + source entity ID
    - tgt_subgraph / tgt_entity: Target subgraph + target entity ID
    """
    src_subgraph: str
    src_entity: str
    tgt_subgraph: str
        tgt_entity: str

class EntityStore:
    """
    Responsible for global/subgraph entity deduplication and indexing.

    Rules:
    - Ignore externally passed entity_id, uniformly assigned by this class as ent:xxxxxx.
    - Prioritize merging by normalized_id, then by normalized name.
    - Maintain:
        - by_id: id -> KGEntity
        - idx_normid: normalized_id.lower() -> id
        - idx_name: norm(name).lower() -> id
    """
    def __init__(self):
        self.by_id: Dict[str, KGEntity] = {}
        self.idx_normid: Dict[str, str] = {}
        self.idx_name: Dict[str, str] = {}

    def _nid(self) -> str:
        return f"ent:{uuid.uuid4().hex[:12]}"

    def _key(self, s: str) -> str:
        return (s or "").strip().lower()

    def upsert(self, e: KGEntity) -> KGEntity:
        e.entity_id = ""

        norm = self._key(e.name)

        if e.normalized_id and e.normalized_id != "N/A":
            k = self._key(e.normalized_id)
            if k in self.idx_normid:
                return self._merge(self.by_id[self.idx_normid[k]], e)

        if norm and norm in self.idx_name:
            return self._merge(self.by_id[self.idx_name[norm]], e)

        new_id = self._nid()
        e.entity_id = new_id
        self.by_id[new_id] = e

        if e.normalized_id and e.normalized_id != "N/A":
            self.idx_normid[self._key(e.normalized_id)] = new_id
        if norm:
            self.idx_name[norm] = new_id

        return e

    def update(self, entities: List[KGEntity]):
        for entity in entities:
            former_entity = self.by_id.get(entity.entity_id)
            if former_entity:
                if former_entity == entity:
                    continue
                former_entity.name = entity.name
                former_entity.entity_type = entity.entity_type
                self.by_id[entity.entity_id] = former_entity

    def _merge(self, base: KGEntity, inc: KGEntity) -> KGEntity:
        if base.entity_type == "Unknown" and inc.entity_type != "Unknown":
            base.entity_type = inc.entity_type

        if inc.name and len(inc.name) > len(base.name):
            if base.name:
                base.aliases.append(base.name)
            base.name = inc.name

        if base.normalized_id == "N/A" and inc.normalized_id and inc.normalized_id != "N/A":
            base.normalized_id = inc.normalized_id
            self.idx_normid[self._key(base.normalized_id)] = base.entity_id

        pool = {self._key(a): a for a in base.aliases}
        for a in ([inc.name] if inc.name else []) + (inc.aliases or []):
            if a:
                pool.setdefault(self._key(a), a)
        base.aliases = sorted(pool.values(), key=str.lower)

        norm = self._key(base.name)
        if norm:
            self.idx_name[norm] = base.entity_id

        return base

    def find_by_norm(self, name_or_alias: str) -> Optional[KGEntity]:
        k = self._key(name_or_alias)
        eid = self.idx_name.get(k)
        return self.by_id.get(eid) if eid else None

    def find_by_normalized_id(self, normalized_id: str) -> Optional[KGEntity]:
        k = self._key(normalized_id)
        eid = self.idx_normid.get(k)
        return self.by_id.get(eid) if eid else None

    def upsert_many(self, entities: List[KGEntity]) -> List[KGEntity]:
        return [self.upsert(e) for e in tqdm(entities)]

    def all(self) -> List[KGEntity]:
        return list(self.by_id.values())

class RelationStore:
    """
    by_head: Get triple list by querying head entity
    by_relation: Get triple list by querying relation type
    by_tail: Get triple list by querying tail entity
    """
    def __init__(self):
        self.triples: List[KGTriple] = []
        self.by_head: Dict[str, List[KGTriple]] = {}
        self.by_relation: Dict[str, List[KGTriple]] = {}
        self.by_tail: Dict[str, List[KGTriple]] = {}

    def _rid(self) -> str:
        return f"rel:{uuid.uuid4().hex[:12]}"

    def add(self, triple: KGTriple):
        """Insert a triple"""
        self.triples.append(triple)

        relation = getattr(triple, "relation", None) or getattr(triple, "rel_type", None)
        head = getattr(triple, "head", None) or getattr(triple, "head_id", None)
        tail = getattr(triple, "tail", None) or getattr(triple, "tail_id", None)

        if relation is not None:
            self.by_relation.setdefault(relation, []).append(triple)
        if head is not None:
            self.by_head.setdefault(head, []).append(triple)
        if tail is not None:
            self.by_tail.setdefault(tail, []).append(triple)

    def add_many(self, triples: List[KGTriple]):
        for triple in triples:
            self.add(triple)

    def find_Triple_by_head_and_tail(self, head: str, tail: str) -> Optional[KGTriple]:
        """Find triple by head entity and tail entity"""
        head_triples = self.by_head.get(head, [])
        for triple in head_triples:
            t = getattr(triple, "tail", None) or getattr(triple, "tail_id", None)
            if t == tail:
                return triple
        return None

    def all(self) -> List[KGTriple]:
        return self.triples

    def reset(self):
        """Reset relationstore"""
        self.triples.clear()
        self.by_head.clear()
        self.by_relation.clear()
        self.by_tail.clear()

class AlignmentStore:
    """
    Only stores "entity-to-entity" alignment results, without IDs or similarities:
    - Each record is a SimpleAlignment
    - Main index: by_source[(src_subgraph, src_entity)] -> List[SimpleAlignment]
    """
    def __init__(self):
        self.by_source: Dict[tuple, List[SimpleAlignment]] = {}

    def add(
        self,
        src_subgraph: str,
        src_entity: str,
        tgt_subgraph: str,
        tgt_entity: str,
    ) -> None:
        """
        Add an alignment edge; do not add duplicates if it already exists.
        """
        key = (src_subgraph, src_entity)
        lst = self.by_source.setdefault(key, [])
        for rec in lst:
            if rec.tgt_subgraph == tgt_subgraph and rec.tgt_entity == tgt_entity:
                return
        lst.append(SimpleAlignment(
            src_subgraph=src_subgraph,
            src_entity=src_entity,
            tgt_subgraph=tgt_subgraph,
            tgt_entity=tgt_entity,
        ))

    def save_from_alignment_dict(
        self,
        alignment_dict: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> None:
        """
        Directly receive refined_alignment from AlignmentTripleAgent.build_entity_alignment:
        {
          src_sg_id: {
            src_eid: [
              {
                "target_subgraph": ...,
                "target_entity": ...,
                ...
              }, ...
            ]
          }
        }
        Only take subgraph + entity ID four fields, other information is ignored.
        """
        for src_sg_id, ent_map in alignment_dict.items():
            for src_eid, matches in ent_map.items():
                for m in matches:
                    tgt_sg_id = m.get("target_subgraph")
                    tgt_eid = m.get("target_entity")
                    if not tgt_sg_id or not tgt_eid:
                        continue
                    self.add(
                        src_subgraph=src_sg_id,
                        src_entity=src_eid,
                        tgt_subgraph=tgt_sg_id,
                        tgt_entity=tgt_eid,
                    )

    def get_for_source(
        self,
        src_subgraph: str,
        src_entity: str,
    ) -> List[SimpleAlignment]:
        """Query all target entities for given (src_subgraph, src_entity)."""
        return list(self.by_source.get((src_subgraph, src_entity), []))

    def all(self) -> List[SimpleAlignment]:
        """Return all alignment edges (flat list)"""
        result: List[SimpleAlignment] = []
        for lst in self.by_source.values():
            result.extend(lst)
        return result

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Export as list[dict] for easy dump_json:
        [
          {
            "src_subgraph": ...,
            "src_entity": ...,
            "tgt_subgraph": ...,
            "tgt_entity": ...
          },
          ...
        ]
        """
        return [asdict(rec) for rec in self.all()]

    def from_list(self, data: List[Dict[str, Any]]) -> None:
        """
        Restore from list[dict]; will clear current content.
        """
        self.by_source.clear()
        for d in data or []:
            self.add(
                src_subgraph=d["src_subgraph"],
                src_entity=d["src_entity"],
                tgt_subgraph=d["tgt_subgraph"],
                tgt_entity=d["tgt_entity"],
            )

class Subgraph:
    """
    Subgraph:
    - Has its own EntityStore / RelationStore
    - Identified by subgraph_id (passed when you create it)
    - Can be exported separately or merge_into global Memory

    NOTICE: subgraph_id format is PMID_index to identify article_paragraph
    """
    def __init__(
        self,
        subgraph_id: str,
        name: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.id = subgraph_id
        self.name = name
        self.meta = dict(meta or {})
        self.entities = EntityStore()
        self.relations = RelationStore()
    def upsert_entity(self, e: KGEntity) -> KGEntity:
        return self.entities.upsert(e)

    def upsert_many_entities(self, ents: List[KGEntity]) -> List[KGEntity]:
        return self.entities.upsert_many(ents)

    def add_relation(self, r: KGTriple) -> KGTriple:
        return self.relations.add(r)

    def add_relations(self, rs: List[KGTriple]) -> List[KGTriple]:
        return self.relations.add_many(rs)

    def get_meta(self) -> Dict[str, Any]:
        return self.meta

    def get_relations(self) -> List[KGTriple]:
        return self.relations.all()

    def get_entities(self) -> List[KGEntity]:
        return self.entities.all()

    def find_by_norm(self, name_or_alias: str) -> Optional[KGEntity]:
        return self.entities.find_by_norm(name_or_alias)

    def find_by_normalized_id(self, nid: str) -> Optional[KGEntity]:
        return self.entities.find_by_normalized_id(nid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "meta": self.meta,
            "entities": [e.to_dict() for e in self.entities.all()],
            "relations": [r.to_dict() for r in self.relations.all()],
        }

    def to_json(self, dirpath: str = ".") -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"subgraph-{self.id or self.name or 'noname'}-{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(path)

    def merge_into(self, mem: "Memory") -> Dict[str, str]:
        """
        Merge current subgraph content into global memory:
        - Entities are deduplicated/merged through mem.entities.upsert
        - Relations rewrite head/tail using merged global entity IDs
        - Return: {old subgraph entity ID -> global entity ID}
        - Also register this subgraph to mem.subgraphs (preserve subgraph view)
        """
        id_map: Dict[str, str] = {}

        for e in self.entities.all():
            old_id = e.entity_id
            e_copy = KGEntity(**e.to_dict())
            merged = mem.entities.upsert(e_copy)
            if old_id:
                id_map[old_id] = merged.entity_id

        for r in self.relations.all():
            head = id_map.get(getattr(r, "head_id", None) or getattr(r, "head", None),
                              getattr(r, "head_id", None) or getattr(r, "head", None))
            tail = id_map.get(getattr(r, "tail_id", None) or getattr(r, "tail", None),
                              getattr(r, "tail_id", None) or getattr(r, "tail", None))
            rel_type = getattr(r, "rel_type", None) or getattr(r, "relation", None)

            mem.relations.add(KGTriple(
                rel_type=rel_type,
                head_id=head,
                tail_id=tail,
                props=dict(getattr(r, "props", {}) or {}),
            ))

        if self.id:
            mem.register_subgraph(self)

        return id_map

class KeyEntityStore:
    """
    Simple list for storing "key entities":
    - No deduplication merging, no index maintenance
    - Only provides basic operations like append, batch append, get all, clear
    """
    def __init__(self):
        self.entities: List[KGEntity] = []

    def add(self, e: KGEntity) -> None:
        """Append a key entity"""
        self.entities.append(e)

    def add_many(self, ents: List[KGEntity]) -> None:
        """Batch append key entities"""
        self.entities.extend(ents)

    def all(self) -> List[KGEntity]:
        """Return all key entities (original list)"""
        return list(self.entities)

    def reset(self) -> None:
        """Clear key entity list"""
        self.entities.clear()
class Memory:
    """
    Global shared memory pool:
    - Holds a global EntityStore / RelationStore
    - Registers multiple Subgraphs (indexed by string ID)
    - Supports exporting unified snapshot JSON (including global + subgraph internal details)
    """
    def __init__(self):
        self.entities = EntityStore()
        self.relations = RelationStore()
        self.subgraphs: Dict[str, Subgraph] = {}
        self.alignments = AlignmentStore()
        self.key_entities = KeyEntityStore()
        self.keyword_entity_map: Dict[str, List[KGEntity]] = {}
        self._extracted_paths: Dict[str, List[dict]] = {}
        self.entity_id_mapping_path: Optional[str] = None
        self.hypothesesdir: str= ""
    def reset(self) -> None:
        """Reset global Memory content (entities, relations, subgraphs, alignments, key entities, etc.)"""
        self.entities = EntityStore()
        self.relations = RelationStore()
        self.subgraphs.clear()
        self.alignments = AlignmentStore()
        self.key_entities = KeyEntityStore()
        self.keyword_entity_map.clear()
        self._extracted_paths.clear()
        self.entity_id_mapping_path = None
        self.hypothesesdir= ""
    
    def add_extracted_path(
        self,
        keyword: str,
        nodes: List[KGEntity],
        edges: List[KGTriple],
    ) -> None:
        if not hasattr(self, "paths"):
            self.paths = {}
        self.paths.setdefault(keyword, []).append(
            {
                "nodes": nodes,
                "edges": edges,
            }
        )
    def add_hypothesesDir(self, hypothesesDir: str) -> None:
        self.hypothesesdir = hypothesesDir
    def get_extracted_paths(self) -> List[dict]:
        """
        Return all stored path records, each record is {'nodes': [...], 'edges': [...]}.
        """
        return list(self._extracted_paths)
    def upsert_many_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        return self.entities.upsert_many(entities)
    def register_subgraph(self, sg: Subgraph) -> None:
        if not sg.id:
            return
        self.subgraphs[sg.id] = sg

    def get_subgraph(self, sg_id: str) -> Optional[Subgraph]:
        return self.subgraphs.get(sg_id)
    
    def remove_subgraph(self, sg_id: str) -> None:
        if sg_id in self.subgraphs:
            del self.subgraphs[sg_id]
    def add_key_entity(self, e: KGEntity) -> None:
        self.key_entities.add(e)
    def add_key_entities(self, ents: List[KGEntity]) -> None:
        self.key_entities.add_many(ents)
    def get_key_entities(self) -> List[KGEntity]:
        return self.key_entities.all()
    def get_allRealationShip(self)-> List[KGTriple]:
        return self.relations.all()
    def add_keyword_entities(self, keyword: str, ents: List[KGEntity]) -> None:
        """
        Override set entity list for a certain keyword.
        """
        keyword = (keyword or "").strip()
        if not keyword:
            return
        self.keyword_entity_map[keyword] = list(ents)

    def append_keyword_entities(self, keyword: str, ents: List[KGEntity]) -> None:
        """
        Append entity list for a certain keyword (extend based on original).
        """
        keyword = (keyword or "").strip()
        if not keyword:
            return
        self.keyword_entity_map.setdefault(keyword, []).extend(ents)

    def get_keyword_entities(self, keyword: str) -> List[KGEntity]:
        """
        Get the entity list currently mapped to a certain keyword.
        """
        return list(self.keyword_entity_map.get(keyword, []))

    def get_keyword_entity_map(self) -> Dict[str, List[KGEntity]]:
        """
        Get complete keyword -> [KGEntity, ...] mapping (shallow copy).
        """
        return {k: list(v) for k, v in self.keyword_entity_map.items()}
    def dump_json(self, dirpath: str = ".") -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirp = Path(dirpath)
        dirp.mkdir(parents=True, exist_ok=True)

        path = dirp / f"memory-{ts}.json"
        data = {
            "entities": [e.to_dict() for e in self.entities.all()],
            "relations": [r.to_dict() for r in self.relations.all()],
            "subgraphs": {
                sg_id: sg.to_dict()
                for sg_id, sg in self.subgraphs.items()
            },
            "alignments": self.alignments.to_list(),
            "key_entities": [e.to_dict() for e in self.key_entities.all()],
            "keyword_entity_map": {
                kw: [e.to_dict() for e in ents]
                for kw, ents in self.keyword_entity_map.items()
            },
            "meta": {"generated_at": ts,"entity_id_mapping_path": self.entity_id_mapping_path,},
            "paths": {
                kw: [
                    {
                        "nodes": [
                            n.to_dict() if hasattr(n, "to_dict") else n
                            for n in path.get("nodes", [])
                        ],
                        "edges": [
                            e.to_dict() if hasattr(e, "to_dict") else e
                            for e in path.get("edges", [])
                        ],
                    }
                    for path in path_list
                ]
                for kw, path_list in getattr(self, "paths", {}).items()
            },
            "hypothesesdir": self.hypothesesdir,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)


def load_memory_from_json(path_or_data: Any) -> Memory:
    """
    Restore to a new Memory instance from snapshot exported by Memory.dump_json().
    - path_or_data: file path (str/Path) or parsed dict
    - Does not modify current global memory variable, returns new Memory object
    - Preserve original entity_id, do not use upsert; rebuild various inverted indexes
    """
    def _coerce_entity(ed: Any) -> KGEntity:
        if isinstance(ed, KGEntity):
            return ed
        try:
            from_dict = getattr(KGEntity, "from_dict", None)
            if callable(from_dict):
                return from_dict(ed)
        except Exception:
            pass
        return KGEntity(**ed)

    def _coerce_triple(rd: Any) -> KGTriple:
        if isinstance(rd, KGTriple):
            return rd
        try:
            from_dict = getattr(KGTriple, "from_dict", None)
            if callable(from_dict):
                return from_dict(rd)
        except Exception:
            pass
            try:
                return KGTriple(**rd)
            except Exception:
                mapped = dict(rd)
            if "relation" in mapped and "rel_type" not in mapped:
                mapped["rel_type"] = mapped["relation"]
            if "head" in mapped and "head_id" not in mapped:
                mapped["head_id"] = mapped["head"]
            if "tail" in mapped and "tail_id" not in mapped:
                mapped["tail_id"] = mapped["tail"]
            allowed = {"rel_type", "relation", "head_id", "tail_id", "head", "tail", "props"}
            slim = {k: v for k, v in mapped.items() if k in allowed}
            return KGTriple(**slim)
    if isinstance(path_or_data, (str, Path)):
        with open(path_or_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(path_or_data, dict):
        data = path_or_data
    else:
        raise TypeError("load_memory_from_json expects a file path (str/Path) or a dict")

    mem = Memory()
    for ed in data.get("entities", []):
        e = _coerce_entity(ed)
        mem.entities.by_id[e.entity_id] = e
        if getattr(e, "normalized_id", None) and e.normalized_id != "N/A":
            mem.entities.idx_normid[mem.entities._key(e.normalized_id)] = e.entity_id
        if getattr(e, "name", None):
            mem.entities.idx_name[mem.entities._key(e.name)] = e.entity_id
    for rd in data.get("relations", []):
        r = _coerce_triple(rd)
        mem.relations.triples.append(r)

        rel = getattr(r, "relation", None) or getattr(r, "rel_type", None)
        head = getattr(r, "head", None) or getattr(r, "head_id", None)
        tail = getattr(r, "tail", None) or getattr(r, "tail_id", None)

        if rel is not None:
            mem.relations.by_relation.setdefault(rel, []).append(r)
        if head is not None:
            mem.relations.by_head.setdefault(head, []).append(r)
        if tail is not None:
            mem.relations.by_tail.setdefault(tail, []).append(r)

    for sg_id, sgd in (data.get("subgraphs") or {}).items():
        sg = Subgraph(
            subgraph_id=sgd.get("id", sg_id),
            name=sgd.get("name", ""),
            meta=sgd.get("meta", {}),
        )

        for ed in sgd.get("entities", []):
            e = _coerce_entity(ed)
            sg.entities.by_id[e.entity_id] = e
            if getattr(e, "normalized_id", None) and e.normalized_id != "N/A":
                sg.entities.idx_normid[sg.entities._key(e.normalized_id)] = e.entity_id
            if getattr(e, "name", None):
                sg.entities.idx_name[sg.entities._key(e.name)] = e.entity_id

        for rd in sgd.get("relations", []):
            r = _coerce_triple(rd)
            sg.relations.triples.append(r)

            rel = getattr(r, "relation", None) or getattr(r, "rel_type", None)
            head = getattr(r, "head", None) or getattr(r, "head_id", None)
            tail = getattr(r, "tail", None) or getattr(r, "tail_id", None)

            if rel is not None:
                sg.relations.by_relation.setdefault(rel, []).append(r)
            if head is not None:
                sg.relations.by_head.setdefault(head, []).append(r)
            if tail is not None:
                sg.relations.by_tail.setdefault(tail, []).append(r)

        mem.register_subgraph(sg)
    align_data = data.get("alignments", [])
    if align_data:
        mem.alignments.from_list(align_data)
    key_ents_data = data.get("key_entities", [])
    for ed in key_ents_data:
        e = _coerce_entity(ed)
        mem.key_entities.add(e)
    kw_map_data = data.get("keyword_entity_map", {}) or {}
    for kw, ents_list in kw_map_data.items():
        ents: List[KGEntity] = []
        for ed in ents_list or []:
            ents.append(_coerce_entity(ed))
        mem.keyword_entity_map[kw] = ents
    paths_data = data.get("paths", [])
    for kw, path_list in paths_data.items():
            for pd in path_list or []:
                node_ents: List[KGEntity] = []
                edge_triples: List[KGTriple] = []

                for ed in pd.get("nodes", []):
                    node_ents.append(_coerce_entity(ed))
                for rd in pd.get("edges", []):
                    edge_triples.append(_coerce_triple(rd))

                mem.add_extracted_path(kw, node_ents, edge_triples)
    mem.entity_id_mapping_path = (
        data.get("entity_id_mapping_path")
        or (data.get("meta", {}) or {}).get("entity_id_mapping_path")
    )
    mem.hypothesesdir = data.get("hypothesesdir","")
    return mem
memory = Memory()