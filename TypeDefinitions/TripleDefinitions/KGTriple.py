from dataclasses import asdict, dataclass
from typing import List, Optional

from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TimeDefinitions.TimeFormat import TimeFormat


@dataclass
class KGTriple:
    """三元组定义信息。

    - head: 头实体名称
    - relation: 关系名称（raw）
    - relation_type: 关系类别（Category）
    - tail: 尾实体名称
    - confidence: 置信度（0-1 之间的浮点数）
    - evidence: 证据（支持该关系的直接引用）
    - temporal_info: 时间信息（如有）
    - mechanism: 机制描述（50-100词）
    - source: 信息来源（如文章pid）
    - subject:链接的头实体对象
    - object:链接的尾实体对象
    - time_info: 抽取的时间信息（分为相对时间、时间范围以及具体时间点）
    """

    head: str
    relation: str
    tail: str
    relation_type: Optional[str]=None
    confidence: Optional[List[float]]=None
    evidence: Optional[List[str]]=None
    mechanism: Optional[str]=None
    source: str = "unknown"
    subject: Optional[KGEntity]=None
    object: Optional[KGEntity]=None
    time_info: Optional[TimeFormat]=None
    
    def get_head(self) -> str:
        return self.head
    def get_relation(self) -> str:
        return self.relation
    def get_relation_type(self) -> Optional[str]:
        return self.relation_type
    def get_tail(self) -> str:
        return self.tail
    def get_confidence(self) -> Optional[List[float]]:
        return self.confidence
    def get_evidence(self) -> Optional[List[str]]:
        return self.evidence
    def get_mechanism(self) -> Optional[str]:
        return self.mechanism
    def get_source(self) -> str:
        return self.source
    def get_subject(self) -> Optional[KGEntity]:
        return self.subject
    def get_object(self) -> Optional[KGEntity]:
        return self.object
    def get_time(self) -> Optional[TimeFormat]:
        return self.time_info
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"({self.head}, -[{self.relation}]->, {self.tail})"
    @classmethod
    def from_dict(cls, data: dict) -> "KGTriple":
        return cls(**data)

def export_triples_to_dicts(triples: list[KGTriple]) -> list[dict]:
    """将 KGTriple 列表导出为字典列表。"""
    triple_dict=[triple.to_dict() for triple in triples]
    return triple_dict