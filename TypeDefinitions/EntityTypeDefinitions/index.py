from ast import main
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class EntityType(str, Enum):
    """实体类型"""
    DISEASE = "disease"
    DRUG = "drug"
    GENE = "gene"
    PROTEIN = "protein"
    PATHWAY = "pathway"
    SYMPTOM = "symptom"
    TREATMENT = "treatment"
    BIOMARKER = "biomarker"
    MECHANISM = "mechanism"
    OTHER = "other"


@dataclass
class EntityDefinition:
    """实体定义信息。

    - name: 实体名称
    - description: 实体解释/定义
    - examples: 例子（样例字符串列表）
    - include: 包含/同义/相关词（用于归一或匹配）
    """

    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)

# 预置示例定义（完整覆盖 EntityType）
ENTITY_DEFINITIONS: Dict[EntityType, EntityDefinition] = {
    # 1) 复用提供资料
    EntityType.DRUG: EntityDefinition(
        name="DRUG",
        description="Pharmaceuticals, therapeutic compounds, medications",
        examples=["aspirin", "ibuprofen", "metformin", "acetylsalicylic acid"],
        include=["brand names", "generic names", "chemical names"],
    ),
    EntityType.DISEASE: EntityDefinition(
        name="DISEASE",
        description="Medical conditions, disorders, syndromes, pathologies",
        examples=["diabetes", "cancer", "hypertension", "myocardial infarction"],
        include=["acute and chronic conditions", "symptoms"],
    ),
    EntityType.GENE: EntityDefinition(
        name="GENE",
        description="Genetic elements, chromosomal regions, genetic variants",
        examples=["BRCA1", "TP53", "APOE", "rs123456"],
        include=["gene symbols", "SNPs", "genetic loci"],
    ),
    EntityType.PROTEIN: EntityDefinition(
        name="PROTEIN",
        description="Enzymes, receptors, antibodies, protein complexes",
        examples=["insulin", "COX-2", "p53", "immunoglobulin"],
        include=["protein names", "enzyme classes"],
    ),
    EntityType.PATHWAY: EntityDefinition(
        name="PATHWAY",
        description="Biological pathways, signaling cascades, metabolic routes",
        examples=["glycolysis", "PI3K/Akt pathway", "cell cycle"],
        include=["regulatory networks", "metabolic pathways"],
    ),

    # 2) 补充定义（与你的类型枚举保持一致）
    EntityType.SYMPTOM: EntityDefinition(
        name="SYMPTOM",
        description="Patient-reported or clinically observed manifestations indicating a possible disease state.",
        examples=["fever", "cough", "chest pain", "fatigue", "dyspnea"],
        include=["clinical manifestations", "signs and symptoms", "patient-reported outcomes"],
    ),
    EntityType.TREATMENT: EntityDefinition(
        name="TREATMENT",
        description="Medical interventions, therapies, or procedures intended to prevent, ameliorate, or cure disease.",
        examples=["surgery", "chemotherapy", "radiotherapy", "insulin therapy", "physical therapy"],
        include=["pharmacologic and nonpharmacologic therapies", "medical procedures", "devices", "lifestyle interventions"],
    ),
    EntityType.BIOMARKER: EntityDefinition(
        name="BIOMARKER",
        description="Measurable indicators of a biological state or condition used for diagnosis, prognosis, or monitoring.",
        examples=["CRP", "troponin I", "HbA1c", "PSA"],
        include=["laboratory tests", "genomic/proteomic/metabolomic markers", "imaging biomarkers"],
    ),
    EntityType.MECHANISM: EntityDefinition(
        name="MECHANISM",
        description="Underlying biological processes or mechanisms of action that explain disease etiology or treatment effects.",
        examples=["apoptosis", "oxidative stress", "inflammation", "receptor binding", "signal transduction"],
        include=["mechanisms of action (MOA)", "molecular processes", "causal mechanisms"],
    ),
    EntityType.OTHER: EntityDefinition(
        name="OTHER",
        description="Biomedical entities or concepts not covered by the above categories but relevant to the task context.",
        examples=["clinical guideline", "exposure", "risk factor", "placebo"],
        include=["miscellaneous biomedical concepts", "study design elements", "context-specific terms"],
    ),
}
@dataclass
class KGEntity:
    """
    Represents a canonical entity in the knowledge graph.

    Attributes:
        entity_id: Unique identifier
        entity_type: Semantic type (e.g. Drug, Disease, Gene, Protein)
        name: Display name
        normalized_id: Reference to standard ontology (e.g., UMLS:C0004238)
        aliases: Alternative names for the entity
    """
    entity_id: str
    entity_type: str = "Unknown"
    name: str = ""
    normalized_id: str = "N/A"
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    def __hash__(self):
        return hash(self.entity_id)
    def __str__(self) -> str:
        """String representation of the entity."""
        return f"{self.name} ({self.entity_type})"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    def get_id(self) -> str:
        return self.entity_id
    def get_type(self) -> str:
        return self.entity_type
    def get_name(self) -> str:
        return self.name
    def get_normalized_id(self) -> str:
        return self.normalized_id
    def get_aliases(self) -> List[str]:
        return self.aliases
    @classmethod
    def from_dict(cls, data: Dict) -> 'KGEntity':
        if isinstance(data, KGEntity):
            return data
        """Create instance from dictionary."""
        return cls(**data)
def format_entity_definition(
    definition: EntityDefinition,
    index: int = 1,
    label: Optional[str] = None,
) -> str:
    """将 EntityDefinition 转换为指定的多行字符串格式。

    示例输出：
    1. DRUG: Pharmaceuticals, therapeutic compounds, medications

       - Examples: aspirin, ibuprofen, metformin, acetylsalicylic acid

       - Include: brand names, generic names, chemical names
    """
    name = label or definition.name
    parts: List[str] = [f"{index}. {name}: {definition.description}"]

    if definition.examples:
        parts.append("   - Examples: " + ", ".join(definition.examples))
    if definition.include:
        parts.append("   - Include: " + ", ".join(definition.include))

    # 在首行与每个条目之间添加一个空行，匹配示例格式
    return "\n\n".join(parts)

def format_all_entity_definitions(
    entity_definitions: Dict[EntityType, EntityDefinition] = ENTITY_DEFINITIONS,
    order: Optional[List[EntityType]] = None,
    labels: Optional[Dict[EntityType, str]] = None,
) -> str:
    """
    将所有实体定义按给定顺序用 format_entity_definition 转化为大字符串并返回。

    Args:
        entity_definitions: 实体定义字典，默认使用全局 ENTITY_DEFINITIONS
        order: 可选，自定义输出顺序（EntityType 列表）。默认按照 EntityType 的枚举顺序
        labels: 可选，为某些类型覆盖显示名称的映射，例如 {EntityType.DRUG: "DRUGS"}

    Returns:
        str: 拼接后的完整多段说明文本
    """
    if order is None:
        order = list(EntityType)  # 严格按枚举顺序

    parts: List[str] = []
    idx = 1
    for et in order:
        definition = entity_definitions.get(et)
        if not definition:
            continue
        label = labels.get(et) if labels else None
        parts.append(format_entity_definition(definition, index=idx, label=label))
        idx += 1

    return "\n\n".join(parts)

if __name__ == "__main__":
    print(format_all_entity_definitions())