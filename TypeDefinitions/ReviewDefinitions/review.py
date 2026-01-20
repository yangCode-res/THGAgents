from dataclasses import dataclass


@dataclass
class Review:
    """综述文章的定义信息。
    - title: 文章标题
    - pubdate: 发表日期
    - citation_count: 引用量
    - abstract: 文章摘要
    - pmid: 文章的 PubMed ID
    """
    title: str
    pubdate: str
    citation_count: int
    abstract: str
    pmid: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "pubdate": self.pubdate,
            "citation_count": self.citation_count,
            "abstract": self.abstract,
            "pmid": self.pmid,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Review":
        return cls(
            title=data.get("title", ""),
            pubdate=data.get("pubdate", ""),
            citation_count=data.get("citation_count", 0),
            abstract=data.get("abstract", ""),
            pmid=data.get("pmid", ""),
        )
    
    def __str__(self) -> str:
        return f"Title: {self.title}\nPubDate: {self.pubdate}\nCitation Count: {self.citation_count}\nAbstract: {self.abstract}\nPMID: {self.pmid}"
    