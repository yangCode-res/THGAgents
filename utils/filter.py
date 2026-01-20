from pathlib import Path

from metapub import FindIt, PubMedFetcher

from utils.download import save_pdfs_from_url_list
from utils.pdf2md import deepseek_pdf_to_md_batch
from utils.search import batch_search_reviews_from_user_query

fetch = PubMedFetcher()


def format_reviews(reviews_metadata):  # 将多篇文章格式化为字符串
    formatted_reviews = []
    for review in reviews_metadata:
        formatted_reviews.append(format_review(review))
    return "\n\n".join(formatted_reviews)


def format_review(article):  # 将标题、日期、引用量、摘要、文章id喂给模型
    return f"""
    标题: {article.title}
    发表日期: {article.pubdate}
    引用量: {fetch.related_pmids(article.pmid).__len__()}
    摘要: {article.abstract}
    文章id: {article.pmid}
    """


def ReviewSelection(reviews_metadata, topk=5) -> list:  # 选择最合适的文章
    selection_prompt = f"""
    从以下{len(reviews_metadata)}篇综述中选择最相关的{topk}篇:
    {format_reviews(reviews_metadata)}
    选择标准:
    1. 覆盖查询主题的不同⽅⾯
    2. ⾼引⽤量和影响因⼦
    3. 最新发表⽇期
    4. 包含机制研究和临床应⽤
    请用,隔开的形式返回所选择的{topk}篇综述的pid，不需要其他额外叙述。
    """
    selected_str = str(generate_text(selection_prompt))
    selected_str = selected_str.replace("[", "").replace("]", "")
    selected_5 = [pid.strip() for pid in selected_str.split(",") if pid.strip()]
    return selected_5


def extract_pdf_paths(download_results) -> list[str]:
    """
    从 save_pdfs_from_url_list 的结果中提取成功的本地 PDF 路径列表。
    """
    pdfs = []
    for item in download_results:
        if item.get("status") in {"OK", "EXISTS"} and item.get("path_or_msg"):
            p = Path(item["path_or_msg"])
            if p.is_file() and p.suffix.lower() == ".pdf":
                pdfs.append(str(p))
    return pdfs


