import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from metapub import FindIt, PubMedFetcher
from openai import OpenAI
from pyexpat import model

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Subgraph
from Store.index import get_memory
from utils.download import save_pdfs_from_url_list
from utils.filter import extract_pdf_paths
from utils.pdf2md import deepseek_pdf_to_md_batch
from utils.pdf2mdOCR import ocr_to_md_files
from utils.process_markdown import split_md_by_mixed_count

fetch=PubMedFetcher()
class ReviewFetcherAgent(Agent):
    def __init__(self,client:OpenAI,model_name:str) -> None:
        self.system_prompt="""You are a specialized Review Fetcher Agent for biomedical knowledge graphs. Your task is to fetch relevant literature reviews based on the user's query.
        You are requested to do the following:
        1.Understand the user's query and identify key biomedical concepts, generate MeSH search strategy.
        2.Based on the given abstracts of the reviews, select the most relevant ones that comprehensively cover different aspects of the query.
        """
        self.memory=get_memory()
        self.logger=get_global_logger()
        self.fetch=PubMedFetcher()
        self.k=2
        super().__init__(client,model_name,self.system_prompt)
    
    def process(self, user_query: str, core_entities: Optional[List[str]] = None, save_dir: str | None = None):
        core_entities = core_entities or []
        strategy = self.generateMeSHStrategy(user_query)
        reviews_metadata = self.fetchReviews(strategy, maxlen=30)
        candidate_topk = self.k
        selected_reviews = self.selectReviews(
            reviews_metadata, query=user_query, core_entities=core_entities, topk=candidate_topk, require_all=True
        )

        if not selected_reviews and core_entities:
            def _matched_set(review):
                text = f"{review.title}\n{review.abstract}".lower()
                return {ent for ent in core_entities if ent.lower() in text}

            scored = [
                (review, len(_matched_set(review)), _matched_set(review))
                for review in reviews_metadata
            ]
            scored = [s for s in scored if s[1] > 0]
            if not scored:
                self.logger.warning(
                    "No reviews mention any key entity; aborting fetch."
                )
                sys.exit(1)

            scored.sort(key=lambda x: x[1], reverse=True)
            anchor_review, _, anchor_set = scored[0]
            anchor_pmid = str(anchor_review.pmid)
            missing_entities = [ent for ent in core_entities if ent not in anchor_set]

            if missing_entities:
                missing_query = user_query + " " + " ".join(missing_entities)
                missing_strategy = self.generateMeSHStrategy(missing_query)
                missing_meta = self.fetchReviews(missing_strategy, maxlen=20)

                def _has_missing(review):
                    text = f"{review.title}\n{review.abstract}".lower()
                    return any(m.lower() in text for m in missing_entities)

                missing_filtered = [r for r in missing_meta if _has_missing(r) and str(r.pmid) != anchor_pmid]
                fallback_pmids = [str(r.pmid) for r in missing_filtered][: max(0, self.k - 1)]

                if not fallback_pmids:
                    def _covers_missing_in_original(review):
                        text = f"{review.title}\n{review.abstract}".lower()
                        return any(m.lower() in text for m in missing_entities)

                    from_original = [r for r in reviews_metadata if _covers_missing_in_original(r) and str(r.pmid) != anchor_pmid]
                    if from_original:
                        fallback_pmids = [str(from_original[0].pmid)]
            else:
                fallback_pmids = []

            selected_reviews = [anchor_pmid] + fallback_pmids

            if len(selected_reviews) < self.k:
                self.logger.warning(
                    f"Only found anchor + {len(selected_reviews)-1} missing-entity reviews; expected {self.k}."
                )


        md_outputs: List[str] = []
        idx_offset = 0
        for pmid in selected_reviews:
            if len(md_outputs) >= self.k:
                break
            try:
                url = FindIt(pmid).url
            except Exception:
                self.logger.warning(f"Failed to fetch URL for PMID: {pmid}")
                continue
            if not url:
                continue

            md_list = ocr_to_md_files(
                [url], save_dir=save_dir or "ocr_md_outputs", start_index=idx_offset + 1
            )
            md_list = [md for md in md_list if md]
            if not md_list:
                self.logger.warning(f"OCR failed for PMID: {pmid}, url: {url}")
                continue

            md_outputs.append(md_list[0])
            idx_offset += 1

        if len(md_outputs) < self.k:
            self.logger.warning(
                f"Only generated {len(md_outputs)} MD files (expected {self.k})."
            )

        self.logger.debug(f"md_outputs=> {md_outputs}")

        for md_output in md_outputs[: self.k]:
            paragraphs = split_md_by_mixed_count(md_output)

            for id, content in paragraphs.items():
                for i, content_chunk in enumerate(content):
                    subgraph_id = f"{id}_{i}"
                    meta = {"text": content_chunk, "source": id}
                    s = Subgraph(subgraph_id=subgraph_id, meta=meta)
                    self.memory.register_subgraph(s)

        if len(md_outputs) == 0:
            self.logger.warning("No review URLs produced valid MD files")
            sys.exit(1)

        return md_outputs


    def generateMeSHStrategy(self,user_query:str)->str:
        prompt = f"""
As an expert Biomedical Information Specialist, generate a high-recall PubMed search strategy to find **Reviews** and **Systematic Reviews** relevant to the user's query.

**User Query:** {user_query}

**Critical Rules for Logic:**
1. **Simplify Concepts:** Extract ONLY the **2 most critical concepts** (usually **Intervention** and **Outcome**).
   - *Example:* For "Tirzepatide vs Semaglutide in obesity for CV outcomes in US insurance data", the ONLY concepts are: (Tirzepatide OR Semaglutide) AND (Cardiovascular Outcomes).
   - **IGNORE** geographic locations (e.g., "USA", "China").
   - **IGNORE** data sources (e.g., "insurance claims", "hospital records").
   - **IGNORE** specific study designs in the query text (e.g., "cohort", "RCT") because we will apply a Review filter later.
   - **IGNORE** specific dates mentioned in the text (e.g., "2018-2025") as we will use a date filter.

2. **Handle Comparisons:** If the query compares two drugs (e.g., Drug A vs Drug B), combine them into **ONE** concept using **OR** (e.g., `("Drug A" OR "Drug B")`), rather than splitting them with AND. This ensures we catch reviews discussing the whole class or either drug.

3. **Handle Population (P):** Do NOT create a separate 'AND' block for the disease (e.g., Diabetes/Obesity) *unless* the drugs are used for multiple wildly different conditions. For GLP-1s, the disease is implied by the outcome (CV risk), so adding "AND Diabetes" might restrict results too much. **Prioritize Broad Search.**

**Step-by-Step Construction:**
1. **Concept 1 (Intervention/Exposure):** Expand with MeSH + Keywords + Brand Names + CAS/Drug Codes.
2. **Concept 2 (Outcome/Main Topic):** Expand with MeSH + Keywords (Synonyms).
3. **Combine:** (Concept 1) OR (Concept 2).
4. **Filters:**
   - Apply "Review"[Publication Type] OR "Systematic Review"[Publication Type].
   - Apply Date Range: "2013/01/01"[Date - Publication] : "2025/12/31"[Date - Publication] (Last 5+ years).

**Output format:**
Return ONLY the raw search query string. No markdown, no explanations.
"""
        result=self.call_llm(prompt)
        self.logger.info(f"mesh strategy=> {result}")
        return str(result)
    
    def fetchReviews(self,search_strategy:str,maxlen=1):
        pmids=self.fetch.pmids_for_query(str(search_strategy),retmax=maxlen)
        reviews_metadata = [self.fetch.article_by_pmid(pmid) for pmid in pmids]
        return reviews_metadata
    
    def selectReviews(self,reviews_metadata, query='', core_entities: Optional[List[str]] = None, topk=2, require_all: bool = False) -> List:
        review_str='\n'.join(self.format_review(review) for review in reviews_metadata)
        self.logger.debug(f"review_str=> {review_str}")
        selection_prompt = f"""
        here is the user query: {query}, and here are the reviews:
        From the following {len(reviews_metadata)} reviews, select the most relevant {topk} ones:
        {review_str}
        Selection criteria:
        1. Cover different aspects of the query topic
        2. High citation count and impact factor
        3. Recent publication date
        4. Include mechanism studies and clinical applications
        5. Select the reviews that most caters to the user query.
        6. You should also take the richness of the review(identified as the range of pages here however the page range is not always available) into consideration so that we could build a knowledge graph with more triples.
        If the page range is not available, you should put other requirements first.
        7. Most importantly, ensure the key entities would have the potential to have connections according to the selected reviews(If there is no possibility to achieve this, return no relation).
        Please return the selected {topk} review pmids in a comma-separated format without any additional description.

        NOTE: key entities: {core_entities}. All key entities MUST co-occur in the abstracts you select 
        (e.g. You opted for two abstacts,there are two situations that meet our need:
        1. for core entity A and B A appears in passage A',B appears in passage B',and A and B have the potential to link each other. 
        2.Both of them appear in one abstract.) 
        Rank among them per criteria above.
        e.g. The query might be "What Can we hypothesize the potential relation between A and B?"
        The priority of the reviews should be:
        1. Both A and B are covered in the review.
        2. Either A or B is covered in the review.
        3. None of A and B is covered in the review.
        However, you should ensure the selected reviews could cover both A and B as much as possible.
        """
        selected_str = str(self.call_llm(selection_prompt))
        selected_str = selected_str.replace("[", "").replace("]", "")
        selected = [pid.strip() for pid in selected_str.split(",") if pid.strip()]

        pmid_order = [str(article.pmid) for article in reviews_metadata]
        for pmid in pmid_order:
            if len(selected) >= topk:
                break
            if pmid and pmid not in selected:
                selected.append(pmid)

        return selected[:topk]
    def format_review(self,article):
        return f"""
        title: {article.title}
        pubdate: {article.pubdate}
        citation_count: {fetch.related_pmids(article.pmid).__len__()}
        abstract: {article.abstract}
        pmid: {article.pmid}
        page-range:{article.pages}
        """
