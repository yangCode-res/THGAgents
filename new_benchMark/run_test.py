from os import system
from tkinter import FALSE
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Alignment_triple.index import AlignmentTripleAgent
from Agents.Fusion_subgraph.index import SubgraphMerger
from Agents.KeywordEntitySearchAgent.index import KeywordEntitySearchAgent
from Agents.Path_extraction.penalty import PathExtractionAgent
from Agents.HypothesisGenerationAgent.index import HypothesisGenerationAgent
from Agents.ReflectionAgent.index import ReflectionAgent
from Agents.Hypotheses_Edit.index import HypothesisEditAgent
from Agents.Query_clarify.index import QueryClarifyAgent
from new_benchMark.Dataloader import BenchmarkDataLoader

from typing import Dict, List, Any, Optional
from Memory.index import Memory, Subgraph ,load_memory_from_json
from openai import OpenAI
from Core.Agent import Agent
from Store.index import get_memory
from Logger.index import get_global_logger
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pathlib import Path
class BenchmarkTestRunner(Agent):
    """
    基准测试运行器类，用于加载和处理 benchmark 数据
    """

    def __init__(self,client:OpenAI,model_name:str,data_path: str = None):
        """
        初始化测试运行器
    
        Args:
            data_path: 数据文件路径，如果为 None 则使用默认路径
        """
        self.data_loader = BenchmarkDataLoader(data_path)
        self.loaded_data = None
        self.processed_data = None
        self.memory=get_memory()
        self.logger=get_global_logger()
        self.system_prompt=""
        super().__init__(client,model_name,self.system_prompt)
    def load_data(self) -> Dict[str, Any]:
        """
        加载 benchmark 数据

        Returns:
            加载的原始数据字典
        """
        print("开始加载 benchmark 数据...")
        self.loaded_data = self.data_loader.run()
        print("数据加载完成！")
        return self.loaded_data
    
    def causal_knowledge_graph_construction(self):
        review_groups=[]
        for group in self.loaded_data:
            reviews=[]
            for review in group['reviews']:
                if review.get('pmc_full_text',None) is not None:
                    reviews.append(review)
            review_groups.append(reviews)

        for group_idx,group in enumerate(review_groups):
            self.memory.reset()
            subgraph_id=0
            for idx,review in enumerate(group):
                full_text='\n'.join([section for section in review.get('pmc_full_text',None).get('sections',None).values()])
                paragraphs = split_text_by_period_boundary(
                    full_text,
                    target_count=1200,
                    min_count=200,
                    name=str(review.get("pmid", "text"))
                )
                for id, content in paragraphs.items():
                    for i, content_chunk in enumerate(content):
                        subgraph_id = f"{review.get('pmid',None)}_{i}"
                        meta = {"text": content_chunk, "source": id}
                        s = Subgraph(subgraph_id=subgraph_id, meta=meta)
                        self.memory.register_subgraph(s)
            self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/graph_register_snapshots")
    def entity_extraction_run(self,memory:Memory,group_idx:int,subgraph_id:Optional[str]=None,subgraph:Optional[Subgraph]=None):
        entity_extraction_agent=EntityExtractionAgent(self.client,self.model_name,memory=memory)
        if subgraph_id is not None and subgraph is not None:
            entity_extraction_agent._process_single_subgraph(subgraph_id,subgraph)
            return
        entity_extraction_agent.process(max_workers=2)
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/entity_extraction_snapshots")
    def relationship_extraction_run(self,memory:Memory,group_idx:int,subgraph_id:Optional[str]=None,subgraph:Optional[Subgraph]=None):
        relationship_extraction_agent=RelationshipExtractionAgent(self.client,self.model_name,memory=memory)
        if subgraph_id is not None and subgraph is not None:
            relationship_extraction_agent.process_subgraph(subgraph)
            return
        relationship_extraction_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/relationship_extraction_snapshots")
    def entity_normalization_run(self,memory:Memory,group_idx:int,subgraph_id:Optional[str]=None,subgraph:Optional[Subgraph]=None):
        entity_normalization_agent=EntityNormalizationAgent(self.client,self.model_name,memory=memory)
        if subgraph_id is not None and subgraph is not None:
            entity_normalization_agent._process_one_subgraph(subgraph_id,subgraph)
            return
        entity_normalization_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/entity_normalization_snapshots")  
    def collaboration_extraction_run(self,memory:Memory,group_idx:int,subgraph_id:Optional[str]=None,subgraph:Optional[Subgraph]=None):
        collaboration_extraction_agent=CollaborationExtractionAgent(self.client,self.model_name,memory=memory)
        if subgraph_id is not None and subgraph is not None:
            collaboration_extraction_agent.process_subgraph(subgraph)
            return
        collaboration_extraction_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/collaboration_extraction_snapshots")
    def causal_extraction_run(self,memory:Memory,group_idx:int,subgraph_id:Optional[str]=None,subgraph:Optional[Subgraph]=None):
        causal_extraction_agent=CausalExtractionAgent(self.client,self.model_name,memory=memory)
        if subgraph_id is not None and subgraph is not None:
            causal_extraction_agent.process_subgraph(subgraph)
            return
        causal_extraction_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/causal_extraction_snapshots")
    def AlignmentTripleAgent_run(self,memory:Memory,group_idx:int):
        alignment_triple_agent=AlignmentTripleAgent(self.client,self.model_name,memory=memory)
        alignment_triple_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/alignment_triple_snapshots")
    def SubgraphMerger_run(self,memory:Memory,group_idx:int):
        subgraph_merger=SubgraphMerger(self.client,self.model_name,memory=memory)
        subgraph_merger.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/subgraph_merger_snapshots")
    def keyEntitySearch_run(self,memory:Memory,group_idx:int,core_entities:List[str],queryId:int):
        key_entity_search_agent=KeywordEntitySearchAgent(self.client,self.model_name,memory=memory,keywords=core_entities)
        key_entity_search_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/key_entity_search_snapshots/{queryId}")
    def pathExtraction_run(self,memory:Memory,group_idx:int,query:str,queryId:int):
        # need query
        path_extraction_agent=PathExtractionAgent(self.client,self.model_name,memory=memory,query=query)
        path_extraction_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/path_extraction_snapshots/{queryId}")
    def hypothesisGeneration_run(self,memory:Memory,group_idx:int,query:str,queryId:int):
        # need query
        hypothesis_generation_agent=HypothesisGenerationAgent(self.client,self.model_name,memory=memory,query=query,output_path=f"./new_benchMark/Group/{group_idx}/hypothesis_generation_snapshots/{queryId}/output.json")
        hypothesis_generation_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/hypothesis_generation_snapshots/{queryId}")
    def reflection_run(self,memory:Memory,group_idx:int,queryId:int):
        reflection_agent=ReflectionAgent(self.client,model_name=self.model_name,memory=memory)
        reflection_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/reflection_snapshots/{queryId}")
    def hypothesisEdit_run(self,memory:Memory,group_idx:int,query:str,queryId:int,output_path:str):
        # need query
        hypothesis_edit_agent=HypothesisEditAgent(self.client,self.model_name,query=query,memory=memory,output_path=output_path)
        hypothesis_edit_agent.process()
        self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/hypothesis_edit_snapshots/{queryId}")
    def query_claify_run(self,group_idx:int,user_query:str):
        queryclarifyagent = QueryClarifyAgent(self.client, self.model_name) # type: ignore
        response = queryclarifyagent.process(user_query)
        clarified_query = response.get("clarified_question", user_query) # type: ignore
        self.clarified_query=clarified_query
        core_entities= response.get("core_entities", []) # type: ignore
        intention= response.get("main_intention", "") # type: ignore
        print(f"Core Entities: {core_entities}")
        print("intention=>",intention)
        print("clarified_query=>",clarified_query)
        
        return clarified_query,core_entities,intention
    def runKgConstruction(self):
        self.load_data()
        for group_idx in range(0,2):
            self.entity_extraction_run(self.memory,group_idx)
            self.entity_normalization_run(self.memory,group_idx)
            self.relationship_extraction_run(self.memory,group_idx)
            self.collaboration_extraction_run(self.memory,group_idx)
            self.causal_extraction_run(self.memory,group_idx)
            def needs_work(sg):
                return len(sg.relations.all()) == 0 or len(sg.entities.all()) == 0

            def process_one(sg_id: str, sg: Subgraph) -> None:
                if len(sg.entities.all()) == 0:
                    self.entity_extraction_run(self.memory, group_idx, sg_id, sg)
                    sg= self.memory.subgraphs[sg_id]  # reload sg after modification
                    self.entity_normalization_run(self.memory, group_idx, sg_id, sg)
                    sg= self.memory.subgraphs[sg_id]  # reload sg after modification
                if len(sg.relations.all()) == 0:
                    self.relationship_extraction_run(self.memory, group_idx, sg_id, sg)
                    sg= self.memory.subgraphs[sg_id]  # reload sg after modification
                self.collaboration_extraction_run(self.memory, group_idx, sg_id, sg)
                sg= self.memory.subgraphs[sg_id]  # reload sg after modification
                self.causal_extraction_run(self.memory, group_idx, sg_id, sg)
            refined_times=1
            pending = [(sid, s) for sid, s in self.memory.subgraphs.items() if needs_work(s)]
            while pending and refined_times<2:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(process_one, sid, sg): sid for sid, sg in pending}
                    for _ in as_completed(futures):
                        pass  # wait for all subgraphs in this wave
                pending = [(sid, sg) for sid, sg in self.memory.subgraphs.items() if needs_work(sg)]
                refined_times+=1
            self.memory.dump_json(f"./new_benchMark/Group/{group_idx}/causal_extraction_snapshots_final")
            self.AlignmentTripleAgent_run(self.memory,group_idx)
            self.SubgraphMerger_run(self.memory,group_idx)
   
    def hypothesisGeneration(self):
        self.load_data()
        for group_idx in range(0,2):
            query_data=self.loaded_data[group_idx]
            merged_hypotheses_queries = self.merge_hypotheses_and_queries(query_data)
            for idx,item in enumerate(merged_hypotheses_queries):

                query_str=item['query']['query_string']
                print(query_str)
                clarified_query,core_entities,intention = self.query_claify_run(group_idx=group_idx,user_query=query_str)
                
                self.keyEntitySearch_run(self.memory,group_idx,core_entities,idx)
                self.pathExtraction_run(self.memory,group_idx,query_str,idx)

                self.hypothesisGeneration_run(self.memory,group_idx,query_str,idx)
                self.reflection_run(self.memory,group_idx,idx)
                edit_output_path=get_group_memory_only_path(group_idx,"Generation_Result",idx)
                self.hypothesisEdit_run(self.memory,group_idx,query_str,idx,output_path=str(edit_output_path))

    def merge_hypotheses_and_queries(self, query_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        合并 hypotheses 和 queries 为一一对应的列表

        Args:
            query_data: 包含 hypotheses 和 queries 的查询数据

        Returns:
            合并后的 hypothesis-query 对列表
        """
        hypotheses = query_data.get("hypotheses", [])
        queries = query_data.get("queries", [])

        # 创建 hypothesis_id 到 hypothesis 的映射
        hypotheses_dict = {h['hypothesis_id']: h for h in hypotheses}

        # 合并对应的 hypothesis 和 query
        merged_items = []
        for query in queries:
            hypothesis_id = query.get('hypothesis_id')
            if hypothesis_id in hypotheses_dict:
                merged_item = {
                    'hypothesis_id': hypothesis_id,
                    'hypothesis': hypotheses_dict[hypothesis_id],
                    'query': query
                }
                merged_items.append(merged_item)
            else:
                print(f"警告: 找不到 hypothesis_id {hypothesis_id} 对应的 hypothesis")

        return merged_items
    def runReaschAgent(self):
        self.load_data()
        for group_idx in range(0,1):
            query_data=self.loaded_data[group_idx]
            merged_hypotheses_queries = self.merge_hypotheses_and_queries(query_data)
            reasch_agent=MultiTrackResearchAgent(client=self.client,model_name=self.model_name,refinement_rounds=1)
            for idx,item in enumerate(merged_hypotheses_queries):
                query_str=item['query']['query_string']
                reasch_agent.run(query_str,output_path=f"./new_benchMark/reaschAgent/{group_idx}/{idx}/output.json")
def get_mixed_word_count(text: str) -> int:
    """
    中文按字，英文/数字按词（空格分隔）
    """
    if not text:
        return 0
    cjk_pattern = re.compile(r'[\u4e00-\u9fff]')
    cjk_count = len(cjk_pattern.findall(text))
    text_without_cjk = cjk_pattern.sub(' ', text)
    english_word_count = len(text_without_cjk.split())
    return cjk_count + english_word_count


def split_text_by_period_boundary(
    text: str,
    target_count: int = 1200,
    min_count: int = 200,
    name: str = "text",
    # 如果你只想用 "."，可改成 r"([.]+)" 并去掉其它符号
    sentence_end_pattern: str = r"([。.!?]+)",
) -> Dict[str, List[str]]:
    """
    用句末符号（默认包含 . 。 ! ?）做“自然完结边界”切块。
    - 优先在句末边界切，不盲切
    - 强制保证每个 chunk >= min_count（不会直接丢弃过短块，而是合并）
    - 若出现“单句超长 > target_count”，会在句内做降级切分（按空白token / 字符串片段）
    """
    if text is None:
        raise ValueError("text is None")
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text)}")
    if target_count <= 0:
        raise ValueError("target_count must be > 0")
    if min_count < 0:
        raise ValueError("min_count must be >= 0")

    # 1) 规范空白：没有 \n 也没关系，压缩多空格即可
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return {name: []}

    # 2) 按句末符号切成“句子单元”，并保留句末符号
    parts = re.split(sentence_end_pattern, t)
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        body = parts[i].strip()
        end = parts[i + 1] if i + 1 < len(parts) else ""
        s = (body + end).strip()
        if s:
            sentences.append(s)

    # 如果完全没匹配到句末符号，就把全文当作一个“句子”
    if not sentences:
        sentences = [t]

    # 3) 处理“单句超长”：句内降级切分（尽量按空白token切；实在不行再硬切）
    def split_very_long_sentence(s: str) -> List[str]:
        if get_mixed_word_count(s) <= target_count:
            return [s]

        # 先按空白保留分隔符切（尽量不破坏文本）
        tokens = re.split(r"(\s+)", s)
        out, cur, cur_count = [], [], 0

        for tok in tokens:
            if not tok:
                continue
            tok_count = get_mixed_word_count(tok)

            # 单 token 仍超长：硬切（防死循环）
            if tok_count > target_count:
                if cur:
                    out.append("".join(cur).strip())
                    cur, cur_count = [], 0
                # 硬切：按字符片段切（这里用长度近似，足够兜底）
                # 如果你希望更精确（按 mixed_count 切），也可以再细化
                step = max(200, len(tok) // (tok_count // target_count + 1))
                for j in range(0, len(tok), step):
                    piece = tok[j:j+step].strip()
                    if piece:
                        out.append(piece)
                continue

            if cur and (cur_count + tok_count > target_count):
                out.append("".join(cur).strip())
                cur = [tok]
                cur_count = tok_count
            else:
                cur.append(tok)
                cur_count += tok_count

        if cur:
            out.append("".join(cur).strip())
        return [x for x in out if x]

    normalized_sentences: List[str] = []
    for s in sentences:
        normalized_sentences.extend(split_very_long_sentence(s))

    # 4) 贪婪合并：优先在句边界结束 chunk
    chunks: List[str] = []
    cur: List[str] = []
    cur_count = 0

    for s in normalized_sentences:
        sc = get_mixed_word_count(s)

        # 若加入后会超过 target_count
        if cur and (cur_count + sc > target_count):
            # 如果当前 chunk 已经达到 min_count，就在这里自然断开
            if cur_count >= min_count:
                chunks.append(" ".join(cur).strip())
                cur = [s]
                cur_count = sc
            else:
                # 当前 chunk 太短：宁可超 target，也要把它补到 >= min_count
                cur.append(s)
                cur_count += sc
        else:
            cur.append(s)
            cur_count += sc

    if cur:
        chunks.append(" ".join(cur).strip())

    # 5) 强制保证 min_count：把过短块与相邻块合并（不丢弃）
    if min_count > 0 and len(chunks) >= 2:
        merged: List[str] = []
        for ch in chunks:
            if not merged:
                merged.append(ch)
                continue

            if get_mixed_word_count(ch) < min_count:
                # 优先并到前一个
                merged[-1] = (merged[-1] + " " + ch).strip()
            else:
                merged.append(ch)

        # 末尾如果仍然过短（极端情况），再向前合并一次
        while len(merged) >= 2 and get_mixed_word_count(merged[-1]) < min_count:
            last = merged.pop()
            merged[-1] = (merged[-1] + " " + last).strip()

        chunks = merged
        print(f"[Chunk statistics] total chunks = {len(chunks)}")
        for i, ch in enumerate(chunks):
            cnt = get_mixed_word_count(ch)
            print(f"  - chunk {i}: {cnt} words/chars")


    return {name: chunks}

def get_group_memory_path(group_id: int,task_type:str,query_id:int) -> Path:
    base_dir = Path(__file__).resolve().parent
    snapshot_dir = (
        base_dir / "Group" / str(group_id) / f"{task_type}_snapshots"/str(query_id)
    )

    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot dir not found: {snapshot_dir}")

    json_files = list(snapshot_dir.glob("memory-*.json"))
    if not json_files:
        raise FileNotFoundError(f"No memory json found in {snapshot_dir}")

    # 🔥 关键：按修改时间排序，取最新
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)

    return latest_json

def get_group_memory_only_path(group_id: int,task_type:str,query_id:int) -> Path:
    base_dir = Path(__file__).resolve().parent
    snapshot_dir = (
        base_dir / "Group" / str(group_id) / f"{task_type}_snapshots"/str(query_id)
    )
    return snapshot_dir