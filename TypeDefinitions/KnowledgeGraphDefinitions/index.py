from dataclasses import dataclass
from typing import List, Optional

from Logger.index import get_global_logger
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


@dataclass
class KnowledgeGraph:
    def __init__(self,relations:Optional[List[KGTriple]]=None):
 
        self.Graph={}
        self.memory=get_memory()
        self.logger=get_global_logger()
        relations=relations or self.memory.relations.all()
        if relations is None:
            self.logger.error("No relations provided to initialize KnowledgeGraph and memory is empty.")
            return
        for triple in relations:
            self.add_edge(triple)
        self.sort_by_confidence()
    
    def add_edge(self,triple:KGTriple):
        # 处理 subject
        if triple.subject is None:
            self.logger.warning(f"Skipping triple with None subject: {triple}")
            return
        if isinstance(triple.subject, KGEntity):
            subj = triple.subject.entity_id
        elif isinstance(triple.subject, dict):
            subj = KGEntity(**triple.subject).entity_id
        else:
            self.logger.warning(f"Skipping triple with invalid subject type: {type(triple.subject)}")
            return
        
        # 处理 object
        if triple.object is None:
            self.logger.warning(f"Skipping triple with None object: {triple}")
            return
        if isinstance(triple.object, KGEntity):
            obj = triple.object.entity_id
        elif isinstance(triple.object, dict):
            obj = KGEntity(**triple.object).entity_id
        else:
            self.logger.warning(f"Skipping triple with invalid object type: {type(triple.object)}")
            return
        
        if subj not in self.Graph:
            self.Graph[subj]=[]
            self.Graph[subj].append((obj,triple))
        else:
            self.Graph[subj].append((obj,triple))
    
    def sort_by_confidence(self):
 
        relation_priority = {
    'CAUSES': 0,          
    'TREATS': 1,           
    'INHIBITS': 2,         
    'ACTIVATES': 3,        
    'REGULATES': 4,        
    'INCREASES/DECREASES': 5,  
    'ASSOCIATED_WITH': 6,  
    'INTERACTS_WITH': 7    
}
        for subj in self.Graph:
            self.Graph[subj].sort(
                # 修改这里的 lambda 函数
                key=lambda x:(relation_priority.get(x[1].relation, 8), x[1].confidence[0] if (x[1].confidence and len(x[1].confidence) > 0) else 0.5),
                reverse=True
            )
    
    def get_subgraph(self,entity:KGEntity,depth:int)->'KnowledgeGraph':

        subgraph_relations=[]
        visited=set()
        
        def dfs(current_entity:KGEntity,current_depth:int):
            if current_depth>depth or current_entity in visited:
                return
            visited.add(current_entity)
            if current_entity in self.Graph:
                for neighbor,triple in self.Graph[current_entity]:
                    subgraph_relations.append(triple)
                    dfs(neighbor,current_depth+1)
        dfs(entity,0)
        return subgraph_relations

       