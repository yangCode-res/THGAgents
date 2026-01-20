import json

from openai import OpenAI

from Core.Agent import Agent
from Store.index import get_memory


class QueryClarifyAgent(Agent):
    def __init__(self,client:OpenAI,model_name:str) -> None:
        self.system_prompt="""You are a specialized Query Clarification Agent for biomedical knowledge graphs. Your task is to help users refine and clarify their queries to ensure accurate and relevant information retrieval.
        OBJECTIVE: Assist users in formulating precise and unambiguous queries related to biomedical topics, ensuring that the queries are well-defined for effective knowledge graph extraction.
        Given the following background information, generate a clear and concise question that can be used for literature retrieval:

        Background Information:
        - **Disease**: Diabetes Mellitus
        - **Treatment**: Insulin Therapy
        - **Research Focus**: The effects of insulin therapy on diabetes management, blood glucose control, and the impact of different insulin dosages.
        - **Literature Type**: Clinical studies, randomized controlled trials, reviews.
        - **Objective**: To gather relevant literature on the efficacy of insulin therapy in diabetes management, particularly focusing on blood glucose control, long-term effects, and potential side effects.

        Please generate a specific question that addresses the research objectives, clearly indicating the disease (Diabetes Mellitus) and treatment (Insulin Therapy). The question should be structured to facilitate the retrieval of relevant clinical research literature.
        Moreover,you should also provide the core entities involved in the question.
        You should also emphasize the main intention of the user query.
        Example of what the question might look like:
        "What is the effect of insulin therapy on blood glucose control and long-term outcomes in patients with diabetes mellitus?"
        And the core entities are "Insulin Therapy" and "Diabetes Mellitus".
        Return the results in the following JSON format:
        {
          "clarified_question": "Your generated question here",
          "core_entities": ["Entity1", "Entity2"...],
          "main_intention": "The main intention of the user query here"
        }
        NOTE:the core entities should be the entities that are most relevant to the user query and can help in retrieving the most pertinent literature.
"""
        self.memory=get_memory()
        super().__init__(client,model_name,self.system_prompt)
    
    def process(self,user_query:str)->str:
        prompt=f"""User Query: {user_query}
Based on the user query above, please generate a clarified and specific question that can be used for literature retrieval in the context of biomedical knowledge graphs. Ensure that the question is precise and unambiguous.
ensure the output must be Json format not List
"""     
        response=self.call_llm(prompt)
        response=self.parse_json(response)
        return response
        try:
            response_json=json.loads(response)
            return response_json
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON.")