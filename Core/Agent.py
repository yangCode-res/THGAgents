from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from multiprocessing import process
from typing import Any, Dict, List, Optional,Union
from urllib import response
from pydantic import Json
from venv import logger
import json
import re
import ast
from openai import OpenAI

from Logger.index import get_global_logger
from Store.index import get_memory


@dataclass
class Agent:
    """
    Universal parent class Agent.

    Required attributes (aligned with planning fields in meta_agent):
    - template_id: Template ID
    - name: Agent name
    - responsibility: Responsibility description of this Agent
    - entity_focus: List of focused entity types (can be strings or enums in different projects)
    - relation_focus: List of focused relation types (can be strings or enums in different projects)
    - priority: Priority of this Agent (smaller value means higher priority)
    """

    template_id: str
    name: str
    responsibility: str
    entity_focus: List[Any] = field(default_factory=list)
    relation_focus: List[Any] = field(default_factory=list)
    priority: int = 1

    metadata: Dict[str, Any] = field(default_factory=dict)

    def configure(
        self,
        *,
        template_id: Optional[str] = None,
        name: Optional[str] = None,
        responsibility: Optional[str] = None,
        entity_focus: Optional[List[Any]] = None,
        relation_focus: Optional[List[Any]] = None,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update Agent configuration as needed."""
        if template_id is not None:
            self.template_id = template_id
        if name is not None:
            self.name = name
        if responsibility is not None:
            self.responsibility = responsibility
        if entity_focus is not None:
            self.entity_focus = list(entity_focus)
        if relation_focus is not None:
            self.relation_focus = list(relation_focus)
        if priority is not None:
            self.priority = int(priority)
        if metadata is not None:
            self.metadata.update(metadata)
    
    def __init__(self,client:OpenAI,model_name:str,system_prompt:str):
        """Initialize Agent base class.
        Args:
            client (OpenAI): OpenAI client instance.
            model_name (str): Model name to use.
            system_prompt (str): Agent prompt.
            metadata (Dict[str, Any], optional): Additional metadata. Defaults to empty dict.
        """
        self.client=client
        self.model_name=model_name
        self.system_prompt=system_prompt
        self.metadata={
            "total_calls":0,
            "total_call_prompt_tokens":0,
            "total_call_completion_tokens":0,
            "total_call_processing_time":0.0
        }
        self.logger=get_global_logger()
        self.memory = get_memory()
    def call_llm(self,prompt:str,temperature:float=0.1,max_tokens:Optional[int]=None,system_prompt:Optional[str]=None):
        """Call language model interface.
        Args:
            prompt (str): User input prompt.
            temperature (float, optional): Randomness of generated text. Defaults to 0.1.
            max_tokens (Optional[int], optional): Maximum length of generated text. Defaults to None.
            system_prompt (Optional[str], optional): System prompt, overrides default system_prompt. Defaults to None."""
        
        start_time=time.time()
        try:
            messages=[
                {"role":"system","content":system_prompt if system_prompt else self.system_prompt},
                {"role":"user","content":prompt}
            ]
            call_kwargs={
                "model":self.model_name,
                "messages":messages,
                "temperature":temperature
            }
            if max_tokens:
                call_kwargs["max_tokens"]=max_tokens
            response=self.client.chat.completions.create(**call_kwargs)

            content=response.choices[0].message.content.strip()
            prompt_tokens=response.usage.prompt_tokens
            completion_tokens=response.usage.completion_tokens
            processing_time=time.time()-start_time
            
            self.metadata["total_calls"]+=1
            self.metadata["total_call_prompt_tokens"]+=prompt_tokens
            self.metadata["total_call_completion_tokens"]+=completion_tokens
            self.metadata["total_call_processing_time"]+=processing_time

            return content
        except Exception as e:
            processing_time=time.time()-start_time
            logger.error(f"LLM call failed: {e}")
            raise e

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for logging/serialization."""
        return asdict(self)
    def parse_json_alignment(self,response:str)->List[Dict]: # type: ignore
        try:
            temp=json.load(response)
        except Exception as e:
            response=self._extract_json_from_markdown_alignment(response)
        """Parse JSON format response returned by LLM.
        Args:
            response (str): String response returned by LLM, expected to be JSON format.
        Returns:
            List[Dict]: Parsed JSON object list."""
        import json
        
        try:
            if "[" in response and "]" in response:
                json_str=response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)
            return json.loads(response)
        except Exception as e:
            logger=get_global_logger()
            logger.info(f"Failed to parse JSON response {e}")
    def parse_json(self, response: Union[str, Any]) -> List[Dict]:

        if isinstance(response, list):
            return [x for x in response if isinstance(x, dict)]
        if isinstance(response, dict):
            return response

        text = str(response).strip()
        if not text:
            return []

        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        def _loads_and_norm(s: str) -> List[Dict]:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                return obj
            return []

        try:
            return _loads_and_norm(text)
        except Exception:
            pass

        extracted = self._extract_json_from_markdown(text)
        extracted = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", extracted)

        extracted = self._extract_first_json_span(extracted)

        extracted = extracted.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        extracted = re.sub(r",\s*([}\]])", r"\1", extracted)  # {"a":1,} -> {"a":1}

        try:
            return _loads_and_norm(extracted)
        except Exception:
            try:
                obj = ast.literal_eval(extracted)
                if isinstance(obj, list):
                    return [x for x in obj if isinstance(x, dict)]
                if isinstance(obj, dict):
                    return obj
            except Exception as e2:
                self.logger.info(f"Failed to parse JSON response {e2}. raw_head={text[:200]!r}")
                return []
    def _extract_json_from_markdown_alignment(self, text: str) -> str:
        """
        Additional processing: if LLM returns Markdown code block format (like ```json\n...\n```),
        extract the JSON content from it.
        """
        if not isinstance(text, str):
            return text
        
        text = text.strip()

        import re

        pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
        match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return text
    def _extract_json_from_markdown(self, text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        text = text.strip()

        import re
        pattern = r"```(?:json)?\s*\n(.*?)\n```"
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text
    def _extract_first_json_span(self, s: str) -> str:
        s = s.strip()
        if not s:
            return s

        start_candidates = [(s.find("{"), "{"), (s.find("["), "[")]
        start_candidates = [(i, ch) for i, ch in start_candidates if i != -1]
        if not start_candidates:
            return s

        start, opener = min(start_candidates, key=lambda x: x[0])
        closer = "}" if opener == "{" else "]"

        stack = []
        in_str = False
        esc = False

        for i in range(start, len(s)):
            c = s[i]

            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue

            if c == '"':
                in_str = True
                continue

            if c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack:
                    continue
                top = stack[-1]
                if (top == "{" and c == "}") or (top == "[" and c == "]"):
                    stack.pop()
                    if not stack:
                        return s[start:i+1]

        return s[start:]
    def process(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        """Execute Agent's main process (needs to be implemented by specific subclasses)."""
        raise NotImplementedError("Subclasses must implement process()")


