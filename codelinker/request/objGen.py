import os
import json
import glob
import uuid
import datetime
import jsonschema
import jsonschema.exceptions
import importlib
import aiofiles
import asyncio

from typing import Literal
from copy import deepcopy
from tenacity import AsyncRetrying, RetryError, stop_after_attempt
from logging import Logger

from ..models import StructuredRet, StructureSchema
from ..config import CodeLinkerConfig
from ..utils import clip_text

class OBJGenerator:
    def __init__(self, config: CodeLinkerConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.chatcompletion_request_funcs = {}
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._load_cache_files())
        if self.config.request.save_completions:
            os.makedirs(self.config.request.save_completions_path, exist_ok=True)
            
    async def _load_cache_files(self):
        if self.config.request.use_cache:
            self.logger.warning("use_cache is enabled, loading completions from cache...")
            self.hash2files = {}
            files = glob.glob(os.path.join(self.config.request.save_completions_path, "*.json"))
            files.sort(key=os.path.getmtime)
            
            async def load_file(file):
                async with aiofiles.open(file, 'r') as f:
                    data = json.loads(await f.read())
                    self.hash2files[hash(json.dumps(data["request"],sort_keys=True))] = file
            tasks = [load_file(file) for file in files]
            await asyncio.gather(*tasks)
            self.logger.warning("Cache loaded and enabled, which may cause unexpected behavior.")

    async def _chatcompletion_request(self, *, request_lib: Literal["openai",] = None, **kwargs) -> dict:
        if self.config.request.use_cache:
            hash_ = hash(json.dumps(kwargs,sort_keys=True))
            self.logger.debug(f"Request hash: {hash_}")
            if hash_ in self.hash2files:
                async with aiofiles.open(self.hash2files[hash_], 'r') as f:
                    data = json.loads(await f.read())
                    if data["request"] == kwargs:
                        self.logger.info(f"Cache hit file {self.hash2files[hash_]}, return cached completions.")
                        # remove cache to avoid duplicate
                        self.hash2files.pop(hash_)
                        return data["response"]

        request_lib = request_lib if request_lib is not None else self.config.request.default_request_lib
        response = await self._get_chatcompletion_request_func(request_lib)(config=self.config,**kwargs)
        
        
        if self.config.request.save_completions:
            async with aiofiles.open(os.path.join(self.config.request.save_completions_path,
                                    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")+f"{uuid.uuid4().hex}.json"
                                    ), 'w') as f:
                data = json.dumps({
                    "request": kwargs,
                    "response": response
                    },)
                await f.write(data)

        return response
    
    def register_request_lib(self, request_type: str, request_func):
        self.chatcompletion_request_funcs[request_type] = request_func

    def _get_chatcompletion_request_func(self, request_type: str):
        if request_type not in self.chatcompletion_request_funcs:
            module = importlib.import_module(
                f'.{request_type}', 'codelinker.request')
            self.chatcompletion_request_funcs[request_type] = getattr(
                module, 'chatcompletion_request')
        return self.chatcompletion_request_funcs[request_type]

    def _get_embedding_request_func(self, request_type: str):
        if request_type not in self.chatcompletion_request_funcs:
            module = importlib.import_module(
                f'.{request_type}', 'codelinker.request')
            self.chatcompletion_request_funcs[request_type] = getattr(
                module, 'embedding_request')
        return self.chatcompletion_request_funcs[request_type]


    async def chatcompletion(self,
                             *,
                             messages: list[dict],
                             schemas: None | StructureSchema | list[StructureSchema |
                                                             list[StructureSchema]] = None,
                             schema_validation: bool = None,
                             dynamic_json_fix: bool = None,
                             max_retry_times: int = None,
                             **kwargs) -> StructuredRet | list[StructuredRet] | list[list[StructuredRet]]:
        for k in list(kwargs.keys()):
            if kwargs[k] is None:
                kwargs.pop(k)

        request_format = kwargs.pop("request_format", self.config.request.format)
        if schemas is None:
            request_format = "messages"

        # filter messages with empty content
        messages = list(filter(lambda x: len(x["content"]) > 0, messages))
            
        max_retry_times = self.config.max_retry_times if max_retry_times is None else max_retry_times
        schema_validation = self.config.request.schema_validation if schema_validation is None else schema_validation
        dynamic_json_fix = self.config.request.dynamic_json_fix if dynamic_json_fix is None else dynamic_json_fix
        
        if not dynamic_json_fix:
            self.logger.debug("Warning: dynamic json fix is disabled, raise error if schema validation failed.")
        
        match request_format:
            case "messages":
                structuredRets = []
                async for attempt in AsyncRetrying(stop=stop_after_attempt(max_retry_times), reraise=True):
                    with attempt:
                        response = await self._chatcompletion_request(messages=messages,**kwargs)
                        for choice in response["choices"]:
                            structuredRets.append(StructuredRet(
                                name="response",
                                content=choice["message"]["content"]
                            ))
            case "tool_call":
                structuredRets = await self._chatcompletion_with_tool_call(
                    messages=messages,
                    schemas=schemas,
                    schema_validation=schema_validation,
                    dynamic_json_fix=dynamic_json_fix,
                    max_retry_times=max_retry_times,
                    **kwargs
                )
            case _:
                raise NotImplementedError(
                    f"Request format {self.config.request.format} is not implemented!")
        if isinstance(structuredRets, list) and len(structuredRets) == 1 and isinstance(structuredRets[0], StructuredRet):
            structuredRets = structuredRets[0]
        return structuredRets

    async def embedding_completion(self, *, text):
        embedding_func = self._get_embedding_request_func(
            request_type="openai")
        embedding = await embedding_func(text)
        return embedding

    async def _chatcompletion_with_tool_call(
        self,
        *,
        messages: list[dict],
        schemas: StructureSchema | list[StructureSchema | list[StructureSchema]],
        schema_validation: bool,
        dynamic_json_fix: bool,
        max_retry_times: int,
        **kwargs,
    ) -> list[StructuredRet] | list[list[StructuredRet]]:
        
        req_template = None

        # construct the request template
        if schemas is not None:
            if "tools" in kwargs or "tool_choice" in kwargs:
                raise ValueError(
                    "You can't provide schemas with tools/tool_choice!")
            
            if isinstance(schemas, StructureSchema):                    
                kwargs["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": schemas.name,
                        "description": schemas.description,
                        "parameters": schemas.parameters
                    }
                }]
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": schemas.name}
                }
                
                constructed_schema = {
                    "name": schemas.name,
                    "description": schemas.description,
                    "parameters": schemas.parameters
                }
                
            elif isinstance(schemas, list):
                req_template = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

                for schema in schemas:
                    if isinstance(schema, StructureSchema):
                        # add schema to the template
                        req_template["properties"][schema.name] = schema.parameters
                        req_template["required"].append(schema.name)
                    elif isinstance(schema, list):
                        combine_name = "arg"+str(schemas.index(schema))
                        req_template["properties"][combine_name] = {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "description": s.description,
                                    "properties": {
                                        s.name: s.parameters
                                    },
                                    "required": [s.name]
                                }
                                for s in schema
                            ]
                        }
                        req_template["required"].append(combine_name)

                constructed_schema = {
                    "name": "structuredRets",
                    "description": "Follow the schema",
                    "parameters": req_template
                }

                kwargs["tools"] = [{
                    "type": "function",
                    "function": constructed_schema
                }]
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {
                        "name": "structuredRets"
                    }
                }

        kwargs["messages"] = messages

        rets = []

        try:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(max_retry_times), reraise=True):
                with attempt:
                    response = await self._chatcompletion_request(**kwargs)
                    
                    for choice in response["choices"]:
                        structuredRets = []
                        for tool_call in choice["message"]["tool_calls"]:
                            if tool_call["type"] != "function":
                                continue

                            if schema_validation:
                                # refine the tool calls
                                d = await self.schema_valiation(
                                    s=tool_call["function"]["arguments"],
                                    schema=constructed_schema,
                                    messages=kwargs["messages"],
                                    dynamic_json_fix=dynamic_json_fix,
                                )
                            else:
                                d = json.loads(
                                    tool_call["function"]["arguments"])
                            if constructed_schema["name"] == "structuredRets":
                                for k, v in d.items():
                                    structuredRets.append(StructuredRet(
                                        name=k,
                                        content=v
                                    ))
                            else:
                                structuredRets.append(StructuredRet(
                                    name=constructed_schema["name"],
                                    content=d
                                ))
                        if len(structuredRets) == 1:
                            rets.append(structuredRets[0])
                        else:
                            rets.append(structuredRets)

        except RetryError as e:
            self.logger.error(f"Chatcompletion Error: Retry failed\n{e}")
            raise e
        return rets


    async def dynamic_json_fixes(self, s, schema, messages: list = [], error_message: str = None) -> dict:

        self.logger.warn(f'Schema Validation on string:\n{s}\nfor schema {schema["name"]} failed, trying to fix it...')

        ret = await self.chatcompletion(
            messages=[
            {
                "role": "system",
                "content": "Your task is to fix the string with schema errors. Remember to keep the target schema in mind. Avoid adding any information about this fix!"
            },
            {
                "role": "user",
                "content": clip_text(f"""Generate Content with Correct Schema.\n# Error String\n{s}\n# Error Message\n{error_message[-1024:]}""",max_tokens=self.config.execution.max_message_tokens,clip_end=True)[0]
            }],
            schemas=StructureSchema(**schema),
            schema_validation=True,
            dynamic_json_fix=False,
            max_retry_times=0,
        )
        return ret.content

    async def schema_valiation(
            self,
            s: str,
            schema: dict,
            messages: list[dict],
            dynamic_json_fix = None) -> dict:
        # load string as instance
        try:
            d = json.loads(s)
        except Exception as e:
            # load error, request a json fixes
            if dynamic_json_fix:
                d = await self.dynamic_json_fixes(s=s, schema=schema, messages=messages, error_message=str(e))
            else:
                raise e

        params = deepcopy(schema["parameters"])
        # recursive remove description

        def remove_description(d: dict):
            if "description" in d:
                d.pop("description")
            for k, v in d.items():
                if isinstance(v, dict):
                    remove_description(v)
                elif isinstance(v, list):
                    for i in v:
                        if isinstance(i, dict):
                            remove_description(i)
        remove_description(params)

        # valid schemas
        try:
            jsonschema.validate(d, params)
        except jsonschema.exceptions.ValidationError as e:
            if dynamic_json_fix:
                d = await self.dynamic_json_fixes(s=s, schema=schema, messages=messages, error_message=str(e))
                jsonschema.validate(d, params)
            else:
                raise e
        except Exception as e:
            self.logger.error("Schema Validation Error:\n"+str(e))

        return d
