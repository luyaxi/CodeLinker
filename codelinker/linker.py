import logging
import inspect
import asyncio

from typing import Callable
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from pydantic import TypeAdapter

from .config import CodeLinkerConfig
from .request import OBJGenerator
from .models import SmartFuncLabel,StructureSchema,StructuredRet
from .utils import clip_text

class CodeLinker:
    '''CodeLinker manage the configuration and the handler of the smart functions.
    
    # Usage
    ```python
    cl = CodeLinker(config)
    @cl.smartFunc()
    def hello_world() -> HelloWorldSchema:
        """Say hello to the world"""
    ```
    
    '''
    def __init__(self, config: CodeLinkerConfig, logger: logging.Logger = None):
        self.config = config
        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        self.objGen = OBJGenerator(config, self.logger)
        self.pool = ThreadPoolExecutor(thread_name_prefix="CodeLinkerHandlerThread")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if not self.pool._shutdown:
            self.pool.shutdown(wait=False, cancel_futures=True)
        
    
    def close(self):
        if not self.pool._shutdown:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def smartFunc(
        self,
        completions_kwargs: dict = {},
    ) -> Callable:
        '''Decorator to wrap a function as a smart function.
        The functions will be executed by language models.
        Note that the code in function will be ignored.
        
        '''
        def decorator(func):
            
            name = func.__name__
            return_type = inspect.signature(func).return_annotation
            if return_type == inspect.Signature.empty:
                raise ValueError(
                    f"Function {func.__name__} does not contain return type annotation!")
            return_type = TypeAdapter(return_type)
            ret_schema = return_type.json_schema()
            def get_ref_schema(refs:str):
                if refs.startswith("#/"):
                    refs = refs[2:]
                refs = refs.split("/")
                schema = ret_schema
                for idx in refs:
                    schema = schema[idx]
                return schema
            
            # replace $defs and $refs
            def replace_refs(schema):
                if isinstance(schema, dict):
                    if "$ref" in schema:
                        return get_ref_schema(schema["$ref"])
                    for k, v in schema.items():
                        schema[k] = replace_refs(v)
                elif isinstance(schema, list):
                    for i, v in enumerate(schema):
                        schema[i] = replace_refs(v)
                return schema
            
            ret_schema = replace_refs(ret_schema)
            
            # get function description from return's schema
            description = ret_schema.pop("description", "")
            prompt = func.__doc__

            label = SmartFuncLabel(
                name=name,
                description=description,
                return_type=return_type,
                schema=ret_schema,
                prompt=prompt,
                completions_kwargs=completions_kwargs,
            )
            setattr(func, "__smart_function_label__", label)

            @wraps(func)
            async def wrapper(
                *args,
                images:list = None,
                messages:list = None,
                tools: list[StructureSchema] = None,
                tool_choice:dict = None,
                reply_format: StructureSchema = None,
                **kwargs):
                if len(args) > 0:
                    raise RuntimeError("Smart functions should not have positional arguments")
                
                label: SmartFuncLabel = getattr(func, "__smart_function_label__")
                self.logger.log(10, f"Executing function: {label.name}")
                
                if len(args) > 0:
                    raise ValueError("Smart functions should not have positional arguments")
                
                for k, v in kwargs.items():
                    kwargs[k] = str(v)
                if messages is None:
                    messages = []
                messages.append(
                    {"role": "user",
                    "content": clip_text(
                        text=label.prompt.format(**kwargs),
                        max_tokens=self.config.execution.max_ai_functions_tokens,
                        clip_end=True)[0]
                    })
                if images is not None and len(images) > 0:
                    messages[-1]["content"] = [
                        {
                            "type": "text",
                            "text": messages[-1]["content"]
                        }
                    ] + images
                    
                if tools is None or len(tools) == 0:
                    rets = await self.objGen.chatcompletion(
                        messages=messages,
                        schemas=StructureSchema(
                            name=label.name,
                            description=label.description,
                            parameters=label.schema
                            ),
                        **label.completions_kwargs,
                    )
                    if isinstance(rets, StructuredRet):
                        return label.return_type.validate_python(rets.content)
                    elif isinstance(rets, list):
                        returns = []
                        for item in rets:
                            if isinstance(item,list):
                                rets = []
                                for i in item:
                                    rets.append(label.return_type.validate_python(i.content))
                                returns.append(rets)
                            elif isinstance(item, StructuredRet):
                                returns.append(label.return_type.validate_python(item.content))
                            else:
                                raise ValueError("Invalid return type")
                    else:
                        raise ValueError("Invalid return type")
                    return returns
                
                else:
                    schemas = []
                    
                    if reply_format is not None:                    
                        schemas.insert(0,reply_format)
                    
                    if tool_choice is not None:
                        # filter tools
                        tools = [tool for tool in tools if tool.name == tool_choice["function"]["name"]]
                    
                    schemas.append(tools)
                    
                    returns =  await self.objGen.chatcompletion(
                        messages=messages,
                        schemas=schemas,
                        **label.completions_kwargs,
                    )
                    
                    if reply_format is None:
                        if isinstance(returns[0],list):
                            return [ret[1:] for ret in returns] # skip the first reply
                        else:
                            return returns[1:]
                    else:
                        return returns
                    
            
            if asyncio.iscoroutinefunction(func):
                return wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        ret = loop.run_until_complete(wrapper(*args, **kwargs))
                        loop.close()
                        return ret
                    
                    # here we are already in an event loop, but calling a sync function
                    # so we need to run it in a separate thread
                    def run_sync_func():
                        return asyncio.run(wrapper(*args, **kwargs))
                    
                    return self.pool.submit(run_sync_func).result()
                    
                return sync_wrapper
        return decorator