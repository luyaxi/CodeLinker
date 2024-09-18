import logging
import inspect
import asyncio

from typing import Callable, Any, TypeVar, Optional
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from pydantic import TypeAdapter,BaseModel
from copy import deepcopy


from .config import CodeLinkerConfig
from .request import OBJGenerator
from .models import SmartFuncLabel, StructureSchema, StructuredRet
from .utils import clip_text

T = TypeVar("T")

def get_ref_schema(refs: str, ret_schema):
    if refs.startswith("#/"):
        refs = refs[2:]
    refs = refs.split("/")
    schema = ret_schema
    for idx in refs:
        schema = schema[idx]
    return schema

# replace $defs and $refs
def replace_refs(schema, ret_schema):
    if isinstance(schema, dict):
        if "$ref" in schema:
            return get_ref_schema(schema["$ref"], ret_schema)
        for k, v in schema.items():
            schema[k] = replace_refs(v,ret_schema)
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            schema[i] = replace_refs(v,ret_schema)
    return schema


async def request(
        return_type: TypeAdapter,
        objGen: OBJGenerator,
        description: str = "",
        prompt: Optional[str] = None,
        request_name: str = "request",
        completions_kwargs: dict = {},
        images: list = None,
        messages: list = [],
        reasoning_format: StructureSchema = None,
        schema_validation: bool = None,
        dynamic_json_fix: bool = None,
        ):
    resolve_none_object = False
    schema = return_type.json_schema()
    if schema == TypeAdapter(str).json_schema():
        schemas = None
    else:
        schema = replace_refs(schema, schema)
        
        if schema["type"] != "object":
            schema = {
                "type": "object",
                "properties": {
                    "content": schema
                }
            }
            resolve_none_object = True
    
        if reasoning_format is not None:
            schemas = [
                reasoning_format,
                StructureSchema(
                    name=request_name,
                    description=description,
                    parameters=schema
                )
            ]
        else:
            schemas = StructureSchema(
                name=request_name,
                description=description,
                parameters=schema
            )
            
    messages = deepcopy(messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    if images is not None and len(images) > 0:
        messages[-1]["content"] = [
            {
                "type": "text",
                "text": messages[-1]["content"]
            }
        ] + images

    
    rets = await objGen.chatcompletion(
        messages=messages,
        schemas=schemas,
        schema_validation=schema_validation,
        dynamic_json_fix=dynamic_json_fix,
        **completions_kwargs,
    )
    if isinstance(rets, StructuredRet):
        if resolve_none_object:
            return return_type.validate_python(rets.content["content"])
        return return_type.validate_python(rets.content)
    elif isinstance(rets, list):
        returns = []
        for item in rets:
            if isinstance(item, list):
                rets = []
                for i in item:
                    if resolve_none_object:
                        rets.append(
                            return_type.validate_python(i.content["content"]))
                    else:
                        rets.append(
                            return_type.validate_python(i.content))
                returns.append(rets)
            elif isinstance(item, StructuredRet):
                if resolve_none_object:
                    returns.append(
                        return_type.validate_python(item.content["content"]))
                else:
                    returns.append(
                        return_type.validate_python(item.content))
            else:
                raise ValueError("Invalid return type")
    else:
        raise ValueError("Invalid return type")
    return returns


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
        self.pool = ThreadPoolExecutor(
            thread_name_prefix="CodeLinkerHandlerThread")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if not self.pool._shutdown:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def close(self):
        if not self.pool._shutdown:
            self.pool.shutdown(wait=False, cancel_futures=True)

    async def exec(
        self,
        return_type: T = str,
        prompt: Optional[str] = None,
        request_name: str = "request",
        model: Optional[str] = None,
        completions_kwargs: dict = {},
        images: list = None,
        messages: list = [],
        reasoning_format: StructureSchema = None,
        schema_validation: bool = None,
        dynamic_json_fix: bool = None,
        request_lib: str = None) -> T:
        if model:
            completions_kwargs["model"] = model
        if request_lib:
            completions_kwargs["request_lib"] = request_lib
            
        return await request(
            prompt=prompt,
            return_type=TypeAdapter(return_type),
            description=return_type.__doc__ if isinstance(return_type, BaseModel) else "",
            objGen=self.objGen,
            request_name=request_name,
            completions_kwargs=completions_kwargs,
            images=images,
            messages=messages,
            reasoning_format=reasoning_format,
            schema_validation=schema_validation,
            dynamic_json_fix=dynamic_json_fix,
        )


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
                
            # get function description from return's schema
            description =  return_type.__doc__ if isinstance(return_type, BaseModel) else ""
            return_type = TypeAdapter(return_type)
            ret_schema = return_type.json_schema()

            def get_ref_schema(refs: str):
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
            description += ret_schema.pop("description", "")

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
                images: list = None,
                messages: list = [],
                reasoning_format: StructureSchema = None,
                **kwargs):

                
                if len(args) > 0:
                    raise RuntimeError(
                        "Smart functions should not have positional arguments")

                label: SmartFuncLabel = getattr(
                    func, "__smart_function_label__")
                self.logger.log(10, f"Executing function: {label.name}")

                if len(args) > 0:
                    raise ValueError(
                        "Smart functions should not have positional arguments")

                for k, v in kwargs.items():
                    kwargs[k] = str(v)
                    
                prompt = clip_text(
                    text=label.prompt.format(**kwargs),
                    max_tokens=self.config.execution.max_ai_functions_tokens,
                    clip_end=True)[0]
                
                return await request(
                    prompt=prompt,
                    return_type=label.return_type,
                    objGen=self.objGen,
                    request_name=label.name,
                    messages=messages,
                    images=images,
                    completions_kwargs=label.completions_kwargs,
                    reasoning_format=reasoning_format,
                )


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
