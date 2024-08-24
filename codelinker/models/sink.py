from pydantic_core.core_schema import str_schema
from pydantic import BaseModel
from typing import Callable, Any, List


class ChannelTag(str):
    """Channel tag is used to identify the channel."""

    def __get_pydantic_core_schema__(self):
        return str_schema()
    

class Channels:
    def __init__(self, prefix: str):
        self.prefix = prefix
    
    @property
    def all(self) -> set[ChannelTag]:
        tags = set()
        for attribute_name in dir(self):
            if attribute_name.startswith("_") or attribute_name == "all":
                continue
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, ChannelTag):
                tags.add(attribute)
            elif isinstance(attribute, Channels):
                tags = tags.union(attribute.all)
        return tags



class SEvent(BaseModel):
    source: str
    time: str  # M-D H:M:S
    tags: List[ChannelTag]
    content: str

    def __str__(self):
        s = ""
        s += f"Time: {self.time} | Source: {self.source} | "
        for tag in self.tags:
            s += f"[{tag}] "
        s += f"| {self.content}"
        return s

class ScheduledCallBack(BaseModel):
    time: str  # M-D H:M:S
    tags: set[ChannelTag]
    called_func: Callable
    callback_args: dict[str, Any] = {}