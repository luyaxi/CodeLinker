from codelinker.models import SEvent, ChannelTag
from .sink import EventSink
from typing import Iterable, Literal, Callable
from functools import partial

class EventProcessor:
    def __init__(self, name: str, sink: EventSink):
        self.name = name
        self.sink = sink
        self.logger = sink.logger.getChild(name)
        
    async def setup(self):
        pass

    def get_tag_lock(self, tag: ChannelTag):
        return self.sink.get_tag_lock(tag)

    def update_time(self, time: str):
        return self.sink.update_time(time)

    def add(self, tags: ChannelTag | Iterable[ChannelTag], content: str | Iterable[str], silent: bool = False):
        return self.sink.add(content=content, tags=tags,
                      source=self.name, silent=silent)

    def listen(self, tags: ChannelTag | Iterable[ChannelTag] = None, max_emit_time: int = None):
        return self.sink.listen(source=self.name, tags=tags, max_emit_time=max_emit_time)

    def unlisten(self, func: callable):
        return self.sink.unlisten(func)
    
    def wait(self, tags: ChannelTag | Iterable[ChannelTag]):
        return self.sink.wait(tags)

    def gather(self, tags: ChannelTag | Iterable[ChannelTag] | None = None, return_dumper: Callable|Literal['str','json','identity'] = 'str') -> Iterable[dict]:
        def tag_filter(event: SEvent):
            if tags is None:
                return True
            if isinstance(tags, ChannelTag):
                if tags in event.tags:
                    return True
                return False
            for tag in tags:
                if tag in event.tags:
                    return True
            return False

        gathered_events = list(filter(tag_filter, self.sink.all_events))
        if isinstance(return_dumper,str):
            match return_dumper:
                case 'str':
                    return_dumper = str
                case 'json':
                    import json
                    return_dumper = partial(json.dumps, ensure_ascii=False,sort_keys=False)
                case 'identity':
                    return_dumper = lambda x: x
                    
        assert callable(return_dumper), "return_dumper must be a callable'"
        
        messages = []
        for event in gathered_events:
            if event.source == self.name:
                messages.append(
                    {'role': 'assistant', 'content': event.content})
            else:
                messages.append(
                    {'role': 'user', 'content': return_dumper(event)})
        return messages

