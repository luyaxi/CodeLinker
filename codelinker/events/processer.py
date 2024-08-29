from codelinker.models import SEvent, ChannelTag
from .sink import EventSink
from typing import Iterable, Literal

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

    def gather(self, tags: ChannelTag | Iterable[ChannelTag] | None = None, return_fmt: Literal['str', 'messages'] = 'str') -> str | Iterable[dict]:
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

        match return_fmt:
            case 'str':
                # string events
                s = "# Past Events"
                for idx, event in enumerate(gathered_events[:-3]):
                    s += f"\n--- Event [{idx}] ---\n{event}"
                s = "# Latest Events"
                for event in gathered_events[-3:]:
                    s += f"\n--- Event ---\n{event}"
                s += "\nYou should pay attention to the events before take further actions."
                return s

            case 'messages':
                messages = []
                for event in gathered_events:
                    if event.source == self.name:
                        messages.append(
                            {'role': 'assistant', 'content': event.content})
                    else:
                        messages.append(
                            {'role': 'user', 'content': str(event)})

                # merge adjacent message with same role
                for i in range(len(messages)-1, 0, -1):
                    if messages[i]['role'] == messages[i-1]['role']:
                        messages[i -
                                 1]['content'] += f"\n\n{messages[i]['content']}"
                        messages.pop(i)

                return messages
            case _:
                raise ValueError(f"Invalid return_fmt {return_fmt}")

