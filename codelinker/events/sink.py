import asyncio

from logging import Logger
from functools import wraps
from typing import Callable, Optional, Iterable
from codelinker.models import SEvent, ScheduledCallBack, Channels, ChannelTag

def get_func_full_name(func: Callable):
    if hasattr(func, "__self__"):
        return f"{func.__self__.__class__}.{func.__name__}"
    else:
        return f"{func.__name__}"


class EventSink:
    def __init__(self, sinkChannels: Channels, logger: Logger,beacon_interval: float = 0.5) -> None:
        self.all_events: list[SEvent] = []  # store all events
        self.time = "Time Freezed."
        self.beacon_interval = beacon_interval

        self.func2wrapper: dict[Callable, Callable] = {}
        self.wrapper2func: dict[Callable, Callable] = {}
        self.subscriber2callcount: dict[Callable, int | None] = {}
        
        self.tag2subscriber: dict[ChannelTag,
                                  list[Callable]] = {}
        self.tag2tasks: dict[ChannelTag, set[asyncio.Task]] = {}
        self.tag2lock: dict[ChannelTag, asyncio.Lock] = {}
        self.all_tags = sinkChannels.all
        for tag in self.all_tags:
            self.tag2subscriber[tag] = []
            self.tag2tasks[tag] = set()
            self.tag2lock[tag] = asyncio.Lock()
            
        

        self.schedule_lock = asyncio.Lock()
        self.scheduled_funcs: set[Callable] = set()
        self.scheduled: list[ScheduledCallBack] = []


        self.cleanup_thd = None
        self.run_scheduled_thd = None
        self.logger = logger.getChild("EventSink")

    def update_time(self, time: str):
        self.time = time
        return self.time

    def init(self, *args, **kwargs):
        """Init event sink"""
        # setup clean thread to clean the tasks
        async def clean():
            while True:
                await asyncio.sleep(self.beacon_interval)
                self.get_tasks(self.all_tags)

        self.cleanup_thd = asyncio.create_task(
            clean(), name="EventSink: Cleanup Thread")

        async def run_scheduled():
            while True:
                await asyncio.sleep(self.beacon_interval)

                for scheduled_callback in list(self.scheduled):
                    tags = scheduled_callback.tags
                    locks = [self.tag2lock[tag] for tag in tags]
                    if all([not lock.locked() for lock in locks]):
                        task = asyncio.create_task(
                            scheduled_callback.called_func(**scheduled_callback.callback_args), name=f"{get_func_full_name(self.wrapper2func[scheduled_callback.called_func])} is triggered by tags {tags}, Time {self.time}")
                        for tag in tags:
                            self.tag2tasks[tag].add(task)
                        self.scheduled.remove(scheduled_callback)
                        self.scheduled_funcs.remove(
                            scheduled_callback.called_func)

        self.run_scheduled_thd = asyncio.create_task(
            run_scheduled(), name="EventSink: Scheduling Thread")

    def get_tasks(self, tags: ChannelTag | Iterable[ChannelTag] | None = None) -> set[asyncio.Task]:
        tasks: set[asyncio.Task] = set()

        if tags is None:
            for ts in self.tag2tasks.values():
                tasks = tasks.union(ts)
        elif isinstance(tags, ChannelTag):
            tasks = self.tag2tasks[tags]
        elif isinstance(tags, Iterable):
            for tag in tags:
                tasks = tasks.union(self.tag2tasks[tag])
        else:
            raise ValueError(f"Invalid tags {tags}")

        for t in list(tasks):
            if t.done():
                tasks.remove(t)
                for tag in self.all_tags:
                    if t in self.tag2tasks[tag]:
                        self.tag2tasks[tag].remove(t)
                try:
                    t.result()
                    self.logger.debug(
                        f"[Task] {t.get_name()} is done."
                    )
                except asyncio.CancelledError:
                    self.logger.debug(
                        f"[Task] {t.get_name()} is canceled.")
                except Exception as e:
                    self.logger.error(
                        f"Error when executing task {t.get_name()}, {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())

        return tasks

    async def wait(self, tags: ChannelTag | Iterable[ChannelTag]):
        if isinstance(tags, ChannelTag):
            tags = [tags]
        while True:
            await asyncio.sleep(delay=self.beacon_interval)

            # check whether all locks are released
            if any([self.tag2lock[tag].locked() for tag in tags]):
                continue

            # check whether there are scheduled tasks
            if len(self.scheduled) > 0:
                ts = set(tags)
                rewait = False
                for sc in self.scheduled:
                    if ts.intersection(sc.tags):
                        rewait = True
                if rewait:
                    continue

            tasks = self.get_tasks(tags)
            if len(tasks) > 0:
                await asyncio.wait(tasks)
            else:
                self.logger.debug("No task to wait, skip.")
                return

    async def close(self, wait: bool = False, timeout: float = 30):
        if not wait:
            tasks = self.get_tasks()
            for task in tasks:
                try:
                    task.cancel()
                except Exception as e:
                    self.logger.error(
                        f"Error when cancel task {task.get_name()}, {e}")
                self.logger.debug(f"Task {task.get_name()} is canceled.")
        else:
            if len(tasks) > 0:
                await asyncio.wait(tasks, timeout=timeout)
        self.cleanup_thd.cancel()

    def get_tag_lock(self, tag: ChannelTag):
        return self.tag2lock[tag]
    
    def unlisten(self, func: Callable):
        if func in self.func2wrapper:
            wrapper = self.func2wrapper[func]
            for tag,subs in self.tag2subscriber.items():
                if wrapper in subs:
                    subs.remove(wrapper)
                    self.logger.debug(
                        f"{get_func_full_name(func)} is unregistered from tag {tag}.")
        else:
            self.logger.debug(
                f"{get_func_full_name(func)} is not registered.")

    def listen(self, source: str, tags: Optional[ChannelTag | Iterable[ChannelTag]] = None, max_emit_time: int = None):
        """listen decorator, callback the registered function when the listened tag is received"""
        def decorator(func: Callable):
            if func not in self.subscriber2callcount:
                self.subscriber2callcount[func] = 0

            if func not in self.func2wrapper:
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    if max_emit_time is not None and self.subscriber2callcount[func] >= max_emit_time:
                        self.logger.debug(
                            f"{get_func_full_name(func)} is cancelled due to max emit time reached.")
                        return

                    self.subscriber2callcount[func] += 1
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                setattr(wrapper, "__sink_source__", source)

                self.func2wrapper[func] = wrapper
                self.wrapper2func[wrapper] = func
            else:
                wrapper = self.func2wrapper[func]

            registered = []
            if tags is None:
                self.tag2subscriber[tags].append(wrapper)
                registered.append(tags)
            elif isinstance(tags, ChannelTag):
                self.tag2subscriber[tags].append(wrapper)
                registered.append(tags)
            elif isinstance(tags, Iterable):
                for tag in tags:
                    self.tag2subscriber[tag].append(wrapper)
                    registered.append(tag)
            else:
                raise ValueError(f"Invalid tags {tags}")

            self.logger.debug(
                f"{get_func_full_name(func)} is registered to tags {registered}.")

            return func
        return decorator

    def add(self, tags: ChannelTag | Iterable[ChannelTag],  content: str | list[str], source: str = "system", silent: bool = False, callback_args: dict = {}) -> list[SEvent]:
        if isinstance(content, str):
            content = [content]
        if isinstance(tags, ChannelTag):
            tags = [tags]

        events = []
        
        for c in content:
            event = SEvent(source=source, time=self.time, tags=tags, content=c)
            self.logger.info(f"{event}")
            self.all_events.append(event)
            events.append(event)
        if silent:
            return events

        # schedule callback execution
        candidate_funcs = []
        for tag in tags:
            for subscriber in self.tag2subscriber[tag]:
                receiver = getattr(subscriber, "__sink_source__")
                if receiver != source:
                    candidate_funcs.append(subscriber)

        for func in candidate_funcs:
            # first check whether there are same function in the scheduled list
            if func in self.scheduled_funcs:
                self.logger.debug(
                    f"{get_func_full_name(self.wrapper2func[func])} is already scheduled and will be skipped.")
                continue

            # add the scheduled callback to the scheduled list
            self.scheduled.append(ScheduledCallBack(time=self.time, tags=set(
                tags), called_func=func, callback_args=callback_args))
            self.scheduled_funcs.add(func)
            self.logger.debug(
                f"Schedule {get_func_full_name(self.wrapper2func[func])} to be triggered by tags {tags} at time {self.time}.")
            
        return events


