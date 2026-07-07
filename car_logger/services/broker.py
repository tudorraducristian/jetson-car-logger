"""Thread-to-async fan-out for SSE.

Worker threads call publish(); the async SSE endpoint subscribes a queue per
client. publish() is safe from any thread because it hands work to the event
loop via call_soon_threadsafe — asyncio.Queue is NOT thread-safe otherwise."""

import asyncio


class EventBroker(object):
    def __init__(self):
        self._subscribers = set()
        self._loop = None

    def set_loop(self, loop):
        self._loop = loop

    async def subscribe(self):
        queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self._subscribers.discard(queue)

    def publish(self, data):
        """Schedule `data` onto every subscriber queue. No-op if no loop yet."""
        loop = self._loop
        if loop is None:
            return
        for queue in list(self._subscribers):
            loop.call_soon_threadsafe(queue.put_nowait, data)
