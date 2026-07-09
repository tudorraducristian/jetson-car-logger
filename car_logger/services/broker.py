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
        """Remember the serving event loop; publish() is a no-op until then."""
        self._loop = loop

    async def subscribe(self):
        """One queue per SSE client. maxsize=1 + drop-on-full coalesces
        bursts: a client that already has an unread change-signal gains
        nothing from a second one (finding: unbounded queues on slow
        clients)."""
        queue = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        """Forget a client's queue; safe to call for an unknown queue."""
        self._subscribers.discard(queue)

    def publish(self, data):
        """Thread-safe: hand the WHOLE fan-out to the loop thread. The
        subscriber set is then only ever touched on the loop (subscribe/
        unsubscribe already run there), so no cross-thread set mutation
        can race the iteration. No-op if no loop yet."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._fanout, data)

    def _fanout(self, data):
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass  # client already has an unread change-signal
