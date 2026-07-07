"""GET /stream/events — Server-Sent Events. One-way server->browser stream.

We send lightweight change-signals ("new_event"); htmx re-fetches the partials
on receipt, so HTML rendering stays server-side (dashboard template change)."""

import asyncio

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter(tags=["stream"])

HEARTBEAT_SECONDS = 30


@router.get("/stream/events")
async def stream_events(request: Request):
    broker = request.app.state.broker
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # keep the connection alive + let the client detect death
                    yield {"event": "heartbeat", "data": "ping"}
                    continue
                yield {"event": "new_event", "data": data}
        finally:
            broker.unsubscribe(queue)

    return EventSourceResponse(event_generator())
