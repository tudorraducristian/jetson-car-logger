import asyncio

import pytest

from car_logger.services.broker import EventBroker


@pytest.mark.asyncio
async def test_publish_reaches_subscriber():
    broker = EventBroker()
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()
    broker.publish("changed")
    data = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert data == "changed"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    broker = EventBroker()
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()
    broker.unsubscribe(queue)
    broker.publish("changed")
    await asyncio.sleep(0.05)
    assert queue.empty()


def test_publish_without_loop_is_noop():
    # publish before any subscriber/loop must not raise
    EventBroker().publish("changed")


@pytest.mark.asyncio
async def test_burst_publishes_coalesce_to_one_signal():
    # Signals carry no payload the client uses — "something changed" twice
    # is worth exactly one re-fetch. maxsize=1 + drop-on-full coalesces.
    broker = EventBroker()
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()
    broker.publish("created")
    broker.publish("updated")
    broker.publish("deleted")
    await asyncio.sleep(0.05)
    assert queue.qsize() == 1
