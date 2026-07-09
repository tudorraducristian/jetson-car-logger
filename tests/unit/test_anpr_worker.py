"""The worker thread must survive ANY exception — a dead thread means every
later event stays 'pending' forever, silently. Found in the stage 4 offline
test. Results are polled with a deadline (no fixed sleeps) to avoid races."""

import time

from car_logger.services.anpr_worker import AnprWorker


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FlakyClient(object):
    """First call raises (like ConnectError did); second call succeeds."""

    def __init__(self):
        self.calls = 0

    def read_plate(self, crop_bytes):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return ("OK", 0.9, "success")


def test_worker_survives_client_exception():
    got = []
    worker = AnprWorker(FlakyClient(), lambda eid, r, c: got.append((eid, r)))
    worker.start()
    worker.submit(1, b"a")
    worker.submit(2, b"b")
    assert _wait_for(lambda: len(got) == 2), "thread died after the exception"
    worker.stop()
    # job 1: the client blew up -> the event still gets a 'failed' result
    assert got[0][0] == 1
    assert got[0][1].status == "failed"
    # job 2: processed normally by the SAME still-alive thread
    assert got[1] == (2, ("OK", 0.9, "success"))


class OkClient(object):
    def read_plate(self, crop_bytes):
        return ("OK", 0.9, "success")


def test_worker_survives_callback_exception():
    seen = []

    def bad_then_good(event_id, result, crop_bytes):
        seen.append(event_id)
        if len(seen) == 1:
            raise RuntimeError("db down")

    worker = AnprWorker(OkClient(), bad_then_good)
    worker.start()
    worker.submit(1, b"a")
    worker.submit(2, b"b")
    assert _wait_for(lambda: len(seen) == 2), "thread died in the callback"
    worker.stop()
    assert seen == [1, 2]


def test_stop_drains_pending_jobs_as_skipped_and_closes_client():
    # codex finding 7: jobs queued at shutdown (daily 04:00 restart!) must
    # not leave their events 'pending' forever.
    calls = []

    class ClosableClient(object):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = ClosableClient()
    worker = AnprWorker(
        client, lambda eid, res, crop: calls.append((eid, res.status)))
    worker.submit(1, b"a")
    worker.submit(2, b"b")
    worker.stop()  # never started: everything is still queued
    assert calls == [(1, "skipped"), (2, "skipped")]
    assert client.closed is True
