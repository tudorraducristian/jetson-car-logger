"""ANPR worker tests.

The Plate Recognizer client is faked; the DB is the in-memory SQLite from
conftest. Deterministic paths call worker._process directly (no thread);
one smoke test runs the real thread end-to-end.
"""

import time

from car_logger import repositories, schemas
from car_logger.services.anpr_client import PlateResult
from car_logger.services.anpr_worker import AnprWorker, cleanup_old_crops


class FakeClient(object):
    """Scripted ANPR client: hands out the queued results in order (raising
    the ones that are exceptions); the last result repeats forever.

    Scripting everything up-front matters: mutating a shared attribute from
    the test thread while the worker thread reads it is a race condition —
    the first version of this file did exactly that and failed flakily."""

    def __init__(self, *results):
        self._results = list(results)
        self.received = []

    def read_plate(self, jpg_bytes):
        self.received.append(jpg_bytes)
        result = (self._results.pop(0) if len(self._results) > 1
                  else self._results[0])
        if isinstance(result, Exception):
            raise result
        return result


def make_event(session_factory):
    db = session_factory()
    try:
        event = repositories.create_event(db, schemas.EventCreate())
        return event.id
    finally:
        db.close()


def get_event(session_factory, event_id):
    db = session_factory()
    try:
        return repositories.get_event(db, event_id)
    finally:
        db.close()


def test_process_success_saves_crop_and_updates_event(session_factory,
                                                      tmp_path):
    event_id = make_event(session_factory)
    client = FakeClient(PlateResult(status="success", plate_text="B123XYZ",
                                    confidence=0.93))
    worker = AnprWorker(client, session_factory,
                        plates_dir=str(tmp_path), maxsize=4)

    worker._process(event_id, b"jpg-bytes")

    saved = tmp_path / ("%d.jpg" % event_id)
    assert saved.read_bytes() == b"jpg-bytes"
    event = get_event(session_factory, event_id)
    assert event.anpr_status == "success"
    assert event.plate_text == "B123XYZ"
    assert event.image_path.endswith("%d.jpg" % event_id)
    assert event.vehicle_id is not None
    assert client.received == [b"jpg-bytes"]


def test_process_failed_keeps_crop_for_debugging(session_factory, tmp_path):
    event_id = make_event(session_factory)
    worker = AnprWorker(FakeClient(PlateResult(status="failed")),
                        session_factory, plates_dir=str(tmp_path), maxsize=4)

    worker._process(event_id, b"blurry")

    assert (tmp_path / ("%d.jpg" % event_id)).exists()
    event = get_event(session_factory, event_id)
    assert event.anpr_status == "failed"
    assert event.plate_text is None


def test_full_queue_sheds_load_as_skipped(session_factory, tmp_path):
    queued_id = make_event(session_factory)
    shed_id = make_event(session_factory)
    worker = AnprWorker(FakeClient(PlateResult(status="success")),
                        session_factory, plates_dir=str(tmp_path), maxsize=1)
    # worker NOT started: the first job fills the queue and stays there

    assert worker.submit(queued_id, b"a") is True
    assert worker.submit(shed_id, b"b") is False

    assert get_event(session_factory, shed_id).anpr_status == "skipped"
    # the queued event is untouched until the worker picks it up
    assert get_event(session_factory, queued_id).anpr_status == "pending"
    assert worker.pending() == 1


def test_worker_thread_survives_a_crashing_job(session_factory, tmp_path):
    bad_id = make_event(session_factory)
    good_id = make_event(session_factory)
    # scripted: first call blows up, every later call succeeds
    client = FakeClient(RuntimeError("boom"),
                        PlateResult(status="success", plate_text="CJ10ABC",
                                    confidence=0.8))
    worker = AnprWorker(client, session_factory,
                        plates_dir=str(tmp_path), maxsize=4)
    worker.start()
    try:
        worker.submit(bad_id, b"bad")
        worker.submit(good_id, b"good")

        # poll until the good job lands (worker thread is async to the test)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if get_event(session_factory, good_id).anpr_status == "success":
                break
            time.sleep(0.05)
    finally:
        worker.stop()

    assert get_event(session_factory, good_id).anpr_status == "success"
    # the crashed job left its event pending — and the thread kept running
    assert get_event(session_factory, bad_id).anpr_status == "pending"


def test_cleanup_removes_only_old_crops(tmp_path):
    old = tmp_path / "1.jpg"
    fresh = tmp_path / "2.jpg"
    old.write_bytes(b"old")
    fresh.write_bytes(b"fresh")
    import os
    forty_days_ago = time.time() - 40 * 86400
    os.utime(str(old), (forty_days_ago, forty_days_ago))

    removed = cleanup_old_crops(plates_dir=str(tmp_path), max_age_days=30)

    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_on_missing_dir_is_a_noop(tmp_path):
    assert cleanup_old_crops(plates_dir=str(tmp_path / "nope"),
                             max_age_days=30) == 0
