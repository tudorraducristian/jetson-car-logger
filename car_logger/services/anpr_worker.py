"""Background ANPR worker: decouples the slow reads from the pipeline.

The pipeline (via the CropCollector) calls submit() with one car's crop
list and returns immediately. This worker thread pulls jobs off a bounded
queue, calls the ANPR client's multi-read (N crops -> one voted result +
the evidence crop), and hands the result to a callback (which persists
it). Under load the queue fills and submit() drops the job rather than
block the pipeline — a dropped plate read is acceptable; a stalled
pipeline is not."""

import logging
import queue
import threading

from car_logger.services.plate_result import PlateResult

log = logging.getLogger(__name__)


class AnprWorker(object):
    def __init__(self, anpr_client, on_result, queue_maxsize=32):
        self._client = anpr_client
        self._on_result = on_result
        self._queue = queue.Queue(maxsize=queue_maxsize)
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, event_id, crops):
        """Enqueue one car's crop list; return False if dropped (queue full)."""
        try:
            self._queue.put_nowait((event_id, crops))
            return True
        except queue.Full:
            return False

    def pending(self):
        """How many jobs wait in the queue (shown on the dashboard)."""
        return self._queue.qsize()

    def _loop(self):
        # Defense in depth (student decision, 2026-07-07): this thread is a
        # single point of failure — if it dies, every later event silently
        # stays 'pending'. So NOTHING a job throws may escape this loop; we
        # log it loudly instead, and the event still gets a result.
        while self._running:
            try:
                event_id, crops = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                try:
                    result, evidence = self._client.read_plate_multi(crops)
                except Exception:
                    log.exception(
                        "ANPR client raised for event %s; marking it failed",
                        event_id)
                    result = PlateResult(None, None, "failed", None)
                    evidence = crops[0] if crops else None
                try:
                    self._on_result(event_id, result, evidence)
                except Exception:
                    log.exception(
                        "on_result callback raised for event %s", event_id)
            finally:
                self._queue.task_done()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Daily-restart reality: jobs still queued at shutdown would leave
        # their events 'pending' forever. Mark them skipped instead.
        while True:
            try:
                event_id, crops = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._on_result(
                    event_id, PlateResult(None, None, "skipped", None),
                    crops[0] if crops else None)
            except Exception:
                log.exception("drain: on_result raised for event %s",
                              event_id)
        self._client.close()
