"""Background ANPR worker: decouples the slow network call from the pipeline.

The pipeline calls submit() and returns immediately. This worker thread pulls
jobs off a bounded queue, calls the ANPR client, and hands the result to a
callback (which persists it). Under load the queue fills and submit() drops the
job rather than block the pipeline — a dropped plate read is acceptable; a
stalled pipeline is not."""

import queue
import threading


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

    def submit(self, event_id, crop_bytes):
        """Enqueue a job; return False if dropped because the queue is full."""
        try:
            self._queue.put_nowait((event_id, crop_bytes))
            return True
        except queue.Full:
            return False

    def _loop(self):
        while self._running:
            try:
                event_id, crop_bytes = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self._client.read_plate(crop_bytes)
                self._on_result(event_id, result, crop_bytes)
            finally:
                self._queue.task_done()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
