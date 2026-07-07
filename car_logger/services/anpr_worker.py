"""ANPR worker: consumes (event_id, jpg_bytes) jobs from a bounded queue.

Why a separate thread + queue? A Plate Recognizer round-trip takes 300-800ms.
If the detection pipeline waited on it, FPS would collapse from ~15 to ~2.
Instead the pipeline drops a job into the queue (microseconds) and moves on;
this worker absorbs the network latency on its own thread.

The queue is BOUNDED (default 32 jobs): if ANPR falls behind — internet down,
API slow — we shed load by marking events "skipped" rather than growing an
unbounded buffer on a 4GB machine.
"""

import logging
import os
import queue
import threading
import time

from car_logger import repositories
from car_logger.config import settings

logger = logging.getLogger(__name__)


class AnprWorker(object):
    def __init__(self, client, session_factory, plates_dir=None,
                 maxsize=None):
        self.client = client
        self.session_factory = session_factory
        self.plates_dir = (plates_dir if plates_dir is not None
                           else settings.plates_dir)
        self._queue = queue.Queue(
            maxsize=maxsize if maxsize is not None
            else settings.anpr_queue_maxsize)
        self._running = False
        self._thread = None

    def start(self):
        os.makedirs(self.plates_dir, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, event_id, jpg_bytes):
        """Called from the PIPELINE thread — must never block.

        Returns True if queued; on a full queue the event is marked skipped
        and we return False (load shedding, not an error)."""
        try:
            self._queue.put_nowait((event_id, jpg_bytes))
            return True
        except queue.Full:
            logger.warning("ANPR queue full, skipping event %s", event_id)
            self._update(event_id, status="skipped")
            return False

    def pending(self):
        """How many jobs wait in the queue (for /api/status)."""
        return self._queue.qsize()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while self._running:
            try:
                event_id, jpg_bytes = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue  # wake up regularly so stop() can take effect
            try:
                self._process(event_id, jpg_bytes)
            except Exception:
                # One bad job must not kill the worker thread for good.
                logger.exception("ANPR job for event %s crashed", event_id)

    def _process(self, event_id, jpg_bytes):
        result = self.client.read_plate(jpg_bytes)
        # Save the crop for every outcome — a failed read is exactly the
        # image you want to look at when debugging.
        image_path = self._save_crop(event_id, jpg_bytes)
        self._update(event_id, status=result.status,
                     plate_text=result.plate_text,
                     confidence=result.confidence,
                     image_path=image_path)

    def _save_crop(self, event_id, jpg_bytes):
        # Forward slashes on purpose: the same string is stored in the DB
        # and later used as a URL path by the dashboard.
        path = "{0}/{1}.jpg".format(self.plates_dir.rstrip("/"), event_id)
        with open(path, "wb") as fh:
            fh.write(jpg_bytes)
        return path

    def _update(self, event_id, status, plate_text=None, confidence=None,
                image_path=None):
        # New short-lived session per outcome: sessions are not thread-safe,
        # so this thread never shares one with the API or the pipeline.
        db = self.session_factory()
        try:
            repositories.update_event_anpr(db, event_id, status=status,
                                           plate_text=plate_text,
                                           confidence=confidence,
                                           image_path=image_path)
        finally:
            db.close()


def cleanup_old_crops(plates_dir=None, max_age_days=None):
    """Delete crops older than the retention window (student: 30 days).

    Called at startup; combined with the daily service restart planned for
    Stage 5, that makes it an automatic daily sweep. Returns the number of
    files removed."""
    plates_dir = plates_dir if plates_dir is not None else settings.plates_dir
    max_age_days = (max_age_days if max_age_days is not None
                    else settings.crop_retention_days)
    if not os.path.isdir(plates_dir):
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for name in os.listdir(plates_dir):
        path = os.path.join(plates_dir, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            logger.warning("could not delete old crop %s", path)
    return removed
