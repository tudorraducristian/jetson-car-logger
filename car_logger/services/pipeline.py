"""Pipeline worker: camera -> detector -> tracker -> on_confirmed callback.

on_confirmed(track, frame) is called once per newly-confirmed track, with the
frame it was confirmed on (so the caller can crop the plate). The callback owns
persistence and ANPR submission; the pipeline stays CV-only."""

import logging
import threading
import time

log = logging.getLogger(__name__)


class PipelineWorker(object):
    def __init__(self, camera, detector, tracker, on_confirmed, target_fps=15):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.on_confirmed = on_confirmed
        self._min_interval = 1.0 / float(target_fps)
        self._running = False
        self._thread = None
        self.last_fps = 0.0
        self.last_event_at = None
        self.frames_processed = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception:
                # A transient failure (SQLite lock, detector hiccup, bad
                # frame) must not kill the appliance's only CV thread.
                # Short sleep so a persistent failure can't spin the CPU.
                log.exception("pipeline tick failed; continuing")
                time.sleep(0.5)

    def _tick(self):
        t0 = time.time()
        frame = self.camera.get_latest_frame()
        if frame is None:
            time.sleep(0.02)
            return
        detections = self.detector.detect(frame)
        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        self.tracker.update(boxes)
        for track in self.tracker.new_confirmed_tracks():
            self.last_event_at = time.time()
            self.on_confirmed(track, frame)
        self.frames_processed += 1
        elapsed = time.time() - t0
        if elapsed > 0:
            self.last_fps = 1.0 / elapsed
        # Throttle to the target FPS so we don't pin the GPU pointlessly.
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
