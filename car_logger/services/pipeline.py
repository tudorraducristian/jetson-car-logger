"""Pipeline worker: camera -> detector -> tracker -> on_event callback.

Runs in its own daemon thread. It reads only the latest frame (drops stale
ones), so it never falls behind and never buffers video."""

import json
import threading
import time


class PipelineWorker(object):
    def __init__(self, camera, detector, tracker, on_event, target_fps=15):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.on_event = on_event
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
            t0 = time.time()
            frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            detections = self.detector.detect(frame)
            boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
            self.tracker.update(boxes)
            for track in self.tracker.new_confirmed_tracks():
                self._emit(track)
            self.frames_processed += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                self.last_fps = 1.0 / elapsed
            # Throttle to the target FPS so we don't pin the GPU pointlessly.
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

    def _emit(self, track):
        event = {
            "bbox_json": json.dumps(list(track.box)),
            "track_id": track.track_id,
            "anpr_status": "pending",
        }
        self.last_event_at = time.time()
        self.on_event(event)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
