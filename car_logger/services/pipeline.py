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
                self._emit(track, frame)
            self.frames_processed += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                self.last_fps = 1.0 / elapsed
            # Throttle to the target FPS so we don't pin the GPU pointlessly.
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

    def _emit(self, track, frame):
        event = {
            "bbox_json": json.dumps(list(track.box)),
            "track_id": track.track_id,
            "anpr_status": "pending",
        }
        # Stage 4: hand the car's pixels over too — ANPR needs the image at
        # the moment the track was confirmed, and this frame is gone by the
        # time the ANPR worker runs.
        crop = self._crop(frame, track.box)
        if crop is None:
            # No usable pixels -> ANPR can never run; don't leave the event
            # stuck on "pending".
            event["anpr_status"] = "skipped"
        self.last_event_at = time.time()
        self.on_event(event, crop)

    @staticmethod
    def _crop(frame, box):
        """Cut the track's bbox out of the frame, clamped to the frame edges.

        Returns None for a degenerate box. The .copy() detaches the small
        crop from the full frame so the frame's memory can be freed while
        the crop waits in the ANPR queue (4GB budget)."""
        height, width = frame.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(width, int(box[2]))
        y2 = min(height, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
