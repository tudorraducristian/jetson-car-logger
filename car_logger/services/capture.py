"""Camera capture worker: cv2.VideoCapture in a daemon thread.

Why a thread? cv2.VideoCapture(0) blocks 3-5s at open and each read() blocks
until a frame arrives. Running it in the request path would freeze the server.
The worker keeps only the *latest* frame (no buffering) to respect the 4GB RAM
budget."""

import threading
import time

import cv2


class CameraWorker(object):
    def __init__(self, device_index=0):
        self.device_index = device_index
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._cap = cv2.VideoCapture(self.device_index)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Hold the lock only for the pointer swap, never during read().
            with self._lock:
                self._frame = frame

    def get_latest_frame(self):
        """Return a private copy of the latest frame, or None if none yet.

        We copy under the lock so the caller can't read a frame the capture
        thread is mid-overwriting (that would be a data race)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
