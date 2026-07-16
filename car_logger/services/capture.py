"""Camera capture worker: cv2.VideoCapture in a daemon thread that heals
itself when the USB webcam drops.

Why a thread? cv2.VideoCapture(0) blocks 3-5s at open and each read() blocks
until a frame arrives. The worker keeps only the *latest* frame (no
buffering) to respect the 4GB RAM budget.

Freshness is the single source of truth: a frame older than stale_after_s
reads as absent (get_latest_frame -> None) and unhealthy, so a frozen frame
can never masquerade as a live camera. cv2 is imported lazily in the default
factory so this module imports without OpenCV (e.g. off-Jetson)."""

import threading
import time

from car_logger.logging_config import get_logger

log = get_logger("car_logger.capture")


def _default_open_capture(device_index):
    import cv2
    return cv2.VideoCapture(device_index)


class CameraWorker(object):
    def __init__(self, device_index=0, stale_after_s=2.0,
                 reopen_backoff_s=2.0, open_capture=None, now=None):
        self.device_index = device_index
        self._stale_after_s = stale_after_s
        self._reopen_backoff_s = reopen_backoff_s
        self._open_capture = open_capture or _default_open_capture
        self._now = now or time.monotonic
        self._cap = None
        self._frame = None
        self._last_frame_at = None
        self._lost = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def _seconds_since_frame(self):
        if self._last_frame_at is None:
            return float("inf")
        return self._now() - self._last_frame_at

    def is_healthy(self):
        """True only if a fresh frame arrived within stale_after_s."""
        with self._lock:
            return (self._frame is not None
                    and self._seconds_since_frame() <= self._stale_after_s)

    def get_latest_frame(self):
        """A private copy of the latest frame, or None if there is none or
        it is stale. Copy under the lock so the caller can't read a frame the
        capture thread is mid-overwriting."""
        with self._lock:
            if (self._frame is None
                    or self._seconds_since_frame() > self._stale_after_s):
                return None
            return self._frame.copy()

    def _run_once(self):
        """One capture iteration: (re)open if needed, read once, store a
        fresh frame, or declare the camera lost after stale_after_s. Returns
        True only when a fresh frame was stored. Sleeps nowhere, so the
        read/reconnect logic is unit-testable without threads."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = self._open_capture(self.device_index)
            return False
        ok, frame = self._cap.read()
        if ok:
            with self._lock:
                self._frame = frame
                self._last_frame_at = self._now()
            if self._lost:
                log.info("camera_reconnected", device_index=self.device_index)
                self._lost = False
            return True
        if self._seconds_since_frame() > self._stale_after_s:
            if not self._lost:
                log.warning("camera_lost", device_index=self.device_index)
                self._lost = True
            self._cap.release()
            self._cap = None
        return False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # A transient read failure must never kill the appliance's only CV
        # feed. _run_once owns the reopen; here we only pace the loop.
        while self._running:
            if self._run_once():
                continue  # read() blocks, so no sleep needed on success
            if self._cap is None or not self._cap.isOpened():
                time.sleep(self._reopen_backoff_s)
            else:
                time.sleep(0.01)  # brief, avoids a busy-spin before loss

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
