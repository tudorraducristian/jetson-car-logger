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
