import numpy as np

from car_logger.services.capture import CameraWorker


class FakeClock(object):
    """Deterministic monotonic clock for tests."""
    def __init__(self, t=0.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


def _worker(clock, **kw):
    return CameraWorker(now=clock, stale_after_s=2.0, **kw)


def test_fresh_frame_is_returned_and_healthy():
    clock = FakeClock(50.0)
    w = _worker(clock)
    frame = np.zeros((4, 4), np.uint8)
    w._frame = frame
    w._last_frame_at = 50.0
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), frame)


def test_stale_frame_reads_as_absent_and_unhealthy():
    clock = FakeClock(50.0)
    w = _worker(clock)
    w._frame = np.zeros((4, 4), np.uint8)
    w._last_frame_at = 50.0
    clock.t = 53.0  # 3s later, past the 2s threshold
    assert w.is_healthy() is False
    assert w.get_latest_frame() is None


def test_no_frame_yet_is_unhealthy():
    w = _worker(FakeClock(0.0))
    assert w.is_healthy() is False
    assert w.get_latest_frame() is None


class FakeCapture(object):
    """Fake cv2.VideoCapture: yields `frame` while alive+opened, else fails."""
    def __init__(self, frame, alive=True, opened=True):
        self.frame = frame
        self.alive = alive
        self._opened = opened
        self.released = False
    def isOpened(self):
        return self._opened
    def read(self):
        if self.alive and self._opened:
            return True, self.frame
        return False, None
    def release(self):
        self.released = True
        self._opened = False


def test_run_once_stores_fresh_frame_and_is_healthy():
    clock = FakeClock(100.0)
    frame = np.zeros((4, 4), np.uint8)
    cap = FakeCapture(frame)
    w = _worker(clock, open_capture=lambda i: cap)
    assert w._run_once() is False  # first call opens the capture
    assert w._run_once() is True   # second call reads a frame
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), frame)


def test_single_dropped_frame_under_threshold_does_not_reopen():
    clock = FakeClock(100.0)
    cap = FakeCapture(np.zeros((4, 4), np.uint8))
    w = _worker(clock, open_capture=lambda i: cap)
    w._run_once(); w._run_once()          # open + one good read
    cap.alive = False                     # a single dropped frame
    assert w._run_once() is False
    assert w.is_healthy() is True         # clock unchanged -> still fresh
    assert w._cap is cap                  # NOT reopened
    assert cap.released is False


def test_loss_after_threshold_releases_and_forces_reopen():
    clock = FakeClock(100.0)
    cap = FakeCapture(np.zeros((4, 4), np.uint8))
    w = _worker(clock, open_capture=lambda i: cap)
    w._run_once(); w._run_once()          # healthy
    cap.alive = False
    clock.advance(3.0)                    # past the 2s threshold
    assert w._run_once() is False
    assert w._lost is True
    assert cap.released is True
    assert w._cap is None                 # dropped -> next call reopens
    assert w.is_healthy() is False
    assert w.get_latest_frame() is None


def test_recovery_after_reopen_clears_lost_flag():
    clock = FakeClock(100.0)
    dead = FakeCapture(np.zeros((4, 4), np.uint8))
    fresh_frame = np.ones((4, 4), np.uint8)
    alive = FakeCapture(fresh_frame)
    caps = [dead, alive]
    w = _worker(clock, open_capture=lambda i: caps.pop(0))
    w._run_once(); w._run_once()          # open dead + one good read
    dead.alive = False
    clock.advance(3.0)
    w._run_once()                         # declared lost, cap dropped
    assert w._run_once() is False         # reopens -> `alive`
    assert w._run_once() is True          # good read on the new capture
    assert w._lost is False
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), fresh_frame)


def test_reopen_that_stays_closed_does_not_crash():
    clock = FakeClock(100.0)
    closed = FakeCapture(np.zeros((4, 4), np.uint8), opened=False)
    w = _worker(clock, open_capture=lambda i: closed)
    assert w._run_once() is False         # opens a not-opened capture
    assert w._run_once() is False         # sees it closed, reopens again
    assert w.is_healthy() is False        # never raises


import time as _time


def test_loop_captures_live_frames_and_stops_cleanly():
    frame = np.zeros((4, 4), np.uint8)
    cap = FakeCapture(frame)
    w = CameraWorker(open_capture=lambda i: cap, now=_time.monotonic,
                     stale_after_s=2.0, reopen_backoff_s=0.05)
    w.start()
    deadline = _time.monotonic() + 1.0
    while _time.monotonic() < deadline and not w.is_healthy():
        _time.sleep(0.01)
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), frame)
    w.stop()
    assert w._running is False
