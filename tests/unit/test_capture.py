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
