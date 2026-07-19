"""The collector implements the spec's Variant 1: crop #1 at track
confirmation, then up to 2 more from later frames spaced >= spacing_s
apart; hand the list over when full or when the track dies. Injectable
clock + crop_fn = deterministic tests, no cv2, no camera."""

from car_logger.services.crop_collector import CropCollector


class FakeTrack(object):
    def __init__(self, track_id, box=(0, 0, 10, 10), missed=0):
        self.track_id = track_id
        self.box = box
        self.missed = missed


class Clock(object):
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _crop_fn(frame, box):
    return (frame, box)  # opaque token; the collector never looks inside


def _collector(calls, clock, reads=3, spacing=0.4):
    return CropCollector(
        on_complete=lambda event_id, crops: calls.append((event_id, crops)),
        reads_per_track=reads, spacing_s=spacing,
        crop_fn=_crop_fn, now=clock)


def test_collects_three_spaced_crops_then_completes_once():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    track = FakeTrack(7)
    collector.start(7, event_id=42, box=track.box, frame="f0")

    collector.tick([track], "f1")          # too soon: only 0.0s elapsed
    assert calls == []
    clock.advance(0.4)
    collector.tick([track], "f2")          # crop #2
    clock.advance(0.4)
    collector.tick([track], "f3")          # crop #3 -> complete
    assert len(calls) == 1
    event_id, crops = calls[0]
    assert event_id == 42
    assert crops == [("f0", (0, 0, 10, 10)), ("f2", (0, 0, 10, 10)),
                     ("f3", (0, 0, 10, 10))]

    clock.advance(1.0)
    collector.tick([track], "f4")          # cleaned up: nothing re-fires
    assert len(calls) == 1


def test_track_death_hands_over_what_it_has():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    collector.start(7, event_id=42, box=(0, 0, 10, 10), frame="f0")
    collector.tick([], "f1")               # the track is gone
    assert calls == [(42, [("f0", (0, 0, 10, 10))])]


def test_missed_track_is_not_cropped_stale_box():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    track = FakeTrack(7, missed=2)
    collector.start(7, event_id=42, box=track.box, frame="f0")
    clock.advance(1.0)
    collector.tick([track], "f1")          # box is stale -> no crop taken
    assert calls == []
    track.missed = 0
    collector.tick([track], "f2")          # fresh again -> crop #2
    clock.advance(0.4)
    collector.tick([track], "f3")          # crop #3 -> complete
    assert len(calls) == 1
    assert len(calls[0][1]) == 3


def test_drain_flushes_partials():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    collector.start(7, event_id=42, box=(0, 0, 10, 10), frame="f0")
    collector.drain()
    assert calls == [(42, [("f0", (0, 0, 10, 10))])]
    collector.drain()                      # idempotent
    assert len(calls) == 1


def test_single_read_config_completes_immediately():
    calls, clock = [], Clock()
    collector = _collector(calls, clock, reads=1)
    collector.start(7, event_id=42, box=(0, 0, 10, 10), frame="f0")
    assert calls == [(42, [("f0", (0, 0, 10, 10))])]
