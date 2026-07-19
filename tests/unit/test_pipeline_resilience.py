"""codex finding 5: one exception must not kill the appliance's CV thread."""

import time

from car_logger.services.pipeline import PipelineWorker


class OneFrameCamera(object):
    def get_latest_frame(self):
        return "frame"


class FlakyDetector(object):
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient CUDA hiccup")
        return []


class NullTracker(object):
    def update(self, boxes):
        return []

    def new_confirmed_tracks(self):
        return []


def test_pipeline_survives_detector_exception():
    detector = FlakyDetector()
    worker = PipelineWorker(camera=OneFrameCamera(), detector=detector,
                            tracker=NullTracker(),
                            on_confirmed=lambda t, f: None, target_fps=200)
    worker.start()
    deadline = time.time() + 3.0
    while worker.frames_processed < 2 and time.time() < deadline:
        time.sleep(0.05)
    worker.stop()
    assert detector.calls >= 2           # kept calling after the raise
    assert worker.frames_processed >= 1  # processed frames post-exception


class NullDetector(object):
    def detect(self, frame):
        return []


class RecordingCollector(object):
    def __init__(self):
        self.ticks = []

    def tick(self, live_tracks, frame):
        self.ticks.append((list(live_tracks), frame))


def test_pipeline_ticks_the_collector_every_frame():
    collector = RecordingCollector()
    worker = PipelineWorker(camera=OneFrameCamera(), detector=NullDetector(),
                            tracker=NullTracker(),
                            on_confirmed=lambda t, f: None,
                            target_fps=200, collector=collector)
    worker.start()
    deadline = time.time() + 3.0
    while worker.frames_processed < 1 and time.time() < deadline:
        time.sleep(0.05)
    worker.stop()
    assert len(collector.ticks) >= 1
    assert collector.ticks[0][1] == "frame"
