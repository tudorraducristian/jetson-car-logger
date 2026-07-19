"""LocalAnprClient with fake engines: no models, no Jetson, no network —
the same injected-dependency pattern the v1 cloud client used with its
injected httpx client."""

import cv2
import numpy as np

from car_logger.services.local_anpr import LocalAnprClient
from car_logger.services.plate_result import PlateResult


def _jpeg(width=64, height=48):
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    assert ok
    return buf.tobytes()


class FakeDetector(object):
    def __init__(self, box):
        self.box = box
        self.calls = 0

    def detect_plate(self, image_bgr):
        self.calls += 1
        return self.box


class FakeOcr(object):
    def __init__(self, text="CJ 45 ARL", confidence=0.97, region="ro",
                 raises=False):
        self.text = text
        self.confidence = confidence
        self.region = region
        self.raises = raises
        self.calls = 0

    def read(self, plate_bgr):
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return self.text, self.confidence, self.region


class SeqOcr(object):
    """Returns a different (text, conf, region) per call — one per crop."""

    def __init__(self, reads):
        self._reads = list(reads)

    def read(self, plate_bgr):
        return self._reads.pop(0)


def test_happy_path_normalizes_text_and_carries_region():
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)), FakeOcr())
    result = client.read_plate(_jpeg())
    assert result == PlateResult("CJ45ARL", 0.97, "success", "ro")


def test_no_plate_found_never_calls_ocr():
    ocr = FakeOcr()
    client = LocalAnprClient(FakeDetector(None), ocr)
    result = client.read_plate(_jpeg())
    assert result.status == "no_plate"
    assert ocr.calls == 0  # the "OCR only when a plate is found" requirement


def test_engine_exception_becomes_failed_never_raises():
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)),
                             FakeOcr(raises=True))
    result = client.read_plate(_jpeg())
    assert result.status == "failed"


def test_corrupt_image_bytes_become_failed():
    client = LocalAnprClient(FakeDetector((0, 0, 1, 1)), FakeOcr())
    result = client.read_plate(b"definitely not a jpeg")
    assert result.status == "failed"


def test_empty_ocr_text_is_failed_plate_seen_but_unreadable():
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)), FakeOcr(text=""))
    result = client.read_plate(_jpeg())
    assert result.status == "failed"


def test_read_plate_multi_votes_and_returns_the_winning_crop():
    crop_a, crop_b, crop_c = _jpeg(64), _jpeg(66), _jpeg(68)  # distinct bytes
    ocr = SeqOcr([("CJ45ARL", 0.91, "ro"),
                  ("CJ45ARI", 0.99, "ro"),
                  ("CJ45ARL", 0.97, "ro")])
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)), ocr)
    result, evidence = client.read_plate_multi([crop_a, crop_b, crop_c])
    assert result.plate_text == "CJ45ARL"
    assert result.confidence == 0.97   # max among the agreeing reads
    assert evidence == crop_c          # the winning read's crop


def test_read_plate_multi_empty_list_is_failed():
    client = LocalAnprClient(FakeDetector(None), FakeOcr())
    result, evidence = client.read_plate_multi([])
    assert result.status == "failed"
    assert evidence == b""


def test_close_is_safe_with_and_without_engine_close():
    class Closable(FakeDetector):
        def __init__(self):
            FakeDetector.__init__(self, None)
            self.closed = False

        def close(self):
            self.closed = True

    detector = Closable()
    client = LocalAnprClient(detector, FakeOcr())  # FakeOcr has no close()
    client.close()
    assert detector.closed is True
