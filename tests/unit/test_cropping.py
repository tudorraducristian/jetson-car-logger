import numpy as np

from car_logger.services.cropping import crop_to_jpeg


def _frame():
    # a 100x100 BGR image
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_crop_returns_jpeg_bytes():
    data = crop_to_jpeg(_frame(), (10, 10, 60, 60))
    assert isinstance(data, bytes)
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_box_clamped_to_frame_bounds():
    # box extends past the 100x100 frame; must not raise and must return bytes
    data = crop_to_jpeg(_frame(), (-20, -20, 500, 500))
    assert isinstance(data, bytes) and len(data) > 0
