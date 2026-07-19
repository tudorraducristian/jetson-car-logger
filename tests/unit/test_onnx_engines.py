"""Pure decode helpers — no onnxruntime, no model files. The shapes and
conventions are facts fixed by the Stage A spike (2026-07-18): detector
rows are [image_id, x1, y1, x2, y2, class_id, score] in 384x384
plain-resize space; OCR heads may come flattened and in any order, so
they are matched by element count."""

import numpy as np

from car_logger.services.onnx_engines import (best_detection,
                                              decode_ocr_outputs,
                                              region_to_code)


def test_best_detection_picks_highest_score_and_scales_back():
    rows = np.array([[0, 10, 10, 100, 50, 0, 0.5],
                     [0, 20, 20, 120, 60, 0, 0.9]], dtype="float32")
    box = best_detection([rows], orig_width=768, orig_height=768,
                         input_side=384, threshold=0.4)
    assert box == (40, 40, 240, 120)  # 768/384 = 2x scale, second row wins


def test_best_detection_below_threshold_is_none():
    rows = np.array([[0, 10, 10, 100, 50, 0, 0.3]], dtype="float32")
    assert best_detection([rows], 768, 768, 384, 0.4) is None


def test_best_detection_accepts_batched_output():
    rows = np.array([[[0, 10, 10, 100, 50, 0, 0.9]]], dtype="float32")
    assert best_detection([rows], 384, 384, 384, 0.4) == (10, 10, 100, 50)


def test_best_detection_degenerate_box_is_none():
    rows = np.array([[0, 50, 50, 50, 50, 0, 0.9]], dtype="float32")
    assert best_detection([rows], 384, 384, 384, 0.4) is None


def test_decode_ocr_argmax_strips_pad_and_reads_the_region_head():
    cfg = {"max_plate_slots": 2, "alphabet": "AB_", "pad_char": "_",
           "plate_regions": ["Romania", "Unknown"]}
    chars = np.zeros((1, 2, 3), dtype="float32")
    chars[0, 0, 0] = 0.9   # slot 0 -> 'A'
    chars[0, 1, 2] = 0.8   # slot 1 -> pad, stripped from the text
    region = np.array([[0.7, 0.3]], dtype="float32")
    text, confidence, region_name = decode_ocr_outputs([chars, region], cfg)
    assert text == "A"
    assert region_name == "Romania"
    assert abs(confidence - (0.9 + 0.8) / 2.0) < 1e-6


def test_decode_ocr_handles_flattened_heads_and_no_region():
    cfg = {"max_plate_slots": 2, "alphabet": "AB_", "pad_char": "_"}
    flat = np.zeros((1, 6), dtype="float32")
    flat[0, 0] = 1.0   # slot 0 -> 'A'
    flat[0, 4] = 1.0   # slot 1 -> 'B'
    text, confidence, region_name = decode_ocr_outputs([flat], cfg)
    assert text == "AB"
    assert region_name is None


def test_region_to_code_student_mapping():
    assert region_to_code("Romania") == "ro"       # the RO gate fires on this
    assert region_to_code("Unknown") is None
    assert region_to_code(None) is None
    assert region_to_code("Czech Republic") == "czech republic"
