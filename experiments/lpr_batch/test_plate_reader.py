from pathlib import Path

from plate_reader import list_images


def test_list_images_returns_only_sorted_image_files(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "sub").mkdir()

    result = list_images(tmp_path)

    assert [p.name for p in result] == ["a.png", "b.jpg"]


from types import SimpleNamespace

import pytest

from plate_reader import best_plate


def _result(text, confidence):
    return SimpleNamespace(ocr=SimpleNamespace(text=text, confidence=confidence))


def test_best_plate_empty_returns_blank():
    assert best_plate([]) == ("", 0.0)


def test_best_plate_picks_highest_confidence():
    results = [_result("AAA111", 0.5), _result("BBB222", 0.9)]
    assert best_plate(results) == ("BBB222", 0.9)


def test_best_plate_averages_per_character_confidence():
    results = [_result("CJ23XZI", [0.8, 1.0])]
    text, confidence = best_plate(results)
    assert text == "CJ23XZI"
    assert confidence == pytest.approx(0.9)


def test_best_plate_skips_results_without_ocr():
    results = [SimpleNamespace(ocr=None), _result("OK123", 0.7)]
    assert best_plate(results) == ("OK123", 0.7)
