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


from plate_reader import PlateRow, read_folder


def test_read_folder_builds_one_row_per_image(tmp_path):
    (tmp_path / "car1.jpg").write_bytes(b"x")
    (tmp_path / "car2.jpg").write_bytes(b"x")

    def fake_predict(image_path):
        if Path(image_path).name == "car1.jpg":
            return [SimpleNamespace(ocr=SimpleNamespace(text="CJ23XZI", confidence=0.95))]
        return []  # no plate found

    rows = read_folder(tmp_path, fake_predict)

    assert rows == [
        PlateRow(filename="car1.jpg", plate_text="CJ23XZI", confidence=0.95),
        PlateRow(filename="car2.jpg", plate_text="", confidence=0.0),
    ]


from openpyxl import load_workbook

from plate_reader import write_excel


def test_write_excel_writes_header_and_rows(tmp_path):
    rows = [PlateRow(filename="car1.jpg", plate_text="CJ23XZI", confidence=0.95)]
    out_path = tmp_path / "plates.xlsx"

    write_excel(rows, out_path)

    worksheet = load_workbook(out_path).active
    assert [c.value for c in worksheet[1]] == ["filename", "plate_text", "confidence"]
    assert [c.value for c in worksheet[2]] == ["car1.jpg", "CJ23XZI", 0.95]
