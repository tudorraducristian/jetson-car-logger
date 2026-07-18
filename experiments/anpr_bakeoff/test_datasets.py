import os

import pytest

from datasets import (list_plateless, load_labels, read_predictions,
                      write_predictions)


def _make_dataset(root):
    os.makedirs(str(root / "images"))
    (root / "images" / "a.jpg").write_bytes(b"\xff\xd8fake")
    (root / "labels.csv").write_text(
        "filename,plate_text\nimages/a.jpg,B123ABC\n")


def test_load_labels_returns_relative_filename_and_text(tmp_path):
    _make_dataset(tmp_path)
    assert load_labels(str(tmp_path)) == [("images/a.jpg", "B123ABC")]


def test_load_labels_rejects_missing_image(tmp_path):
    _make_dataset(tmp_path)
    (tmp_path / "labels.csv").write_text(
        "filename,plate_text\nimages/ghost.jpg,B123ABC\n")
    with pytest.raises(ValueError):
        load_labels(str(tmp_path))


def test_load_labels_rejects_empty_labels(tmp_path):
    os.makedirs(str(tmp_path / "images"))
    (tmp_path / "labels.csv").write_text("filename,plate_text\n")
    with pytest.raises(ValueError):
        load_labels(str(tmp_path))


def test_list_plateless_empty_when_folder_absent(tmp_path):
    _make_dataset(tmp_path)
    assert list_plateless(str(tmp_path)) == []


def test_list_plateless_returns_relative_sorted(tmp_path):
    _make_dataset(tmp_path)
    os.makedirs(str(tmp_path / "plateless"))
    (tmp_path / "plateless" / "b.jpg").write_bytes(b"x")
    (tmp_path / "plateless" / "a.jpg").write_bytes(b"x")
    assert list_plateless(str(tmp_path)) == [
        "plateless/a.jpg", "plateless/b.jpg"]


def test_predictions_round_trip(tmp_path):
    path = str(tmp_path / "preds.csv")
    write_predictions(path, [
        ("images/a.jpg", "B123ABC", 0.97, 812.3),
        ("images/b.jpg", None, None, 401.0),   # candidate read nothing
    ])
    assert read_predictions(path) == {
        "images/a.jpg": ("B123ABC", 0.97, 812.3),
        "images/b.jpg": (None, None, 401.0),
    }
