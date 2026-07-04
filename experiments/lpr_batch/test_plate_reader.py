from pathlib import Path

from plate_reader import list_images


def test_list_images_returns_only_sorted_image_files(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "sub").mkdir()

    result = list_images(tmp_path)

    assert [p.name for p in result] == ["a.png", "b.jpg"]
