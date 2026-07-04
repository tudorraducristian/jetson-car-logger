"""Folder -> plate rows -> Excel. Pure, testable helpers (no model here)."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder):
    """Return image files directly inside ``folder``, sorted by name."""
    folder = Path(folder)
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
