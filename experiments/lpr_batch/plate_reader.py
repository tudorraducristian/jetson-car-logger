"""Folder -> plate rows -> Excel. Pure, testable helpers (no model here)."""

import statistics
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


def _aggregate_confidence(confidence):
    """Collapse a float or per-character list into one float."""
    if isinstance(confidence, list):
        return statistics.mean(confidence) if confidence else 0.0
    return float(confidence)


def best_plate(results):
    """Return (text, confidence) for the highest-confidence reading with text."""
    best_text = ""
    best_confidence = 0.0
    found = False
    for result in results:
        ocr = getattr(result, "ocr", None)
        if ocr is None or not ocr.text:
            continue
        confidence = _aggregate_confidence(ocr.confidence)
        if not found or confidence > best_confidence:
            best_text = ocr.text
            best_confidence = confidence
            found = True
    return best_text, best_confidence
