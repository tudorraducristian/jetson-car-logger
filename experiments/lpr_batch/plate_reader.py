"""Folder -> plate rows -> Excel. Pure, testable helpers (no model here)."""

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook

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


@dataclass
class PlateRow:
    filename: str
    plate_text: str
    confidence: float


def read_folder(folder, predict):
    """Run ``predict`` on every image in ``folder`` and return one row each.

    A failing image (corrupt file, inference error) gets a blank row and a
    warning on stderr instead of aborting the rest of the batch.
    """
    rows = []
    for image_path in list_images(folder):
        try:
            results = predict(str(image_path))
        except Exception as exc:
            print(f"WARNING: {image_path.name}: {exc}", file=sys.stderr)
            results = []
        text, confidence = best_plate(results)
        rows.append(
            PlateRow(
                filename=image_path.name,
                plate_text=text,
                confidence=round(confidence, 4),
            )
        )
    return rows


def write_excel(rows, out_path):
    """Write rows to an .xlsx with a header line."""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "plates"
    worksheet.append(["filename", "plate_text", "confidence"])
    for row in rows:
        worksheet.append([row.filename, row.plate_text, row.confidence])
    workbook.save(str(out_path))
