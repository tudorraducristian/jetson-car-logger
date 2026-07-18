"""Canonical bake-off dataset layout + predictions CSV I/O.

Dataset dir: images/ (*.jpg with a plate) + labels.csv
(header filename,plate_text; filename RELATIVE to the dataset dir) +
optional plateless/ (*.jpg guaranteed plate-free, for false-positive rate).

Predictions CSV: filename,plate_text,confidence,latency_ms — empty
plate_text means "read nothing" (the candidate's no-plate answer);
confidence is 0-1 across ALL candidates (runners convert)."""

import csv
import os

PREDICTIONS_HEADER = ["filename", "plate_text", "confidence", "latency_ms"]
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_labels(dataset_dir):
    path = os.path.join(dataset_dir, "labels.csv")
    rows = []
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            img = os.path.join(dataset_dir, row["filename"])
            if not os.path.isfile(img):
                raise ValueError(
                    "labels.csv references missing image: {0}".format(img))
            rows.append((row["filename"], row["plate_text"]))
    if not rows:
        raise ValueError("no labels found in {0}".format(path))
    return rows


def list_plateless(dataset_dir):
    d = os.path.join(dataset_dir, "plateless")
    if not os.path.isdir(d):
        return []
    return sorted(
        "plateless/" + name for name in os.listdir(d)
        if name.lower().endswith(_IMAGE_EXTS)
    )


def write_predictions(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(PREDICTIONS_HEADER)
        for filename, plate_text, confidence, latency_ms in rows:
            writer.writerow([
                filename,
                plate_text if plate_text else "",
                "" if confidence is None else "{0:.4f}".format(confidence),
                "{0:.1f}".format(latency_ms),
            ])


def read_predictions(path):
    out = {}
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["filename"]] = (
                row["plate_text"] or None,
                float(row["confidence"]) if row["confidence"] else None,
                float(row["latency_ms"]),
            )
    return out
