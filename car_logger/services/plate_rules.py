"""Pure decision rules for plate data quality.

OCR output is a hypothesis with a confidence score, not a fact. These
functions decide — in one place — when a reading is trustworthy enough to
mint a Vehicle identity. Events always keep the raw reading regardless."""

import re

# Bucharest: B + 2-3 digits + 3 letters. Counties: 2 letters + 2 digits +
# 3 letters. Applied to normalized text (uppercase, no separators).
_RO_PLATE_RE = re.compile(r"^(B\d{2,3}|[A-Z]{2}\d{2})[A-Z]{3}$")


def normalize_plate(text):
    """Uppercase and strip spaces/dashes; None passes through."""
    if text is None:
        return None
    return text.replace(" ", "").replace("-", "").upper()


def is_valid_ro_plate(text):
    """True if the normalized text looks like a Romanian plate."""
    if not text:
        return False
    return _RO_PLATE_RE.match(normalize_plate(text)) is not None


def should_create_vehicle(plate_text, confidence, region, min_confidence):
    """The identity gate: trustworthy enough to create/update a Vehicle?

    STUDENT DECISIONS (2026-07-08): threshold configurable (default 0.85);
    the RO format check applies ONLY when the API says region == "ro" — a
    Romanian regex applied blindly rejects correct foreign reads (the
    CZ-plate lesson)."""
    if not plate_text:
        return False
    if confidence is None or confidence < min_confidence:
        return False
    if region == "ro" and not is_valid_ro_plate(plate_text):
        return False
    return True
