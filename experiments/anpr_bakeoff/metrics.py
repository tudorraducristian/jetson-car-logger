"""Pure scoring functions for the ANPR bake-off.

Predictions are normalized with the SAME rules production uses
(car_logger.services.plate_rules.normalize_plate), so candidates are
scored on what the app would actually store."""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from car_logger.services.plate_rules import normalize_plate  # noqa: E402


def exact_match(predicted, truth):
    """True when normalized prediction equals normalized truth.

    A missing prediction (None/empty) is always a miss."""
    if not predicted:
        return False
    return normalize_plate(predicted) == normalize_plate(truth)


def levenshtein(a, b):
    """Edit distance (insert/delete/substitute), classic DP, O(len*len)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + cost))
        prev = cur
    return prev[-1]


def cer(predicted, truth):
    """Character error rate against the normalized truth. 0.0 = perfect;
    a missing prediction deletes every character (1.0)."""
    norm_truth = normalize_plate(truth) or ""
    if not norm_truth:
        raise ValueError("truth plate text must not be empty")
    norm_pred = normalize_plate(predicted) or ""
    return levenshtein(norm_pred, norm_truth) / float(len(norm_truth))
