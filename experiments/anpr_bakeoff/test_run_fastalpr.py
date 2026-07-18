"""_best is the only local logic in the fast-alpr runner; stub results
mirror the API surface verified in the repr check (2026-07-18):
ocr.text plus ocr.confidence as a per-character LIST (fast-alpr 0.4.0).
The list is reduced to its mean — the same convention fast-alpr's own
draw code uses (statistics.mean in alpr.py)."""

from collections import namedtuple

from run_fastalpr import _best, _mean_confidence

Ocr = namedtuple("Ocr", ["text", "confidence"])
Res = namedtuple("Res", ["ocr"])


def test_best_none_when_nothing_detected():
    assert _best([]) is None
    assert _best([Res(ocr=None)]) is None


def test_best_picks_highest_mean_ocr_confidence():
    a = Res(ocr=Ocr("B123ABC", [0.90, 0.92]))
    b = Res(ocr=Ocr("B123ABD", [0.70, 0.74]))
    assert _best([b, a]) is a


def test_mean_confidence_handles_list_and_scalar():
    assert abs(_mean_confidence([0.8, 0.6]) - 0.7) < 1e-9
    assert _mean_confidence(0.5) == 0.5
