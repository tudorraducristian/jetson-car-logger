from evaluate import calibration_rows, score_candidate

LABELS = [("images/a.jpg", "B123ABC"), ("images/b.jpg", "CJ07XYZ")]
PLATELESS = ["plateless/w.jpg"]
PREDS = {
    "images/a.jpg": ("B123ABC", 0.95, 800.0),   # correct
    "images/b.jpg": ("CJ99XYZ", 0.60, 1200.0),  # wrong (2 chars)
    "plateless/w.jpg": ("FAKE123", 0.30, 700.0),  # false positive
}


def test_score_candidate_headline_numbers():
    s = score_candidate(LABELS, PLATELESS, PREDS)
    assert s["n"] == 2
    assert s["exact_match_rate"] == 0.5
    assert s["read_rate"] == 1.0
    assert s["fp_rate"] == 1.0
    assert abs(s["mean_cer"] - (0.0 + 2.0 / 7.0) / 2) < 1e-9
    assert s["mean_latency_ms"] == 1000.0


def test_score_candidate_missing_row_counts_as_no_read():
    s = score_candidate(LABELS, [], {"images/a.jpg": ("B123ABC", 0.95, 800.0)})
    assert s["read_rate"] == 0.5
    assert s["exact_match_rate"] == 0.5


def test_score_candidate_fp_rate_none_without_plateless():
    assert score_candidate(LABELS, [], PREDS)["fp_rate"] is None


def test_calibration_rows_only_actual_reads_with_correctness():
    rows = calibration_rows(LABELS, PREDS)
    assert rows == [(0.95, True), (0.60, False)]
