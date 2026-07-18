import json

from run_openalpr import parse_alpr_json


def test_parse_alpr_json_takes_best_result_and_scales_confidence():
    payload = json.dumps({
        "processing_time_ms": 421.7,
        "results": [{"plate": "M5XSX", "confidence": 89.5}],
    })
    assert parse_alpr_json(payload) == ("M5XSX", 0.895, 421.7)


def test_parse_alpr_json_no_results_means_no_read():
    payload = json.dumps({"processing_time_ms": 380.0, "results": []})
    assert parse_alpr_json(payload) == (None, None, 380.0)
