"""Metrics are the referee of the bake-off — they get the strictest TDD."""

from metrics import cer, exact_match, levenshtein


def test_exact_match_ignores_case_and_separators():
    # Same normalization as production: 'b-123 abc' == 'B123ABC'.
    assert exact_match("b-123 abc", "B123ABC") is True


def test_exact_match_missing_prediction_is_a_miss():
    assert exact_match(None, "B123ABC") is False
    assert exact_match("", "B123ABC") is False


def test_exact_match_wrong_text_is_a_miss():
    assert exact_match("B123ABD", "B123ABC") is False


def test_levenshtein_known_distances():
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("", "ABC") == 3
    assert levenshtein("ABC", "ABD") == 1
    assert levenshtein("AB", "ABC") == 1


def test_cer_missing_prediction_is_total_error():
    assert cer(None, "B123ABC") == 1.0


def test_cer_one_wrong_char_out_of_seven():
    assert abs(cer("B123ABD", "B123ABC") - 1.0 / 7.0) < 1e-9
