"""The vote is Stage B's real error filter — the bake-off proved the
OCR's confidence cannot tell right from wrong (all 7 wrong reads sat at
conf >= 0.9997). Every rule here is a student decision from 2026-07-19;
see the Stage B spec."""

from car_logger.services.plate_result import PlateResult
from car_logger.services.plate_voting import vote_on_reads


def _ok(text, conf=0.99, region="ro"):
    return PlateResult(text, conf, "success", region)


def _no_plate():
    return PlateResult(None, None, "no_plate", None)


def _failed():
    return PlateResult(None, None, "failed", None)


def test_three_identical_reads_win():
    result, _ = vote_on_reads([_ok("CJ45ARL"), _ok("CJ45ARL"), _ok("CJ45ARL")])
    assert result.status == "success"
    assert result.plate_text == "CJ45ARL"


def test_two_of_three_agreeing_beat_the_odd_one_out():
    result, _ = vote_on_reads([_ok("CJ45ARL"), _ok("CJ45ARI"), _ok("CJ45ARL")])
    assert result.status == "success"
    assert result.plate_text == "CJ45ARL"


def test_three_different_texts_fail():
    result, _ = vote_on_reads([_ok("AAA111"), _ok("BBB222"), _ok("CCC333")])
    assert result.status == "failed"
    assert result.plate_text is None


def test_no_plate_abstains_so_a_single_text_is_accepted():
    result, _ = vote_on_reads([_ok("CJ45ARL"), _no_plate(), _no_plate()])
    assert result.status == "success"
    assert result.plate_text == "CJ45ARL"


def test_two_way_tie_fails():
    result, _ = vote_on_reads([_ok("AAA111"), _ok("BBB222"), _no_plate()])
    assert result.status == "failed"


def test_single_read_is_accepted():
    # graceful degradation to v1's single-read behavior (fast cars)
    result, _ = vote_on_reads([_ok("CJ45ARL")])
    assert result.status == "success"


def test_all_no_plate_is_no_plate():
    result, _ = vote_on_reads([_no_plate(), _no_plate(), _no_plate()])
    assert result.status == "no_plate"


def test_technical_failures_without_texts_fail():
    result, _ = vote_on_reads([_failed(), _no_plate(), _failed()])
    assert result.status == "failed"


def test_empty_input_fails():
    result, _ = vote_on_reads([])
    assert result.status == "failed"


def test_verdict_carries_max_confidence_of_the_agreeing_reads():
    reads = [_ok("CJ45ARL", 0.91), _ok("CJ45ARL", 0.97), _ok("XX99XXX", 0.99)]
    result, winner_index = vote_on_reads(reads)
    assert result.confidence == 0.97
    assert winner_index == 1


def test_verdict_region_comes_from_the_winning_read():
    reads = [PlateResult("CJ45ARL", 0.91, "success", None),
             PlateResult("CJ45ARL", 0.97, "success", "ro")]
    result, _ = vote_on_reads(reads)
    assert result.region == "ro"
