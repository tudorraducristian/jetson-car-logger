"""The identity gate: when is an OCR reading trustworthy enough for a Vehicle?"""

from car_logger.services.plate_rules import (
    is_valid_ro_plate, normalize_plate, should_create_vehicle)


def test_normalize_uppercases_and_strips_separators():
    assert normalize_plate("b 123-abc") == "B123ABC"


def test_normalize_none_passthrough():
    assert normalize_plate(None) is None


def test_ro_county_format_valid():
    assert is_valid_ro_plate("CJ45XYZ") is True


def test_ro_bucharest_format_valid():
    assert is_valid_ro_plate("B123ABC") is True


def test_ro_rejects_four_trailing_digits():
    assert is_valid_ro_plate("ELT4740") is False  # one of today's phantoms


def test_gate_rejects_below_threshold():
    assert should_create_vehicle("MMM8748", 0.60, None, 0.85) is False


def test_gate_accepts_confident_foreign_read():
    # region "cz": no RO regex — confidence alone decides (the CZ lesson)
    assert should_create_vehicle("EL147AD", 0.97, "cz", 0.85) is True


def test_gate_rejects_ro_region_with_bad_format():
    assert should_create_vehicle("ELT4740", 0.95, "ro", 0.85) is False


def test_gate_accepts_ro_region_with_good_format():
    assert should_create_vehicle("B123ABC", 0.95, "ro", 0.85) is True


def test_gate_rejects_missing_text_or_confidence():
    assert should_create_vehicle(None, 0.99, "ro", 0.85) is False
    assert should_create_vehicle("B123ABC", None, "ro", 0.85) is False
