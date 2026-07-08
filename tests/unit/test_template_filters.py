"""The localtime filter: display-side conversion of stored-UTC timestamps."""

from datetime import datetime, timezone

from car_logger.api.routes_dashboard import localtime


def test_localtime_keeps_the_instant():
    utc_naive = datetime(2026, 7, 8, 6, 53, 42)
    local = localtime(utc_naive)
    # Same instant, different clock face: converting the result back to UTC
    # must land exactly on the stored value, whatever the OS timezone is.
    assert local.utcoffset() is not None
    assert local.astimezone(timezone.utc).replace(tzinfo=None) == utc_naive


def test_localtime_none_passthrough():
    assert localtime(None) is None
