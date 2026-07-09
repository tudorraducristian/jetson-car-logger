"""on_result: ANPR completion — deleted-event check + the identity gate.

crop_bytes is b"" in every test (falsy) so no crop files get written."""

from car_logger import main as app_main
from car_logger import repositories, schemas
from car_logger.models import Vehicle
from car_logger.services.anpr_client import PlateResult


class FakeBroker(object):
    def __init__(self):
        self.published = []

    def publish(self, data):
        self.published.append(data)


def _on_result(monkeypatch, db_session):
    monkeypatch.setattr(app_main, "SessionLocal", lambda: db_session)
    broker = FakeBroker()
    return app_main._make_on_result(broker), broker


def test_deleted_event_creates_nothing(monkeypatch, db_session):
    # codex finding 1: event deleted while the ANPR call was in flight
    on_result, broker = _on_result(monkeypatch, db_session)
    on_result(9999, PlateResult("B123ABC", 0.99, "success", "ro"), b"")
    assert db_session.query(Vehicle).count() == 0
    assert broker.published == []


def test_low_confidence_keeps_text_but_no_vehicle(monkeypatch, db_session):
    on_result, broker = _on_result(monkeypatch, db_session)
    event = repositories.create_event(db_session, schemas.EventCreate())
    on_result(event.id, PlateResult("EL4740", 0.60, "success", "cz"), b"")
    refreshed = repositories.get_event(db_session, event.id)
    assert refreshed.plate_text == "EL4740"
    assert refreshed.region == "cz"
    assert refreshed.vehicle_id is None
    assert db_session.query(Vehicle).count() == 0
    assert broker.published == ["updated"]


def test_confident_read_creates_vehicle(monkeypatch, db_session):
    on_result, broker = _on_result(monkeypatch, db_session)
    event = repositories.create_event(db_session, schemas.EventCreate())
    on_result(event.id, PlateResult("B123ABC", 0.95, "success", "ro"), b"")
    refreshed = repositories.get_event(db_session, event.id)
    assert refreshed.vehicle_id is not None
    assert db_session.query(Vehicle).count() == 1
    assert broker.published == ["updated"]
