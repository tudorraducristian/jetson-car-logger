from car_logger import repositories, schemas
from car_logger.models import Vehicle


def _make(plate=None, status="pending"):
    return schemas.EventCreate(plate_text=plate, anpr_status=status)


def test_create_and_get_event(db_session):
    created = repositories.create_event(db_session, _make(plate="B123XYZ"))
    assert created.id is not None
    fetched = repositories.get_event(db_session, created.id)
    assert fetched.plate_text == "B123XYZ"


def test_get_missing_event_returns_none(db_session):
    assert repositories.get_event(db_session, 999) is None


def test_list_events_empty(db_session):
    assert repositories.list_events(db_session) == []


def test_list_events_newest_first(db_session):
    a = repositories.create_event(db_session, _make(plate="AAA"))
    b = repositories.create_event(db_session, _make(plate="BBB"))
    rows = repositories.list_events(db_session)
    assert [r.id for r in rows] == [b.id, a.id]


def test_list_events_plate_filter_is_partial(db_session):
    repositories.create_event(db_session, _make(plate="B123XYZ"))
    repositories.create_event(db_session, _make(plate="CJ99ABC"))
    rows = repositories.list_events(db_session, plate_text="123")
    assert len(rows) == 1
    assert rows[0].plate_text == "B123XYZ"


def test_list_events_caps_limit(db_session):
    for i in range(5):
        repositories.create_event(db_session, _make(plate="P" + str(i)))
    rows = repositories.list_events(db_session, limit=1000)
    assert len(rows) == 5  # all 5 returned, but limit was capped, not errored
    assert repositories.MAX_LIST_LIMIT == 100


def test_upsert_vehicle_creates_then_bumps(db_session):
    v1 = repositories.upsert_vehicle_for_plate(db_session, "B123XYZ")
    assert v1.total_sightings == 1
    v2 = repositories.upsert_vehicle_for_plate(db_session, "B123XYZ")
    assert v2.id == v1.id
    assert v2.total_sightings == 2
    assert db_session.query(Vehicle).count() == 1


def test_update_event_anpr_sets_plate_and_status(db_session):
    ev = repositories.create_event(db_session, _make())
    updated = repositories.update_event_anpr(
        db_session, ev.id, plate_text="B123XYZ", confidence=0.9,
        status="success", image_path="data/plates/1.jpg",
    )
    assert updated.plate_text == "B123XYZ"
    assert updated.anpr_status == "success"
    assert updated.image_path == "data/plates/1.jpg"


def test_event_stats_counts(db_session):
    repositories.create_event(db_session, _make(plate=None))
    ev = repositories.create_event(db_session, _make(plate=None))
    repositories.update_event_anpr(db_session, ev.id, "B1", 0.9, "success",
                                   "p.jpg")
    repositories.upsert_vehicle_for_plate(db_session, "B1")
    stats = repositories.event_stats(db_session)
    assert stats["total_events"] == 2
    assert stats["plates_read"] == 1
    assert stats["unique_vehicles"] == 1
