from car_logger import repositories, schemas


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


# --- Stage 4: ANPR outcome + vehicles ---------------------------------------


def test_anpr_success_creates_and_links_vehicle(db_session):
    event = repositories.create_event(db_session, _make())
    updated = repositories.update_event_anpr(
        db_session, event.id, status="success",
        plate_text="B123XYZ", confidence=0.9,
        image_path="data/plates/1.jpg",
    )
    assert updated.anpr_status == "success"
    assert updated.plate_text == "B123XYZ"
    assert updated.image_path == "data/plates/1.jpg"
    assert updated.vehicle_id is not None
    assert updated.vehicle.plate_text == "B123XYZ"
    assert updated.vehicle.total_sightings == 1


def test_second_sighting_reuses_vehicle_and_bumps_count(db_session):
    first = repositories.create_event(db_session, _make())
    second = repositories.create_event(db_session, _make())
    repositories.update_event_anpr(db_session, first.id, status="success",
                                   plate_text="B123XYZ", confidence=0.9)
    updated = repositories.update_event_anpr(db_session, second.id,
                                             status="success",
                                             plate_text="B123XYZ",
                                             confidence=0.8)
    vehicles = repositories.list_vehicles(db_session)
    assert len(vehicles) == 1  # same plate → same vehicle, not a duplicate
    assert vehicles[0].total_sightings == 2
    assert updated.vehicle_id == vehicles[0].id


def test_anpr_failure_saves_status_but_no_vehicle(db_session):
    event = repositories.create_event(db_session, _make())
    updated = repositories.update_event_anpr(
        db_session, event.id, status="failed",
        image_path="data/plates/9.jpg",
    )
    assert updated.anpr_status == "failed"
    assert updated.plate_text is None
    assert updated.vehicle_id is None
    assert repositories.list_vehicles(db_session) == []


def test_update_event_anpr_missing_id_returns_none(db_session):
    assert repositories.update_event_anpr(db_session, 999,
                                          status="failed") is None


def test_get_stats_counts(db_session):
    a = repositories.create_event(db_session, _make())
    repositories.create_event(db_session, _make())
    repositories.update_event_anpr(db_session, a.id, status="success",
                                   plate_text="B123XYZ", confidence=0.9)
    stats = repositories.get_stats(db_session)
    assert stats["total_events"] == 2
    assert stats["events_last_24h"] == 2
    assert stats["total_vehicles"] == 1
    assert stats["plates_read"] == 1
