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
