"""Dashboard partials rendered through the real app (TestClient + in-memory
DB). These are HTML fragments, so we assert on markers in the text."""


def _create_event(client, plate=None):
    payload = {"anpr_status": "pending"}
    if plate:
        payload["plate_text"] = plate
        payload["anpr_status"] = "success"
    response = client.post("/api/events", json=payload)
    assert response.status_code == 200
    return response.json()["id"]


def test_events_feed_empty_state(client):
    response = client.get("/partials/events-feed")
    assert response.status_code == 200
    assert "Niciun eveniment" in response.text


def test_events_feed_shows_created_event(client):
    _create_event(client, plate="B123XYZ")
    response = client.get("/partials/events-feed")
    assert response.status_code == 200
    assert "B123XYZ" in response.text


def test_vehicles_list_empty_state(client):
    response = client.get("/partials/vehicles-list")
    assert response.status_code == 200
    assert "Nicio pl" in response.text  # "Nicio plăcuță..."


def test_stats_partial_renders_counters(client):
    _create_event(client)
    response = client.get("/partials/stats")
    assert response.status_code == 200
    assert "Evenimente" in response.text
    assert "pipeline" in response.text


def test_event_detail_renders_fields(client):
    event_id = _create_event(client, plate="CJ10ABC")
    response = client.get("/partials/event/%d" % event_id)
    assert response.status_code == 200
    assert "CJ10ABC" in response.text
    assert "eveniment #%d" % event_id in response.text


def test_event_detail_missing_returns_404(client):
    assert client.get("/partials/event/9999").status_code == 404


def test_event_detail_empty_placeholder(client):
    response = client.get("/partials/event-detail")
    assert response.status_code == 200
    assert "Alege" in response.text


def test_fresh_event_row_is_marked_new(client):
    # created "now" -> inside the freshness window -> animated entrance
    client.post("/api/events", json={"plate_text": "NEW111"})
    resp = client.get("/partials/events-feed")
    assert resp.status_code == 200
    assert "row-new" in resp.text


def test_old_event_row_is_not_marked_new(client, db_session):
    from datetime import datetime, timedelta

    from car_logger.models import Event

    db_session.add(Event(timestamp=datetime.utcnow() - timedelta(minutes=5),
                         anpr_status="pending"))
    db_session.commit()
    resp = client.get("/partials/events-feed")
    assert resp.status_code == 200
    assert "row-new" not in resp.text


def _seed_read_and_no_plate_events(db_session):
    from car_logger import repositories, schemas
    read = repositories.create_event(
        db_session, schemas.EventCreate(anpr_status="pending"))
    repositories.update_event_anpr(
        db_session, read.id, "CJ45ARL", 0.97, "success", None)
    unread = repositories.create_event(
        db_session, schemas.EventCreate(anpr_status="pending"))
    repositories.update_event_anpr(
        db_session, unread.id, None, None, "no_plate", None)


def test_feed_default_shows_only_plate_read_events(client, db_session):
    _seed_read_and_no_plate_events(db_session)
    response = client.get("/partials/events-feed")
    assert response.status_code == 200
    assert "CJ45ARL" in response.text
    assert "fără plăcuță" not in response.text


def test_feed_filter_all_shows_everything_with_the_new_badge(client,
                                                             db_session):
    _seed_read_and_no_plate_events(db_session)
    response = client.get("/partials/events-feed?filter=all")
    assert "CJ45ARL" in response.text
    assert "fără plăcuță" in response.text
