def test_create_event_returns_id(client):
    resp = client.post("/api/events", json={"plate_text": "B123XYZ"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] > 0
    assert body["plate_text"] == "B123XYZ"
    assert body["anpr_status"] == "pending"


def test_get_event_roundtrip(client):
    created = client.post("/api/events", json={"plate_text": "CJ01AAA"}).json()
    resp = client.get("/api/events/" + str(created["id"]))
    assert resp.status_code == 200
    assert resp.json()["plate_text"] == "CJ01AAA"


def test_get_missing_event_is_404(client):
    resp = client.get("/api/events/9999")
    assert resp.status_code == 404


def test_list_events_empty(client):
    resp = client.get("/api/events")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_events_plate_filter(client):
    client.post("/api/events", json={"plate_text": "B123XYZ"})
    client.post("/api/events", json={"plate_text": "CJ99ABC"})
    resp = client.get("/api/events", params={"plate": "123"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_list_events_limit_over_100_rejected(client):
    resp = client.get("/api/events", params={"limit": 500})
    assert resp.status_code == 422  # Query(le=100) enforces the ceiling


def test_delete_event(client):
    created = client.post("/api/events", json={"plate_text": "DEL123"}).json()
    resp = client.delete("/api/events/" + str(created["id"]))
    assert resp.status_code == 204
    assert client.get("/api/events/" + str(created["id"])).status_code == 404


def test_delete_missing_event_is_404(client):
    assert client.delete("/api/events/9999").status_code == 404


def test_event_region_roundtrip(client):
    created = client.post(
        "/api/events", json={"plate_text": "B123ABC", "region": "ro"}).json()
    assert created["region"] == "ro"
    resp = client.get("/api/events/" + str(created["id"]))
    assert resp.json()["region"] == "ro"


def test_delete_last_event_removes_vehicle(client, db_session):
    from datetime import datetime

    from car_logger.models import Vehicle

    vehicle = Vehicle(plate_text="GONE123", first_seen_at=datetime.utcnow(),
                      last_seen_at=datetime.utcnow(), total_sightings=1)
    db_session.add(vehicle)
    db_session.commit()
    created = client.post("/api/events", json={
        "plate_text": "GONE123", "vehicle_id": vehicle.id}).json()
    assert client.delete("/api/events/" + str(created["id"])).status_code == 204
    assert db_session.query(Vehicle).count() == 0


def test_delete_one_of_two_recomputes_sightings(client, db_session):
    from datetime import datetime

    from car_logger.models import Vehicle

    vehicle = Vehicle(plate_text="STAY123", first_seen_at=datetime.utcnow(),
                      last_seen_at=datetime.utcnow(), total_sightings=2)
    db_session.add(vehicle)
    db_session.commit()
    first = client.post("/api/events", json={
        "plate_text": "STAY123", "vehicle_id": vehicle.id}).json()
    client.post("/api/events", json={
        "plate_text": "STAY123", "vehicle_id": vehicle.id})
    assert client.delete("/api/events/" + str(first["id"])).status_code == 204
    db_session.refresh(vehicle)
    assert vehicle.total_sightings == 1
