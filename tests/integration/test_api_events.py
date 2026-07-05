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
