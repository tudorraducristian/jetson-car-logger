from fastapi.testclient import TestClient

from car_logger.main import app

client = TestClient(app)


def test_root_returns_greeting_and_version():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Car Logger is running"
    assert body["version"] == "0.2.0"


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
