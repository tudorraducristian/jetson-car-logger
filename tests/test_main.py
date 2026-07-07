from fastapi.testclient import TestClient

from car_logger.main import app

client = TestClient(app)


def test_root_serves_the_dashboard_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Car Logger" in response.text


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
