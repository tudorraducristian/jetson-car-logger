from fastapi.testclient import TestClient

from car_logger.main import app


class FakeCam(object):
    def __init__(self, healthy):
        self._healthy = healthy
    def is_healthy(self):
        return self._healthy


class FakePipeline(object):
    last_fps = 12.0
    frames_processed = 5
    last_event_at = None


def test_camera_ok_reflects_health():
    client = TestClient(app)
    app.state.pipeline = FakePipeline()

    app.state.camera = FakeCam(healthy=True)
    assert client.get("/api/status").json()["camera_ok"] is True

    app.state.camera = FakeCam(healthy=False)
    assert client.get("/api/status").json()["camera_ok"] is False
