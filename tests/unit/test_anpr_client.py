import httpx
import pytest

from car_logger.services.anpr_client import AnprClient


def _client_returning(responses):
    """Build an AnprClient whose httpx.Client replays the given responses in
    order (each is (status_code, json))."""
    calls = {"n": 0}

    def handler(request):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        status, payload = responses[i]
        return httpx.Response(status, json=payload)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    ac = AnprClient("http://anpr.test", "tok", client=http, max_retries=2)
    return ac, calls


def test_200_returns_plate(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, calls = _client_returning([
        (200, {"results": [{"plate": "b123xyz", "score": 0.92}]}),
    ])
    result = ac.read_plate(b"jpegbytes")
    assert result.status == "success"
    assert result.plate_text == "b123xyz"
    assert abs(result.confidence - 0.92) < 1e-9
    assert calls["n"] == 1


def test_201_created_returns_plate(monkeypatch):
    # The real Plate Recognizer answers 201 Created, not 200 — discovered
    # live in stage 4 task 7, after six green tests that all mocked 200.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, calls = _client_returning([
        (201, {"results": [{"plate": "mmm8748", "score": 1.0}]}),
    ])
    result = ac.read_plate(b"jpegbytes")
    assert result.status == "success"
    assert result.plate_text == "mmm8748"
    assert calls["n"] == 1


def test_200_no_results_is_failed(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, _ = _client_returning([(200, {"results": []})])
    assert ac.read_plate(b"x").status == "failed"


def test_429_throttled_no_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, calls = _client_returning([(429, {})])
    result = ac.read_plate(b"x")
    assert result.status == "throttled"
    assert calls["n"] == 1  # did NOT retry


def test_500_retries_then_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, calls = _client_returning([(500, {}), (500, {}), (500, {})])
    result = ac.read_plate(b"x")
    assert result.status == "failed"
    assert calls["n"] == 3  # initial + 2 retries


def test_500_then_200_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, calls = _client_returning([
        (500, {}),
        (200, {"results": [{"plate": "cj01aaa", "score": 0.8}]}),
    ])
    result = ac.read_plate(b"x")
    assert result.status == "success"
    assert result.plate_text == "cj01aaa"
    assert calls["n"] == 2


def test_timeout_retries_then_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def handler(request):
        raise httpx.TimeoutException("slow", request=request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    ac = AnprClient("http://anpr.test", "tok", client=http, max_retries=2)
    assert ac.read_plate(b"x").status == "failed"
