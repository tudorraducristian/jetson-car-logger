"""ANPR client tests — the network is faked with httpx.MockTransport.

Each test hands the client a `handler(request)` function that plays the role
of the Plate Recognizer server: it can answer 200, 429, 500 or raise a
transport error, and it counts how many times it was called — which is how
we prove the retry policy does exactly what the student decided.
"""

import httpx
import pytest

from car_logger.services.anpr_client import AnprClient, PlateResult


def make_client(handler, max_retries=2, api_key="test-key"):
    client = AnprClient(
        api_url="https://anpr.test/v1/plate-reader/",
        api_key=api_key,
        timeout=1.0,
        max_retries=max_retries,
        transport=httpx.MockTransport(handler),
    )
    # record backoff pauses instead of actually sleeping
    client.sleeps = []
    client._sleep = client.sleeps.append
    return client


def test_200_returns_success_with_uppercased_plate():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={
            "results": [{"plate": "b123xyz", "score": 0.91}],
        })

    result = make_client(handler).read_plate(b"fake-jpg")

    assert result.status == "success"
    assert result.plate_text == "B123XYZ"
    assert result.confidence == 0.91
    assert len(calls) == 1


def test_200_with_no_results_is_failed():
    def handler(request):
        return httpx.Response(200, json={"results": []})

    result = make_client(handler).read_plate(b"fake-jpg")

    assert result.status == "failed"
    assert result.plate_text is None


def test_timeout_retries_then_fails():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("too slow", request=request)

    client = make_client(handler, max_retries=2)
    result = client.read_plate(b"fake-jpg")

    assert result.status == "failed"
    # 1 initial attempt + 2 retries
    assert len(calls) == 3
    # exponential backoff: pauses double
    assert client.sleeps == [0.5, 1.0]


def test_429_fails_immediately_as_throttled():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"detail": "quota exceeded"})

    client = make_client(handler)
    result = client.read_plate(b"fake-jpg")

    assert result.status == "throttled"
    # no retry, no backoff — retrying would burn even more quota
    assert len(calls) == 1
    assert client.sleeps == []


def test_500_is_retried_and_can_recover():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 2:
            return httpx.Response(500)
        return httpx.Response(201, json={
            "results": [{"plate": "cj10abc", "score": 0.88}],
        })

    result = make_client(handler).read_plate(b"fake-jpg")

    assert result.status == "success"
    assert result.plate_text == "CJ10ABC"
    assert len(calls) == 2


def test_500_every_time_exhausts_retries():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    result = make_client(handler, max_retries=2).read_plate(b"fake-jpg")

    assert result.status == "failed"
    assert len(calls) == 3


def test_network_error_fails_gracefully():
    def handler(request):
        raise httpx.ConnectError("no route to host", request=request)

    # must not raise — the pipeline worker can't afford an unhandled crash
    result = make_client(handler).read_plate(b"fake-jpg")

    assert result.status == "failed"


def test_other_4xx_fails_without_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"detail": "bad token"})

    result = make_client(handler).read_plate(b"fake-jpg")

    assert result.status == "failed"
    assert len(calls) == 1


def test_missing_api_key_skips_without_calling_api():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": []})

    result = make_client(handler, api_key="").read_plate(b"fake-jpg")

    assert result.status == "skipped"
    assert calls == []
