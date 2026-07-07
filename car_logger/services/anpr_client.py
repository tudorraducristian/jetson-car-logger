"""Plate Recognizer HTTP client.

Retry policy (student's decisions, Stage 4):
- timeout per request: 5s (tolerates home Wi-Fi jitter; the caller is a
  background worker, so a slow call never blocks the detection pipeline)
- 5xx / timeout / network error: retry up to 2 times with exponential
  backoff (0.5s, 1s) — server-side trouble is usually transient
- 429: NO retry — it means *we* ran out of quota; retrying burns more of it
- other 4xx: NO retry — the request itself is wrong, repeating won't help
"""

import time
from typing import NamedTuple, Optional

import httpx

from car_logger.config import settings


class PlateResult(NamedTuple):
    # status mirrors Event.anpr_status: success | failed | throttled | skipped
    status: str
    plate_text: Optional[str] = None
    confidence: Optional[float] = None


class AnprClient(object):
    def __init__(self, api_url=None, api_key=None, timeout=None,
                 max_retries=None, transport=None):
        self.api_url = api_url if api_url is not None else settings.anpr_api_url
        self.api_key = api_key if api_key is not None else settings.anpr_api_key
        self.timeout = timeout if timeout is not None else settings.anpr_timeout_seconds
        self.max_retries = (max_retries if max_retries is not None
                            else settings.anpr_max_retries)
        # transport is injectable so tests can use httpx.MockTransport
        # instead of a real network.
        self._transport = transport
        # injectable sleep so retry tests don't actually wait
        self._sleep = time.sleep

    def read_plate(self, jpg_bytes):
        """POST one JPEG to Plate Recognizer; return a PlateResult.

        Never raises: every failure mode collapses into a PlateResult status
        so the worker can write it straight to Event.anpr_status."""
        if not self.api_key:
            # No key configured (e.g. fresh install): don't waste a call.
            return PlateResult(status="skipped")

        attempts = 1 + self.max_retries
        for attempt in range(attempts):
            retryable = False
            try:
                with httpx.Client(transport=self._transport,
                                  timeout=self.timeout) as client:
                    response = client.post(
                        self.api_url,
                        headers={"Authorization": "Token " + self.api_key},
                        files={"upload": ("plate.jpg", jpg_bytes,
                                          "image/jpeg")},
                    )
            except httpx.TransportError:
                # covers timeouts and connection errors alike
                retryable = True
            else:
                if response.status_code in (200, 201):
                    return self._parse(response)
                if response.status_code == 429:
                    return PlateResult(status="throttled")
                if response.status_code >= 500:
                    retryable = True
                else:
                    return PlateResult(status="failed")

            if retryable and attempt < attempts - 1:
                # 0.5s, then 1s, then 2s... — doubling gives a struggling
                # server room to recover instead of hammering it.
                self._sleep(0.5 * (2 ** attempt))

        return PlateResult(status="failed")

    def _parse(self, response):
        """Extract the best plate from a 200/201 body.

        Student's decision: an image where the API finds no plate counts as
        "failed" (we keep the crop on disk so we can inspect why)."""
        results = response.json().get("results") or []
        if not results:
            return PlateResult(status="failed")
        best = results[0]
        plate = best.get("plate")
        if not plate:
            return PlateResult(status="failed")
        return PlateResult(status="success",
                           plate_text=plate.upper(),
                           confidence=best.get("score"))
