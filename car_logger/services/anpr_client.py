"""Plate Recognizer HTTP client with a deliberate retry policy.

STUDENT DECISIONS (confirmed 2026-07-07):
- timeout      = 5.0 : per-request timeout in seconds.
- max_retries  = 2   : retry count for 5xx / timeouts, exponential backoff.
- success = exactly 200 or 201 (the real API answers 201 Created; we list
  the documented codes explicitly instead of accepting any 2xx).
- 429 -> no retry, status='throttled' (respect the published rate limit).
- 4xx -> no retry, status='failed' (our request is wrong; retrying repeats it).
"""

import time
from collections import namedtuple

import httpx

PlateResult = namedtuple(
    "PlateResult", ["plate_text", "confidence", "status"]
)  # status: success | failed | throttled


class AnprClient(object):
    def __init__(self, api_url, api_key, client=None, timeout=5.0,
                 max_retries=2):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client if client is not None else httpx.Client(
            timeout=timeout
        )

    def read_plate(self, image_bytes):
        """POST the image to Plate Recognizer; return a PlateResult.

        Never raises for expected network/API failures — the pipeline must keep
        running whether or not the plate is read."""
        headers = {"Authorization": "Token " + self.api_key}
        attempt = 0
        while True:
            try:
                resp = self._client.post(
                    self.api_url,
                    files={"upload": image_bytes},
                    headers=headers,
                    timeout=self.timeout,
                )
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                return PlateResult(None, None, "failed")

            if resp.status_code in (200, 201):
                return self._parse(resp.json())
            if resp.status_code == 429:
                return PlateResult(None, None, "throttled")
            if 500 <= resp.status_code < 600 and attempt < self.max_retries:
                attempt += 1
                time.sleep(0.1 * (2 ** attempt))
                continue
            return PlateResult(None, None, "failed")

    def _parse(self, payload):
        results = payload.get("results", [])
        if not results:
            return PlateResult(None, None, "failed")
        best = results[0]
        return PlateResult(best.get("plate"), best.get("score"), "success")
