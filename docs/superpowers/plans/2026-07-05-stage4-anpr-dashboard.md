# Stage 4 — ANPR + Dashboard (car_logger) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read plates via the Plate Recognizer API without slowing the detection pipeline, store the cropped plate image, upsert a `Vehicle` per plate, and serve a live-refreshing web dashboard (events feed + vehicles + stats) built with Jinja2 + Tailwind (CDN) + htmx (CDN, 2s polling).

**Architecture:** ANPR is offloaded to a dedicated `AnprWorker` thread fed by a bounded queue, so the pipeline never blocks on the 300–800ms network call (it drops work under load rather than stall). The pipeline, on a confirmed track, crops the frame to the bbox, persists a `pending` event to get its id, then submits `(event_id, crop_bytes)` to the ANPR worker. When a result returns, a callback saves the crop to `data/plates/<id>.jpg`, updates the event, and upserts the `Vehicle`. The dashboard is server-rendered: `GET /` returns the shell, htmx polls partial routes every 2s to refresh the three panels.

**Tech Stack:** httpx 0.22.0 (sync `Client` + `MockTransport` for tests), `queue.Queue` + `threading`, cv2 `imencode` for cropping, Jinja2 3.0.3 (`Jinja2Templates`), Tailwind Play CDN, htmx CDN, the Stage 2 repository, pytest.

## Global Constraints

- **Python 3.6.9 target.** No 3.7+ syntax.
- **ANPR must not block the pipeline.** The network call runs in a separate worker; the pipeline submits and moves on. Verify the non-blocking claim explicitly.
- **No secrets in git.** The Plate Recognizer token lives in `.env` as `ANPR_API_KEY`. `.gitignore` must cover `.env` and `data/`.
- **Graceful offline behaviour.** Internet down → events still created, `anpr_status="failed"`, no crash. Reconnect → new events get plates.
- **Retry policy is the student's decision** (defaults documented below).
- **Tailwind + htmx via CDN only.** No npm/build step.
- **Sync everywhere** except where FastAPI needs otherwise (nothing async this stage).
- **Split execution:** **[LAPTOP — Claude]** writes/commits/pushes; **[JETSON — student]** pulls and runs. Paste output at each **CHECKPOINT**.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Student amendments (2026-07-07, before re-execution)

A first pass was executed off PLAN.md instead of this plan and reverted
(`8691fbd`). Decisions the student locked in for this re-run; where they
conflict with a task below, the amendment wins:

- **Retry policy confirmed as documented:** timeout 5.0s, max_retries 2
  (exponential backoff), 429 → no retry (`throttled`), 4xx → no retry.
- **Crops are saved for every ANPR outcome, not only success** — a failed
  read's image is the debugging evidence (amends Task 5 `_make_on_result`).
- **Retention cleanup lands in this stage, not Stage 5:** delete crops
  older than 30 days at startup (Stage 5's plan doesn't actually contain
  it). Small addition in Task 5.
- **Missing API key behaves as the plan says** (client calls and fails) —
  no special-casing.
- **Task 6 templates:** reuse the richer ui-ux-pro-max set from the first
  pass (this plan marks its templates as "the floor, not the ceiling");
  context keys must match `event_stats`.
- Commit trailer uses the current model:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File structure (what this stage creates)

- `car_logger/services/anpr_client.py` — `AnprClient` (httpx + retry policy; unit-tested with `MockTransport`)
- `car_logger/services/anpr_worker.py` — `AnprWorker` (thread + bounded queue)
- `car_logger/services/cropping.py` — `crop_to_jpeg(frame_bgr, box) -> bytes`
- `car_logger/repositories.py` — add `update_event_anpr(...)` and `upsert_vehicle_for_plate(...)`, `list_vehicles(...)`, `count stats`
- `car_logger/api/routes_dashboard.py` — `GET /`, `GET /partials/events-feed`, `/partials/vehicles-list`, `/partials/stats`
- `car_logger/templates/base.html`, `dashboard.html`, `partials/events_feed.html`, `partials/vehicles_list.html`, `partials/stats.html`
- `car_logger/services/pipeline.py` — extend `on_event` to pass the crop; wire ANPR submit
- `car_logger/main.py` — build ANPR client + worker on startup, pass to pipeline
- `tests/unit/test_anpr_client.py`, `tests/unit/test_cropping.py`, `tests/integration/test_dashboard.py`

**Prereq (student, manual):** sign up at platerecognizer.com (free tier, 2500/month), put the token in `.env` as `ANPR_API_KEY=...`, and smoke-test with `curl` + a sample image before wiring code.

---

### Task 1: ANPR client with retry policy (test-first)

**Files:**
- Create: `car_logger/services/anpr_client.py`
- Test: `tests/unit/test_anpr_client.py`

**Interfaces:**
- Produces:
  - `PlateResult` namedtuple `(plate_text: Optional[str], confidence: Optional[float], status: str)` where `status ∈ {"success", "failed", "throttled"}`.
  - `AnprClient(api_url, api_key, client=None, timeout=5.0, max_retries=2)` with `read_plate(image_bytes: bytes) -> PlateResult`. `client` is an injectable `httpx.Client` (tests pass one backed by `MockTransport`).

> **STUDENT DECISION — confirm or tune, justify each:**
> - `timeout = 5.0s` per request.
> - `max_retries = 2` for 5xx (transient server errors), exponential backoff.
> - `429` (rate-limited) → **no retry**, `status="throttled"` (respect the limit).
> - `4xx` (bad request) → **no retry**, `status="failed"`.

- [x] **Step 1: Write the failing tests** **[LAPTOP — Claude]** *(pushed `0682802`)*

`tests/unit/test_anpr_client.py`:
```python
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
```

- [x] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]** *(RED confirmed on the Jetson 2026-07-07: `ModuleNotFoundError`)*

```bash
git add tests/unit/test_anpr_client.py
git commit -m "test(anpr): failing tests for retry policy and parsing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_anpr_client.py -v
```
Expected: FAIL — module doesn't exist.

- [x] **Step 3: Implement the ANPR client** **[LAPTOP — Claude]** *(pushed `31ef73e`)*

`car_logger/services/anpr_client.py`:
```python
"""Plate Recognizer HTTP client with a deliberate retry policy.

STUDENT DECISIONS (defaults; justify each):
- timeout      = 5.0 : per-request timeout in seconds.
- max_retries  = 2   : retry count for 5xx / timeouts, exponential backoff.
- 429 -> no retry, status='throttled' (respect the published rate limit).
- 4xx -> no retry, status='failed'.
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

            if resp.status_code == 200:
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
```

- [x] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]** *(student confirmed 2026-07-07: `6 passed`)*

```bash
git add car_logger/services/anpr_client.py
git commit -m "feat(anpr): Plate Recognizer client with retry/throttle policy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_anpr_client.py -v
```
Expected: `6 passed`.

**CHECKPOINT:** paste the pytest output before Task 2.

---

### Task 2: Frame cropping helper (test-first)

**Files:**
- Create: `car_logger/services/cropping.py`
- Test: `tests/unit/test_cropping.py`

**Interfaces:**
- Produces: `crop_to_jpeg(frame_bgr, box) -> bytes` — crops `(x1,y1,x2,y2)` (clamped to frame bounds) and returns JPEG bytes. Consumed by the pipeline (Task 5).

- [x] **Step 1: Write the failing tests** **[LAPTOP — Claude]** *(pushed `2c1d6f3`)*

`tests/unit/test_cropping.py`:
```python
import numpy as np

from car_logger.services.cropping import crop_to_jpeg


def _frame():
    # a 100x100 BGR image
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_crop_returns_jpeg_bytes():
    data = crop_to_jpeg(_frame(), (10, 10, 60, 60))
    assert isinstance(data, bytes)
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_box_clamped_to_frame_bounds():
    # box extends past the 100x100 frame; must not raise and must return bytes
    data = crop_to_jpeg(_frame(), (-20, -20, 500, 500))
    assert isinstance(data, bytes) and len(data) > 0
```

- [x] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]** *(RED confirmed 2026-07-07: `ModuleNotFoundError`)*

```bash
git add tests/unit/test_cropping.py
git commit -m "test(cropping): failing tests for bbox crop to jpeg

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**: `git pull && python3 -m pytest tests/unit/test_cropping.py -v` → FAIL.

- [x] **Step 3: Implement cropping** **[LAPTOP — Claude]** *(pushed `0ccb99b`)*

`car_logger/services/cropping.py`:
```python
"""Crop a detection bbox out of a frame and JPEG-encode it for ANPR/storage."""

import cv2


def crop_to_jpeg(frame_bgr, box):
    """Return JPEG bytes of the box region, clamped to the frame bounds."""
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    crop = frame_bgr[y1:y2, x1:x2]
    ok, buffer = cv2.imencode(".jpg", crop)
    if not ok:
        return b""
    return buffer.tobytes()
```

- [x] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]** *(student confirmed 2026-07-07: `2 passed`)*

```bash
git add car_logger/services/cropping.py
git commit -m "feat(cropping): clamp bbox and JPEG-encode the crop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**: `git pull && python3 -m pytest tests/unit/test_cropping.py -v` → `2 passed`.

**CHECKPOINT:** paste output before Task 3.

---

### Task 3: Repository additions (event update + vehicle upsert + stats, test-first)

**Files:**
- Modify: `car_logger/repositories.py`
- Test: append to `tests/unit/test_repositories.py`

**Interfaces:**
- Produces:
  - `update_event_anpr(db, event_id, plate_text, confidence, status, image_path, vehicle_id=None) -> Optional[Event]`
  - `upsert_vehicle_for_plate(db, plate_text) -> Vehicle` (create-or-bump `total_sightings` + `last_seen_at`)
  - `list_vehicles(db, skip=0, limit=50) -> List[Vehicle]` (newest `last_seen_at` first)
  - `event_stats(db) -> dict` → `{"total_events": int, "plates_read": int, "unique_vehicles": int}`

- [ ] **Step 1: Write the failing tests** **[LAPTOP — Claude]**

Append to `tests/unit/test_repositories.py`:
```python
from car_logger.models import Vehicle


def test_upsert_vehicle_creates_then_bumps(db_session):
    v1 = repositories.upsert_vehicle_for_plate(db_session, "B123XYZ")
    assert v1.total_sightings == 1
    v2 = repositories.upsert_vehicle_for_plate(db_session, "B123XYZ")
    assert v2.id == v1.id
    assert v2.total_sightings == 2
    assert db_session.query(Vehicle).count() == 1


def test_update_event_anpr_sets_plate_and_status(db_session):
    ev = repositories.create_event(db_session, _make())
    updated = repositories.update_event_anpr(
        db_session, ev.id, plate_text="B123XYZ", confidence=0.9,
        status="success", image_path="data/plates/1.jpg",
    )
    assert updated.plate_text == "B123XYZ"
    assert updated.anpr_status == "success"
    assert updated.image_path == "data/plates/1.jpg"


def test_event_stats_counts(db_session):
    repositories.create_event(db_session, _make(plate=None))
    ev = repositories.create_event(db_session, _make(plate=None))
    repositories.update_event_anpr(db_session, ev.id, "B1", 0.9, "success",
                                   "p.jpg")
    repositories.upsert_vehicle_for_plate(db_session, "B1")
    stats = repositories.event_stats(db_session)
    assert stats["total_events"] == 2
    assert stats["plates_read"] == 1
    assert stats["unique_vehicles"] == 1
```

- [ ] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add tests/unit/test_repositories.py
git commit -m "test(repositories): failing tests for vehicle upsert + stats

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**: `git pull && python3 -m pytest tests/unit/test_repositories.py -v` → the 3 new tests FAIL.

- [ ] **Step 3: Implement the repository additions** **[LAPTOP — Claude]**

Append to `car_logger/repositories.py` (add `from datetime import datetime` and `from car_logger.models import Vehicle` to the imports):
```python
def update_event_anpr(db, event_id, plate_text, confidence, status,
                      image_path, vehicle_id=None):
    """Fill in ANPR results on an existing event. Returns the event or None."""
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        return None
    event.plate_text = plate_text
    event.plate_confidence = confidence
    event.anpr_status = status
    event.image_path = image_path
    if vehicle_id is not None:
        event.vehicle_id = vehicle_id
    db.commit()
    db.refresh(event)
    return event


def upsert_vehicle_for_plate(db, plate_text):
    """Create the vehicle for this plate or bump its sighting counters."""
    now = datetime.utcnow()
    vehicle = db.query(Vehicle).filter(
        Vehicle.plate_text == plate_text
    ).first()
    if vehicle is None:
        vehicle = Vehicle(plate_text=plate_text, first_seen_at=now,
                          last_seen_at=now, total_sightings=1)
        db.add(vehicle)
    else:
        vehicle.last_seen_at = now
        vehicle.total_sightings += 1
    db.commit()
    db.refresh(vehicle)
    return vehicle


def list_vehicles(db, skip=0, limit=50):
    capped = min(limit, MAX_LIST_LIMIT)
    return (db.query(Vehicle)
              .order_by(Vehicle.last_seen_at.desc())
              .offset(skip).limit(capped).all())


def event_stats(db):
    total = db.query(Event).count()
    plates = db.query(Event).filter(Event.plate_text.isnot(None)).count()
    vehicles = db.query(Vehicle).count()
    return {
        "total_events": total,
        "plates_read": plates,
        "unique_vehicles": vehicles,
    }
```

- [ ] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/repositories.py
git commit -m "feat(repositories): event ANPR update, vehicle upsert, stats

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**: `git pull && python3 -m pytest tests/unit/test_repositories.py -v` → all green.

**CHECKPOINT:** paste output before Task 4.

---

### Task 4: ANPR worker thread (bounded queue)

**Files:**
- Create: `car_logger/services/anpr_worker.py`

**Interfaces:**
- Consumes: an `AnprClient`, an `on_result(event_id, plate_result, crop_bytes)` callback.
- Produces: `AnprWorker(anpr_client, on_result, queue_maxsize=32)` with `start()`, `submit(event_id, crop_bytes) -> bool` (False if dropped because full), `stop()`. The worker never blocks the caller: `submit` uses `put_nowait` and drops under load.

- [ ] **Step 1: Write the ANPR worker** **[LAPTOP — Claude]**

`car_logger/services/anpr_worker.py`:
```python
"""Background ANPR worker: decouples the slow network call from the pipeline.

The pipeline calls submit() and returns immediately. This worker thread pulls
jobs off a bounded queue, calls the ANPR client, and hands the result to a
callback (which persists it). Under load the queue fills and submit() drops the
job rather than block the pipeline — a dropped plate read is acceptable; a
stalled pipeline is not."""

import queue
import threading


class AnprWorker(object):
    def __init__(self, anpr_client, on_result, queue_maxsize=32):
        self._client = anpr_client
        self._on_result = on_result
        self._queue = queue.Queue(maxsize=queue_maxsize)
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, event_id, crop_bytes):
        """Enqueue a job; return False if dropped because the queue is full."""
        try:
            self._queue.put_nowait((event_id, crop_bytes))
            return True
        except queue.Full:
            return False

    def _loop(self):
        while self._running:
            try:
                event_id, crop_bytes = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self._client.read_plate(crop_bytes)
                self._on_result(event_id, result, crop_bytes)
            finally:
                self._queue.task_done()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/services/anpr_worker.py
git commit -m "feat(anpr): bounded-queue worker thread, drops under load

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Smoke-test the worker in isolation** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "
import time
from car_logger.services.anpr_worker import AnprWorker
class FakeClient:
    def read_plate(self, b): return ('FAKE', 0.5, 'success')
got = []
w = AnprWorker(FakeClient(), lambda eid, r, c: got.append((eid, r)))
w.start(); w.submit(1, b'x'); time.sleep(0.5); w.stop()
print('result:', got)
"
```
Expected: `result: [(1, ('FAKE', 0.5, 'success'))]`.

**CHECKPOINT:** paste output before Task 5.

---

### Task 5: Wire ANPR into the pipeline (non-blocking)

**Files:**
- Modify: `car_logger/services/pipeline.py`
- Modify: `car_logger/main.py`

**Interfaces:**
- Pipeline's `on_event` now receives the crop and event id path: pipeline persists a `pending` event, crops the frame, and submits to the ANPR worker. `main.py` builds `AnprClient` + `AnprWorker`, whose `on_result` saves the crop and updates the event + vehicle.

- [ ] **Step 1: Extend the pipeline to crop + submit** **[LAPTOP — Claude]**

Modify `car_logger/services/pipeline.py`. Change the constructor to accept an optional `on_confirmed(track, frame)` callback that fully owns persistence+ANPR (keeps the pipeline itself dependency-light):
```python
"""Pipeline worker: camera -> detector -> tracker -> on_confirmed callback.

on_confirmed(track, frame) is called once per newly-confirmed track, with the
frame it was confirmed on (so the caller can crop the plate). The callback owns
persistence and ANPR submission; the pipeline stays CV-only."""

import threading
import time


class PipelineWorker(object):
    def __init__(self, camera, detector, tracker, on_confirmed, target_fps=15):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.on_confirmed = on_confirmed
        self._min_interval = 1.0 / float(target_fps)
        self._running = False
        self._thread = None
        self.last_fps = 0.0
        self.last_event_at = None
        self.frames_processed = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            t0 = time.time()
            frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            detections = self.detector.detect(frame)
            boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
            self.tracker.update(boxes)
            for track in self.tracker.new_confirmed_tracks():
                self.last_event_at = time.time()
                self.on_confirmed(track, frame)
            self.frames_processed += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                self.last_fps = 1.0 / elapsed
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
```

> This replaces the Stage 3 `on_event(event_dict)` shape. The Stage 3 live-run snippets that called `PipelineWorker(..., emitted.append, ...)` were verification scaffolding, not shipped code — nothing else imports the old signature.

- [ ] **Step 2: Update `main.py` to build the full ANPR path** **[LAPTOP — Claude]**

`car_logger/main.py` — replace the `_persist_event`/`_startup` section from Stage 3 with:
```python
"""Car Logger API entrypoint - the app object everything else attaches to."""

import json
import os

from fastapi import FastAPI

from car_logger.api.routes_events import router as events_router
from car_logger.api.routes_status import router as status_router
from car_logger.api.routes_dashboard import router as dashboard_router
from car_logger.config import settings
from car_logger.database import SessionLocal
from car_logger import repositories, schemas

APP_VERSION = "0.4.0"

PLATES_DIR = "data/plates"

app = FastAPI(title="Car Logger", version=APP_VERSION)

app.include_router(events_router)
app.include_router(status_router)
app.include_router(dashboard_router)


def _make_on_result():
    """Build the ANPR result callback: save crop, update event, upsert vehicle."""
    def on_result(event_id, plate_result, crop_bytes):
        db = SessionLocal()
        try:
            image_path = None
            if plate_result.status == "success" and plate_result.plate_text:
                os.makedirs(PLATES_DIR, exist_ok=True)
                image_path = os.path.join(PLATES_DIR, str(event_id) + ".jpg")
                with open(image_path, "wb") as fh:
                    fh.write(crop_bytes)
                vehicle = repositories.upsert_vehicle_for_plate(
                    db, plate_result.plate_text
                )
                repositories.update_event_anpr(
                    db, event_id, plate_result.plate_text,
                    plate_result.confidence, "success", image_path, vehicle.id,
                )
            else:
                repositories.update_event_anpr(
                    db, event_id, None, None, plate_result.status, None,
                )
        finally:
            db.close()
    return on_result


@app.on_event("startup")
def _startup():
    if not settings.enable_pipeline:
        return
    from car_logger.services.capture import CameraWorker
    from car_logger.services.detector import Detector
    from car_logger.services.tracker import IoUTracker
    from car_logger.services.pipeline import PipelineWorker
    from car_logger.services.anpr_client import AnprClient
    from car_logger.services.anpr_worker import AnprWorker
    from car_logger.services.cropping import crop_to_jpeg

    camera = CameraWorker(device_index=settings.camera_index)
    camera.start()

    anpr_client = AnprClient(settings.anpr_api_url, settings.anpr_api_key)
    anpr_worker = AnprWorker(anpr_client, _make_on_result())
    anpr_worker.start()

    def on_confirmed(track, frame):
        # 1) persist a pending event to get its id
        db = SessionLocal()
        try:
            event = repositories.create_event(db, schemas.EventCreate(
                bbox_json=json.dumps(list(track.box)),
                track_id=track.track_id,
                anpr_status="pending",
            ))
            event_id = event.id
        finally:
            db.close()
        # 2) crop and hand off to ANPR — pipeline does NOT wait for the network
        crop_bytes = crop_to_jpeg(frame, track.box)
        submitted = anpr_worker.submit(event_id, crop_bytes)
        if not submitted:
            db2 = SessionLocal()
            try:
                repositories.update_event_anpr(
                    db2, event_id, None, None, "skipped", None,
                )
            finally:
                db2.close()

    pipeline = PipelineWorker(
        camera=camera,
        detector=Detector(threshold=settings.detector_threshold),
        tracker=IoUTracker(),
        on_confirmed=on_confirmed,
        target_fps=settings.max_pipeline_fps,
    )
    pipeline.start()
    app.state.camera = camera
    app.state.pipeline = pipeline
    app.state.anpr_worker = anpr_worker


@app.on_event("shutdown")
def _shutdown():
    for name in ("pipeline", "anpr_worker", "camera"):
        worker = getattr(app.state, name, None)
        if worker is not None:
            worker.stop()


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
```

> `GET /` is now served by the dashboard router (Task 6), so the old `root()` handler is removed. `tests/test_main.py`'s root test must move to assert HTML instead — updated in Task 6 Step 4.

- [ ] **Step 3: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/services/pipeline.py car_logger/main.py
git commit -m "feat(pipeline): non-blocking ANPR via crop + worker submit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 4: Verify non-blocking claim** **[JETSON — student]**

Temporarily add `time.sleep(1)` at the top of `AnprClient.read_plate` on the Jetson (edit locally, do not commit), run the Stage 3 Task 5 live snippet adapted to the new `on_confirmed` signature, and confirm `last_fps` stays ≥ 10 despite the 1s fake latency. Then remove the sleep. This proves the pipeline isn't waiting on ANPR. *(Per CLAUDE.md, this experiment is the student's to run and reason about.)*

**CHECKPOINT:** report the observed FPS with and without the fake sleep before Task 6.

---

### Task 6: Dashboard templates + routes

**Files:**
- Create: `car_logger/templates/base.html`, `dashboard.html`, `partials/events_feed.html`, `partials/vehicles_list.html`, `partials/stats.html`
- Create: `car_logger/api/routes_dashboard.py`
- Modify: `tests/test_main.py` (root now returns HTML)
- Test: `tests/integration/test_dashboard.py`

**Interfaces:**
- Consumes: `repositories.list_events`, `list_vehicles`, `event_stats`, `get_db`.
- Produces: `GET /` (full page), `GET /partials/events-feed`, `GET /partials/vehicles-list`, `GET /partials/stats` (HTML fragments htmx swaps in).

> **Design note:** for a polished dark editorial look, invoke the `ui-ux-pro-max` (or `frontend-design`) skill at execution time and let it own the CSS/markup. The templates below are a complete, working baseline so the task has no placeholder — treat them as the floor, not the ceiling.

- [ ] **Step 1: Write the base template** **[LAPTOP — Claude]**

`car_logger/templates/base.html`:
```html
<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Car Logger{% endblock %}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>
</head>
<body class="bg-neutral-950 text-neutral-100 min-h-screen">
  <header class="border-b border-neutral-800 px-6 py-4">
    <h1 class="text-2xl font-semibold tracking-tight">Car Logger</h1>
    <p class="text-neutral-400 text-sm">Live vehicle detection on the LAN</p>
  </header>
  <main class="p-6">{% block content %}{% endblock %}</main>
</body>
</html>
```

- [ ] **Step 2: Write the dashboard + partial templates** **[LAPTOP — Claude]**

`car_logger/templates/dashboard.html`:
```html
{% extends "base.html" %}
{% block content %}
<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <section class="lg:col-span-2">
    <h2 class="text-lg font-medium mb-3">Live events</h2>
    <div id="events-feed"
         hx-get="/partials/events-feed"
         hx-trigger="load, every 2s"
         hx-swap="innerHTML">
      Loading…
    </div>
  </section>
  <aside class="space-y-6">
    <div>
      <h2 class="text-lg font-medium mb-3">Stats</h2>
      <div id="stats" hx-get="/partials/stats"
           hx-trigger="load, every 2s" hx-swap="innerHTML">…</div>
    </div>
    <div>
      <h2 class="text-lg font-medium mb-3">Vehicles</h2>
      <div id="vehicles" hx-get="/partials/vehicles-list"
           hx-trigger="load, every 5s" hx-swap="innerHTML">…</div>
    </div>
  </aside>
</div>
{% endblock %}
```

`car_logger/templates/partials/events_feed.html`:
```html
<ul class="divide-y divide-neutral-800">
  {% for e in events %}
  <li class="py-2 flex items-center justify-between">
    <div>
      <span class="font-mono">{{ e.plate_text or "—" }}</span>
      <span class="text-xs text-neutral-500 ml-2">{{ e.anpr_status }}</span>
    </div>
    <time class="text-xs text-neutral-500">{{ e.timestamp }}</time>
  </li>
  {% else %}
  <li class="py-2 text-neutral-500">No events yet.</li>
  {% endfor %}
</ul>
```

`car_logger/templates/partials/vehicles_list.html`:
```html
<ul class="space-y-1">
  {% for v in vehicles %}
  <li class="flex justify-between text-sm">
    <span class="font-mono">{{ v.plate_text }}</span>
    <span class="text-neutral-500">{{ v.total_sightings }}×</span>
  </li>
  {% else %}
  <li class="text-neutral-500 text-sm">No vehicles yet.</li>
  {% endfor %}
</ul>
```

`car_logger/templates/partials/stats.html`:
```html
<dl class="grid grid-cols-3 gap-2 text-center">
  <div><dt class="text-xs text-neutral-500">Events</dt>
    <dd class="text-xl font-semibold">{{ stats.total_events }}</dd></div>
  <div><dt class="text-xs text-neutral-500">Plates</dt>
    <dd class="text-xl font-semibold">{{ stats.plates_read }}</dd></div>
  <div><dt class="text-xs text-neutral-500">Vehicles</dt>
    <dd class="text-xl font-semibold">{{ stats.unique_vehicles }}</dd></div>
</dl>
```

- [ ] **Step 3: Write the dashboard router** **[LAPTOP — Claude]**

`car_logger/api/routes_dashboard.py`:
```python
"""Server-rendered dashboard: the full page plus htmx partial fragments."""

import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from car_logger import repositories
from car_logger.database import get_db

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "templates")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

router = APIRouter(tags=["dashboard"])


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/partials/events-feed", response_class=HTMLResponse)
def events_feed(request: Request, db: Session = Depends(get_db)):
    events = repositories.list_events(db, limit=25)
    return templates.TemplateResponse(
        "partials/events_feed.html", {"request": request, "events": events}
    )


@router.get("/partials/vehicles-list", response_class=HTMLResponse)
def vehicles_list(request: Request, db: Session = Depends(get_db)):
    vehicles = repositories.list_vehicles(db, limit=25)
    return templates.TemplateResponse(
        "partials/vehicles_list.html",
        {"request": request, "vehicles": vehicles},
    )


@router.get("/partials/stats", response_class=HTMLResponse)
def stats(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "partials/stats.html",
        {"request": request, "stats": repositories.event_stats(db)},
    )
```

- [ ] **Step 4: Write dashboard tests + fix the root test** **[LAPTOP — Claude]**

Replace the root test in `tests/test_main.py`:
```python
def test_root_serves_dashboard_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Car Logger" in response.text
    assert "text/html" in response.headers["content-type"]
```
> This test now needs the `client` fixture (DB override), so move `test_root_serves_dashboard_html` and `test_health_returns_ok` to take `client` (from `conftest.py`) instead of the module-level `TestClient`. `test_health_returns_ok` becomes `def test_health_returns_ok(client): ...`.

`tests/integration/test_dashboard.py`:
```python
def test_events_feed_partial_renders(client):
    client.post("/api/events", json={"plate_text": "B123XYZ"})
    resp = client.get("/partials/events-feed")
    assert resp.status_code == 200
    assert "B123XYZ" in resp.text


def test_stats_partial_renders_counts(client):
    resp = client.get("/partials/stats")
    assert resp.status_code == 200
    assert "Events" in resp.text


def test_vehicles_partial_empty(client):
    resp = client.get("/partials/vehicles-list")
    assert resp.status_code == 200
    assert "No vehicles yet." in resp.text
```

- [ ] **Step 5: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/templates car_logger/api/routes_dashboard.py tests/test_main.py tests/integration/test_dashboard.py
git commit -m "feat(dashboard): jinja+htmx dashboard with polling partials

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/ -v
```
Expected: full suite green (Stage 1–3 + anpr client 6 + cropping 2 + repo additions 3 + dashboard 3).

**CHECKPOINT:** paste the full pytest output before Task 7.

---

### Task 7: End-to-end live verification

**Files:** none (add `data/` to `.gitignore` if missing).

- [ ] **Step 1: Confirm `.gitignore` covers secrets + data** **[LAPTOP — Claude]**

Ensure `.gitignore` contains `.env` and `data/`. If not, add them and commit:
```bash
git add .gitignore
git commit -m "chore: ignore .env and data/ (crops, db)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 2: Run the full appliance** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
alembic upgrade head
uvicorn car_logger.main:app --host 0.0.0.0 --port 8000
```
Open `http://192.168.0.232:8000/` on the laptop/phone. Hold a printed plate or a car photo (with a visible plate) to the webcam.

Expected: within a few seconds an event appears in the feed; shortly after, its `plate_text` fills in and a vehicle appears in the sidebar. DevTools → Network shows htmx GETs to `/partials/*` every 2s.

- [ ] **Step 3: Internet-down test** **[JETSON — student]**

Disconnect the Jetson from the internet (unplug Ethernet / disable Wi-Fi). Show a car to the camera. Expected: events still created, `anpr_status="failed"` (visible in the feed), no crash. Reconnect → new events get plates again.

- [ ] **Step 4: Resource check** **[JETSON — student]**

In another terminal: `tegrastats`. Expected: full-stack RAM < 2.5GB; pipeline FPS (via `/api/status`) ≥ 8 with ANPR integrated.

**CHECKPOINT:** report: plate read live? internet-down behaviour? FPS + RAM? Stage 4 is done when a plate is read end-to-end and the offline test passes.

---

## Self-Review

**1. Spec coverage** (against `PLAN.md` Week 4):
- ANPR client, httpx, student retry policy, mocked tests (200/429/500/timeout): Task 1 (6 tests). ✓
- Non-blocking integration via separate worker + queue, FPS-unaffected proof: Tasks 4–5. ✓
- Crop storage to `data/plates/<id>.jpg`, event update with plate/status/path: Tasks 2, 3, 5. ✓
- Vehicle upsert per plate: Task 3. ✓
- Dashboard base + dashboard.html + three panels, Tailwind+htmx CDN: Task 6. ✓
- Dashboard routes: `/`, `/partials/events-feed|vehicles-list|stats`: Task 6. ✓
- Events feed auto-refresh every 2s: dashboard.html `hx-trigger="every 2s"`. ✓
- Network-down → events created, `anpr_status="failed"`: Tasks 5 + 7 Step 3. ✓
- Mobile browser, RAM < 2.5GB, FPS ≥ 8, all tests green: Task 7. ✓

**2. Placeholder scan:** every code/template step is complete; the retention-cleanup helper (PLAN 4.4) is deferred to Stage 5 polish and noted there, not left as a TODO here. ✓

**3. Type consistency:** `PlateResult(plate_text, confidence, status)` fields identical in client, worker callback, and `on_result`. `update_event_anpr` / `upsert_vehicle_for_plate` / `list_vehicles` / `event_stats` signatures match between impl (Task 3), tests, and callers (Task 5, Task 6). `on_confirmed(track, frame)` signature matches pipeline impl (Task 5 Step 1) and the `main.py` closure. Template context keys (`events`, `vehicles`, `stats`) match the router. ✓

## Notes for the executor

- **Retry policy is the student's call** — the defaults (5s timeout, 2 retries, no-retry on 429/4xx) are documented; confirm or change with justification.
- The `on_event(event_dict)` shape from Stage 3 is intentionally replaced by `on_confirmed(track, frame)` — only Stage 3's throwaway verification snippet used the old shape; no shipped module depends on it.
- Debugging threading/SQLite issues that surface here (multiple sessions across camera/pipeline/ANPR threads) is the student's per CLAUDE.md — Claude explains, the student fixes.
- Do not add SSE yet — the dashboard polls in this stage; SSE replaces polling in Stage 5.
