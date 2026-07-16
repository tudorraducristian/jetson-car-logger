# Camera Self-Healing + Honest Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the camera layer recover on its own when the USB webcam drops, and make `camera_ok` tell the truth (fresh frame vs frozen frame).

**Architecture:** Approach A — reconnection lives inside `CameraWorker`. A monotonic freshness timestamp is the single source of truth: `get_latest_frame()` returns `None` when the last frame is older than `stale_after_s`, `is_healthy()` reports the same, and the capture loop reopens the device (retry-forever with backoff) once a read has been failing past the threshold. The read/reconnect decision is factored into a sleepless `_run_once()` so it is unit-testable without threads via an injected fake camera and fake clock.

**Tech Stack:** Python 3.6, OpenCV (`cv2.VideoCapture`, lazily imported), FastAPI, Pydantic v1 `BaseSettings`, structlog, pytest.

## Global Constraints

- **Python 3.6 only** — no walrus `:=`, no f-string `=`, no `typing.Literal`, no dict-union `|`. `float("inf")` and kwargs-logging are fine.
- **Pydantic v1** — settings via `BaseSettings`, nested `class Config`. New fields are plain typed defaults.
- **No new dependencies.** Reuse `structlog` via `car_logger.logging_config.get_logger`.
- **`cv2` must NOT be imported at module top** of `capture.py` — it goes in the default factory (mirrors `detector.py`'s lazy `jetson.inference` import) so the module imports without OpenCV.
- **Tests run on the Jetson** (the laptop's Python 3.14 cannot install the pinned stack). `numpy` is available there for test frames.
- **Loss is time-based:** default `stale_after_s = 2.0`, `reopen_backoff_s = 2.0`, both from config.
- **Split execution:** Claude writes/commits/pushes on the laptop; the student runs RED/GREEN on the Jetson and owns the assertions.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `car_logger/services/capture.py` | Self-healing camera worker + honest freshness | rewrite |
| `tests/unit/test_capture.py` | Unit tests: freshness, read/reconnect (fakes) | create |
| `car_logger/api/routes_status.py` | `camera_ok` = `is_healthy()` | modify 1 line |
| `tests/integration/test_api_status.py` | `/api/status` honesty | create |
| `car_logger/config.py` | Two tunables | modify |
| `tests/unit/test_config.py` | Settings defaults | create |
| `car_logger/main.py` | Pass tunables to `CameraWorker` | modify |
| `.env.example`, `README.md` | Document the two settings | modify |

---

### Task 1: Honest freshness core in `CameraWorker`

Rewrites `capture.py` down to an importable class with the full constructor and the two honest read-side methods. The capture loop and reconnect come in Tasks 2-3; this task delivers the freshness truth (`get_latest_frame` → `None` when stale, `is_healthy`).

**Files:**
- Modify (rewrite): `car_logger/services/capture.py`
- Test: `tests/unit/test_capture.py`

**Interfaces:**
- Produces:
  - `_default_open_capture(device_index) -> cv2.VideoCapture` (module-level, lazy `import cv2`).
  - `CameraWorker(device_index=0, stale_after_s=2.0, reopen_backoff_s=2.0, open_capture=None, now=None)` — `open_capture` defaults to `_default_open_capture`, `now` defaults to `time.monotonic`.
  - `CameraWorker.is_healthy() -> bool`
  - `CameraWorker.get_latest_frame() -> Optional[frame]` (copy, or `None` when absent/stale)
  - `CameraWorker._seconds_since_frame() -> float`
  - State fields: `_cap`, `_frame`, `_last_frame_at`, `_lost`, `_lock`, `_running`, `_thread`.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_capture.py`:

```python
import numpy as np

from car_logger.services.capture import CameraWorker


class FakeClock(object):
    """Deterministic monotonic clock for tests."""
    def __init__(self, t=0.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


def _worker(clock, **kw):
    return CameraWorker(now=clock, stale_after_s=2.0, **kw)


def test_fresh_frame_is_returned_and_healthy():
    clock = FakeClock(50.0)
    w = _worker(clock)
    frame = np.zeros((4, 4), np.uint8)
    w._frame = frame
    w._last_frame_at = 50.0
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), frame)


def test_stale_frame_reads_as_absent_and_unhealthy():
    clock = FakeClock(50.0)
    w = _worker(clock)
    w._frame = np.zeros((4, 4), np.uint8)
    w._last_frame_at = 50.0
    clock.t = 53.0  # 3s later, past the 2s threshold
    assert w.is_healthy() is False
    assert w.get_latest_frame() is None


def test_no_frame_yet_is_unhealthy():
    w = _worker(FakeClock(0.0))
    assert w.is_healthy() is False
    assert w.get_latest_frame() is None
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_capture.py -v`
Expected: FAIL — `ImportError`/`AttributeError` (the new constructor signature and `is_healthy` don't exist yet).

- [x] **Step 3: Write the minimal implementation**

Replace the entire contents of `car_logger/services/capture.py` with:

```python
"""Camera capture worker: cv2.VideoCapture in a daemon thread that heals
itself when the USB webcam drops.

Why a thread? cv2.VideoCapture(0) blocks 3-5s at open and each read() blocks
until a frame arrives. The worker keeps only the *latest* frame (no
buffering) to respect the 4GB RAM budget.

Freshness is the single source of truth: a frame older than stale_after_s
reads as absent (get_latest_frame -> None) and unhealthy, so a frozen frame
can never masquerade as a live camera. cv2 is imported lazily in the default
factory so this module imports without OpenCV (e.g. off-Jetson)."""

import threading
import time


def _default_open_capture(device_index):
    import cv2
    return cv2.VideoCapture(device_index)


class CameraWorker(object):
    def __init__(self, device_index=0, stale_after_s=2.0,
                 reopen_backoff_s=2.0, open_capture=None, now=None):
        self.device_index = device_index
        self._stale_after_s = stale_after_s
        self._reopen_backoff_s = reopen_backoff_s
        self._open_capture = open_capture or _default_open_capture
        self._now = now or time.monotonic
        self._cap = None
        self._frame = None
        self._last_frame_at = None
        self._lost = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def _seconds_since_frame(self):
        if self._last_frame_at is None:
            return float("inf")
        return self._now() - self._last_frame_at

    def is_healthy(self):
        """True only if a fresh frame arrived within stale_after_s."""
        with self._lock:
            return (self._frame is not None
                    and self._seconds_since_frame() <= self._stale_after_s)

    def get_latest_frame(self):
        """A private copy of the latest frame, or None if there is none or
        it is stale. Copy under the lock so the caller can't read a frame the
        capture thread is mid-overwriting."""
        with self._lock:
            if (self._frame is None
                    or self._seconds_since_frame() > self._stale_after_s):
                return None
            return self._frame.copy()
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_capture.py -v`
Expected: PASS (3 passed).

- [x] **Step 5: Commit**

```bash
git add car_logger/services/capture.py tests/unit/test_capture.py
git commit -m "feat(camera): honest freshness — stale frame reads as absent"
```

---

### Task 2: Self-healing read/reconnect (`_run_once`)

Adds the sleepless per-iteration method that reads, stores a fresh frame, and declares/recovers the camera. This is where reconnection lives, fully unit-testable with a fake camera + fake clock.

**Files:**
- Modify: `car_logger/services/capture.py`
- Test: `tests/unit/test_capture.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: `CameraWorker._run_once() -> bool` — returns `True` only when a fresh frame was stored this call; opens `self._cap` via the factory when missing/closed; on a read failure older than `stale_after_s` it logs `camera_lost` once, releases and drops `self._cap`; on the first success after a loss it logs `camera_reconnected` and clears the lost flag. Never sleeps.

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_capture.py`:

```python
class FakeCapture(object):
    """Fake cv2.VideoCapture: yields `frame` while alive+opened, else fails."""
    def __init__(self, frame, alive=True, opened=True):
        self.frame = frame
        self.alive = alive
        self._opened = opened
        self.released = False
    def isOpened(self):
        return self._opened
    def read(self):
        if self.alive and self._opened:
            return True, self.frame
        return False, None
    def release(self):
        self.released = True
        self._opened = False


def test_run_once_stores_fresh_frame_and_is_healthy():
    clock = FakeClock(100.0)
    frame = np.zeros((4, 4), np.uint8)
    cap = FakeCapture(frame)
    w = _worker(clock, open_capture=lambda i: cap)
    assert w._run_once() is False  # first call opens the capture
    assert w._run_once() is True   # second call reads a frame
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), frame)


def test_single_dropped_frame_under_threshold_does_not_reopen():
    clock = FakeClock(100.0)
    cap = FakeCapture(np.zeros((4, 4), np.uint8))
    w = _worker(clock, open_capture=lambda i: cap)
    w._run_once(); w._run_once()          # open + one good read
    cap.alive = False                     # a single dropped frame
    assert w._run_once() is False
    assert w.is_healthy() is True         # clock unchanged -> still fresh
    assert w._cap is cap                  # NOT reopened
    assert cap.released is False


def test_loss_after_threshold_releases_and_forces_reopen():
    clock = FakeClock(100.0)
    cap = FakeCapture(np.zeros((4, 4), np.uint8))
    w = _worker(clock, open_capture=lambda i: cap)
    w._run_once(); w._run_once()          # healthy
    cap.alive = False
    clock.advance(3.0)                    # past the 2s threshold
    assert w._run_once() is False
    assert w._lost is True
    assert cap.released is True
    assert w._cap is None                 # dropped -> next call reopens
    assert w.is_healthy() is False
    assert w.get_latest_frame() is None


def test_recovery_after_reopen_clears_lost_flag():
    clock = FakeClock(100.0)
    dead = FakeCapture(np.zeros((4, 4), np.uint8))
    fresh_frame = np.ones((4, 4), np.uint8)
    alive = FakeCapture(fresh_frame)
    caps = [dead, alive]
    w = _worker(clock, open_capture=lambda i: caps.pop(0))
    w._run_once(); w._run_once()          # open dead + one good read
    dead.alive = False
    clock.advance(3.0)
    w._run_once()                         # declared lost, cap dropped
    assert w._run_once() is False         # reopens -> `alive`
    assert w._run_once() is True          # good read on the new capture
    assert w._lost is False
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), fresh_frame)


def test_reopen_that_stays_closed_does_not_crash():
    clock = FakeClock(100.0)
    closed = FakeCapture(np.zeros((4, 4), np.uint8), opened=False)
    w = _worker(clock, open_capture=lambda i: closed)
    assert w._run_once() is False         # opens a not-opened capture
    assert w._run_once() is False         # sees it closed, reopens again
    assert w.is_healthy() is False        # never raises
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_capture.py -v`
Expected: FAIL — `AttributeError: 'CameraWorker' object has no attribute '_run_once'`.

- [x] **Step 3: Write the minimal implementation**

At the top of `car_logger/services/capture.py`, add the logger import below `import time`:

```python
from car_logger.logging_config import get_logger

log = get_logger("car_logger.capture")
```

Add this method to `CameraWorker` (after `get_latest_frame`):

```python
    def _run_once(self):
        """One capture iteration: (re)open if needed, read once, store a
        fresh frame, or declare the camera lost after stale_after_s. Returns
        True only when a fresh frame was stored. Sleeps nowhere, so the
        read/reconnect logic is unit-testable without threads."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = self._open_capture(self.device_index)
            return False
        ok, frame = self._cap.read()
        if ok:
            with self._lock:
                self._frame = frame
                self._last_frame_at = self._now()
            if self._lost:
                log.info("camera_reconnected", device_index=self.device_index)
                self._lost = False
            return True
        if self._seconds_since_frame() > self._stale_after_s:
            if not self._lost:
                log.warning("camera_lost", device_index=self.device_index)
                self._lost = True
            self._cap.release()
            self._cap = None
        return False
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_capture.py -v`
Expected: PASS (8 passed total).

- [x] **Step 5: Commit**

```bash
git add car_logger/services/capture.py tests/unit/test_capture.py
git commit -m "feat(camera): self-healing read — reopen on loss, retry with lost/reconnected logs"
```

---

### Task 3: Thread loop + start/stop

Wraps `_run_once` in the daemon thread with the backoff sleeps and clean start/stop. `start()` no longer opens the device synchronously — the loop opens it — so startup no longer blocks 3-5s.

**Files:**
- Modify: `car_logger/services/capture.py`
- Test: `tests/unit/test_capture.py`

**Interfaces:**
- Consumes: `_run_once` from Task 2.
- Produces: `CameraWorker.start()`, `CameraWorker.stop()`, `CameraWorker._loop()`. Contract unchanged from callers' view (`start` then `get_latest_frame`/`is_healthy`, `stop` on shutdown).

- [x] **Step 1: Write the failing test**

Append to `tests/unit/test_capture.py`:

```python
import time as _time


def test_loop_captures_live_frames_and_stops_cleanly():
    frame = np.zeros((4, 4), np.uint8)
    cap = FakeCapture(frame)
    w = CameraWorker(open_capture=lambda i: cap, now=_time.monotonic,
                     stale_after_s=2.0, reopen_backoff_s=0.05)
    w.start()
    deadline = _time.monotonic() + 1.0
    while _time.monotonic() < deadline and not w.is_healthy():
        _time.sleep(0.01)
    assert w.is_healthy() is True
    assert np.array_equal(w.get_latest_frame(), frame)
    w.stop()
    assert w._running is False
```

- [x] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_capture.py::test_loop_captures_live_frames_and_stops_cleanly -v`
Expected: FAIL — `AttributeError: 'CameraWorker' object has no attribute 'start'` (Task 1 removed the old start/stop).

- [x] **Step 3: Write the minimal implementation**

Add these three methods to `CameraWorker` (after `_run_once`):

```python
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # A transient read failure must never kill the appliance's only CV
        # feed. _run_once owns the reopen; here we only pace the loop.
        while self._running:
            if self._run_once():
                continue  # read() blocks, so no sleep needed on success
            if self._cap is None or not self._cap.isOpened():
                time.sleep(self._reopen_backoff_s)
            else:
                time.sleep(0.01)  # brief, avoids a busy-spin before loss

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_capture.py -v`
Expected: PASS (9 passed total).

- [x] **Step 5: Commit**

```bash
git add car_logger/services/capture.py tests/unit/test_capture.py
git commit -m "feat(camera): daemon loop over _run_once with backoff; start no longer blocks"
```

---

### Task 4: Honest `/api/status`

`camera_ok` stops meaning "has a frame" and starts meaning "is healthy".

**Files:**
- Modify: `car_logger/api/routes_status.py:25`
- Test: `tests/integration/test_api_status.py`

**Interfaces:**
- Consumes: `CameraWorker.is_healthy()`.
- Produces: `/api/status` JSON with a truthful `camera_ok`.

- [x] **Step 1: Write the failing test**

Create `tests/integration/test_api_status.py`:

```python
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
```

- [x] **Step 2: Run the test to verify it fails**

Run: `pytest tests/integration/test_api_status.py -v`
Expected: FAIL — `AttributeError: 'FakeCam' object has no attribute 'get_latest_frame'` (the route still calls the old method).

- [x] **Step 3: Write the minimal implementation**

In `car_logger/api/routes_status.py`, change the `camera_ok` line inside the non-None-pipeline return (line ~25):

```python
        "camera_ok": camera is not None and camera.is_healthy(),
```

- [x] **Step 4: Run the test to verify it passes**

Run: `pytest tests/integration/test_api_status.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add car_logger/api/routes_status.py tests/integration/test_api_status.py
git commit -m "feat(status): camera_ok reports is_healthy, not just a present frame"
```

---

### Task 5: Config tunables + wiring + docs

Expose the two knobs in settings, pass them to the worker, and document them.

**Files:**
- Modify: `car_logger/config.py`
- Modify: `car_logger/main.py:128`
- Modify: `.env.example`, `README.md`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `CameraWorker(...)` full signature from Task 1.
- Produces: `settings.camera_stale_after_s: float`, `settings.camera_reopen_backoff_s: float`.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_config.py`:

```python
from car_logger.config import Settings


def test_camera_healing_settings_have_defaults():
    s = Settings()
    assert s.camera_stale_after_s == 2.0
    assert s.camera_reopen_backoff_s == 2.0
```

- [x] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL — `AttributeError`/validation: fields don't exist yet.

- [x] **Step 3: Write the minimal implementation**

In `car_logger/config.py`, add two fields after `camera_index`:

```python
    # camera self-healing (student decision 2026-07-15): no fresh frame for
    # this long => camera lost => camera_ok False + reopen. Reopen retries
    # this often until the device returns.
    camera_stale_after_s: float = 2.0
    camera_reopen_backoff_s: float = 2.0
```

In `car_logger/main.py`, replace the `CameraWorker(...)` construction in `_startup`:

```python
    camera = CameraWorker(
        device_index=settings.camera_index,
        stale_after_s=settings.camera_stale_after_s,
        reopen_backoff_s=settings.camera_reopen_backoff_s,
    )
    camera.start()
```

Append to `.env.example`:

```
# How long (seconds) with no fresh camera frame before the camera is
# declared lost (camera_ok goes False and the worker reopens it).
CAMERA_STALE_AFTER_S=2.0
# Delay (seconds) between reopen attempts while the camera is gone.
CAMERA_REOPEN_BACKOFF_S=2.0
```

In `README.md`, add two rows to the configuration table (match the existing column layout):

```
| `CAMERA_STALE_AFTER_S` | 2.0 | Seconds with no fresh frame before the camera is declared lost and reopened. |
| `CAMERA_REOPEN_BACKOFF_S` | 2.0 | Seconds between reopen attempts while the camera is gone. |
```

- [x] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add car_logger/config.py car_logger/main.py .env.example README.md tests/unit/test_config.py
git commit -m "feat(config): CAMERA_STALE_AFTER_S / CAMERA_REOPEN_BACKOFF_S wired into the worker"
```

---

### Task 6: Ops prerequisite + live verification on the Jetson

Not code. Disable USB autosuspend (the drop's root cause) and prove end-to-end that the appliance self-heals and `camera_ok` is honest. Student runs these on the device.

**Files:** none (device config + live checkpoint).

- [x] **Step 1: Disable USB autosuspend (persistent)** — done 2026-07-16; `cat /sys/module/usbcore/parameters/autosuspend` → `-1` after reboot (nano had to be apt-installed first).

```bash
sudo cp /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.bak
# Append " usbcore.autosuspend=-1" to the APPEND line, then:
sudo reboot
```
After reboot verify:
```bash
cat /sys/module/usbcore/parameters/autosuspend    # expect: -1
```

- [x] **Step 2: Deploy the code and run the full suite** — **91 passed**. Deviation: bare `pytest` died collecting `experiments/` (needs openpyxl) + a stray nested clone `~/jetson-car-logger/jetson-car-logger/` on the Jetson; fixed durably with `pytest.ini` `testpaths = tests` (`8fca0ca`). The nested clone still needs manual deletion (student's call).

```bash
cd ~/jetson-car-logger && git pull
source venv/bin/activate && pytest -q      # expect: all green (existing + new)
sudo systemctl restart car-logger
```

- [x] **Step 3: Confirm the chain works (baseline)** — `camera_ok: true`, fps 18.3-18.4, `frames_processed` climbing (389 → 2250 across checks).

Point a real car at the camera; on the dashboard (`http://<jetson-ip>:8000`) a new event appears. `curl`-free check over the running service:
```bash
python3 -c "import urllib.request,json; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/status')))"
```
Expect `camera_ok: true`, `frames_processed` climbing.

- [x] **Step 4: Prove self-healing (the real test)** — proven 2026-07-16, second run. **Live finding on the first run:** healing worked (reopen every ~2s, recovery on replug) but `camera_lost`/`camera_reconnected` never logged — a real unplug makes GStreamer flip `isOpened()` to False instead of failing `read()`, so the loss-declaring branch never ran. Fixed RED→GREEN (test `409eeef`, fix `7ee34cd`): loss is declared on the closed-handle path too, gated on ever having seen a frame. Second run journal: `camera_lost` once at 15:24:45, reopen attempts every ~2s, `camera_reconnected` once at 15:24:56; `camera_ok` false→true; frames resumed — **no manual restart**.

Unplug the webcam USB for ~5s, watch the journal, then replug — WITHOUT restarting the service:
```bash
journalctl -u car-logger -f
```
Expect: `camera_lost` logged once when unplugged; `/api/status` `camera_ok` flips to `false` and `frames_processed` stops climbing; on replug, `camera_reconnected` logged once, `camera_ok` back to `true`, frames resume — no manual restart.

- [x] **Step 5: Record the result** — this file + [[car-logger-progress]] updated 2026-07-16; camera repair DONE. Extra (in-spirit deviation): dashboard stats partial also switched to `is_healthy()` (`7c78b94`) — same boolean, no needless frame copy.

Tick the checkboxes in this plan with the observed outcome (journal lines, the camera_ok False→True transition). Update [[car-logger-progress]] to mark the camera repair DONE.

---

## Notes for the executor

- Steps that show code are complete — type them as written; the student owns
  and reviews the assertions per the project's ritual.
- Tasks 1-5 run RED→GREEN on the Jetson (unit + integration, no hardware).
  Task 6 is the live, observed checkpoint and needs the physical webcam.
- Do not weaken `camera_ok` back to a presence check "to make a test pass" —
  the whole point is that it stops lying.
