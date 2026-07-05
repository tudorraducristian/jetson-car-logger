# Stage 3 — Camera + Detection Pipeline (car_logger) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Jetson captures webcam video in a background thread, runs SSD-Mobilenet-v2 detection, tracks vehicles with a simple IoU tracker to deduplicate, and writes one `Event` per confirmed track to the DB. Add `GET /api/status`. No ANPR yet (plate stays null).

**Architecture:** A producer/consumer split across threads. `CameraWorker` (producer) reads frames in a daemon thread and exposes the latest frame thread-safely. `PipelineWorker` (consumer) pulls the latest frame, runs `Detector` (the black-box CV), feeds boxes to `IoUTracker`, and calls an `on_event` callback for each newly *confirmed* track. The callback persists via the Stage 2 repository using its own DB session (SQLite `check_same_thread=False` makes cross-thread writes legal). Pure-logic modules (`geometry`, `tracker`) are unit-tested; hardware modules (`capture`, `detector`) and the wiring are verified live on the Jetson.

**Tech Stack:** `threading`, `cv2` (system site-package), `jetson.inference` / `jetson.utils` (SSD-Mobilenet-v2), the Stage 2 repository + SQLAlchemy session, FastAPI startup/shutdown events, pytest.

## Global Constraints

- **Python 3.6.9 target.** No 3.7+ syntax.
- **CV is a black box:** `jetson.inference.detectNet("ssd-mobilenet-v2")`. Do not expand the CV layer (no ultralytics, no cv2.cuda, no custom models).
- **Memory-conscious:** process the *latest* frame only; never queue/buffer frames. If RAM > 2.5GB, investigate.
- **Threads + SQLite:** each worker opens its own `Session` from `SessionLocal`; never share one Session across threads. The engine already has `check_same_thread=False` (Stage 2).
- **No state in code:** events go to SQLite via the repository. In-memory tracker state (track ids) is deliberately transient — it is not persisted.
- **Clean shutdown:** Ctrl+C must stop the threads and release the camera (no "device busy" on restart).
- **Split execution:** **[LAPTOP — Claude]** writes/commits/pushes; **[JETSON — student]** pulls and runs. Paste output at each **CHECKPOINT**.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File structure (what this stage creates)

- `car_logger/services/__init__.py` (empty)
- `car_logger/services/geometry.py` — `iou(box_a, box_b)` (pure, tested)
- `car_logger/services/tracker.py` — `IoUTracker` + `Track` (pure, tested; **student tunes thresholds**)
- `car_logger/services/capture.py` — `CameraWorker` (hardware, live-verified)
- `car_logger/services/detector.py` — `Detector` wrapping detectNet (hardware, live-verified)
- `car_logger/services/pipeline.py` — `PipelineWorker` (glue, live-verified)
- `car_logger/api/routes_status.py` — `GET /api/status`
- `car_logger/main.py` — startup/shutdown handlers, `include_router`
- `tests/unit/test_geometry.py`, `tests/unit/test_tracker.py`

**Box convention (contract for the whole stage):** a box is a 4-tuple `(x1, y1, x2, y2)` in pixels, top-left / bottom-right. A `Detection` is a namedtuple `(x1, y1, x2, y2, confidence, class_id)`.

---

### Task 1: IoU geometry helper (test-first)

**Files:**
- Create: `car_logger/services/__init__.py` (empty)
- Create: `car_logger/services/geometry.py`
- Test: `tests/unit/test_geometry.py`

**Interfaces:**
- Produces: `iou(box_a, box_b) -> float` in `[0.0, 1.0]`, where a box is `(x1, y1, x2, y2)`. Consumed by the tracker (Task 2).

- [ ] **Step 1: Write the failing tests** **[LAPTOP — Claude]**

`car_logger/services/__init__.py`: empty file.

`tests/unit/test_geometry.py`:
```python
from car_logger.services.geometry import iou


def test_identical_boxes_iou_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_disjoint_boxes_iou_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_half_overlap_iou():
    # two 10x10 boxes overlapping in a 10x5 strip:
    # intersection = 50, union = 100 + 100 - 50 = 150 -> 1/3
    result = iou((0, 0, 10, 10), (0, 5, 10, 15))
    assert abs(result - (1.0 / 3.0)) < 1e-9


def test_zero_area_box_is_zero():
    assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0
```

- [ ] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/__init__.py tests/unit/test_geometry.py
git commit -m "test(geometry): failing tests for IoU

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_geometry.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'car_logger.services.geometry'`.

- [ ] **Step 3: Implement `iou`** **[LAPTOP — Claude]**

`car_logger/services/geometry.py`:
```python
"""Pure geometry for the tracker — no CV, no hardware, fully unit-testable."""


def iou(box_a, box_b):
    """Intersection-over-Union of two (x1, y1, x2, y2) boxes. Returns 0.0..1.0."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
```

- [ ] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/geometry.py
git commit -m "feat(geometry): IoU helper for the tracker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_geometry.py -v
```
Expected: `4 passed`.

**CHECKPOINT:** paste the pytest output before Task 2.

---

### Task 2: IoU tracker (STUDENT-LED business logic, test-first)

**Files:**
- Create: `car_logger/services/tracker.py`
- Test: `tests/unit/test_tracker.py`

**Interfaces:**
- Consumes: `iou` (Task 1).
- Produces:
  - `Track` object with `.track_id: int`, `.box: tuple`, `.hits: int`, `.missed: int`, `.emitted: bool`.
  - `IoUTracker(iou_threshold=0.3, max_missed=5, min_hits=5)` with:
    - `update(boxes: List[tuple]) -> List[Track]` — feed one frame's boxes, returns live tracks.
    - `new_confirmed_tracks() -> List[Track]` — tracks that just crossed `min_hits` and haven't emitted yet (marks them emitted). Consumed by the pipeline (Task 5).

> **STUDENT DECISION — confirm or tune these three, and write one sentence of justification per value in the docstring:**
> - `iou_threshold = 0.3` — below this, two boxes are treated as different vehicles.
> - `max_missed = 5` — frames a track survives with no match before it dies (handles brief detection dropouts).
> - `min_hits = 5` — frames a track must be seen before we trust it enough to emit an event (kills single-frame false positives).

- [ ] **Step 1: Write the failing tests (the scenarios ARE the spec)** **[LAPTOP — Claude]**

`tests/unit/test_tracker.py`:
```python
from car_logger.services.tracker import IoUTracker


def test_overlapping_box_keeps_same_track_id():
    t = IoUTracker(iou_threshold=0.3, max_missed=5, min_hits=5)
    t.update([(0, 0, 10, 10)])
    first_id = t.tracks[0].track_id
    # a slightly shifted box that still overlaps a lot -> same track
    t.update([(1, 1, 11, 11)])
    assert len(t.tracks) == 1
    assert t.tracks[0].track_id == first_id
    assert t.tracks[0].hits == 2


def test_non_overlapping_boxes_make_two_tracks():
    t = IoUTracker()
    t.update([(0, 0, 10, 10), (100, 100, 110, 110)])
    assert len(t.tracks) == 2
    assert t.tracks[0].track_id != t.tracks[1].track_id


def test_track_dies_after_max_missed_frames():
    t = IoUTracker(max_missed=2)
    t.update([(0, 0, 10, 10)])          # born
    t.update([])                        # missed 1
    t.update([])                        # missed 2
    assert len(t.tracks) == 1           # still alive at == max_missed
    t.update([])                        # missed 3 -> death
    assert t.tracks == []


def test_confirmed_only_after_min_hits_and_emitted_once():
    t = IoUTracker(min_hits=3)
    box = (0, 0, 10, 10)
    t.update([box])
    assert t.new_confirmed_tracks() == []   # 1 hit
    t.update([box])
    assert t.new_confirmed_tracks() == []   # 2 hits
    t.update([box])
    confirmed = t.new_confirmed_tracks()     # 3 hits -> confirmed
    assert len(confirmed) == 1
    # a track emits exactly once, not every subsequent frame:
    t.update([box])
    assert t.new_confirmed_tracks() == []
```

- [ ] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add tests/unit/test_tracker.py
git commit -m "test(tracker): failing scenarios for IoU dedup rules

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_tracker.py -v
```
Expected: FAIL — `car_logger.services.tracker` doesn't exist.

- [ ] **Step 3: Implement the tracker** **[LAPTOP — Claude]**

`car_logger/services/tracker.py`:
```python
"""Simple greedy IoU tracker — the deduplication brain of the pipeline.

STUDENT DECISIONS (defaults; justify each, then tune on real footage):
- iou_threshold = 0.3 : below this, boxes are different vehicles.
- max_missed    = 5   : a track survives 5 frames of no match (dropout tolerance).
- min_hits      = 5   : seen 5 frames before we emit an event (kills flicker).
"""

from car_logger.services.geometry import iou


class Track(object):
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.hits = 1
        self.missed = 0
        self.emitted = False

    def is_confirmed(self, min_hits):
        return self.hits >= min_hits


class IoUTracker(object):
    def __init__(self, iou_threshold=0.3, max_missed=5, min_hits=5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks = []

    def update(self, boxes):
        """Feed one frame's detection boxes; return the live Track list."""
        unmatched = list(range(len(boxes)))

        # Greedy: match each existing track to its best still-free box.
        for track in self.tracks:
            best_score = self.iou_threshold
            best_j = -1
            for j in unmatched:
                score = iou(track.box, boxes[j])
                if score >= best_score:
                    best_score = score
                    best_j = j
            if best_j >= 0:
                track.box = boxes[best_j]
                track.hits += 1
                track.missed = 0
                unmatched.remove(best_j)
            else:
                track.missed += 1

        # Births: any detection that matched nothing starts a new track.
        for j in unmatched:
            self.tracks.append(Track(self._next_id, boxes[j]))
            self._next_id += 1

        # Deaths: drop tracks unseen for more than max_missed frames.
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks

    def new_confirmed_tracks(self):
        """Tracks that just became confirmed and have not emitted yet.

        Marks them emitted so a track produces exactly one event."""
        ready = []
        for t in self.tracks:
            if t.is_confirmed(self.min_hits) and not t.emitted:
                t.emitted = True
                ready.append(t)
        return ready
```

- [ ] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/tracker.py
git commit -m "feat(tracker): greedy IoU tracker with confirm/emit-once rules

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_tracker.py -v
```
Expected: `4 passed`.

**CHECKPOINT:** paste the pytest output before Task 3.

---

### Task 3: Camera capture worker (hardware — live-verified)

**Files:**
- Create: `car_logger/services/capture.py`

**Interfaces:**
- Produces: `CameraWorker(device_index=0)` with `start()`, `get_latest_frame() -> Optional[ndarray]` (a BGR copy, thread-safe), `stop()`. Consumed by the pipeline (Task 5).

- [ ] **Step 1: Write the camera worker** **[LAPTOP — Claude]**

`car_logger/services/capture.py`:
```python
"""Camera capture worker: cv2.VideoCapture in a daemon thread.

Why a thread? cv2.VideoCapture(0) blocks 3-5s at open and each read() blocks
until a frame arrives. Running it in the request path would freeze the server.
The worker keeps only the *latest* frame (no buffering) to respect the 4GB RAM
budget."""

import threading
import time

import cv2


class CameraWorker(object):
    def __init__(self, device_index=0):
        self.device_index = device_index
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._cap = cv2.VideoCapture(self.device_index)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Hold the lock only for the pointer swap, never during read().
            with self._lock:
                self._frame = frame

    def get_latest_frame(self):
        """Return a private copy of the latest frame, or None if none yet.

        We copy under the lock so the caller can't read a frame the capture
        thread is mid-overwriting (that would be a data race)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/services/capture.py
git commit -m "feat(capture): threaded CameraWorker with latest-frame access

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Live-verify the camera opens and yields frames** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "
import time
from car_logger.services.capture import CameraWorker
c = CameraWorker(0); c.start(); time.sleep(3)
f = c.get_latest_frame()
print('frame shape:', None if f is None else f.shape)
c.stop(); print('stopped cleanly')
"
```
Expected: a shape like `(480, 640, 3)` then `stopped cleanly`. If `None`, the webcam index may not be 0 — check `ls /dev/video*`.

**CHECKPOINT:** paste the output before Task 4.

---

### Task 4: Detector wrapper (black-box CV — live-verified)

**Files:**
- Create: `car_logger/services/detector.py`
- Modify: `car_logger/config.py` (add `detector_threshold` and `camera_index`)

**Interfaces:**
- Consumes: `jetson.inference.detectNet`, a BGR frame.
- Produces: `Detection` namedtuple `(x1, y1, x2, y2, confidence, class_id)`; `Detector(threshold=0.5).detect(frame_bgr) -> List[Detection]`, filtered to vehicle COCO classes.

- [ ] **Step 1: Add the two settings** **[LAPTOP — Claude]**

In `car_logger/config.py`, add inside `Settings` (after `max_pipeline_fps`):
```python
    detector_threshold: float = 0.5
    camera_index: int = 0
    enable_pipeline: bool = True
```
And add to `.env.example`:
```
DETECTOR_THRESHOLD=0.5
CAMERA_INDEX=0
ENABLE_PIPELINE=true
```

- [ ] **Step 2: Write the detector** **[LAPTOP — Claude]**

`car_logger/services/detector.py`:
```python
"""The entire CV layer: a thin wrapper over jetson.inference SSD-Mobilenet-v2.

jetson.inference/jetson.utils are imported lazily inside __init__ so importing
this module (e.g. during test collection or on a non-Jetson box) does not
require CUDA to be present."""

from collections import namedtuple

import cv2

Detection = namedtuple(
    "Detection", ["x1", "y1", "x2", "y2", "confidence", "class_id"]
)

# COCO class ids we keep: 3=car, 4=motorcycle, 6=bus, 8=truck.
VEHICLE_CLASS_IDS = frozenset([3, 4, 6, 8])


class Detector(object):
    def __init__(self, threshold=0.5):
        import jetson.inference
        import jetson.utils
        self._utils = jetson.utils
        self._net = jetson.inference.detectNet(
            "ssd-mobilenet-v2", threshold=threshold
        )

    def detect(self, frame_bgr):
        """Run detection on one BGR frame; return vehicle Detections only."""
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        cuda_img = self._utils.cudaFromNumpy(rgba)
        raw = self._net.Detect(cuda_img, overlay="none")
        results = []
        for d in raw:
            if int(d.ClassID) not in VEHICLE_CLASS_IDS:
                continue
            results.append(Detection(
                int(d.Left), int(d.Top), int(d.Right), int(d.Bottom),
                float(d.Confidence), int(d.ClassID),
            ))
        return results
```

- [ ] **Step 3: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/services/detector.py car_logger/config.py .env.example
git commit -m "feat(detector): SSD-Mobilenet-v2 wrapper filtered to vehicle classes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 4: Live-verify detection on a real frame** **[JETSON — student]**

Point the webcam at a phone/screen showing a car photo, then:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "
import time
from car_logger.services.capture import CameraWorker
from car_logger.services.detector import Detector
c = CameraWorker(0); c.start(); time.sleep(3)
d = Detector(threshold=0.5)
dets = d.detect(c.get_latest_frame())
print('detections:', dets)
c.stop()
"
```
Expected: first run downloads/builds the TensorRT engine (can take a few minutes — normal). Then a list of `Detection(...)` when a car is visible, `[]` otherwise.

**CHECKPOINT:** paste the output before Task 5.

---

### Task 5: Pipeline worker (glue — live-verified)

**Files:**
- Create: `car_logger/services/pipeline.py`

**Interfaces:**
- Consumes: `CameraWorker`, `Detector`, `IoUTracker`, an `on_event(event_dict)` callback.
- Produces: `PipelineWorker(camera, detector, tracker, on_event, target_fps=15)` with `start()`, `stop()`, and read-only attributes `last_fps: float`, `last_event_at: Optional[float]`, `frames_processed: int`. Each confirmed track triggers `on_event({"bbox_json": ..., "track_id": ..., "anpr_status": "pending"})`. Consumed by `main.py` (Task 7) and `/api/status` (Task 6).

- [ ] **Step 1: Write the pipeline worker** **[LAPTOP — Claude]**

`car_logger/services/pipeline.py`:
```python
"""Pipeline worker: camera -> detector -> tracker -> on_event callback.

Runs in its own daemon thread. It reads only the latest frame (drops stale
ones), so it never falls behind and never buffers video."""

import json
import threading
import time


class PipelineWorker(object):
    def __init__(self, camera, detector, tracker, on_event, target_fps=15):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.on_event = on_event
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
                self._emit(track)
            self.frames_processed += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                self.last_fps = 1.0 / elapsed
            # Throttle to the target FPS so we don't pin the GPU pointlessly.
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

    def _emit(self, track):
        event = {
            "bbox_json": json.dumps(list(track.box)),
            "track_id": track.track_id,
            "anpr_status": "pending",
        }
        self.last_event_at = time.time()
        self.on_event(event)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/services/pipeline.py
git commit -m "feat(pipeline): threaded camera->detect->track->emit worker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Live-verify the pipeline emits deduplicated events** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "
import time
from car_logger.services.capture import CameraWorker
from car_logger.services.detector import Detector
from car_logger.services.tracker import IoUTracker
from car_logger.services.pipeline import PipelineWorker
emitted = []
c = CameraWorker(0); c.start()
p = PipelineWorker(c, Detector(0.5), IoUTracker(), emitted.append, target_fps=15)
p.start(); time.sleep(15)   # hold a car image to the camera for these 15s
p.stop(); c.stop()
print('events emitted:', len(emitted), '| last_fps:', round(p.last_fps, 1))
for e in emitted[:5]: print(e)
"
```
Expected: a **small** number of events (single digits, not hundreds) for one car held in view — proving dedup works — and `last_fps` ≥ 10.

**CHECKPOINT:** paste the output before Task 6.

---

### Task 6: `GET /api/status` endpoint

**Files:**
- Create: `car_logger/api/routes_status.py`

**Interfaces:**
- Consumes: the running `PipelineWorker` and `CameraWorker`, stashed on `app.state` in Task 7.
- Produces: `GET /api/status` → `{"pipeline_running": bool, "fps": float, "frames_processed": int, "camera_ok": bool, "last_event_at": Optional[float]}`.

- [ ] **Step 1: Write the status router** **[LAPTOP — Claude]**

`car_logger/api/routes_status.py`:
```python
"""GET /api/status — pipeline health for monitoring and the dashboard."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def status(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    camera = getattr(request.app.state, "camera", None)
    if pipeline is None:
        return {
            "pipeline_running": False,
            "fps": 0.0,
            "frames_processed": 0,
            "camera_ok": False,
            "last_event_at": None,
        }
    return {
        "pipeline_running": True,
        "fps": round(pipeline.last_fps, 1),
        "frames_processed": pipeline.frames_processed,
        "camera_ok": camera is not None and camera.get_latest_frame() is not None,
        "last_event_at": pipeline.last_event_at,
    }
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/api/routes_status.py
git commit -m "feat(api): GET /api/status reporting pipeline health

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

**CHECKPOINT:** none yet — wired and verified together in Task 7.

---

### Task 7: Wire pipeline into the app lifecycle

**Files:**
- Modify: `car_logger/main.py`

**Interfaces:**
- Consumes: `CameraWorker`, `Detector`, `IoUTracker`, `PipelineWorker`, `SessionLocal`, `repositories`, `schemas`, `settings`, the status router.
- Produces: on startup, a running camera + pipeline whose `on_event` persists via a fresh session; on shutdown, both stopped. `app.state.pipeline` / `app.state.camera` exposed for `/api/status`. Gated by `settings.enable_pipeline` (off in tests).

- [ ] **Step 1: Update `main.py`** **[LAPTOP — Claude]**

`car_logger/main.py`:
```python
"""Car Logger API entrypoint - the app object everything else attaches to."""

from fastapi import FastAPI

from car_logger.api.routes_events import router as events_router
from car_logger.api.routes_status import router as status_router
from car_logger.config import settings
from car_logger.database import SessionLocal
from car_logger import repositories, schemas

APP_VERSION = "0.3.0"

app = FastAPI(title="Car Logger", version=APP_VERSION)

app.include_router(events_router)
app.include_router(status_router)


def _persist_event(event_dict):
    """on_event callback: open a short-lived session in the pipeline thread and
    write the event. A new session per event keeps thread ownership simple."""
    db = SessionLocal()
    try:
        repositories.create_event(db, schemas.EventCreate(**event_dict))
    finally:
        db.close()


@app.on_event("startup")
def _startup():
    if not settings.enable_pipeline:
        return
    # Imported here (not at module top) so importing main.py without a camera
    # (e.g. test collection) never needs cv2/jetson at import time.
    from car_logger.services.capture import CameraWorker
    from car_logger.services.detector import Detector
    from car_logger.services.tracker import IoUTracker
    from car_logger.services.pipeline import PipelineWorker

    camera = CameraWorker(device_index=settings.camera_index)
    camera.start()
    pipeline = PipelineWorker(
        camera=camera,
        detector=Detector(threshold=settings.detector_threshold),
        tracker=IoUTracker(),
        on_event=_persist_event,
        target_fps=settings.max_pipeline_fps,
    )
    pipeline.start()
    app.state.camera = camera
    app.state.pipeline = pipeline


@app.on_event("shutdown")
def _shutdown():
    pipeline = getattr(app.state, "pipeline", None)
    camera = getattr(app.state, "camera", None)
    if pipeline is not None:
        pipeline.stop()
    if camera is not None:
        camera.stop()


@app.get("/")
def root():
    """Greeting endpoint - proves the server is reachable from the LAN."""
    return {"message": "Car Logger is running", "version": APP_VERSION}


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
```

Update the version assertion in `tests/test_main.py`:
```python
    assert body["version"] == "0.3.0"
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/main.py tests/test_main.py
git commit -m "feat(app): start/stop camera+pipeline on lifecycle, expose /api/status

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Confirm the whole test suite still passes** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/ -v
```
Expected: all Stage 1+2 tests plus geometry (4) + tracker (4) still green. The pipeline does NOT start during tests (TestClient isn't used as a context manager, and `enable_pipeline` can also be set false via env).

- [ ] **Step 4: Full live run + observe events accumulating** **[JETSON — student]**

Terminal A (leave running):
```bash
cd ~/jetson-car-logger && source venv/bin/activate
alembic upgrade head   # ensure the DB exists
uvicorn car_logger.main:app --host 0.0.0.0 --port 8000
```
Terminal B (or laptop browser), while holding a car image to the camera:
```bash
curl http://192.168.0.232:8000/api/status
curl "http://192.168.0.232:8000/api/events?limit=20"
```
Expected: `/api/status` shows `fps` ≥ 10 and `pipeline_running: true`; `/api/events` returns a handful of events with `bbox_json` populated and `plate_text` null.

- [ ] **Step 5: Clean-shutdown check** **[JETSON — student]**

Press Ctrl+C in Terminal A. Then immediately restart uvicorn. Expected: no "device busy" / "camera in use" error — the shutdown handler released the camera. Confirm with `tegrastats` (separate terminal) that RAM stayed < 2.5GB during the run.

**CHECKPOINT:** paste `/api/status`, a sample of `/api/events`, and confirm the clean restart. Stage 3 is done.

---

## Self-Review

**1. Spec coverage** (against `PLAN.md` Week 3):
- Threaded `CameraWorker` with thread-safe latest-frame + start/stop + daemon: Task 3. ✓
- Detector wrapping detectNet, vehicle-class filter, configurable threshold: Task 4. ✓
- IoU tracker, student-owned thresholds, ≥ 4 test scenarios: Task 2 (4 scenarios). ✓
- Pipeline orchestration + event-emission rule (confirmed tracks, emit once): Tasks 2 + 5. ✓
- Startup/shutdown wiring, DB persistence via repository from the thread: Task 7. ✓
- `GET /api/status` with FPS/health/last event: Tasks 6–7. ✓
- Dedup produces few events (not 30/sec): Task 5 Step 3 + Task 7 Step 4. ✓
- FPS ≥ 10, RAM < 2.5GB, clean Ctrl+C: Task 7 Steps 4–5. ✓
- All Week 2 tests still pass: Task 7 Step 3. ✓

**2. Placeholder scan:** every code step is complete; no TBD/TODO. ✓

**3. Type consistency:** box tuple `(x1,y1,x2,y2)` used identically in geometry, tracker, `Detection`, and the pipeline's `boxes` comprehension. `new_confirmed_tracks()` / `update()` names match across tracker impl, tests, and pipeline. `on_event` dict keys (`bbox_json`, `track_id`, `anpr_status`) match `EventCreate` fields (Stage 2). `app.state.pipeline`/`camera` set in Task 7 match reads in Task 6. `settings.detector_threshold`/`camera_index`/`enable_pipeline` added in Task 4 match uses in Task 7. ✓

## Notes for the executor

- **Business logic is the student's:** the three tracker thresholds AND the event-emission rule (confirmed-track-emits-once) are theirs to confirm/justify. Defaults are provided so the plan is runnable, not to be accepted blindly.
- First `Detector(...)` call builds a TensorRT engine — minutes on the Nano, one-time. Not a hang.
- If the camera never yields frames, check `ls /dev/video*` and try `CAMERA_INDEX=1` in `.env`. Per the CLAUDE.md debugging rule, chase device/threading errors yourself — Claude explains, you fix.
- Do not add ANPR here — plate stays null until Stage 4.
