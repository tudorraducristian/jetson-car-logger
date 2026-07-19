# v2 Stage B — Local ANPR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cloud Plate Recognizer client with the bake-off
winner (fast-alpr ONNX stack on onnxruntime 1.9 CPU), filtered by a
3-read multi-frame vote, with a plate-read-only dashboard default — then
delete the cloud client after a live validation window.

**Architecture:** The `PlateResult` contract and worker-queue semantics
stay; the pipeline gains a per-track `CropCollector` (3 crops, ≥0.4 s
apart, one job per car), the worker calls
`LocalAnprClient.read_plate_multi` (plate detection → OCR per crop →
pure-function vote), and the dashboard feed gains a `filter` param
(default: `success` only). Spec: `docs/superpowers/specs/2026-07-19-v2b-local-anpr-integration-design.md`.

**Tech Stack:** Python 3.6 on the Jetson (JetPack 4.6), onnxruntime
1.9.0 CPU (+ numpy 1.19.5), cv2, FastAPI 0.67 + htmx (pinned stack),
pytest on the Jetson.

## Global Constraints

- **Python 3.6 syntax only.** No walrus, no f-string `=`, no dict `|`.
  Codebase style: `.format()` over f-strings. Pydantic v1, SQLAlchemy 1.3
  Query API, sync endpoints.
- **Split execution (established since Stage 1):** LAPTOP = Claude
  writes/commits/pushes (the pinned 3.6 app deps do NOT install on the
  laptop — the app suite runs only on the Jetson). JETSON = student (or
  Claude over ssh) runs every RED/GREEN checkpoint:
  `ssh tudor@192.168.0.188` (IP is DHCP — rediscover via `arp -a`, MAC
  `00-04-4b`), then `cd ~/jetson-car-logger && git pull` and
  `venv/bin/pytest ... -v`.
- **One task at a time** (standing student rule): after each task, a
  short written summary + pause. Do not batch tasks.
- **From Task 10 on** (numpy 1.19.5 installed), EVERY manual python/pytest
  run on the Jetson needs `OPENBLAS_CORETYPE=ARMV8` exported (numpy
  SIGILLs on the Tegra X1 without it). Harmless before Task 10; the
  commands below include it throughout for muscle memory.
- **Suite baseline:** 91 tests green on the Jetson before this plan.
  Every task ends with the full suite green (old + new).
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Task 11 is GATED:** it must NOT run until the validation window in
  `docs/v2-stage-b-validation-log.md` is closed (≥5 days AND ≥15
  verified events, max 1 wrong read).

## File Structure

```
models/anpr/                                  NEW  committed ONNX models (~11 MB)
├── yolo-v9-t-384-license-plates-end2end-opset15.onnx
├── cct_xs_v2_global.onnx
├── cct_xs_v2_global_plate_config.yaml
└── README.md                                      provenance + licences
car_logger/services/
├── plate_result.py                           NEW  PlateResult namedtuple (neutral home)
├── plate_voting.py                           NEW  pure vote function
├── local_anpr.py                             NEW  LocalAnprClient (2-stage, injected engines)
├── onnx_engines.py                           NEW  OnnxPlateDetector + OnnxPlateOcr + pure decode helpers
├── crop_collector.py                         NEW  per-track 3-crop collection
├── anpr_client.py                            MOD  re-exports PlateResult (deleted in Task 11)
├── anpr_worker.py                            MOD  job = (event_id, [crops]) → read_plate_multi
└── pipeline.py                               MOD  optional collector, ticked each frame
car_logger/
├── config.py                                 MOD  model paths + vote knobs (API key removed in Task 11)
├── main.py                                   MOD  wiring: engines, collector, version 0.6.0
├── models.py                                 MOD  status comment only
├── repositories.py                           MOD  list_events(only_read=)
├── api/routes_dashboard.py                   MOD  filter param on events-feed
└── templates/ (dashboard.html, partials/macros.html)  MOD  toggle + no_plate badge
deployment/car-logger.service                 MOD  Environment=OPENBLAS_CORETYPE=ARMV8
requirements.txt                              MOD  + onnxruntime==1.9.0, numpy==1.19.5 (− httpx in Task 11)
docs/v2-stage-b-validation-log.md             NEW  the window's evidence table
tests/unit/: test_plate_voting.py, test_local_anpr.py, test_onnx_engines.py,
             test_crop_collector.py            NEW
tests/unit/: test_anpr_worker.py, test_pipeline_resilience.py,
             test_repositories.py, test_config.py  MOD
tests/integration/test_dashboard.py           MOD  filter + badge
```

---

### Task 1: The models enter git (`models/anpr/`) — LAPTOP only

**Files:**
- Create: `models/anpr/cct_xs_v2_global.onnx` (copied from cache)
- Create: `models/anpr/cct_xs_v2_global_plate_config.yaml` (copied)
- Create: `models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx` (generated)
- Create: `models/anpr/README.md`

**Interfaces:**
- Consumes: laptop caches `C:\Users\40747\.cache\fast-plate-ocr\cct-xs-v2-global-model\` and `C:\Users\40747\.cache\open-image-models\yolo-v9-t-384-license-plate-end2end\` (left by the Stage A bake-off).
- Produces: the three model files at the exact paths above — Task 9's config defaults and Task 10's smoke test use these paths verbatim.

- [x] **Step 1: Copy the OCR model + config from the laptop cache**

Run (laptop, git-bash, from repo root):
```bash
mkdir -p models/anpr
cp "/c/Users/40747/.cache/fast-plate-ocr/cct-xs-v2-global-model/cct_xs_v2_global.onnx" models/anpr/
cp "/c/Users/40747/.cache/fast-plate-ocr/cct-xs-v2-global-model/cct_xs_v2_global_plate_config.yaml" models/anpr/
ls -la models/anpr/
```
Expected: two files, ~3.3 MB + ~1.7 KB.

- [x] **Step 2: Re-stamp the detector to opset 15**

The hub detector is opset 17; ORT 1.9 (the Jetson's last cp36 wheel)
supports ≤15. The Stage A spike proved the conversion is output-identical.

Run (laptop):
```bash
.venv/Scripts/pip install onnx
.venv/Scripts/python - <<'PY'
import onnx
from onnx import version_converter
SRC = r"C:\Users\40747\.cache\open-image-models\yolo-v9-t-384-license-plate-end2end\yolo-v9-t-384-license-plates-end2end.onnx"
DST = "models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx"
model = onnx.load(SRC)
print("original opset:", model.opset_import[0].version)
converted = version_converter.convert_version(model, 15)
onnx.save(converted, DST)
print("saved:", DST, "opset:", converted.opset_import[0].version)
PY
```
Expected: `original opset: 17`, `saved: ... opset: 15`.

- [x] **Step 3: Verify the opset-15 model is output-identical to the original**

Run (laptop — `.venv` has onnxruntime via `fast-alpr[onnx]` from Stage A):
```bash
.venv/Scripts/python - <<'PY'
import numpy as np
import onnxruntime as ort
SRC = r"C:\Users\40747\.cache\open-image-models\yolo-v9-t-384-license-plate-end2end\yolo-v9-t-384-license-plates-end2end.onnx"
DST = "models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx"
blob = np.random.RandomState(0).rand(1, 3, 384, 384).astype("float32")
outs = []
for path in (SRC, DST):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs.append(sess.run(None, {sess.get_inputs()[0].name: blob}))
same = all(np.array_equal(a, b) for a, b in zip(outs[0], outs[1]))
print("bit-identical outputs:", same)
assert same
PY
```
Expected: `bit-identical outputs: True`.

- [x] **Step 4: Write `models/anpr/README.md`**

```markdown
# Local ANPR models (v2 Stage B)

Committed on purpose: clone + pip install = working appliance, no runtime
downloads (student decision, 2026-07-19). ~11 MB total.

| file | role | origin |
|---|---|---|
| `yolo-v9-t-384-license-plates-end2end-opset15.onnx` | plate detector | ankandrew/open-image-models hub model `yolo-v9-t-384-license-plate-end2end`, re-stamped opset 17 → 15 with `onnx.version_converter` (outputs verified bit-identical; the Jetson's onnxruntime 1.9 supports opset ≤ 15) |
| `cct_xs_v2_global.onnx` | OCR — plate text + region head | ankandrew/fast-plate-ocr hub model `cct-xs-v2-global-model` (already opset 15) |
| `cct_xs_v2_global_plate_config.yaml` | OCR preprocessing config | ships with the OCR model |

Both upstream projects are MIT-licensed. Chosen by the Stage A bake-off
(`experiments/anpr_bakeoff/RESULTS.md`): 93.5% exact-match on
eu_benchmark, 100% on our real crops, ~337 ms/crop + 110 MB on the
Jetson CPU. Do NOT swap or upgrade these files without re-running the
bake-off.
```

- [x] **Step 5: Commit**

```bash
git add models/anpr
git commit -m "feat(v2b): ship the bake-off winner's ONNX models in git

opset-15 re-stamped detector (verified bit-identical) + cct-xs-v2-global
OCR + config. Repo becomes self-sufficient: clone + pip install = app.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `PlateResult` moves to a neutral module

The namedtuple lives in `anpr_client.py`, which Task 11 deletes;
`anpr_worker.py` imports it from there. Move first, delete later.

**Files:**
- Create: `car_logger/services/plate_result.py`
- Modify: `car_logger/services/anpr_client.py:15-24`
- Modify: `car_logger/services/anpr_worker.py:13`

**Interfaces:**
- Produces: `from car_logger.services.plate_result import PlateResult` —
  every later task imports it from here. Fields unchanged:
  `(plate_text, confidence, status, region)`.

- [x] **Step 1: Create `car_logger/services/plate_result.py`**

```python
"""The one result type every ANPR engine speaks.

Lives in its own module so engines can come and go (cloud client in v1,
local ONNX stack in v2) without the worker or the callbacks caring where
a result came from."""

from collections import namedtuple

PlateResult = namedtuple(
    "PlateResult", ["plate_text", "confidence", "status", "region"]
)  # status: success | failed | no_plate | throttled | skipped
```

- [x] **Step 2: Re-export from `anpr_client.py`**

Replace lines 15-24 (the `time`/`namedtuple` imports and the namedtuple
definition) so the top of the file reads:

```python
import time

import httpx

from car_logger.services.plate_result import PlateResult  # noqa: F401 (re-export: v1 call sites import it from here)
from car_logger.services.plate_rules import normalize_plate
```

- [x] **Step 3: Point the worker at the new home**

In `anpr_worker.py` replace
`from car_logger.services.anpr_client import PlateResult` with:

```python
from car_logger.services.plate_result import PlateResult
```

- [x] **Step 4: Full suite green (pure refactor — no behavior change)**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest -q`
Expected: `91 passed`.

- [x] **Step 5: Commit**

```bash
git add car_logger/services/plate_result.py car_logger/services/anpr_client.py car_logger/services/anpr_worker.py
git commit -m "refactor(v2b): PlateResult moves to a neutral module

anpr_client.py gets deleted at the end of Stage B; the result type it
defined outlives it. Cloud client re-exports for compatibility.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `plate_voting.py` — the vote as a pure function (TDD)

**Files:**
- Create: `car_logger/services/plate_voting.py`
- Test: `tests/unit/test_plate_voting.py`

**Interfaces:**
- Consumes: `PlateResult` from Task 2.
- Produces: `vote_on_reads(reads) -> (PlateResult, winner_index)` where
  `reads` is a list of per-crop `PlateResult`s (texts already
  normalized) and `winner_index` is the index of the read whose
  confidence/region the verdict carries (0 when there is no winner).
  Task 4's `read_plate_multi` uses `winner_index` to pick the evidence
  crop.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_plate_voting.py`:

```python
"""The vote is Stage B's real error filter — the bake-off proved the
OCR's confidence cannot tell right from wrong (all 7 wrong reads sat at
conf >= 0.9997). Every rule here is a student decision from 2026-07-19;
see the Stage B spec."""

from car_logger.services.plate_result import PlateResult
from car_logger.services.plate_voting import vote_on_reads


def _ok(text, conf=0.99, region="ro"):
    return PlateResult(text, conf, "success", region)


def _no_plate():
    return PlateResult(None, None, "no_plate", None)


def _failed():
    return PlateResult(None, None, "failed", None)


def test_three_identical_reads_win():
    result, _ = vote_on_reads([_ok("CJ45ARL"), _ok("CJ45ARL"), _ok("CJ45ARL")])
    assert result.status == "success"
    assert result.plate_text == "CJ45ARL"


def test_two_of_three_agreeing_beat_the_odd_one_out():
    result, _ = vote_on_reads([_ok("CJ45ARL"), _ok("CJ45ARI"), _ok("CJ45ARL")])
    assert result.status == "success"
    assert result.plate_text == "CJ45ARL"


def test_three_different_texts_fail():
    result, _ = vote_on_reads([_ok("AAA111"), _ok("BBB222"), _ok("CCC333")])
    assert result.status == "failed"
    assert result.plate_text is None


def test_no_plate_abstains_so_a_single_text_is_accepted():
    result, _ = vote_on_reads([_ok("CJ45ARL"), _no_plate(), _no_plate()])
    assert result.status == "success"
    assert result.plate_text == "CJ45ARL"


def test_two_way_tie_fails():
    result, _ = vote_on_reads([_ok("AAA111"), _ok("BBB222"), _no_plate()])
    assert result.status == "failed"


def test_single_read_is_accepted():
    # graceful degradation to v1's single-read behavior (fast cars)
    result, _ = vote_on_reads([_ok("CJ45ARL")])
    assert result.status == "success"


def test_all_no_plate_is_no_plate():
    result, _ = vote_on_reads([_no_plate(), _no_plate(), _no_plate()])
    assert result.status == "no_plate"


def test_technical_failures_without_texts_fail():
    result, _ = vote_on_reads([_failed(), _no_plate(), _failed()])
    assert result.status == "failed"


def test_empty_input_fails():
    result, _ = vote_on_reads([])
    assert result.status == "failed"


def test_verdict_carries_max_confidence_of_the_agreeing_reads():
    reads = [_ok("CJ45ARL", 0.91), _ok("CJ45ARL", 0.97), _ok("XX99XXX", 0.99)]
    result, winner_index = vote_on_reads(reads)
    assert result.confidence == 0.97
    assert winner_index == 1


def test_verdict_region_comes_from_the_winning_read():
    reads = [PlateResult("CJ45ARL", 0.91, "success", None),
             PlateResult("CJ45ARL", 0.97, "success", "ro")]
    result, _ = vote_on_reads(reads)
    assert result.region == "ro"
```

- [x] **Step 2: RED**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_plate_voting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'car_logger.services.plate_voting'`.

- [x] **Step 3: Implement `car_logger/services/plate_voting.py`**

```python
"""Multi-frame vote: N reads of one track -> one verdict. Pure function.

STUDENT DECISIONS (2026-07-19, Stage B spec):
- a text with >= 2 votes wins;
- no_plate reads and technical failures abstain (carry no text);
- no majority but exactly ONE distinct text seen -> accept it (a single
  usable read degrades gracefully to v1's single-read behavior);
- no majority and >= 2 distinct texts -> failed (can't break the tie);
- zero texts: all usable reads said no_plate -> no_plate; anything with
  a technical failure in the mix (or an empty list) -> failed."""

from car_logger.services.plate_result import PlateResult


def vote_on_reads(reads):
    """Return (PlateResult, winner_index).

    winner_index points at the read whose confidence/region the verdict
    carries — the caller saves that crop as the event's visual evidence.
    It is 0 when there is no winning read."""
    votes = {}
    saw_no_plate = False
    saw_technical = False
    for i, read in enumerate(reads):
        if read.status == "success" and read.plate_text:
            votes.setdefault(read.plate_text, []).append(i)
        elif read.status == "no_plate":
            saw_no_plate = True
        else:
            saw_technical = True

    if votes:
        best_text = max(votes, key=lambda text: len(votes[text]))
        indexes = votes[best_text]
        if len(indexes) >= 2 or len(votes) == 1:
            winner = max(indexes,
                         key=lambda i: reads[i].confidence or 0.0)
            best = reads[winner]
            return (PlateResult(best_text, best.confidence, "success",
                                best.region), winner)
        return (PlateResult(None, None, "failed", None), 0)

    if saw_no_plate and not saw_technical:
        return (PlateResult(None, None, "no_plate", None), 0)
    return (PlateResult(None, None, "failed", None), 0)
```

- [x] **Step 4: GREEN**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_plate_voting.py -v`
Expected: 12 passed. Then full suite: `venv/bin/pytest -q` → 103 passed.

- [x] **Step 5: Commit**

```bash
git add car_logger/services/plate_voting.py tests/unit/test_plate_voting.py
git commit -m "feat(v2b): multi-frame vote as a pure function (TDD)

2-of-3 agreement wins; no_plate abstains; single usable read accepted;
ties and text-less failures are honest 'failed'. The winner index picks
the evidence crop.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `LocalAnprClient` — two stages, injected engines (TDD)

**Files:**
- Create: `car_logger/services/local_anpr.py`
- Test: `tests/unit/test_local_anpr.py`

**Interfaces:**
- Consumes: `PlateResult` (Task 2), `vote_on_reads` (Task 3),
  `normalize_plate` (existing). Engine duck-types:
  `detector_engine.detect_plate(image_bgr) -> (x1, y1, x2, y2) or None`;
  `ocr_engine.read(plate_bgr) -> (text, confidence, region_code_or_None)`;
  both optionally `close()`.
- Produces: `LocalAnprClient(detector_engine, ocr_engine)` with
  `read_plate(image_bytes) -> PlateResult`,
  `read_plate_multi(crops) -> (PlateResult, evidence_bytes)`, `close()`.
  Task 7's worker calls `read_plate_multi`; Task 5's engines implement
  the duck-types; Task 9 wires it all in `main.py`.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_local_anpr.py`:

```python
"""LocalAnprClient with fake engines: no models, no Jetson, no network —
the same injected-dependency pattern the v1 cloud client used with its
injected httpx client."""

import cv2
import numpy as np

from car_logger.services.local_anpr import LocalAnprClient
from car_logger.services.plate_result import PlateResult


def _jpeg(width=64, height=48):
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    assert ok
    return buf.tobytes()


class FakeDetector(object):
    def __init__(self, box):
        self.box = box
        self.calls = 0

    def detect_plate(self, image_bgr):
        self.calls += 1
        return self.box


class FakeOcr(object):
    def __init__(self, text="CJ 45 ARL", confidence=0.97, region="ro",
                 raises=False):
        self.text = text
        self.confidence = confidence
        self.region = region
        self.raises = raises
        self.calls = 0

    def read(self, plate_bgr):
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return self.text, self.confidence, self.region


class SeqOcr(object):
    """Returns a different (text, conf, region) per call — one per crop."""

    def __init__(self, reads):
        self._reads = list(reads)

    def read(self, plate_bgr):
        return self._reads.pop(0)


def test_happy_path_normalizes_text_and_carries_region():
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)), FakeOcr())
    result = client.read_plate(_jpeg())
    assert result == PlateResult("CJ45ARL", 0.97, "success", "ro")


def test_no_plate_found_never_calls_ocr():
    ocr = FakeOcr()
    client = LocalAnprClient(FakeDetector(None), ocr)
    result = client.read_plate(_jpeg())
    assert result.status == "no_plate"
    assert ocr.calls == 0  # the "OCR only when a plate is found" requirement


def test_engine_exception_becomes_failed_never_raises():
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)),
                             FakeOcr(raises=True))
    result = client.read_plate(_jpeg())
    assert result.status == "failed"


def test_corrupt_image_bytes_become_failed():
    client = LocalAnprClient(FakeDetector((0, 0, 1, 1)), FakeOcr())
    result = client.read_plate(b"definitely not a jpeg")
    assert result.status == "failed"


def test_empty_ocr_text_is_failed_plate_seen_but_unreadable():
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)), FakeOcr(text=""))
    result = client.read_plate(_jpeg())
    assert result.status == "failed"


def test_read_plate_multi_votes_and_returns_the_winning_crop():
    crop_a, crop_b, crop_c = _jpeg(64), _jpeg(66), _jpeg(68)  # distinct bytes
    ocr = SeqOcr([("CJ45ARL", 0.91, "ro"),
                  ("CJ45ARI", 0.99, "ro"),
                  ("CJ45ARL", 0.97, "ro")])
    client = LocalAnprClient(FakeDetector((2, 2, 30, 20)), ocr)
    result, evidence = client.read_plate_multi([crop_a, crop_b, crop_c])
    assert result.plate_text == "CJ45ARL"
    assert result.confidence == 0.97   # max among the agreeing reads
    assert evidence == crop_c          # the winning read's crop


def test_read_plate_multi_empty_list_is_failed():
    client = LocalAnprClient(FakeDetector(None), FakeOcr())
    result, evidence = client.read_plate_multi([])
    assert result.status == "failed"
    assert evidence == b""


def test_close_is_safe_with_and_without_engine_close():
    class Closable(FakeDetector):
        def __init__(self):
            FakeDetector.__init__(self, None)
            self.closed = False

        def close(self):
            self.closed = True

    detector = Closable()
    client = LocalAnprClient(detector, FakeOcr())  # FakeOcr has no close()
    client.close()
    assert detector.closed is True
```

- [x] **Step 2: RED**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_local_anpr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'car_logger.services.local_anpr'`.

- [x] **Step 3: Implement `car_logger/services/local_anpr.py`**

```python
"""Local two-stage ANPR client: plate detection -> OCR -> vote.

Drop-in replacement for the v1 cloud client behind the same PlateResult
contract. Engines are injected so unit tests run with fakes — no models,
no Jetson (the same pattern as the injected httpx client before it).

"Never raises" contract preserved: corrupt image, missing model file,
engine exception -> PlateResult(None, None, "failed", None); the worker
loop's defense-in-depth stays as the second net."""

import logging

import cv2
import numpy as np

from car_logger.services.plate_result import PlateResult
from car_logger.services.plate_rules import normalize_plate
from car_logger.services.plate_voting import vote_on_reads

log = logging.getLogger(__name__)


class LocalAnprClient(object):
    def __init__(self, detector_engine, ocr_engine):
        self._detector = detector_engine
        self._ocr = ocr_engine

    def close(self):
        """Release the engines (symmetric with the v1 client's close)."""
        for engine in (self._detector, self._ocr):
            close = getattr(engine, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:
                log.exception("engine close failed")

    def read_plate(self, image_bytes):
        """One vehicle crop (JPEG bytes) -> one PlateResult. Never raises."""
        try:
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                return PlateResult(None, None, "failed", None)
            box = self._detector.detect_plate(image)
            if box is None:
                # Stage 1 found nothing -> OCR never runs (spec requirement)
                return PlateResult(None, None, "no_plate", None)
            x1, y1, x2, y2 = box
            text, confidence, region = self._ocr.read(image[y1:y2, x1:x2])
            text = normalize_plate(text)
            if not text:
                # a plate was seen but nothing decodable came off it
                return PlateResult(None, None, "failed", None)
            return PlateResult(text, confidence, "success", region)
        except Exception:
            log.exception("local ANPR read failed")
            return PlateResult(None, None, "failed", None)

    def read_plate_multi(self, crops):
        """N crops of one track -> (verdict, evidence_bytes). Never raises.

        evidence_bytes is the winning read's crop — the image saved for
        the event matches the text it claims; first crop otherwise."""
        try:
            reads = [self.read_plate(crop) for crop in crops]
            result, winner_index = vote_on_reads(reads)
            evidence = crops[winner_index] if crops else b""
            return result, evidence
        except Exception:
            log.exception("local ANPR multi-read failed")
            return (PlateResult(None, None, "failed", None),
                    crops[0] if crops else b"")
```

- [x] **Step 4: GREEN**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_local_anpr.py -v`
Expected: 8 passed. Full suite: 111 passed.

- [x] **Step 5: Commit**

```bash
git add car_logger/services/local_anpr.py tests/unit/test_local_anpr.py
git commit -m "feat(v2b): LocalAnprClient - two-stage local read behind the v1 contract (TDD)

Injected engines, never-raises, OCR only when a plate is found,
read_plate_multi votes and returns the winning crop as evidence.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `onnx_engines.py` — the real detector + OCR (TDD on the pure decode helpers)

onnxruntime is imported **lazily in the constructors** (the same pattern
as `detector.py`'s jetson imports) so this module imports fine on
machines without ORT; the decode logic lives in pure helpers and that is
what the unit tests cover. The real models are exercised on the Jetson
in Task 10's smoke test.

**Files:**
- Create: `car_logger/services/onnx_engines.py`
- Test: `tests/unit/test_onnx_engines.py`

**Interfaces:**
- Consumes: model/config paths (Task 1's files, via Task 9's settings).
- Produces: `OnnxPlateDetector(model_path, threshold=0.4)` and
  `OnnxPlateOcr(model_path, config_path)` implementing Task 4's engine
  duck-types; pure helpers `best_detection`, `decode_ocr_outputs`,
  `region_to_code`.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_onnx_engines.py`:

```python
"""Pure decode helpers — no onnxruntime, no model files. The shapes and
conventions are facts fixed by the Stage A spike (2026-07-18): detector
rows are [image_id, x1, y1, x2, y2, class_id, score] in 384x384
plain-resize space; OCR heads may come flattened and in any order, so
they are matched by element count."""

import numpy as np

from car_logger.services.onnx_engines import (best_detection,
                                              decode_ocr_outputs,
                                              region_to_code)


def test_best_detection_picks_highest_score_and_scales_back():
    rows = np.array([[0, 10, 10, 100, 50, 0, 0.5],
                     [0, 20, 20, 120, 60, 0, 0.9]], dtype="float32")
    box = best_detection([rows], orig_width=768, orig_height=768,
                         input_side=384, threshold=0.4)
    assert box == (40, 40, 240, 120)  # 768/384 = 2x scale, second row wins


def test_best_detection_below_threshold_is_none():
    rows = np.array([[0, 10, 10, 100, 50, 0, 0.3]], dtype="float32")
    assert best_detection([rows], 768, 768, 384, 0.4) is None


def test_best_detection_accepts_batched_output():
    rows = np.array([[[0, 10, 10, 100, 50, 0, 0.9]]], dtype="float32")
    assert best_detection([rows], 384, 384, 384, 0.4) == (10, 10, 100, 50)


def test_best_detection_degenerate_box_is_none():
    rows = np.array([[0, 50, 50, 50, 50, 0, 0.9]], dtype="float32")
    assert best_detection([rows], 384, 384, 384, 0.4) is None


def test_decode_ocr_argmax_strips_pad_and_reads_the_region_head():
    cfg = {"max_plate_slots": 2, "alphabet": "AB_", "pad_char": "_",
           "plate_regions": ["Romania", "Unknown"]}
    chars = np.zeros((1, 2, 3), dtype="float32")
    chars[0, 0, 0] = 0.9   # slot 0 -> 'A'
    chars[0, 1, 2] = 0.8   # slot 1 -> pad, stripped from the text
    region = np.array([[0.7, 0.3]], dtype="float32")
    text, confidence, region_name = decode_ocr_outputs([chars, region], cfg)
    assert text == "A"
    assert region_name == "Romania"
    assert abs(confidence - (0.9 + 0.8) / 2.0) < 1e-6


def test_decode_ocr_handles_flattened_heads_and_no_region():
    cfg = {"max_plate_slots": 2, "alphabet": "AB_", "pad_char": "_"}
    flat = np.zeros((1, 6), dtype="float32")
    flat[0, 0] = 1.0   # slot 0 -> 'A'
    flat[0, 4] = 1.0   # slot 1 -> 'B'
    text, confidence, region_name = decode_ocr_outputs([flat], cfg)
    assert text == "AB"
    assert region_name is None


def test_region_to_code_student_mapping():
    assert region_to_code("Romania") == "ro"       # the RO gate fires on this
    assert region_to_code("Unknown") is None
    assert region_to_code(None) is None
    assert region_to_code("Czech Republic") == "czech republic"
```

- [x] **Step 2: RED**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_onnx_engines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'car_logger.services.onnx_engines'`.

- [x] **Step 3: Implement `car_logger/services/onnx_engines.py`**

```python
"""ONNX engines for the local ANPR: YOLO plate detector + CCT OCR.

onnxruntime is imported lazily in the constructors (the same pattern as
detector.py's jetson imports) so importing this module never needs ORT;
the pure helpers below hold the decode logic and carry the unit tests.

Facts fixed by the Stage A spike (2026-07-18, RESULTS.md):
- detector input (1, 3, 384, 384) float32 RGB /255, PLAIN resize — the
  output coordinates come back in that space and are scaled back here;
- detector output rows: [image_id, x1, y1, x2, y2, class_id, score];
- OCR input per its yaml config (RGB 128x64 NHWC for the global model);
  output heads: per-slot char probabilities + a plate_regions head,
  possibly flattened, matched by element count;
- the OCR must see the TIGHT detector crop — whole frames decode garbage.
"""

import logging

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)


def best_detection(outputs, orig_width, orig_height, input_side, threshold):
    """Decode detector outputs -> best (x1, y1, x2, y2) in original-image
    coordinates, or None when nothing clears the threshold."""
    rows = np.asarray(outputs[0]).reshape(-1, 7)
    rows = rows[rows[:, 6] >= threshold]
    if rows.shape[0] == 0:
        return None
    best = rows[np.argmax(rows[:, 6])]
    scale_x = orig_width / float(input_side)
    scale_y = orig_height / float(input_side)
    x1 = max(0, int(best[1] * scale_x))
    y1 = max(0, int(best[2] * scale_y))
    x2 = min(orig_width, int(best[3] * scale_x))
    y2 = min(orig_height, int(best[4] * scale_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def decode_ocr_outputs(outputs, config):
    """Decode OCR heads -> (text, confidence, region_name_or_None)."""
    slots = config["max_plate_slots"]
    alphabet = config["alphabet"]
    pad = config.get("pad_char", "_")
    regions = config.get("plate_regions") or []
    char_probs = None
    region_probs = None
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.size == slots * len(alphabet):
            char_probs = arr.reshape(slots, len(alphabet))
        elif regions and arr.size == len(regions):
            region_probs = arr.reshape(len(regions))
    if char_probs is None:
        raise ValueError("no OCR output has {0}x{1} elements".format(
            slots, len(alphabet)))
    indexes = char_probs.argmax(axis=-1)
    text = "".join(alphabet[i] for i in indexes).replace(pad, "")
    confidence = float(char_probs.max(axis=-1).mean())
    region_name = None
    if region_probs is not None:
        region_name = regions[int(region_probs.argmax())]
    return text, confidence, region_name


def region_to_code(region_name):
    """Country name from the OCR's region head -> our region code.

    STUDENT DECISION (2026-07-19): Romania -> "ro" (the RO regex gate in
    should_create_vehicle fires only on "ro"); "Unknown" -> None; any
    other country -> its lowercased name, kept as information."""
    if not region_name or region_name == "Unknown":
        return None
    if region_name == "Romania":
        return "ro"
    return region_name.lower()


class OnnxPlateDetector(object):
    """Stage 1: find the plate inside a vehicle crop. CPU-only."""

    def __init__(self, model_path, threshold=0.4):
        import onnxruntime  # lazy: only the Jetson venv carries ORT 1.9
        self.threshold = threshold
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        meta = self._session.get_inputs()[0]
        self._input_name = meta.name
        self._side = int(meta.shape[-1])

    def detect_plate(self, image_bgr):
        height, width = image_bgr.shape[:2]
        blob = cv2.resize(image_bgr, (self._side, self._side))
        blob = blob.astype("float32") / 255.0
        blob = blob[:, :, ::-1].transpose(2, 0, 1)[np.newaxis]  # BGR->RGB, NCHW
        outputs = self._session.run(
            None, {self._input_name: np.ascontiguousarray(blob)})
        return best_detection(outputs, width, height, self._side,
                              self.threshold)

    def close(self):
        self._session = None


class OnnxPlateOcr(object):
    """Stage 2: read the text (and region) off a tight plate crop."""

    def __init__(self, model_path, config_path):
        import onnxruntime
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        meta = self._session.get_inputs()[0]
        self._input_name = meta.name
        self._wants_float = meta.type == "tensor(float)"

    def read(self, plate_bgr):
        cfg = self.config
        if cfg.get("image_color_mode", "grayscale") == "rgb":
            img = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (cfg["img_width"], cfg["img_height"]))
        if img.ndim == 2:
            arr = img[np.newaxis, :, :, np.newaxis]  # NHWC, gray
        else:
            arr = img[np.newaxis, :, :, :]           # NHWC, rgb
        arr = arr.astype("float32" if self._wants_float else "uint8")
        outputs = self._session.run(None, {self._input_name: arr})
        text, confidence, region_name = decode_ocr_outputs(outputs, cfg)
        return text, confidence, region_to_code(region_name)

    def close(self):
        self._session = None
```

- [x] **Step 4: GREEN**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_onnx_engines.py -v`
Expected: 7 passed. Full suite: 118 passed.

- [x] **Step 5: Commit**

```bash
git add car_logger/services/onnx_engines.py tests/unit/test_onnx_engines.py
git commit -m "feat(v2b): ONNX detector + OCR engines with pure decode helpers (TDD)

Lazy ORT import (detector.py pattern); decode conventions are the Stage A
spike facts; region head mapped Romania->ro, Unknown->None.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `CropCollector` — 3 crops per track (TDD)

**Files:**
- Create: `car_logger/services/crop_collector.py`
- Test: `tests/unit/test_crop_collector.py`

**Interfaces:**
- Consumes: `crop_to_jpeg(frame, box)` (existing, injectable); tracker
  `Track` objects with `.track_id`, `.box`, `.missed` (existing).
- Produces: `CropCollector(on_complete, reads_per_track=3,
  spacing_s=0.4, crop_fn=None, now=None)` with
  `start(track_id, event_id, box, frame)`,
  `tick(live_tracks, frame)`, `drain()`.
  `on_complete(event_id, crops_list)` fires exactly once per event.
  Task 9 wires `start` into `on_confirmed` and `tick` into the pipeline.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_crop_collector.py`:

```python
"""The collector implements the spec's Variant 1: crop #1 at track
confirmation, then up to 2 more from later frames spaced >= spacing_s
apart; hand the list over when full or when the track dies. Injectable
clock + crop_fn = deterministic tests, no cv2, no camera."""

from car_logger.services.crop_collector import CropCollector


class FakeTrack(object):
    def __init__(self, track_id, box=(0, 0, 10, 10), missed=0):
        self.track_id = track_id
        self.box = box
        self.missed = missed


class Clock(object):
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _crop_fn(frame, box):
    return (frame, box)  # opaque token; the collector never looks inside


def _collector(calls, clock, reads=3, spacing=0.4):
    return CropCollector(
        on_complete=lambda event_id, crops: calls.append((event_id, crops)),
        reads_per_track=reads, spacing_s=spacing,
        crop_fn=_crop_fn, now=clock)


def test_collects_three_spaced_crops_then_completes_once():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    track = FakeTrack(7)
    collector.start(7, event_id=42, box=track.box, frame="f0")

    collector.tick([track], "f1")          # too soon: only 0.0s elapsed
    assert calls == []
    clock.advance(0.4)
    collector.tick([track], "f2")          # crop #2
    clock.advance(0.4)
    collector.tick([track], "f3")          # crop #3 -> complete
    assert len(calls) == 1
    event_id, crops = calls[0]
    assert event_id == 42
    assert crops == [("f0", (0, 0, 10, 10)), ("f2", (0, 0, 10, 10)),
                     ("f3", (0, 0, 10, 10))]

    clock.advance(1.0)
    collector.tick([track], "f4")          # cleaned up: nothing re-fires
    assert len(calls) == 1


def test_track_death_hands_over_what_it_has():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    collector.start(7, event_id=42, box=(0, 0, 10, 10), frame="f0")
    collector.tick([], "f1")               # the track is gone
    assert calls == [(42, [("f0", (0, 0, 10, 10))])]


def test_missed_track_is_not_cropped_stale_box():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    track = FakeTrack(7, missed=2)
    collector.start(7, event_id=42, box=track.box, frame="f0")
    clock.advance(1.0)
    collector.tick([track], "f1")          # box is stale -> no crop taken
    assert calls == []
    track.missed = 0
    collector.tick([track], "f2")          # fresh again -> crop #2
    clock.advance(0.4)
    collector.tick([track], "f3")          # crop #3 -> complete
    assert len(calls) == 1
    assert len(calls[0][1]) == 3


def test_drain_flushes_partials():
    calls, clock = [], Clock()
    collector = _collector(calls, clock)
    collector.start(7, event_id=42, box=(0, 0, 10, 10), frame="f0")
    collector.drain()
    assert calls == [(42, [("f0", (0, 0, 10, 10))])]
    collector.drain()                      # idempotent
    assert len(calls) == 1


def test_single_read_config_completes_immediately():
    calls, clock = [], Clock()
    collector = _collector(calls, clock, reads=1)
    collector.start(7, event_id=42, box=(0, 0, 10, 10), frame="f0")
    assert calls == [(42, [("f0", (0, 0, 10, 10))])]
```

- [x] **Step 2: RED**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_crop_collector.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Implement `car_logger/services/crop_collector.py`**

```python
"""Per-track crop collection for the multi-frame vote (Stage B spec,
Variant 1): crop #1 at confirmation, then up to reads_per_track-1 more
from later frames spaced spacing_s apart — consecutive frames are nearly
identical and would fail identically; spacing decorrelates the reads.
Hands the crop list to on_complete when full or when the track dies.

State is one bounded dict keyed by track_id, cleaned on completion and
on track death — it can never outgrow the tracker's own track list."""

import time

from car_logger.services.cropping import crop_to_jpeg


class _Collection(object):
    def __init__(self, event_id, first_crop, taken_at):
        self.event_id = event_id
        self.crops = [first_crop]
        self.last_taken_at = taken_at


class CropCollector(object):
    def __init__(self, on_complete, reads_per_track=3, spacing_s=0.4,
                 crop_fn=None, now=None):
        self._on_complete = on_complete
        self._reads_per_track = reads_per_track
        self._spacing_s = spacing_s
        self._crop_fn = crop_fn if crop_fn is not None else crop_to_jpeg
        self._now = now if now is not None else time.monotonic
        self._pending = {}

    def start(self, track_id, event_id, box, frame):
        """Take crop #1 at confirmation; register the collection.

        Completes immediately when the config asks for a single read."""
        crop = self._crop_fn(frame, box)
        if self._reads_per_track <= 1:
            self._on_complete(event_id, [crop])
            return
        self._pending[track_id] = _Collection(event_id, crop, self._now())

    def tick(self, live_tracks, frame):
        """Called once per pipeline tick with the tracker's live tracks."""
        live = dict((t.track_id, t) for t in live_tracks)
        for track_id in list(self._pending):
            collection = self._pending[track_id]
            track = live.get(track_id)
            if track is None:
                # track died — vote with what we have (student decision)
                del self._pending[track_id]
                self._on_complete(collection.event_id, collection.crops)
                continue
            if track.missed > 0:
                continue  # box is stale; cropping now would frame empty road
            if self._now() - collection.last_taken_at < self._spacing_s:
                continue
            collection.crops.append(self._crop_fn(frame, track.box))
            collection.last_taken_at = self._now()
            if len(collection.crops) >= self._reads_per_track:
                del self._pending[track_id]
                self._on_complete(collection.event_id, collection.crops)

    def drain(self):
        """Shutdown path: flush partial collections so their events still
        get a result (they become 'skipped' via the worker's own drain)."""
        for track_id in list(self._pending):
            collection = self._pending.pop(track_id)
            self._on_complete(collection.event_id, collection.crops)
```

- [x] **Step 4: GREEN**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_crop_collector.py -v`
Expected: 5 passed. Full suite: 123 passed.

- [x] **Step 5: Commit**

```bash
git add car_logger/services/crop_collector.py tests/unit/test_crop_collector.py
git commit -m "feat(v2b): per-track CropCollector - 3 spaced crops, one handover (TDD)

Track death hands over partials; stale (missed) boxes are never cropped;
drain() flushes at shutdown. Injectable clock+crop_fn for hardware-free tests.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `AnprWorker` speaks crop lists

One job stays one car — only the payload grows from one crop to a list,
and the client call becomes `read_plate_multi`. Queue semantics, drop
policy, drain-as-skipped, defense-in-depth: all unchanged.

**Files:**
- Modify: `car_logger/services/anpr_worker.py`
- Test: `tests/unit/test_anpr_worker.py` (fakes updated to the new client API)

**Interfaces:**
- Consumes: `client.read_plate_multi(crops) -> (result, evidence_bytes)`
  (Task 4), `client.close()`.
- Produces: `submit(event_id, crops)` where `crops` is a list of JPEG
  bytes; `on_result(event_id, result, evidence_bytes)` — the existing
  `_make_on_result` signature, untouched.

- [x] **Step 1: Update the tests to the new client API**

Replace the fakes and submits in `tests/unit/test_anpr_worker.py` so the
whole file reads:

```python
"""The worker thread must survive ANY exception — a dead thread means every
later event stays 'pending' forever, silently. Found in the stage 4 offline
test. Results are polled with a deadline (no fixed sleeps) to avoid races.

v2 (Stage B): a job is (event_id, [crops]) — one job per car — and the
client call is read_plate_multi, which returns (result, evidence_crop)."""

import time

from car_logger.services.anpr_worker import AnprWorker


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FlakyClient(object):
    """First call raises (like ConnectError did); second call succeeds."""

    def __init__(self):
        self.calls = 0

    def read_plate_multi(self, crops):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return ("OK", 0.9, "success"), crops[0]

    def close(self):
        pass


def test_worker_survives_client_exception():
    got = []
    worker = AnprWorker(FlakyClient(), lambda eid, r, c: got.append((eid, r)))
    worker.start()
    worker.submit(1, [b"a1", b"a2"])
    worker.submit(2, [b"b1"])
    assert _wait_for(lambda: len(got) == 2), "thread died after the exception"
    worker.stop()
    # job 1: the client blew up -> the event still gets a 'failed' result
    assert got[0][0] == 1
    assert got[0][1].status == "failed"
    # job 2: processed normally by the SAME still-alive thread
    assert got[1] == (2, ("OK", 0.9, "success"))


class OkClient(object):
    def read_plate_multi(self, crops):
        return ("OK", 0.9, "success"), crops[0]

    def close(self):
        pass


def test_worker_survives_callback_exception():
    seen = []

    def bad_then_good(event_id, result, crop_bytes):
        seen.append(event_id)
        if len(seen) == 1:
            raise RuntimeError("db down")

    worker = AnprWorker(OkClient(), bad_then_good)
    worker.start()
    worker.submit(1, [b"a"])
    worker.submit(2, [b"b"])
    assert _wait_for(lambda: len(seen) == 2), "thread died in the callback"
    worker.stop()
    assert seen == [1, 2]


def test_worker_passes_the_evidence_crop_to_the_callback():
    got = []
    worker = AnprWorker(OkClient(), lambda eid, r, c: got.append(c))
    worker.start()
    worker.submit(1, [b"first", b"second"])
    assert _wait_for(lambda: len(got) == 1)
    worker.stop()
    assert got[0] == b"first"  # OkClient returns crops[0] as evidence


def test_stop_drains_pending_jobs_as_skipped_and_closes_client():
    # codex finding 7: jobs queued at shutdown (daily 04:00 restart!) must
    # not leave their events 'pending' forever.
    calls = []

    class ClosableClient(object):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = ClosableClient()
    worker = AnprWorker(
        client, lambda eid, res, crop: calls.append((eid, res.status)))
    worker.submit(1, [b"a"])
    worker.submit(2, [b"b"])
    worker.stop()  # never started: everything is still queued
    assert calls == [(1, "skipped"), (2, "skipped")]
    assert client.closed is True
```

- [x] **Step 2: RED**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_anpr_worker.py -v`
Expected: FAIL — the worker still calls `read_plate` (AttributeError on
the fakes) and unpacks a single result.

- [x] **Step 3: Adapt the worker**

In `car_logger/services/anpr_worker.py`, update the docstring's third
sentence and the three touched methods:

Module docstring becomes:
```python
"""Background ANPR worker: decouples the slow reads from the pipeline.

The pipeline (via the CropCollector) calls submit() with one car's crop
list and returns immediately. This worker thread pulls jobs off a bounded
queue, calls the ANPR client's multi-read (N crops -> one voted result +
the evidence crop), and hands the result to a callback (which persists
it). Under load the queue fills and submit() drops the job rather than
block the pipeline — a dropped plate read is acceptable; a stalled
pipeline is not."""
```

`submit` becomes:
```python
    def submit(self, event_id, crops):
        """Enqueue one car's crop list; return False if dropped (queue full)."""
        try:
            self._queue.put_nowait((event_id, crops))
            return True
        except queue.Full:
            return False
```

`_loop`'s body between `get` and `finally` becomes:
```python
            try:
                event_id, crops = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                try:
                    result, evidence = self._client.read_plate_multi(crops)
                except Exception:
                    log.exception(
                        "ANPR client raised for event %s; marking it failed",
                        event_id)
                    result = PlateResult(None, None, "failed", None)
                    evidence = crops[0] if crops else None
                try:
                    self._on_result(event_id, result, evidence)
                except Exception:
                    log.exception(
                        "on_result callback raised for event %s", event_id)
            finally:
                self._queue.task_done()
```

`stop`'s drain loop becomes:
```python
        while True:
            try:
                event_id, crops = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._on_result(
                    event_id, PlateResult(None, None, "skipped", None),
                    crops[0] if crops else None)
            except Exception:
                log.exception("drain: on_result raised for event %s",
                              event_id)
```

- [x] **Step 4: GREEN**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_anpr_worker.py tests/unit/test_on_result.py -v`
Expected: all pass — `test_on_result.py` UNTOUCHED and green (the
callback contract did not move). Full suite: 124 passed.

- [x] **Step 5: Commit**

```bash
git add car_logger/services/anpr_worker.py tests/unit/test_anpr_worker.py
git commit -m "feat(v2b): AnprWorker jobs carry one car's crop list (TDD)

read_plate_multi returns (verdict, evidence crop); queue/drop/drain
semantics and the on_result contract unchanged - on_result tests untouched.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Dashboard — `filter` param, toggle, `no_plate` badge (TDD)

**Files:**
- Modify: `car_logger/repositories.py:33-43` (list_events)
- Modify: `car_logger/api/routes_dashboard.py:50-57` (events_feed)
- Modify: `car_logger/templates/partials/macros.html` (status_badge)
- Modify: `car_logger/templates/dashboard.html:30-46` (controls + feed div)
- Modify: `car_logger/models.py:44` (status comment only)
- Test: `tests/unit/test_repositories.py` (append), `tests/integration/test_dashboard.py` (append)

**Interfaces:**
- Produces: `list_events(db, skip, limit, plate_text, only_read=False)`;
  `GET /partials/events-feed?filter=read|all&q=...` (default `read`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_repositories.py`:

```python
def test_list_events_only_read_narrows_to_success(db_session):
    from car_logger import repositories, schemas
    read = repositories.create_event(
        db_session, schemas.EventCreate(anpr_status="pending"))
    repositories.update_event_anpr(
        db_session, read.id, "CJ45ARL", 0.97, "success", None)
    unread = repositories.create_event(
        db_session, schemas.EventCreate(anpr_status="pending"))
    repositories.update_event_anpr(
        db_session, unread.id, None, None, "no_plate", None)

    assert len(repositories.list_events(db_session)) == 2
    only_read = repositories.list_events(db_session, only_read=True)
    assert [e.plate_text for e in only_read] == ["CJ45ARL"]
```
(imports local to the test so it never depends on the file's header; the
`db_session` fixture comes from `tests/conftest.py`.)

Append to `tests/integration/test_dashboard.py`:

```python
def _seed_read_and_no_plate_events(db_session):
    from car_logger import repositories, schemas
    read = repositories.create_event(
        db_session, schemas.EventCreate(anpr_status="pending"))
    repositories.update_event_anpr(
        db_session, read.id, "CJ45ARL", 0.97, "success", None)
    unread = repositories.create_event(
        db_session, schemas.EventCreate(anpr_status="pending"))
    repositories.update_event_anpr(
        db_session, unread.id, None, None, "no_plate", None)


def test_feed_default_shows_only_plate_read_events(client, db_session):
    _seed_read_and_no_plate_events(db_session)
    response = client.get("/partials/events-feed")
    assert response.status_code == 200
    assert "CJ45ARL" in response.text
    assert "fără plăcuță" not in response.text


def test_feed_filter_all_shows_everything_with_the_new_badge(client,
                                                             db_session):
    _seed_read_and_no_plate_events(db_session)
    response = client.get("/partials/events-feed?filter=all")
    assert "CJ45ARL" in response.text
    assert "fără plăcuță" in response.text
```

- [ ] **Step 2: RED**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_repositories.py tests/integration/test_dashboard.py -v`
Expected: the three new tests FAIL (`only_read` unexpected keyword; the
default feed still shows the no_plate event; badge text missing).

- [ ] **Step 3: Implement**

`repositories.py` — `list_events` becomes:

```python
def list_events(db: Session, skip: int = 0, limit: int = 50,
                plate_text: Optional[str] = None,
                only_read: bool = False) -> List[Event]:
    """Newest-first page of events; optional plate substring filter.

    only_read narrows to plate-read events (anpr_status == "success") —
    the dashboard's default view since v2 (student decision 2026-07-19:
    filter in the UI, not in the data; everything stays persisted)."""
    capped = min(limit, MAX_LIST_LIMIT)
    query = db.query(Event)
    if plate_text:
        query = query.filter(Event.plate_text.like("%" + plate_text + "%"))
    if only_read:
        query = query.filter(Event.anpr_status == "success")
    return (query.order_by(Event.timestamp.desc(), Event.id.desc())
                 .offset(skip)
                 .limit(capped)
                 .all())
```

`routes_dashboard.py` — `events_feed` becomes:

```python
@router.get("/partials/events-feed")
def events_feed(request: Request, q: str = "", filter: str = "read",
                db: Session = Depends(get_db)):
    """Feed fragment, newest first; `q` filters by plate substring.

    `filter` is the dashboard toggle: "read" (default) shows only
    plate-read events; "all" shows everything — the audit view for the
    Stage B validation window."""
    events = repositories.list_events(db, limit=15, plate_text=(q or None),
                                      only_read=(filter != "all"))
    fresh_cutoff = datetime.utcnow() - timedelta(seconds=FRESH_ROW_SECONDS)
    return templates.TemplateResponse(
        "partials/events_feed.html",
        {"request": request, "events": events, "fresh_cutoff": fresh_cutoff})
```

`macros.html` — add the `no_plate` branch to `status_badge`, between the
`pending` and `failed` branches:

```jinja
  {%- elif status == 'no_plate' -%}
    {%- set dot, label, cls = 'bg-paper-faint', 'fără plăcuță', 'text-paper-dim border-paper-faint/30' -%}
```

`dashboard.html` — replace the header div + feed div of the events
section (current lines 31-45) with:

```html
      <div class="mb-4 flex flex-wrap items-end justify-between gap-4">
        <h2 class="font-display text-2xl font-medium">Evenimente</h2>
        <div id="feed-controls" class="flex flex-wrap items-center gap-3">
          {# The toggle: student decision 2026-07-19 — filter in the UI,
             not in the data. "Toate" is the validation-window audit view. #}
          <fieldset class="flex items-center gap-3 rounded border border-ink-700 px-3 py-1.5 font-mono text-[11px] text-paper-dim">
            <label class="flex cursor-pointer items-center gap-1.5">
              <input type="radio" name="filter" value="read" checked
                     class="accent-gold"
                     hx-get="/partials/events-feed" hx-target="#events-feed"
                     hx-trigger="change" hx-include="#feed-controls input">
              Cu plăcuță
            </label>
            <label class="flex cursor-pointer items-center gap-1.5">
              <input type="radio" name="filter" value="all"
                     class="accent-gold"
                     hx-get="/partials/events-feed" hx-target="#events-feed"
                     hx-trigger="change" hx-include="#feed-controls input">
              Toate
            </label>
          </fieldset>
          <input type="search" name="q" placeholder="caut&#259; dup&#259; plac&#259;&#8230;"
                 class="w-48 rounded border border-ink-700 bg-ink-900 px-3 py-1.5
                        font-mono text-xs text-paper placeholder-paper-faint
                        focus:border-gold focus:outline-none"
                 hx-get="/partials/events-feed" hx-target="#events-feed"
                 hx-trigger="input changed delay:300ms"
                 hx-include="#feed-controls input">
        </div>
      </div>
      <div id="events-feed"
           hx-get="/partials/events-feed"
           hx-trigger="load, sse:new_event"
           hx-swap="innerHTML"
           hx-include="#feed-controls input"
           class="min-h-[16rem]">
      </div>
```
(The controls sit OUTSIDE `#events-feed`, so SSE swaps never reset the
radio state, and `hx-include="#feed-controls input"` makes every refresh
— load, SSE, search, toggle — carry both `q` and the checked `filter`.)

`models.py` line 44 comment becomes:

```python
    # anpr_status: pending | success | failed | no_plate | skipped |
    # throttled (throttled: historical rows only — the v1 cloud rate limit)
```

- [ ] **Step 4: GREEN**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_repositories.py tests/integration/test_dashboard.py -v`
Expected: all pass. Full suite: 127 passed.

- [ ] **Step 5: Commit**

```bash
git add car_logger/repositories.py car_logger/api/routes_dashboard.py car_logger/templates/partials/macros.html car_logger/templates/dashboard.html car_logger/models.py tests/unit/test_repositories.py tests/integration/test_dashboard.py
git commit -m "feat(v2b): dashboard defaults to plate-read events + no_plate badge (TDD)

filter=read|all on the feed (radios outside the swap target keep state,
hx-include carries q+filter on every refresh); data stays fully persisted.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Config + wiring — the local engine goes live in `main.py`

**Files:**
- Modify: `car_logger/config.py` (add v2 settings; API key fields STAY until Task 11)
- Modify: `car_logger/services/pipeline.py` (optional collector)
- Modify: `car_logger/main.py` (engines + collector wiring, APP_VERSION)
- Test: `tests/unit/test_config.py` (append), `tests/unit/test_pipeline_resilience.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: settings `anpr_detector_model_path`, `anpr_ocr_model_path`,
  `anpr_ocr_config_path`, `plate_detection_threshold`,
  `anpr_reads_per_track`, `anpr_read_spacing_s`;
  `PipelineWorker(..., collector=None)`; `app.state.crop_collector`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config.py`:

```python
def test_v2_local_anpr_defaults():
    from car_logger.config import Settings
    settings = Settings(_env_file=None)
    assert settings.anpr_detector_model_path == (
        "models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx")
    assert settings.anpr_ocr_model_path == "models/anpr/cct_xs_v2_global.onnx"
    assert settings.anpr_ocr_config_path == (
        "models/anpr/cct_xs_v2_global_plate_config.yaml")
    assert settings.plate_detection_threshold == 0.4
    assert settings.anpr_reads_per_track == 3
    assert settings.anpr_read_spacing_s == 0.4
    # Stage A verdict: 0.90 is the garbage floor for the LOCAL engine's
    # confidence scale (the 0.85 default was calibrated for the cloud API)
    assert settings.min_vehicle_confidence == 0.90
```

Append to `tests/unit/test_pipeline_resilience.py`:

```python
class NullDetector(object):
    def detect(self, frame):
        return []


class RecordingCollector(object):
    def __init__(self):
        self.ticks = []

    def tick(self, live_tracks, frame):
        self.ticks.append((list(live_tracks), frame))


def test_pipeline_ticks_the_collector_every_frame():
    collector = RecordingCollector()
    worker = PipelineWorker(camera=OneFrameCamera(), detector=NullDetector(),
                            tracker=NullTracker(),
                            on_confirmed=lambda t, f: None,
                            target_fps=200, collector=collector)
    worker.start()
    deadline = time.time() + 3.0
    while worker.frames_processed < 1 and time.time() < deadline:
        time.sleep(0.05)
    worker.stop()
    assert len(collector.ticks) >= 1
    assert collector.ticks[0][1] == "frame"
```

- [ ] **Step 2: RED**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest tests/unit/test_config.py tests/unit/test_pipeline_resilience.py -v`
Expected: new tests FAIL (missing settings attrs; `collector` unexpected
keyword).

- [ ] **Step 3: Implement**

`config.py` — first, update the existing `min_vehicle_confidence` field
(the 0.85 was calibrated for Plate Recognizer's score scale; the Stage A
verdict recalibrated it for the local engine):

```python
    # identity gate: a plate reading below this confidence never creates a
    # Vehicle (the event still keeps the reading). Recalibrated for the
    # LOCAL engine from the bake-off distributions (student decision
    # 2026-07-18): 0.90 is a garbage floor ONLY — this model's confidence
    # does NOT separate correct from wrong reads; the multi-frame vote is
    # the real error filter.
    min_vehicle_confidence: float = 0.90
```

Also update `.env.example`'s `MIN_VEHICLE_CONFIDENCE=0.85` line to
`MIN_VEHICLE_CONFIDENCE=0.90`, and in Task 10 Step 2 the student must
delete (or update) any `MIN_VEHICLE_CONFIDENCE=0.85` line in the
Jetson's `.env` — a stale .env would silently override the new default.

Then, after the `camera_reopen_backoff_s` field, add:

```python
    # v2 local ANPR (Stage B, spec 2026-07-19). Paths are relative to the
    # repo root — both `uvicorn` in dev and systemd's WorkingDirectory
    # run from there. Committed models: see models/anpr/README.md.
    anpr_detector_model_path: str = (
        "models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx")
    anpr_ocr_model_path: str = "models/anpr/cct_xs_v2_global.onnx"
    anpr_ocr_config_path: str = "models/anpr/cct_xs_v2_global_plate_config.yaml"
    # 0.4 = fast-alpr 0.4.0's default detector threshold — the setting the
    # bake-off accuracy (93.5% / 100%) was measured under.
    plate_detection_threshold: float = 0.4
    # The multi-frame vote (the REAL error filter — confidence is not,
    # per the bake-off calibration): 3 reads per track, >= 0.4 s apart.
    anpr_reads_per_track: int = 3
    anpr_read_spacing_s: float = 0.4
```

`pipeline.py` — constructor gains `collector=None` and `_tick` uses the
tracker's return value:

```python
    def __init__(self, camera, detector, tracker, on_confirmed,
                 target_fps=15, collector=None):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.on_confirmed = on_confirmed
        self.collector = collector
        self._min_interval = 1.0 / float(target_fps)
        self._running = False
        self._thread = None
        self.last_fps = 0.0
        self.last_event_at = None
        self.frames_processed = 0
```

and in `_tick`, replace the `self.tracker.update(boxes)` line and the
confirmed-tracks loop with:

```python
        tracks = self.tracker.update(boxes)
        for track in self.tracker.new_confirmed_tracks():
            self.last_event_at = time.time()
            self.on_confirmed(track, frame)
        if self.collector is not None:
            self.collector.tick(tracks, frame)
```

Also update the module docstring's second paragraph to:

```python
"""Pipeline worker: camera -> detector -> tracker -> on_confirmed callback.

on_confirmed(track, frame) is called once per newly-confirmed track, with
the frame it was confirmed on. The callback owns persistence and starts
the crop collection; the collector (v2) is then ticked every frame to
gather the remaining spaced crops. The pipeline stays CV-only."""
```

`main.py` — bump `APP_VERSION` to `"0.6.0"`, then in `_startup` replace
the import block and the ANPR wiring (current lines 124-163) with:

```python
    from car_logger.services.capture import CameraWorker
    from car_logger.services.detector import Detector
    from car_logger.services.tracker import IoUTracker
    from car_logger.services.pipeline import PipelineWorker
    from car_logger.services.local_anpr import LocalAnprClient
    from car_logger.services.onnx_engines import (OnnxPlateDetector,
                                                  OnnxPlateOcr)
    from car_logger.services.anpr_worker import AnprWorker
    from car_logger.services.crop_collector import CropCollector

    camera = CameraWorker(
        device_index=settings.camera_index,
        stale_after_s=settings.camera_stale_after_s,
        reopen_backoff_s=settings.camera_reopen_backoff_s,
    )
    camera.start()

    # v2: fully local ANPR — models load ONCE here, never per crop.
    anpr_client = LocalAnprClient(
        OnnxPlateDetector(settings.anpr_detector_model_path,
                          threshold=settings.plate_detection_threshold),
        OnnxPlateOcr(settings.anpr_ocr_model_path,
                     settings.anpr_ocr_config_path))
    anpr_worker = AnprWorker(anpr_client, _make_on_result(app.state.broker))
    anpr_worker.start()

    def submit_crops(event_id, crops):
        # collector says a car's crop list is complete -> one queued job
        if anpr_worker.submit(event_id, crops):
            return
        db2 = SessionLocal()
        try:
            repositories.update_event_anpr(
                db2, event_id, None, None, "skipped", None,
            )
        finally:
            db2.close()
        app.state.broker.publish("updated")

    collector = CropCollector(
        submit_crops,
        reads_per_track=settings.anpr_reads_per_track,
        spacing_s=settings.anpr_read_spacing_s)

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
        app.state.broker.publish("created")
        # 2) crop #1 now; the collector gathers the rest across frames
        collector.start(track.track_id, event_id, track.box, frame)
```

then pass the collector to the pipeline and store it on app.state (the
`PipelineWorker(...)` call and the `app.state` lines become):

```python
    pipeline = PipelineWorker(
        camera=camera,
        detector=Detector(threshold=settings.detector_threshold),
        tracker=IoUTracker(),
        on_confirmed=on_confirmed,
        target_fps=settings.max_pipeline_fps,
        collector=collector,
    )
    pipeline.start()
    app.state.camera = camera
    app.state.pipeline = pipeline
    app.state.crop_collector = collector
    app.state.anpr_worker = anpr_worker
    log.info("pipeline_started", target_fps=settings.max_pipeline_fps)
```

`_shutdown` becomes (pipeline first, then flush the collector so partial
collections reach the worker's drain, then worker, then camera):

```python
@app.on_event("shutdown")
def _shutdown():
    # Stop order matters: pipeline first (no new submits), then flush the
    # collector's partial collections into the queue (they drain as
    # 'skipped'), then the ANPR worker, then the camera it read from.
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        pipeline.stop()
    collector = getattr(app.state, "crop_collector", None)
    if collector is not None:
        collector.drain()
    for name in ("anpr_worker", "camera"):
        worker = getattr(app.state, name, None)
        if worker is not None:
            worker.stop()
    log.info("app_shutdown")
```

- [ ] **Step 4: GREEN + full suite**

Run (JETSON): `OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest -q`
Expected: 129 passed (127 + 2 new; startup wiring itself is exercised
live in Task 10 — TestClient never fires startup, by design of the
fixtures).

- [ ] **Step 5: Commit**

```bash
git add car_logger/config.py car_logger/services/pipeline.py car_logger/main.py .env.example tests/unit/test_config.py tests/unit/test_pipeline_resilience.py
git commit -m "feat(v2b): wire the local ANPR stack - engines, collector, vote (app 0.6.0)

min_vehicle_confidence recalibrated 0.85 -> 0.90 (Stage A verdict: garbage
floor only, the vote is the filter). Cloud client no longer constructed;
anpr_api_* settings stay until the post-validation cleanup commit.
Shutdown drains the collector before the worker so partials become
honest 'skipped'.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Jetson deployment, smoke tests, live E2E — the window opens

**Files:**
- Modify: `requirements.txt` (+ onnxruntime, numpy pins)
- Modify: `deployment/car-logger.service` (+ OPENBLAS env)
- Create: `docs/v2-stage-b-validation-log.md`

**Interfaces:**
- Consumes: everything shipped in Tasks 1-9; Jetson paths
  `~/jetson-car-logger`, app venv `venv/`, spike assets in
  `~/anpr_spike/` (the `event_22_plate.jpg` tight crop), `car_test.jpg`
  in the repo root on the Jetson, `~/e2e_fake_cam.py`.
- Produces: the running v2 appliance + the open validation window.

- [ ] **Step 1: Pin the new Jetson dependencies (LAPTOP, commit + push)**

Append to `requirements.txt`:

```
onnxruntime==1.9.0           # v2 local ANPR — LAST cp36 aarch64 wheel (opset<=15); CPU provider only
numpy==1.19.5                # pulled by onnxruntime; on Tegra X1 needs OPENBLAS_CORETYPE=ARMV8 or it SIGILLs
```

Edit `deployment/car-logger.service` — after the `PYTHONUNBUFFERED`
line, add:

```
# numpy>=1.19.5 wheels crash (SIGILL) on the Tegra X1 without this
# (Stage A Task 9 spike finding).
Environment=OPENBLAS_CORETYPE=ARMV8
```

Commit:
```bash
git add requirements.txt deployment/car-logger.service
git commit -m "feat(v2b): pin onnxruntime 1.9 + numpy 1.19.5, OPENBLAS_CORETYPE in the unit

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 2: Install on the Jetson**

Run (JETSON):
```bash
cd ~/jetson-car-logger && git pull
venv/bin/pip install onnxruntime==1.9.0 numpy==1.19.5
venv/bin/python -c "import onnxruntime, numpy; print(onnxruntime.__version__, numpy.__version__)"
grep MIN_VEHICLE_CONFIDENCE .env 2>/dev/null || echo "no override - default 0.90 applies"
```
Expected: `1.9.0 1.19.5`. (If the last python command SIGILLs — that IS
the OPENBLAS symptom; prefix it with `OPENBLAS_CORETYPE=ARMV8` and it
must print cleanly. From here on, every manual run needs the prefix.)
If the grep shows `MIN_VEHICLE_CONFIDENCE=0.85`, the student deletes or
updates that line — a stale `.env` would silently override the new 0.90
default from Task 9.

- [ ] **Step 3: Smoke A — the committed models load and read, in the APP venv**

The Stage A spike ran in a separate venv; this proves the same thing in
the venv that will run the appliance, against the files in git.

Run (JETSON):
```bash
OPENBLAS_CORETYPE=ARMV8 venv/bin/python experiments/anpr_bakeoff/spike_onnx_jetson.py \
    --ocr-model models/anpr/cct_xs_v2_global.onnx \
    --ocr-config models/anpr/cct_xs_v2_global_plate_config.yaml \
    --detector-model models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx \
    --image ~/anpr_spike/event_22_plate.jpg \
    --expect CJ45ARL
```
Expected: `ocr read: 'CJ45ARL'` and `SPIKE PASS`.

- [ ] **Step 4: Smoke B — jetson-inference still works under numpy 1.19.5**

The vehicle detector (cudaFromNumpy) has NEVER run against the new numpy
— known risk carried in the spec. Run (JETSON):
```bash
cd ~/jetson-car-logger && OPENBLAS_CORETYPE=ARMV8 venv/bin/python -c "
import numpy
print('numpy', numpy.__version__)
import cv2
from car_logger.services.detector import Detector
frame = cv2.imread('car_test.jpg')
detections = Detector(threshold=0.5).detect(frame)
print('detections:', [(d.x1, d.y1, d.x2, d.y2) for d in detections])
"
```
Expected: `numpy 1.19.5` and at least one detection box for the BMW —
no SIGILL, no cudaFromNumpy error. **If this fails, STOP the task and
debug (student-led) before anything goes live: this is the one risk the
laptop cannot test.**

- [ ] **Step 5: Full suite + E2E with the fake camera, OFFLINE**

Run (JETSON):
```bash
OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest -q          # expected: 129 passed
sudo systemctl stop car-logger                       # :8000 must be free
sudo ip route del default                            # INTERNET OFF (LAN stays)
OPENBLAS_CORETYPE=ARMV8 venv/bin/python ~/e2e_fake_cam.py
```
From the laptop browser: `http://<jetson-ip>:8000` → the fake-cam car
(`car_test.jpg`, plate `MMM8748`) must appear with badge `citită` and
plate `MMM8748` — **read with zero internet**. This is v2's whole point;
savor it. Also flip the new toggle to „Toate" and back. Then restore:
```bash
sudo ip route add default via 192.168.0.1 dev eth0
```
(Ctrl+C the fake cam first.)

- [ ] **Step 6: Install the new unit + start the real service**

Run (JETSON):
```bash
sudo cp deployment/car-logger.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart car-logger
systemctl status car-logger --no-pager
journalctl -u car-logger -n 30 --no-pager | grep -E "pipeline_started|error" || true
```
Expected: `active (running)`, `pipeline_started` in the journal, no
errors. Dashboard reachable, camera live. Note RAM: `free -h` and one
`tegrastats` sample (< 3 GB total).

- [ ] **Step 7: Open the validation window (LAPTOP, commit + push)**

Create `docs/v2-stage-b-validation-log.md`:

```markdown
# Stage B — live validation window (the gate for the cleanup commit)

**Opened:** <fill in the deploy date> · **Criteria (spec 2026-07-19):**
window closes only when ALL hold — ≥ 5 calendar days of live running AND
≥ 15 manually verified events with **max 1 wrong read**, RAM < 3 GB
through day 5, pipeline FPS unchanged. Task 11 is FORBIDDEN until then.

A **wrong read** = the displayed text does not match the plate visible in
the event's photo. An honest `fără plăcuță`/`eșuat` on a visible plate is
a miss, NOT a wrong read — the bar guards lies, not misses. Verify ~3
events/day, spread across lighting (morning/evening/rain). No
cherry-picking: a wrong plate noticed outside this table counts too.
Verify in the „Toate" view so misses are also seen.

## Verified events (need ≥ 15)

| # | date | event id | photo shows | app read | status | verdict (ok/WRONG) | note |
|---|------|----------|-------------|----------|--------|--------------------|------|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |  |

## Daily health (need ≥ 5 days)

| day | date | RAM (`free -h` / tegrastats) | 04:00 restart clean? | FPS ok? | notes |
|-----|------|------------------------------|----------------------|---------|-------|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |

## Wrong-read investigations (every WRONG needs one)

_(none yet)_
```

Commit:
```bash
git add docs/v2-stage-b-validation-log.md
git commit -m "docs(v2b): validation window opened - the log that gates the cleanup commit

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

**The window is now open. STOP here. Task 11 waits for the log to show
all criteria met (student updates the log over the following days).**

---

### Task 11: The cleanup commit — GATED on the closed validation window

**PRECONDITION (verify before ANY step):** `docs/v2-stage-b-validation-log.md`
shows ≥ 5 days, ≥ 15 verified events, ≤ 1 wrong read, RAM/FPS ok, and
every WRONG has an investigation note. If not — STOP.

**Files:**
- Delete: `car_logger/services/anpr_client.py`, `tests/unit/test_anpr_client.py`
- Modify: `car_logger/config.py` (remove `anpr_api_key`, `anpr_api_url`)
- Modify: `.env.example` (remove the two ANPR_API lines)
- Modify: `requirements.txt` (remove httpx)
- Modify: `CLAUDE.md` (rule 7, scope-creep list, stack, diagram)
- Modify: `car_logger/main.py` (APP_VERSION `"2.0.0"`)

**Interfaces:**
- Consumes: the closed validation log; `PlateResult` already lives in
  `plate_result.py` (Task 2), so nothing imports from `anpr_client`.

- [ ] **Step 1: Prove nothing still needs the cloud client**

Run (LAPTOP, git-bash):
```bash
grep -rn "anpr_client\|AnprClient\|anpr_api" --include="*.py" car_logger/ tests/ | grep -v "services/anpr_client.py"
grep -rn "httpx" --include="*.py" car_logger/ tests/ | grep -v "services/anpr_client.py"
```
Expected: only `tests/unit/test_anpr_client.py` hits (which is deleted
next) — no other file imports the client, httpx, or the key settings.
If anything else shows up, fix that first.

- [ ] **Step 2: Delete + strip**

```bash
git rm car_logger/services/anpr_client.py tests/unit/test_anpr_client.py
```

`config.py`: delete the two lines
```python
    anpr_api_key: str = ""
    anpr_api_url: str = "https://api.platerecognizer.com/v1/plate-reader/"
```

`.env.example`: delete the `ANPR_API_KEY=...` and `ANPR_API_URL=...`
lines and change the header comment to:
```
# Copy to .env (gitignored) to override defaults. Everything has a sane
# default — since v2 the appliance needs NO secrets and .env is optional.
```

`requirements.txt`: delete the line
`httpx==0.22.0              # for Plate Recognizer API calls`.

`main.py`: `APP_VERSION = "2.0.0"`.

- [ ] **Step 3: Amend CLAUDE.md (the spec's list, exact edits)**

1. Hard rule 7 — replace its text with:
   ```
   7. **No modern CV libraries.** No ultralytics, no YOLOv5+, no `cv2.cuda`.
      Vehicle detection is `jetson.inference.detectNet("ssd-mobilenet-v2")`.
      **v2 exception (bake-off verdict, 2026-07-18):** plate detection + OCR
      run locally as the two pinned ONNX models in `models/anpr/` on
      onnxruntime 1.9 CPU. That is the whole CV layer. Do not expand it
      and do not swap models without re-running the bake-off.
   ```
2. "Rejecting scope creep" list — delete the line
   `- Self-hosted OCR → "that's the CV v2 project next semester"`.
3. Stack block — remove the `httpx==0.22.0` line; add under it:
   ```
   onnxruntime==1.9.0           # v2 local ANPR (Jetson: last cp36 aarch64 wheel)
   numpy==1.19.5                # onnxruntime dep; Tegra needs OPENBLAS_CORETYPE=ARMV8
   ```
4. Architecture diagram — the ANPR box no longer crosses the device
   boundary. Replace the three lines
   ```
   │                            │  ANPR stage  │─────────────────────┼──>  Plate Recognizer
   │                            │  (async)     │<────────────────────┼──   API (external)
   │                            └──────┬───────┘   plate text         │
   ```
   with
   ```
   │                            │  ANPR stage  │  local ONNX:         │
   │                            │  (worker)    │  plate det + OCR     │
   │                            └──────┬───────┘   plate text          │
   ```
   and delete the now-dangling `crop + POST` label on the ANPR box's
   incoming arrow (leave the arrow).

- [ ] **Step 4: The final commit (LAPTOP)**

```bash
git add -A
git commit -m "feat(v2)!: delete the cloud ANPR client - the appliance is fully offline (2.0.0)

Validation window closed (docs/v2-stage-b-validation-log.md): the numbers
authorize this. No API key, no httpx, no external calls; CLAUDE.md
amended (rule 7 v2 exception, scope list, diagram).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Verify on the device (JETSON)**

```bash
cd ~/jetson-car-logger && git pull
OPENBLAS_CORETYPE=ARMV8 venv/bin/pytest -q
```
Expected: all pass — the count DROPS versus Task 10's 129 by exactly the
deleted `test_anpr_client.py`'s tests, and nothing else fails.

The student then deletes the `ANPR_API_KEY`/`ANPR_API_URL` lines from
the Jetson's `.env` (or deletes the whole file — it is optional now),
and restarts:
```bash
grep -n "ANPR_API" .env 2>/dev/null || echo "no key on device"
sudo systemctl restart car-logger && systemctl status car-logger --no-pager
```
Expected: `no key on device`, service `active (running)`, dashboard
alive, cars still being read — with no secret anywhere on the machine.

Remaining for the student afterwards (his own voice, not this plan):
README + `docs/architecture.md` retell the story cloud → local; the v1
demo video is still owed.

---

## Execution log

- **Task 1 DONE (2026-07-19, laptop):** models copied from the Stage A
  caches, detector re-stamped opset 17 → 15, outputs verified
  bit-identical (`True`), README written. Commit `1549452`. Jetson not
  needed yet.
- **Task 2 DONE (2026-07-19):** `PlateResult` lives in
  `services/plate_result.py`; cloud client re-exports, worker imports the
  new home. Jetson checkpoint: **91 passed** (pure refactor). Commit
  `2acd9b7`.
- **Task 3 DONE (2026-07-19):** vote_on_reads implemented TDD — RED
  proven on the Jetson (collection error, module missing, commit
  `bfa9339`), then GREEN: **11 passed** + full suite **102 passed**
  (commit `2393371`). NOTE: the plan's expected counts were off by one —
  its own test file has 11 tests, not 12; every later "full suite"
  target shifts by −1 (Task 4: 110, T5: 117, T6: 122, T7: 123, T8: 126,
  T9/T10: 128).
- **Task 4 DONE (2026-07-19):** LocalAnprClient implemented TDD — RED
  proven on the Jetson (collection error, commit `fec6c01`), then GREEN:
  **8 passed** + full suite **110 passed** (commit `22bafc1`). Two-stage
  read behind the v1 contract, never-raises, OCR only after a detected
  plate, read_plate_multi returns the winning crop as evidence. **Next:
  Task 5** (onnx_engines, TDD on the pure decode helpers).
- **Task 5 DONE (2026-07-19):** onnx_engines implemented TDD — RED on the
  Jetson (collection error, commit `5bf5d84`), then GREEN: **7 passed** +
  full suite **117 passed** (commit `ae47dac`). Pure decode helpers
  (best_detection / decode_ocr_outputs / region_to_code) are unit-tested;
  the real OnnxPlateDetector + OnnxPlateOcr import ORT lazily and are
  exercised live only in Task 10's smoke test. **Next: Task 6**
  (CropCollector, TDD).
- **Task 6 DONE (2026-07-19):** CropCollector implemented TDD — RED on the
  Jetson (collection error, commit `9ff30c7`), then GREEN: **5 passed** +
  full suite **122 passed** (commit `b2dd55e`). Per-track dict, crop #1 at
  confirmation + spaced follow-ups, track-death handover, stale-box skip,
  drain() at shutdown; injectable clock + crop_fn. **Next: Task 7**
  (AnprWorker speaks crop lists — modifies existing worker + its tests).
- **Task 7 DONE (2026-07-19):** AnprWorker adapted TDD — RED on the Jetson
  (2 failed: worker still called read_plate, commit `2b4a9bd`), then
  GREEN: worker + on_result **7 passed** (on_result UNTOUCHED — callback
  contract intact) + full suite **123 passed** (commit `932a012`). Job is
  now (event_id, [crops]) → read_plate_multi → (verdict, evidence);
  queue/drop/drain semantics unchanged. **Next: Task 8** (dashboard filter
  + no_plate badge, TDD — first UI/template task).
