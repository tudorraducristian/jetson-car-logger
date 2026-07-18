# v2.0 "Local ANPR" — offline plate reading — Design

**Date:** 2026-07-18 · **Status:** approved by student · **Sequencing:**
this officially opens the CV v2 project (CLAUDE.md's "next semester" item,
started deliberately early by student decision).

## Problem

Plate reading currently requires internet: every confirmed track POSTs a
crop to the Plate Recognizer cloud API (`anpr_client.py`). No ethernet /
hotspot → every event stays `failed`. The appliance promise ("power it on,
it works") holds only where there is connectivity, and the API key is a
recurring dependency (rate limits, expiry, secret management).

**Goal in one line:** same functionality as today — detect car, read
plate, store event, live dashboard — with zero internet and zero API key.

## Requirements (from brainstorming, 2026-07-18)

1. Fully on-device ANPR on the original Jetson Nano 4GB (JetPack 4.6,
   Ubuntu 18.04, Python 3.6, CUDA 10.2, TensorRT 8.2). This environment is
   fixed: JetPack 4.6 is the last release supporting the original Nano.
2. OCR runs **only when a plate is actually found** in the vehicle crop —
   a two-stage local pipeline (plate detection → OCR), not blind OCR.
3. Pure offline: the cloud client is **removed**, not kept as fallback
   (student decision — no hybrid mode).
4. Evidence-based engine choice: a bake-off on real data before
   integration (student decision — staged approach).
5. Everything else unchanged: worker queue semantics, DB schema (one new
   status value only), dashboard, SSE, deployment flow.

## Non-goals (YAGNI)

- Hybrid cloud fallback (decided against).
- Fine-tuning / training models — if local accuracy disappoints, that is
  the *next* project, on real numbers.
- GPU inference for ANPR — CPU is preferred so the SSD vehicle detector
  keeps the GPU (revisit only if CPU latency fails the bake-off bar).
- Plate regions beyond Europe.

## Research findings (2026-07-18)

Candidates that can plausibly run on Python 3.6 / JetPack 4.6:

| Candidate | What it is | For | Against |
|---|---|---|---|
| **OpenALPR** (open source) | C++ lib, apt-packaged in Ubuntu 18.04, CPU, `-c eu` region config, does its own plate detection + OCR | Trivial install, proven on Jetson, low RAM, no GPU contention, no Python-version issues | Unmaintained (~2017), moderate accuracy |
| **fast-alpr stack** (ankandrew) | YOLO-tiny plate detector (ONNX) + fast-plate-ocr **European model** (MobileViTV2, 40+ countries incl. RO) | Modern, plate-specialised, best expected accuracy, predicts region | PyPI package needs modern Python — on 3.6 we must drive the ONNX models manually (old cp36 onnxruntime from Jetson Zoo, or TensorRT 8.2 engine); opset compatibility is the real risk |
| NVIDIA LPDNet+LPRNet (TAO) | Official TensorRT models | Fast, NVIDIA-blessed | **Rejected:** pretrained LPRNet is US/China only — poor on RO plates without fine-tuning |
| EasyOCR | Generic OCR on PyTorch | Works on Nano | **Rejected:** ~1GB+ RAM on a 4GB board already using 1.5–2.2GB, slow, not plate-specialised, needs its own detector anyway |

## Design

### 1. Target architecture: swap the inside of the box, not the box

`read_plate(image_bytes) -> PlateResult(plate_text, confidence, status,
region)` stays the contract. `AnprWorker`, its bounded queue and drop
policy, `on_result`, repositories, SSE, dashboard — untouched.

New `car_logger/services/local_anpr.py` — `LocalAnprClient`:

- **Stage 1 — plate detection** on the vehicle crop. Nothing found →
  `PlateResult(None, None, "no_plate", None)`; OCR never runs (this *is*
  the "read only when a plate is identified" requirement).
- **Stage 2 — OCR** on the detected plate rectangle → text + confidence,
  through the existing `normalize_plate`.
- Engines (detector, OCR) are **injected** in the constructor as small
  interfaces — unit tests use fakes, no models, no Jetson (same pattern as
  today's injected httpx client).
- Models load **once at construction** (startup), never per crop.
  `close()` releases them, symmetric with the current client.
- **"Never raises" contract preserved:** missing model file, inference
  exception, corrupt image → `PlateResult(None, None, "failed", None)`;
  the worker loop's defense-in-depth stays as a second net.

### 2. Behavioural consequences

- **Statuses:** events still start `pending`; final states become
  `success | failed | no_plate | skipped`. `no_plate` is new (dashboard
  badge: "no plate visible"). `throttled` is no longer produced;
  historical rows keep it and the UI keeps rendering it. **No migration
  needed** (verified: `anpr_status` is an unconstrained String column and
  the Pydantic schemas do not validate its values) — only the model
  comment and the badge template change.
- **`region`:** OpenALPR does not predict country → `None` (the RO regex
  gate in `should_create_vehicle` already applies only when
  `region == "ro"`). fast-plate-ocr global/EU models do predict region →
  pass it through.
- **`min_vehicle_confidence` (0.85):** calibrated for Plate Recognizer's
  score scale. The local engine's scale differs → **recalibrated by the
  student from the bake-off score distributions**.
- **Config:** `anpr_api_key` / `anpr_api_url` removed; added: model/config
  paths for the chosen engine + plate-detection threshold. `.env` becomes
  optional — no secrets remain.
- **CLAUDE.md amendments:** rule 7 gains the v2 exception (the CV layer
  now includes local plate detection + OCR), "Self-hosted OCR" leaves the
  scope-creep list, architecture diagram updated (ANPR box no longer
  crosses the device boundary).

### 3. Stage A — the bake-off (`experiments/anpr_bakeoff/`)

Dataset, two layers (we have <50 real crops — too few to carry the
decision alone):

- **Base:** a public European-plates dataset (~500–1000 annotated images,
  permissive licence, chosen at implementation time from Kaggle/Roboflow;
  criteria: EU plates, varied conditions, ideally some RO).
- **Real-world check:** our crops from `data/plates/` + the DB's
  `success` cloud readings as ground truth, exported from the Jetson by a
  small script into `folder + CSV (filename, plate_text)`.

Candidates:

1. **OpenALPR `-c eu`** — apt install on the Jetson, measured in the final
   environment directly.
2. **fast-alpr stack** — accuracy measured on the laptop with modern
   Python (accuracy is a property of the model, not the machine);
   on-device feasibility proven separately (below).

Metrics per candidate: **exact-match rate** (primary), character error
rate, false-positive rate on plate-less images (feeds the `no_plate`
threshold), latency per crop and added RAM (on Jetson).

**Feasibility spike (Jetson, only if the ONNX stack wins on accuracy):**

1. cp36 `onnxruntime` wheel from the Jetson Zoo (CPU execution provider is
   enough — a few crops per minute, not per frame);
2. if the models' opset is too new for that runtime → convert to a
   TensorRT 8.2 engine with `trtexec`, drive it from JetPack's native
   Python bindings;
3. both fail → **OpenALPR wins by feasibility**, regardless of accuracy.

**Decision rule (fixed in advance):** the most accurate candidate on
exact-match that runs on the Jetson at **< 2 s/crop** and **< 500 MB
added RAM** (total stack stays under CLAUDE.md's 3 GB alarm line).

Deliverable: `experiments/anpr_bakeoff/RESULTS.md` — metric tables + the
argued decision; also the data the student uses to recalibrate
`min_vehicle_confidence`.

### 4. Stage B — integration of the winner

- TDD as usual; existing `anpr_worker` / `on_result` tests stay green
  untouched (the interface is unchanged). New tests: `LocalAnprClient`
  with fake engines (both stages, error paths), `no_plate` handling,
  dashboard badge.
- Inference runs where the network call runs today: the `AnprWorker`
  thread, same queue, same drop policy. CPU-only by default.
- Live validation on the Jetson: `tegrastats` for RAM (< 3 GB), pipeline
  FPS unaffected, real traffic for a few days.

### 5. Rollout / rollback

- Branch → push → pull on Jetson → restart systemd, as usual.
- **Deleting `anpr_client.py` and the key from `.env` is the last commit,
  not the first** — it happens only after the live validation window.
- Rollback = `git revert`; the cloud client stays in git history.

## Student decisions recorded (2026-07-18)

1. v2 starts now; CLAUDE.md is amended rather than obeyed on this point.
2. Pure offline — no hybrid cloud fallback.
3. Staged approach: bake-off first, integrate the winner second.
4. Real-crop dataset is small (<50) → public EU dataset carries the
   benchmark, real crops are the sanity check.

## Success criteria

1. Ethernet cable unplugged, hotspot off: a car passes, the event appears
   on the dashboard with a plate reading (or an honest `no_plate`).
2. No API key anywhere in the repo, `.env`, or the running service.
3. Bake-off report exists with real numbers; the engine choice and the new
   `min_vehicle_confidence` cite it.
4. RAM under 3 GB, pipeline FPS unchanged, appliance survives the daily
   restart cycle.
5. The student can explain the two-stage design and the bake-off numbers
   to Radu in a 10-minute review.
