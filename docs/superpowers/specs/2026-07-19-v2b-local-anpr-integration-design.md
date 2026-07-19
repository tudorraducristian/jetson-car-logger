# v2 Stage B — integrating the bake-off winner — Design

**Date:** 2026-07-19 · **Status:** approved by student · **Parent spec:**
`2026-07-18-v2-local-anpr-design.md` (Stage A closed 2026-07-18, verdict
in `experiments/anpr_bakeoff/RESULTS.md`).

## Problem

Stage A picked the engine: **fast-alpr with the `cct-xs-v2-global` OCR**
(+ the `yolo-v9-t-384` detector re-stamped to opset 15) on onnxruntime
1.9.0 CPU — 93.5% / 100% exact-match, ≈337 ms/crop, 110 MB RSS. Stage B
wires it into `car_logger`, replacing the cloud client.

The bake-off also proved the hard constraint this design is built
around: **the OCR's confidence does not separate correct from wrong
reads** (all 7 wrong reads sat at conf ≥ 0.9997). The error filter must
therefore be **multi-frame track agreement** — reading the same plate on
several frames and requiring the readings to agree — not a confidence
threshold.

## Student decisions recorded (2026-07-19)

1. **Models ship in git** (`models/anpr/`, ~10 MB total).
2. **Region mapping:** the global OCR predicts the plate's country;
   Romania maps to `"ro"` so the existing RO-regex gate in
   `should_create_vehicle` keeps working unchanged.
3. **Votes per track: 3 reads, agreement = at least 2 identical** (after
   `normalize_plate`). All three different → `failed`.
4. **Partial collections vote with what they have** — 2 identical →
   `success`; 2 differing → `failed` (tie); a single usable read →
   accepted (graceful degradation to v1's single-read behaviour).
5. **`no_plate` reads are abstentions** — they do not out-vote a real
   reading. All reads `no_plate` → event status `no_plate`.
6. **Architecture: the pipeline collects, one job per car** (Variant 1
   below); queue semantics unchanged.
7. **Dashboard: filter in the UI, not in the data.** Default feed shows
   plate-read events only; a "Toate" toggle reveals everything. All
   events keep being persisted.
8. **Validation window: ≥ 5 calendar days AND ≥ 15 manually verified
   events, with at most 1 wrong read among the verified 15.** Details
   below.

## Design

### 1. Components and data flow

New files:

- **`car_logger/services/local_anpr.py` — `LocalAnprClient`.** Replaces
  `AnprClient` behind the same contract. Two injected engine interfaces
  (plate detector, OCR) — unit tests use fakes, no models, no Jetson
  (same pattern as the injected httpx client today). Models load **once
  at construction** (ORT 1.9, CPU); `close()` releases them. "Never
  raises" preserved: any exception → `PlateResult(None, None, "failed",
  None)`. API: `read_plate(image_bytes)` (single crop, two stages:
  detect plate → OCR only if found) plus `read_plate_multi(crops)`
  (reads each crop, then calls the vote).
- **`car_logger/services/plate_voting.py`** — the vote as a **pure
  function**: list of per-crop `PlateResult`s in, one verdict out.
  Rules, precisely:
  1. Normalize texts (`normalize_plate`) before comparing.
  2. `no_plate` reads and technically-failed reads carry no text — they
     abstain.
  3. A text with ≥ 2 votes wins → `success`.
  4. No majority, exactly one distinct text seen → accept it →
     `success` (single-read acceptance).
  5. No majority, ≥ 2 distinct texts (1-1 or 1-1-1) → `failed`.
  6. Zero texts: all usable reads were `no_plate` → `no_plate`;
     otherwise (technical failures in the mix, or empty input) →
     `failed`.
  - Winner's `plate_confidence` = max confidence among the reads that
    voted for the winning text; `region` comes from the
    highest-confidence winning read.
- **`car_logger/services/crop_collector.py`** — per-track collection:
  on confirmation keep crop #1, then take up to 2 more from later
  frames spaced ≥ `anpr_read_spacing_s` apart (consecutive frames are
  near-identical and would fail identically — spacing decorrelates the
  reads). At 3 crops or track death, hand over the list. Per-track
  state lives in a bounded instance dict, cleaned on track death.

Modified: **`pipeline.py`** (ticks the collector every frame; the event
is still created at confirmation, `pending`, exactly as today) and
**`anpr_worker.py`** (a job becomes `(event_id, [crops])`; one job = one
car, so the bounded queue and drop policy are untouched).

Untouched: tracker, capture, broker/SSE, repositories, `on_result`, DB
schema.

Full flow: track confirmed → `pending` event in DB (+ SSE) → collector
gathers up to 3 crops (~1 s) → one queue job → worker: stage-1 plate
detection per crop, OCR only where a plate was found → vote → one
`PlateResult` → `on_result` persists (+ SSE), as today. The event's
saved image is the winning read's crop (the visual evidence matches the
displayed text); on `no_plate`/`failed`, the first crop.

### 2. Statuses, data, dashboard

| status | when | badge |
|---|---|---|
| `success` | the vote produced a text | `citită` (as today) |
| `no_plate` | no plate visible in any crop | `fără plăcuță` (new, grey) |
| `failed` | plate seen but reads diverged / technical error | `eșuat` (narrower meaning) |
| `skipped` | queue full, job dropped | `omis` (as today) |
| `throttled` | never produced again; historical rows keep rendering | `limitat` (history only) |

No DB migration (verified in the parent spec: `anpr_status` is an
unconstrained String and the schemas don't validate its values) — only
the model comment and the badge template change.

`min_vehicle_confidence = 0.90` (Stage A verdict) stays a garbage floor
only; the vote is the real error filter. The RO gate in
`should_create_vehicle` works unchanged via the `"ro"` region mapping.

Dashboard:

- Default feed shows **`success` only** — a card appears ~1.5–2 s after
  detection, when the vote has concluded; intermediate `pending` cards
  are not shown in the default view.
- Toggle **"Cu plăcuță / Toate"** next to the search box: a `filter`
  query param on the events-list route, a `WHERE` in the repository,
  htmx swaps the list — same pattern as the existing plate search.
  "Toate" shows everything (including in-flight `pending`) and is the
  audit view for the validation window.
- SSE unchanged: on any update the list is re-requested **with the
  current filter**.

### 3. Config, models, deployment

- **`models/anpr/`** in git: the opset-15 re-stamped detector (~7 MB,
  the exact artifact verified bit-identical in the Task 9 spike) + the
  `cct-xs-v2-global` OCR with its preprocessing config (~2 MB). Clone +
  pip install = working app, no runtime downloads.
- **`config.py`:** remove `anpr_api_key` / `anpr_api_url`; add
  `anpr_detector_model_path`, `anpr_ocr_model_path` (+ OCR config
  path), `plate_detection_threshold = 0.4` (fast-alpr 0.4.0's default
  detector threshold — the setting the bake-off accuracy was measured
  under), `anpr_reads_per_track = 3`, `anpr_read_spacing_s = 0.4`. All
  with sane defaults — **`.env` becomes optional; no secrets remain.**
- **Jetson dependencies:** `onnxruntime==1.9.0` (the cp36 aarch64 wheel
  proven in the spike), pinned in the Jetson requirements only —
  laptop unit tests use fakes and don't need ORT.
- **systemd unit:** add `Environment=OPENBLAS_CORETYPE=ARMV8` (numpy
  SIGILLs on the Tegra X1 without it — Task 9 finding). Daily restart
  stays.

Rollout, in safety order:

1. New branch → TDD on the laptop → push → pull on the Jetson →
   restart systemd. The cloud client **stays in the repo, unused**
   (rollback = `git revert`).
2. **Validation window** (see below).
3. Only after the window closes: the **cleanup commit** — delete
   `anpr_client.py`, its tests, and the API key from `.env` /
   `.env.example`; amend CLAUDE.md (rule 7 gains the v2 exception,
   "Self-hosted OCR" leaves the scope-creep list, the architecture
   diagram no longer crosses the device boundary).

### 4. Validation window (gate for the cleanup commit)

Closes only when **all** of these hold:

- **≥ 5 calendar days** of the service running live on the Jetson
  (exercises ~5 daily systemd restarts, morning/evening/rain light,
  weekday vs weekend traffic, long-run RAM behaviour).
- **≥ 15 events manually verified** on the dashboard, spread across
  days and lighting conditions (~3/day), not taken from a single sunny
  morning.
- **At most 1 wrong read among the verified 15.** A wrong read = the
  event displays a text that does not match the plate in the photo. An
  honest `no_plate`/`failed` on a visible plate is a miss, not a wrong
  read — the bar guards lies, not misses. Any wrong read (including
  ones noticed outside the formal sample — no cherry-picking) gets
  investigated and its cause noted; a second one keeps the window open.
- **RAM < 3 GB** on `tegrastats` through day 5; pipeline FPS unchanged.

### 5. Testing

Unit (laptop, fakes only):

- `test_plate_voting.py` — every voting rule above as a small case:
  3 identical; 2+1; 1-1-1; abstentions; all-`no_plate`; single read;
  two-way tie; empty list. Student writes the assertions.
- `test_local_anpr.py` — fake engines: stage 1 finds nothing → OCR
  **not called** (counting fake); exception in either stage → `failed`,
  never raised; happy path → text + confidence + region.
- `test_crop_collector.py` — 3 crops at correct spacing; track dies
  after 1 crop → hands over 1; state cleaned on track death (dict
  bounded).
- `anpr_worker` — existing tests minimally adapted (`submit` takes a
  crop list); queue/drop semantics tests unchanged. `on_result` tests
  untouched.

Integration (TestClient): `?filter=read` returns only `success`;
`?filter=all` returns everything; default route uses the plate-read
filter; the `fără plăcuță` badge renders.

Jetson smoke test before the window opens: the Task 9 spike script,
repurposed — the `models/anpr/` files load on ORT 1.9 and read the
known `CJ45ARL` crop correctly.

## Success criteria

1. Ethernet unplugged: a passing car shows up on the dashboard with a
   correct plate (or an honest `fără plăcuță`), read entirely
   on-device.
2. The default dashboard feed contains only plate-read events; the
   "Toate" view still tells the full story.
3. The validation window closed on its recorded numbers (5 days / 15
   verified / ≤ 1 wrong), and the cleanup commit exists: no API key,
   no cloud client, CLAUDE.md amended.
4. RAM < 3 GB, pipeline FPS unchanged, daily restarts survived.
5. The student can explain the voting rules and why confidence is not
   the filter — with the bake-off numbers as evidence — in a 10-minute
   review with Radu.
