# v2 Stage A — ANPR Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `experiments/anpr_bakeoff/RESULTS.md` with real numbers that pick the local ANPR engine (OpenALPR `-c eu` vs the fast-alpr ONNX stack) per the decision rule fixed in the spec.

**Architecture:** A small evaluation harness in `experiments/anpr_bakeoff/`: pure metric functions (TDD), a canonical dataset layout, one runner script per candidate emitting a common predictions CSV, and one evaluator that turns labels+predictions into a markdown report plus confidence-calibration data. Two datasets: a public EU benchmark (openalpr/benchmarks `endtoend/eu`) and our real crops exported from the Jetson's DB. A conditional Jetson feasibility spike proves the ONNX models run on Python 3.6 before they may win.

**Tech Stack:** Python (3.6-compatible style everywhere; scripts that run on the Jetson use stdlib only), pytest, OpenALPR CLI (apt, Jetson), fast-alpr (pip, laptop), onnxruntime 1.9.0 (Jetson spike — verified cp36 aarch64 wheel exists on PyPI).

**Spec:** `docs/superpowers/specs/2026-07-18-v2-local-anpr-design.md` (approved 2026-07-18).

## Global Constraints

- **Split execution (project convention):** LAPTOP = Claude writes/commits/pushes and runs laptop-side steps; JETSON = the student runs the marked steps and pastes output back. The Jetson may also be reachable via ssh (IP moves with DHCP — rediscover via `arp -a`, MAC `00-04-4b`; see memory note). Jetson steps are **checkpoints**: do not proceed past them without the student's pasted result.
- **Python 3.6 style in every bake-off file** (they may all end up running on the Jetson): no walrus, no f-strings (repo convention is `.format()`/concatenation), stdlib only for Jetson-run scripts (`export_real_crops.py`, `run_openalpr.py`; `spike_onnx_jetson.py` additionally uses onnxruntime/numpy/yaml/cv2).
- **No new dependencies in the app's `requirements.txt`** — Stage A touches only `experiments/` (plus this plan's docs). The app itself does not change in Stage A.
- **Datasets are never committed:** the existing root `.gitignore` pattern `data/` already ignores any directory named `data` at any depth, so `experiments/anpr_bakeoff/data/` is ignored with no gitignore change. Predictions CSVs and reports under `experiments/anpr_bakeoff/predictions/` ARE committed (they're the evidence).
- **Tests for the harness live next to it** (`experiments/anpr_bakeoff/test_*.py`) and are run explicitly: `pytest experiments/anpr_bakeoff -v`. `pytest.ini` (`testpaths = tests`) keeps them out of the app's default suite — run `pytest` alone afterwards to prove the app suite (91 tests) is untouched.
- **Decision rule (verbatim from spec, fixed in advance):** the most accurate candidate on exact-match that runs on the Jetson at **< 2 s/crop** and **< 500 MB added RAM**; ONNX feasibility order: PyPI `onnxruntime==1.9.0` (CPU) → TensorRT 8.2 `trtexec` → both fail ⇒ OpenALPR wins by feasibility.
- **Laptop Python:** 3.14 is installed (`py -3.14`). If a pip resolver refuses a package on 3.14, install Python 3.12 from python.org and substitute `py -3.12` — do not downgrade the package.

## File Structure (all new, under `experiments/anpr_bakeoff/`)

```
experiments/anpr_bakeoff/
├── README.md                       # how to run everything, on which machine
├── metrics.py                      # exact_match, levenshtein, cer (pure)
├── datasets.py                     # canonical layout + predictions CSV I/O
├── evaluate.py                     # labels + predictions → report + calibration
├── convert_openalpr_benchmarks.py  # upstream benchmark → canonical dataset
├── export_real_crops.py            # JETSON: DB + data/plates → canonical dataset
├── run_openalpr.py                 # JETSON: alpr CLI → predictions CSV
├── run_fastalpr.py                 # LAPTOP: fast-alpr (2 OCR variants) → CSVs
├── spike_onnx_jetson.py            # JETSON: ORT 1.9 feasibility (conditional)
├── test_metrics.py
├── test_datasets.py
├── test_evaluate.py
├── test_convert_openalpr_benchmarks.py
├── test_export_real_crops.py
├── test_run_openalpr.py
├── test_run_fastalpr.py
├── predictions/                    # committed: *.csv, report_*.md, calib_*.csv
├── data/                           # gitignored: eu_benchmark/, real_crops/, _src/
└── RESULTS.md                      # the deliverable (Task 10)
```

**Canonical dataset layout** (produced by the two converter/export scripts, consumed by every runner and the evaluator):

```
data/<name>/
├── images/*.jpg          # images with a visible plate
├── labels.csv            # header: filename,plate_text — filename is RELATIVE
│                         #   to the dataset dir, e.g. "images/eu1.jpg"
└── plateless/*.jpg       # OPTIONAL: images guaranteed to contain no plate
```

**Predictions CSV** (one per candidate per dataset, in `predictions/`, named `<candidate>__<dataset>.csv`): header `filename,plate_text,confidence,latency_ms`; `filename` uses the same dataset-relative form (`images/…` or `plateless/…`); empty `plate_text` = the candidate read nothing (its no-plate answer); `confidence` is 0–1 (OpenALPR's 0–100 divided by 100 in its runner).

Candidate ids used in filenames and reports: `openalpr_eu`, `fastalpr_eu`, `fastalpr_global`. Dataset ids: `eu_benchmark`, `real_crops`.

---

### Task 1: Scaffolding + metrics module (LAPTOP)

**Files:**
- Create: `experiments/anpr_bakeoff/metrics.py`
- Create: `experiments/anpr_bakeoff/test_metrics.py`
- Create: `experiments/anpr_bakeoff/README.md`
- Create: `experiments/anpr_bakeoff/predictions/.gitkeep`

**Interfaces:**
- Consumes: `car_logger.services.plate_rules.normalize_plate` (existing; pure, no deps).
- Produces: `exact_match(predicted, truth) -> bool`, `levenshtein(a, b) -> int`, `cer(predicted, truth) -> float` — used by `evaluate.py` (Task 3).

- [ ] **Step 1: Create the laptop venv (one-time) and install pytest**

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install pytest
```

Expected: venv created at repo root (`.venv/` is already gitignored). All laptop `pytest`/`python` commands below mean `.venv\Scripts\pytest` / `.venv\Scripts\python`.

- [ ] **Step 2: Write the failing tests**

`experiments/anpr_bakeoff/test_metrics.py`:

```python
"""Metrics are the referee of the bake-off — they get the strictest TDD."""

from metrics import cer, exact_match, levenshtein


def test_exact_match_ignores_case_and_separators():
    # Same normalization as production: 'b-123 abc' == 'B123ABC'.
    assert exact_match("b-123 abc", "B123ABC") is True


def test_exact_match_missing_prediction_is_a_miss():
    assert exact_match(None, "B123ABC") is False
    assert exact_match("", "B123ABC") is False


def test_exact_match_wrong_text_is_a_miss():
    assert exact_match("B123ABD", "B123ABC") is False


def test_levenshtein_known_distances():
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("", "ABC") == 3
    assert levenshtein("ABC", "ABD") == 1
    assert levenshtein("AB", "ABC") == 1


def test_cer_missing_prediction_is_total_error():
    assert cer(None, "B123ABC") == 1.0


def test_cer_one_wrong_char_out_of_seven():
    assert abs(cer("B123ABD", "B123ABC") - 1.0 / 7.0) < 1e-9
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 4: Write the implementation**

`experiments/anpr_bakeoff/metrics.py`:

```python
"""Pure scoring functions for the ANPR bake-off.

Predictions are normalized with the SAME rules production uses
(car_logger.services.plate_rules.normalize_plate), so candidates are
scored on what the app would actually store."""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from car_logger.services.plate_rules import normalize_plate  # noqa: E402


def exact_match(predicted, truth):
    """True when normalized prediction equals normalized truth.

    A missing prediction (None/empty) is always a miss."""
    if not predicted:
        return False
    return normalize_plate(predicted) == normalize_plate(truth)


def levenshtein(a, b):
    """Edit distance (insert/delete/substitute), classic DP, O(len*len)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + cost))
        prev = cur
    return prev[-1]


def cer(predicted, truth):
    """Character error rate against the normalized truth. 0.0 = perfect;
    a missing prediction deletes every character (1.0)."""
    norm_truth = normalize_plate(truth) or ""
    if not norm_truth:
        raise ValueError("truth plate text must not be empty")
    norm_pred = normalize_plate(predicted) or ""
    return levenshtein(norm_pred, norm_truth) / float(len(norm_truth))
```

Note on imports: bake-off modules import each other as plain siblings (`from metrics import …`) — both `python experiments/anpr_bakeoff/x.py` and `pytest experiments/anpr_bakeoff` put the script's own directory on `sys.path`. Only the repo root (for `car_logger`) needs the explicit insert above.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_metrics.py -v`
Expected: 6 passed

- [ ] **Step 6: Create README stub and predictions dir**

`experiments/anpr_bakeoff/README.md`:

```markdown
# ANPR bake-off (v2 Stage A)

Picks the local plate-reading engine by measurement, per
`docs/superpowers/specs/2026-07-18-v2-local-anpr-design.md`.

Machines: LAPTOP = harness dev + fast-alpr runs; JETSON = OpenALPR runs,
real-crop export, ONNX feasibility spike.

## Layout
- `data/<name>/images/*.jpg` + `labels.csv` (`filename,plate_text`,
  filename relative to the dataset dir) + optional `plateless/*.jpg`.
  `data/` is gitignored — regenerate with the scripts below.
- `predictions/<candidate>__<dataset>.csv` — committed evidence.

## Recipes (details in each script's docstring)
1. Public dataset:   `python convert_openalpr_benchmarks.py` (laptop)
2. Real crops:       `python3 export_real_crops.py` (Jetson) + scp back
3. OpenALPR runs:    `python3 run_openalpr.py …` (Jetson)
4. fast-alpr runs:   `python run_fastalpr.py …` (laptop)
5. Score:            `python evaluate.py …` (laptop)
6. Spike (if ONNX wins accuracy): `python3 spike_onnx_jetson.py` (Jetson)

Harness tests: `pytest experiments/anpr_bakeoff -v` (kept out of the app
suite by pytest.ini's `testpaths = tests`).

Verdict: see `RESULTS.md`.
```

Create empty `experiments/anpr_bakeoff/predictions/.gitkeep`.

- [ ] **Step 7: Prove the harness stays out of the app suite**

The laptop cannot run the app suite at all (no fastapi in `.venv` — the app
runs on the Jetson), so the check here is isolation only:

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff --collect-only -q`
Expected: exactly the 6 metrics tests, nothing from `tests/`.

The converse guard (default `pytest` on the Jetson still collects only the
app's 91) is a student checkpoint in Task 5 Step 6 — `pytest.ini`'s
`testpaths = tests` is what protects it.

- [ ] **Step 8: Commit**

```bash
git add experiments/anpr_bakeoff
git commit -m "feat(bakeoff): metrics module - the referee of the ANPR bake-off (TDD)"
```

---

### Task 2: Dataset layout + predictions CSV I/O (LAPTOP)

**Files:**
- Create: `experiments/anpr_bakeoff/datasets.py`
- Create: `experiments/anpr_bakeoff/test_datasets.py`

**Interfaces:**
- Produces (used by every runner and `evaluate.py`):
  - `PREDICTIONS_HEADER = ["filename", "plate_text", "confidence", "latency_ms"]`
  - `load_labels(dataset_dir) -> list[(filename, plate_text)]` — filename dataset-relative (`images/…`); raises `ValueError` on missing image or empty labels.
  - `list_plateless(dataset_dir) -> list[filename]` — dataset-relative (`plateless/…`), `[]` when the folder is absent.
  - `write_predictions(path, rows)` — rows of `(filename, plate_text_or_None, confidence_or_None, latency_ms)`.
  - `read_predictions(path) -> dict[filename -> (plate_text_or_None, confidence_or_None, latency_ms)]`.

- [ ] **Step 1: Write the failing tests**

`experiments/anpr_bakeoff/test_datasets.py`:

```python
import os

import pytest

from datasets import (list_plateless, load_labels, read_predictions,
                      write_predictions)


def _make_dataset(root):
    os.makedirs(str(root / "images"))
    (root / "images" / "a.jpg").write_bytes(b"\xff\xd8fake")
    (root / "labels.csv").write_text(
        "filename,plate_text\nimages/a.jpg,B123ABC\n")


def test_load_labels_returns_relative_filename_and_text(tmp_path):
    _make_dataset(tmp_path)
    assert load_labels(str(tmp_path)) == [("images/a.jpg", "B123ABC")]


def test_load_labels_rejects_missing_image(tmp_path):
    _make_dataset(tmp_path)
    (tmp_path / "labels.csv").write_text(
        "filename,plate_text\nimages/ghost.jpg,B123ABC\n")
    with pytest.raises(ValueError):
        load_labels(str(tmp_path))


def test_load_labels_rejects_empty_labels(tmp_path):
    os.makedirs(str(tmp_path / "images"))
    (tmp_path / "labels.csv").write_text("filename,plate_text\n")
    with pytest.raises(ValueError):
        load_labels(str(tmp_path))


def test_list_plateless_empty_when_folder_absent(tmp_path):
    _make_dataset(tmp_path)
    assert list_plateless(str(tmp_path)) == []


def test_list_plateless_returns_relative_sorted(tmp_path):
    _make_dataset(tmp_path)
    os.makedirs(str(tmp_path / "plateless"))
    (tmp_path / "plateless" / "b.jpg").write_bytes(b"x")
    (tmp_path / "plateless" / "a.jpg").write_bytes(b"x")
    assert list_plateless(str(tmp_path)) == [
        "plateless/a.jpg", "plateless/b.jpg"]


def test_predictions_round_trip(tmp_path):
    path = str(tmp_path / "preds.csv")
    write_predictions(path, [
        ("images/a.jpg", "B123ABC", 0.97, 812.3),
        ("images/b.jpg", None, None, 401.0),   # candidate read nothing
    ])
    assert read_predictions(path) == {
        "images/a.jpg": ("B123ABC", 0.97, 812.3),
        "images/b.jpg": (None, None, 401.0),
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'datasets'`

- [ ] **Step 3: Write the implementation**

`experiments/anpr_bakeoff/datasets.py`:

```python
"""Canonical bake-off dataset layout + predictions CSV I/O.

Dataset dir: images/ (*.jpg with a plate) + labels.csv
(header filename,plate_text; filename RELATIVE to the dataset dir) +
optional plateless/ (*.jpg guaranteed plate-free, for false-positive rate).

Predictions CSV: filename,plate_text,confidence,latency_ms — empty
plate_text means "read nothing" (the candidate's no-plate answer);
confidence is 0-1 across ALL candidates (runners convert)."""

import csv
import os

PREDICTIONS_HEADER = ["filename", "plate_text", "confidence", "latency_ms"]
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_labels(dataset_dir):
    path = os.path.join(dataset_dir, "labels.csv")
    rows = []
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            img = os.path.join(dataset_dir, row["filename"])
            if not os.path.isfile(img):
                raise ValueError(
                    "labels.csv references missing image: {0}".format(img))
            rows.append((row["filename"], row["plate_text"]))
    if not rows:
        raise ValueError("no labels found in {0}".format(path))
    return rows


def list_plateless(dataset_dir):
    d = os.path.join(dataset_dir, "plateless")
    if not os.path.isdir(d):
        return []
    return sorted(
        "plateless/" + name for name in os.listdir(d)
        if name.lower().endswith(_IMAGE_EXTS)
    )


def write_predictions(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(PREDICTIONS_HEADER)
        for filename, plate_text, confidence, latency_ms in rows:
            writer.writerow([
                filename,
                plate_text if plate_text else "",
                "" if confidence is None else "{0:.4f}".format(confidence),
                "{0:.1f}".format(latency_ms),
            ])


def read_predictions(path):
    out = {}
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["filename"]] = (
                row["plate_text"] or None,
                float(row["confidence"]) if row["confidence"] else None,
                float(row["latency_ms"]),
            )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_datasets.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add experiments/anpr_bakeoff/datasets.py experiments/anpr_bakeoff/test_datasets.py
git commit -m "feat(bakeoff): canonical dataset layout + predictions CSV round-trip (TDD)"
```

---

### Task 3: Evaluator (LAPTOP)

**Files:**
- Create: `experiments/anpr_bakeoff/evaluate.py`
- Create: `experiments/anpr_bakeoff/test_evaluate.py`

**Interfaces:**
- Consumes: `metrics.exact_match/cer`, `datasets.load_labels/list_plateless/read_predictions`.
- Produces:
  - `score_candidate(labels, plateless_names, preds) -> dict` with keys `n`, `exact_match_rate`, `mean_cer`, `read_rate`, `fp_rate` (None when no plateless data), `mean_latency_ms`, `p95_latency_ms`.
  - `calibration_rows(labels, preds) -> list[(confidence, correct_bool)]` (only rows where the candidate read something) — the raw material for recalibrating `min_vehicle_confidence` in Stage B.
  - CLI: `python evaluate.py --dataset data/<name> --pred <candidate>=<csv> [--pred …] --out predictions/report_<name>.md` — also writes `predictions/calib_<candidate>__<name>.csv` per candidate.

- [ ] **Step 1: Write the failing tests**

`experiments/anpr_bakeoff/test_evaluate.py`:

```python
from evaluate import calibration_rows, score_candidate

LABELS = [("images/a.jpg", "B123ABC"), ("images/b.jpg", "CJ07XYZ")]
PLATELESS = ["plateless/w.jpg"]
PREDS = {
    "images/a.jpg": ("B123ABC", 0.95, 800.0),   # correct
    "images/b.jpg": ("CJ99XYZ", 0.60, 1200.0),  # wrong (2 chars)
    "plateless/w.jpg": ("FAKE123", 0.30, 700.0),  # false positive
}


def test_score_candidate_headline_numbers():
    s = score_candidate(LABELS, PLATELESS, PREDS)
    assert s["n"] == 2
    assert s["exact_match_rate"] == 0.5
    assert s["read_rate"] == 1.0
    assert s["fp_rate"] == 1.0
    assert abs(s["mean_cer"] - (0.0 + 2.0 / 7.0) / 2) < 1e-9
    assert s["mean_latency_ms"] == 1000.0


def test_score_candidate_missing_row_counts_as_no_read():
    s = score_candidate(LABELS, [], {"images/a.jpg": ("B123ABC", 0.95, 800.0)})
    assert s["read_rate"] == 0.5
    assert s["exact_match_rate"] == 0.5


def test_score_candidate_fp_rate_none_without_plateless():
    assert score_candidate(LABELS, [], PREDS)["fp_rate"] is None


def test_calibration_rows_only_actual_reads_with_correctness():
    rows = calibration_rows(LABELS, PREDS)
    assert rows == [(0.95, True), (0.60, False)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Write the implementation**

`experiments/anpr_bakeoff/evaluate.py`:

```python
"""Score candidates' predictions against a dataset's labels.

Usage (laptop, repo root):
  .venv\\Scripts\\python experiments/anpr_bakeoff/evaluate.py \\
      --dataset experiments/anpr_bakeoff/data/eu_benchmark \\
      --pred openalpr_eu=experiments/anpr_bakeoff/predictions/openalpr_eu__eu_benchmark.csv \\
      --pred fastalpr_eu=experiments/anpr_bakeoff/predictions/fastalpr_eu__eu_benchmark.csv \\
      --out experiments/anpr_bakeoff/predictions/report_eu_benchmark.md

Writes the markdown comparison table to --out (and stdout), plus one
calib_<candidate>__<dataset>.csv per candidate next to --out: rows of
confidence,correct for every image the candidate actually read — Stage B
recalibrates min_vehicle_confidence from these."""

import argparse
import csv
import os

from datasets import list_plateless, load_labels, read_predictions
from metrics import cer, exact_match

_EMPTY = (None, None, 0.0)


def score_candidate(labels, plateless_names, preds):
    hits = 0
    reads = 0
    cers = []
    latencies = []
    for filename, truth in labels:
        pred_text, _conf, latency = preds.get(filename, _EMPTY)
        latencies.append(latency)
        if pred_text:
            reads += 1
        if exact_match(pred_text, truth):
            hits += 1
        cers.append(cer(pred_text, truth))
    fp_rate = None
    if plateless_names:
        false_reads = sum(
            1 for name in plateless_names if preds.get(name, _EMPTY)[0])
        fp_rate = false_reads / float(len(plateless_names))
    n = len(labels)
    latencies.sort()
    return {
        "n": n,
        "exact_match_rate": hits / float(n),
        "mean_cer": sum(cers) / float(n),
        "read_rate": reads / float(n),
        "fp_rate": fp_rate,
        "mean_latency_ms": sum(latencies) / float(n),
        "p95_latency_ms": latencies[max(0, int(0.95 * n) - 1)],
    }


def calibration_rows(labels, preds):
    rows = []
    for filename, truth in labels:
        pred_text, conf, _latency = preds.get(filename, _EMPTY)
        if pred_text:
            rows.append((conf, exact_match(pred_text, truth)))
    return rows


def _fmt_pct(value):
    return "n/a" if value is None else "{0:.1%}".format(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred", action="append", required=True,
                        help="candidate=predictions.csv (repeatable)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = load_labels(args.dataset)
    plateless = list_plateless(args.dataset)
    dataset_name = os.path.basename(os.path.normpath(args.dataset))

    lines = [
        "## Dataset `{0}` — {1} labeled, {2} plateless".format(
            dataset_name, len(labels), len(plateless)),
        "",
        "| candidate | n | exact match | mean CER | read rate | FP rate | mean ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for spec in args.pred:
        candidate, _, csv_path = spec.partition("=")
        preds = read_predictions(csv_path)
        s = score_candidate(labels, plateless, preds)
        lines.append(
            "| {0} | {1} | {2} | {3:.3f} | {4} | {5} | {6:.0f} | {7:.0f} |".format(
                candidate, s["n"], _fmt_pct(s["exact_match_rate"]),
                s["mean_cer"], _fmt_pct(s["read_rate"]),
                _fmt_pct(s["fp_rate"]), s["mean_latency_ms"],
                s["p95_latency_ms"]))
        calib_path = os.path.join(
            os.path.dirname(args.out),
            "calib_{0}__{1}.csv".format(candidate, dataset_name))
        with open(calib_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["confidence", "correct"])
            for conf, correct in calibration_rows(labels, preds):
                writer.writerow([
                    "" if conf is None else "{0:.4f}".format(conf),
                    "1" if correct else "0"])
    report = "\n".join(lines) + "\n"
    with open(args.out, "w") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_evaluate.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add experiments/anpr_bakeoff/evaluate.py experiments/anpr_bakeoff/test_evaluate.py
git commit -m "feat(bakeoff): evaluator - comparison table + confidence calibration dump (TDD)"
```

---

### Task 4: Public EU dataset (LAPTOP)

**Files:**
- Create: `experiments/anpr_bakeoff/convert_openalpr_benchmarks.py`
- Create: `experiments/anpr_bakeoff/test_convert_openalpr_benchmarks.py`

**Interfaces:**
- Consumes: upstream repo `openalpr/benchmarks`, folder `endtoend/eu` — verified format (2026-07-18): pairs `<name>.jpg` + `<name>.txt`; each txt line is TAB-separated `filename x y w h plate_text` (e.g. `eu1.jpg	396	340	203	46	M5XSX`).
- Produces: `data/eu_benchmark/` in canonical layout; `parse_annotation_line(line) -> (filename, plate_text)`.

- [ ] **Step 1: Write the failing test**

`experiments/anpr_bakeoff/test_convert_openalpr_benchmarks.py`:

```python
import pytest

from convert_openalpr_benchmarks import parse_annotation_line


def test_parse_annotation_line_takes_filename_and_last_field():
    line = "eu1.jpg\t396\t340\t203\t46\tM5XSX\n"
    assert parse_annotation_line(line) == ("eu1.jpg", "M5XSX")


def test_parse_annotation_line_rejects_garbage():
    with pytest.raises(ValueError):
        parse_annotation_line("not-an-annotation\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_convert_openalpr_benchmarks.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`experiments/anpr_bakeoff/convert_openalpr_benchmarks.py`:

```python
"""Convert openalpr/benchmarks endtoend/eu into the canonical layout.

One-time, laptop:
  git clone --depth 1 --filter=blob:none --sparse \\
      https://github.com/openalpr/benchmarks \\
      experiments/anpr_bakeoff/data/_src/benchmarks
  git -C experiments/anpr_bakeoff/data/_src/benchmarks sparse-checkout set endtoend/eu
  .venv\\Scripts\\python experiments/anpr_bakeoff/convert_openalpr_benchmarks.py

Each image has a sibling .txt whose tab-separated line ends with the
ground-truth plate text (filename x y w h plate)."""

import csv
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "data", "_src", "benchmarks", "endtoend", "eu")
OUT = os.path.join(_HERE, "data", "eu_benchmark")


def parse_annotation_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 6:
        raise ValueError("unexpected annotation line: {0!r}".format(line))
    return parts[0], parts[-1]


def convert(src_dir, out_dir):
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(src_dir, name), "r") as fh:
            first_line = fh.readline()
        img_name, plate_text = parse_annotation_line(first_line)
        src_img = os.path.join(src_dir, img_name)
        if not os.path.isfile(src_img):
            print("skip (no image): {0}".format(name))
            continue
        shutil.copyfile(src_img, os.path.join(images_dir, img_name))
        labels.append(("images/" + img_name, plate_text))
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "plate_text"])
        writer.writerows(labels)
    return len(labels)


if __name__ == "__main__":
    count = convert(SRC, OUT)
    print("eu_benchmark: {0} labeled images".format(count))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_convert_openalpr_benchmarks.py -v`
Expected: 2 passed

- [ ] **Step 5: Clone the source and run the conversion**

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/openalpr/benchmarks experiments/anpr_bakeoff/data/_src/benchmarks
git -C experiments/anpr_bakeoff/data/_src/benchmarks sparse-checkout set endtoend/eu
.venv\Scripts\python experiments/anpr_bakeoff/convert_openalpr_benchmarks.py
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'experiments/anpr_bakeoff'); from datasets import load_labels; print(len(load_labels('experiments/anpr_bakeoff/data/eu_benchmark')))"
```

Expected: conversion prints ~100+ labeled images; the loader validates and prints the same count. If the format differs from the verified sample (multi-line txt, different separator), STOP and adapt `parse_annotation_line` with a new test using a real line — do not guess silently.

**Known deviation from spec, on record:** the spec preferred ~500–1000 public images; this source has ~108 with verified ground-truth text. Rule (see Task 8): if the top two candidates land within 10 percentage points of exact-match on this dataset, we add a second public source before deciding. Verified text labels beat unverifiable bulk.

- [ ] **Step 6: Commit**

```bash
git add experiments/anpr_bakeoff/convert_openalpr_benchmarks.py experiments/anpr_bakeoff/test_convert_openalpr_benchmarks.py
git commit -m "feat(bakeoff): openalpr-benchmarks eu converter + dataset recipe (TDD)"
```

---

### Task 5: Real-crops export (LAPTOP writes, JETSON runs — CHECKPOINT)

**Files:**
- Create: `experiments/anpr_bakeoff/export_real_crops.py`
- Create: `experiments/anpr_bakeoff/test_export_real_crops.py`

**Interfaces:**
- Consumes: the Jetson's `car_logger.db` (table `events`: `id`, `plate_text`, `anpr_status`) and `data/plates/<event_id>.jpg` (both live only on the Jetson).
- Produces: `data/real_crops/` in canonical layout; `export(db_path, plates_dir, out_dir) -> int`.

- [ ] **Step 1: Write the failing test**

`experiments/anpr_bakeoff/test_export_real_crops.py`:

```python
import csv
import os
import sqlite3

from export_real_crops import export


def _seed(db_path, plates_dir, rows):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, plate_text TEXT, "
        "anpr_status TEXT NOT NULL)")
    conn.executemany("INSERT INTO events VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    os.makedirs(plates_dir)


def test_export_only_successful_reads_with_existing_crop(tmp_path):
    db = str(tmp_path / "car_logger.db")
    plates = str(tmp_path / "plates")
    out = str(tmp_path / "real_crops")
    _seed(db, plates, [
        (1, "B123ABC", "success"),   # exported
        (2, None, "failed"),         # not success -> skipped
        (3, "CJ07XYZ", "success"),   # crop missing on disk -> skipped
    ])
    (tmp_path / "plates" / "1.jpg").write_bytes(b"\xff\xd8fake")

    assert export(db, plates, out) == 1

    with open(os.path.join(out, "labels.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [{"filename": "images/event_1.jpg",
                     "plate_text": "B123ABC"}]
    assert os.path.isfile(os.path.join(out, "images", "event_1.jpg"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_export_real_crops.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`experiments/anpr_bakeoff/export_real_crops.py`:

```python
"""Export real plate crops + their cloud readings as a bake-off dataset.

Runs ON THE JETSON (python3.6, stdlib only), from the app directory:
  cd ~/jetson-car-logger
  python3 experiments/anpr_bakeoff/export_real_crops.py \\
      --db car_logger.db --plates data/plates \\
      --out experiments/anpr_bakeoff/data/real_crops

The cloud API's successful reads are the ground truth the local
candidates are measured against. Read-only on the DB."""

import argparse
import csv
import os
import shutil
import sqlite3


def export(db_path, plates_dir, out_dir):
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, plate_text FROM events "
            "WHERE anpr_status = 'success' AND plate_text IS NOT NULL "
            "ORDER BY id").fetchall()
    finally:
        conn.close()
    labels = []
    for event_id, plate_text in rows:
        src = os.path.join(plates_dir, "{0}.jpg".format(event_id))
        if not os.path.isfile(src):
            continue
        rel = "images/event_{0}.jpg".format(event_id)
        shutil.copyfile(src, os.path.join(out_dir, rel.replace("/", os.sep)))
        labels.append((rel, plate_text))
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "plate_text"])
        writer.writerows(labels)
    return len(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="car_logger.db")
    parser.add_argument("--plates", default="data/plates")
    parser.add_argument("--out",
                        default="experiments/anpr_bakeoff/data/real_crops")
    args = parser.parse_args()
    count = export(args.db, args.plates, args.out)
    print("real_crops: {0} labeled images".format(count))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_export_real_crops.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit and push**

```bash
git add experiments/anpr_bakeoff/export_real_crops.py experiments/anpr_bakeoff/test_export_real_crops.py
git commit -m "feat(bakeoff): Jetson real-crops exporter - cloud reads become ground truth (TDD)"
git push
```

- [ ] **Step 6: CHECKPOINT (JETSON, student) — run the export**

```bash
cd ~/jetson-car-logger && git pull
pytest
python3 experiments/anpr_bakeoff/export_real_crops.py
```

Expected: `pytest` still reports **91 passed** (proof the harness stayed out of the app suite — the counterpart of Task 1 Step 7), then `real_crops: N labeled images` with N < 50 (per current DB estimate). Student pastes both.

- [ ] **Step 7: CHECKPOINT (JETSON→LAPTOP) — copy the dataset to the laptop**

On the laptop (find the current IP first if it moved: `arp -a | findstr "00-04-4b"`):

```powershell
scp -r tudor@<JETSON_IP>:~/jetson-car-logger/experiments/anpr_bakeoff/data/real_crops experiments/anpr_bakeoff/data/real_crops
.venv\Scripts\python -c "import sys; sys.path.insert(0, 'experiments/anpr_bakeoff'); from datasets import load_labels; print(len(load_labels('experiments/anpr_bakeoff/data/real_crops')))"
```

Expected: loader prints the same N.

- [ ] **Step 8: CHECKPOINT (student, manual) — plateless triage**

The student eyeballs `data/real_crops/images/` for crops where no plate is
visible at all (bad angle, rear without plate). For each such image: move
it into `data/real_crops/plateless/` and delete its row from `labels.csv`.
This is judgment work — small N, done by hand, on the laptop copy. If none
qualify, skip: the FP-rate column will read n/a and the harness handles it.
(Sync the same moves back to the Jetson copy later only if we rerun there —
the Jetson runners in Task 6 pull the laptop-curated copy via scp anyway.)

---

### Task 6: OpenALPR runner (LAPTOP writes, JETSON runs — CHECKPOINT)

**Files:**
- Create: `experiments/anpr_bakeoff/run_openalpr.py`
- Create: `experiments/anpr_bakeoff/test_run_openalpr.py`

**Interfaces:**
- Consumes: `datasets.load_labels/list_plateless/write_predictions`; the `alpr` CLI (`-c eu -j -n 1`), whose JSON has `results[0].plate`, `results[0].confidence` (0–100) and top-level `processing_time_ms`.
- Produces: `predictions/openalpr_eu__<dataset>.csv` per dataset; `parse_alpr_json(text) -> (plate_text_or_None, confidence_0_1_or_None, processing_ms)`.

- [ ] **Step 1: Write the failing tests**

`experiments/anpr_bakeoff/test_run_openalpr.py`:

```python
import json

from run_openalpr import parse_alpr_json


def test_parse_alpr_json_takes_best_result_and_scales_confidence():
    payload = json.dumps({
        "processing_time_ms": 421.7,
        "results": [{"plate": "M5XSX", "confidence": 89.5}],
    })
    assert parse_alpr_json(payload) == ("M5XSX", 0.895, 421.7)


def test_parse_alpr_json_no_results_means_no_read():
    payload = json.dumps({"processing_time_ms": 380.0, "results": []})
    assert parse_alpr_json(payload) == (None, None, 380.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_run_openalpr.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`experiments/anpr_bakeoff/run_openalpr.py`:

```python
"""Run the OpenALPR CLI candidate over a dataset. JETSON, py3.6, stdlib.

  python3 experiments/anpr_bakeoff/run_openalpr.py \\
      --dataset experiments/anpr_bakeoff/data/real_crops \\
      --out experiments/anpr_bakeoff/predictions/openalpr_eu__real_crops.csv

Latency honesty: every alpr invocation pays process start + model load,
so the CSV's wall latency OVERSTATES an in-process integration. The JSON's
processing_time_ms (model time only) understates it; we print both
averages at the end and record wall time in the CSV as the worst case."""

import argparse
import json
import subprocess
import sys
import time

from datasets import list_plateless, load_labels, write_predictions


def parse_alpr_json(output_text):
    payload = json.loads(output_text)
    ms = float(payload.get("processing_time_ms", 0.0))
    results = payload.get("results", [])
    if not results:
        return None, None, ms
    best = results[0]
    return best.get("plate"), float(best.get("confidence", 0.0)) / 100.0, ms


def run(dataset_dir, out_csv):
    names = [f for f, _ in load_labels(dataset_dir)]
    names += list_plateless(dataset_dir)
    rows = []
    model_ms = []
    for rel in names:
        img = dataset_dir.rstrip("/") + "/" + rel
        t0 = time.time()
        proc = subprocess.run(
            ["alpr", "-c", "eu", "-j", "-n", "1", img],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wall_ms = (time.time() - t0) * 1000.0
        if proc.returncode != 0:
            sys.stderr.write("alpr failed on {0}: {1}\n".format(
                rel, proc.stderr.decode("utf-8", "replace").strip()))
            rows.append((rel, None, None, wall_ms))
            continue
        plate, conf, ms = parse_alpr_json(proc.stdout.decode("utf-8"))
        model_ms.append(ms)
        rows.append((rel, plate, conf, wall_ms))
    write_predictions(out_csv, rows)
    walls = [r[3] for r in rows]
    print("{0} images | wall avg {1:.0f} ms | model-only avg {2:.0f} ms".format(
        len(rows), sum(walls) / len(walls),
        sum(model_ms) / len(model_ms) if model_ms else 0.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.dataset, args.out)
```

(py3.6 note: `subprocess.run` exists; `capture_output=` does not — hence explicit `stdout=/stderr=`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_run_openalpr.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit and push**

```bash
git add experiments/anpr_bakeoff/run_openalpr.py experiments/anpr_bakeoff/test_run_openalpr.py
git commit -m "feat(bakeoff): OpenALPR CLI runner with honest dual latency reporting (TDD)"
git push
```

- [ ] **Step 6: CHECKPOINT (JETSON, student) — install OpenALPR and smoke-test**

```bash
sudo apt-get update && sudo apt-get install -y openalpr
which alpr && alpr --version
```

Expected: a path and a version string. If `alpr` is missing after install, try `sudo apt-get install -y openalpr-utils` (Debian sometimes splits the CLI) and paste what apt says either way.

- [ ] **Step 7: CHECKPOINT (LAPTOP→JETSON) — ship both curated datasets, run both, fetch CSVs**

Laptop (ship the curated copies — the Task 5 triage happened on the laptop):

```powershell
scp -r experiments/anpr_bakeoff/data/eu_benchmark tudor@<JETSON_IP>:~/jetson-car-logger/experiments/anpr_bakeoff/data/eu_benchmark
scp -r experiments/anpr_bakeoff/data/real_crops tudor@<JETSON_IP>:~/jetson-car-logger/experiments/anpr_bakeoff/data/real_crops
```

Jetson (student; `/usr/bin/time -v` captures peak RSS = the RAM number for the decision rule):

```bash
cd ~/jetson-car-logger && git pull
mkdir -p experiments/anpr_bakeoff/predictions
/usr/bin/time -v python3 experiments/anpr_bakeoff/run_openalpr.py \
    --dataset experiments/anpr_bakeoff/data/eu_benchmark \
    --out experiments/anpr_bakeoff/predictions/openalpr_eu__eu_benchmark.csv 2> openalpr_time_eu.txt
tail -n 20 openalpr_time_eu.txt
python3 experiments/anpr_bakeoff/run_openalpr.py \
    --dataset experiments/anpr_bakeoff/data/real_crops \
    --out experiments/anpr_bakeoff/predictions/openalpr_eu__real_crops.csv
```

Student pastes: both runners' summary lines + the `Maximum resident set size` line from `openalpr_time_eu.txt`. Laptop then fetches the CSVs:

```powershell
scp "tudor@<JETSON_IP>:~/jetson-car-logger/experiments/anpr_bakeoff/predictions/openalpr_eu__*.csv" experiments/anpr_bakeoff/predictions/
git add experiments/anpr_bakeoff/predictions/openalpr_eu__*.csv
git commit -m "data(bakeoff): OpenALPR eu predictions on both datasets (Jetson run)"
```

---

### Task 7: fast-alpr runner (LAPTOP)

**Files:**
- Create: `experiments/anpr_bakeoff/run_fastalpr.py`
- Create: `experiments/anpr_bakeoff/test_run_fastalpr.py`

**Interfaces:**
- Consumes: `fast_alpr.ALPR(detector_model=…, ocr_model=…)`, `.predict(path)` → list of results with `.ocr.text` / `.ocr.confidence` (verified API surface 2026-07-18); `datasets.write_predictions`.
- Produces: `predictions/fastalpr_eu__<dataset>.csv` and `predictions/fastalpr_global__<dataset>.csv` for both datasets. OCR variants (verified names from the fast-plate-ocr model hub): `european-plates-mobile-vit-v2-model`, `cct-xs-v2-global-model`; detector: `yolo-v9-t-384-license-plate-end2end`. Models cache under `~/.cache/fast-plate-ocr/` (OCR) — the spike (Task 9) scp's the cached `.onnx`+config files to the Jetson.

- [ ] **Step 1: Install fast-alpr in the laptop venv**

```powershell
.venv\Scripts\python -m pip install "fast-alpr[onnx]"
```

Expected: installs cleanly on Python 3.14 (pulls onnxruntime CPU + opencv). If the resolver refuses 3.14, per Global Constraints: install Python 3.12, recreate `.venv` with `py -3.12`, reinstall pytest + fast-alpr, rerun the harness tests (they are interpreter-agnostic).

- [ ] **Step 2: API verification on one image (before writing the batch loop around it)**

```powershell
.venv\Scripts\python -c "from fast_alpr import ALPR; a = ALPR(detector_model='yolo-v9-t-384-license-plate-end2end', ocr_model='european-plates-mobile-vit-v2-model'); r = a.predict('experiments/anpr_bakeoff/data/eu_benchmark/images/eu1.jpg'); print(repr(r))"
```

Expected: a list with one result whose OCR text resembles `M5XSX`; note the exact attribute names in the printed repr. If they differ from `.ocr.text`/`.ocr.confidence`, adjust `_best` and its test below to the observed names before the batch run — this repr is the contract check.

- [ ] **Step 3: Write the failing test**

`experiments/anpr_bakeoff/test_run_fastalpr.py`:

```python
"""_best is the only local logic in the fast-alpr runner; stub results
mirror the API surface verified in the repr check (ocr.text/confidence)."""

from collections import namedtuple

from run_fastalpr import _best

Ocr = namedtuple("Ocr", ["text", "confidence"])
Res = namedtuple("Res", ["ocr"])


def test_best_none_when_nothing_detected():
    assert _best([]) is None
    assert _best([Res(ocr=None)]) is None


def test_best_picks_highest_ocr_confidence():
    a = Res(ocr=Ocr("B123ABC", 0.91))
    b = Res(ocr=Ocr("B123ABD", 0.72))
    assert _best([b, a]) is a
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_run_fastalpr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_fastalpr'`

- [ ] **Step 5: Write the runner**

`experiments/anpr_bakeoff/run_fastalpr.py`:

```python
"""Run the fast-alpr candidate (detector + one OCR variant). LAPTOP.

  .venv\\Scripts\\python experiments/anpr_bakeoff/run_fastalpr.py \\
      --dataset experiments/anpr_bakeoff/data/eu_benchmark \\
      --ocr european-plates-mobile-vit-v2-model \\
      --out experiments/anpr_bakeoff/predictions/fastalpr_eu__eu_benchmark.csv

Laptop latency is INDICATIVE ONLY (different CPU than the Jetson) — the
on-device number comes from the Task 9 spike. Accuracy, however, is a
property of the models, and that is what this run measures."""

import argparse
import os
import time

from datasets import list_plateless, load_labels, write_predictions

from fast_alpr import ALPR

DETECTOR = "yolo-v9-t-384-license-plate-end2end"


def _best(results):
    """Highest-OCR-confidence detection, or None when nothing was found."""
    best = None
    for r in results:
        if r.ocr is None:
            continue
        if best is None or r.ocr.confidence > best.ocr.confidence:
            best = r
    return best


def run(dataset_dir, ocr_model, out_csv):
    alpr = ALPR(detector_model=DETECTOR, ocr_model=ocr_model)
    names = [f for f, _ in load_labels(dataset_dir)]
    names += list_plateless(dataset_dir)
    rows = []
    for rel in names:
        img = os.path.join(dataset_dir, rel.replace("/", os.sep))
        t0 = time.time()
        results = alpr.predict(img)
        wall_ms = (time.time() - t0) * 1000.0
        best = _best(results)
        if best is None:
            rows.append((rel, None, None, wall_ms))
        else:
            rows.append((rel, best.ocr.text,
                         float(best.ocr.confidence), wall_ms))
    write_predictions(out_csv, rows)
    walls = [r[3] for r in rows]
    print("{0} images | wall avg {1:.0f} ms (laptop, indicative)".format(
        len(rows), sum(walls) / len(walls)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.dataset, args.ocr, args.out)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff/test_run_fastalpr.py -v`
Expected: 2 passed

- [ ] **Step 7: Run all four combinations**

```powershell
.venv\Scripts\python experiments/anpr_bakeoff/run_fastalpr.py --dataset experiments/anpr_bakeoff/data/eu_benchmark --ocr european-plates-mobile-vit-v2-model --out experiments/anpr_bakeoff/predictions/fastalpr_eu__eu_benchmark.csv
.venv\Scripts\python experiments/anpr_bakeoff/run_fastalpr.py --dataset experiments/anpr_bakeoff/data/real_crops --ocr european-plates-mobile-vit-v2-model --out experiments/anpr_bakeoff/predictions/fastalpr_eu__real_crops.csv
.venv\Scripts\python experiments/anpr_bakeoff/run_fastalpr.py --dataset experiments/anpr_bakeoff/data/eu_benchmark --ocr cct-xs-v2-global-model --out experiments/anpr_bakeoff/predictions/fastalpr_global__eu_benchmark.csv
.venv\Scripts\python experiments/anpr_bakeoff/run_fastalpr.py --dataset experiments/anpr_bakeoff/data/real_crops --ocr cct-xs-v2-global-model --out experiments/anpr_bakeoff/predictions/fastalpr_global__real_crops.csv
```

Expected: four CSVs, each printing its image count + laptop wall average.

- [ ] **Step 8: Commit**

```bash
git add experiments/anpr_bakeoff/run_fastalpr.py experiments/anpr_bakeoff/test_run_fastalpr.py experiments/anpr_bakeoff/predictions/fastalpr_*.csv
git commit -m "feat(bakeoff): fast-alpr runner + predictions for both OCR variants (TDD)"
```

---

### Task 8: Score everything, draft RESULTS.md, close-call gate (LAPTOP)

**Files:**
- Create: `experiments/anpr_bakeoff/RESULTS.md` (draft)
- Create: `experiments/anpr_bakeoff/predictions/report_eu_benchmark.md`, `…/report_real_crops.md` (generated)

**Interfaces:**
- Consumes: `evaluate.py` CLI (Task 3), all six predictions CSVs (Tasks 6–7).
- Produces: accuracy tables + calibration CSVs; the input for the spike-or-not branch (Task 9) and the final decision (Task 10).

- [ ] **Step 1: Generate both reports**

```powershell
.venv\Scripts\python experiments/anpr_bakeoff/evaluate.py --dataset experiments/anpr_bakeoff/data/eu_benchmark --pred openalpr_eu=experiments/anpr_bakeoff/predictions/openalpr_eu__eu_benchmark.csv --pred fastalpr_eu=experiments/anpr_bakeoff/predictions/fastalpr_eu__eu_benchmark.csv --pred fastalpr_global=experiments/anpr_bakeoff/predictions/fastalpr_global__eu_benchmark.csv --out experiments/anpr_bakeoff/predictions/report_eu_benchmark.md
.venv\Scripts\python experiments/anpr_bakeoff/evaluate.py --dataset experiments/anpr_bakeoff/data/real_crops --pred openalpr_eu=experiments/anpr_bakeoff/predictions/openalpr_eu__real_crops.csv --pred fastalpr_eu=experiments/anpr_bakeoff/predictions/fastalpr_eu__real_crops.csv --pred fastalpr_global=experiments/anpr_bakeoff/predictions/fastalpr_global__real_crops.csv --out experiments/anpr_bakeoff/predictions/report_real_crops.md
```

Expected: two markdown tables on stdout + six `calib_*.csv` files.

- [ ] **Step 2: Apply the close-call gate**

Read the `eu_benchmark` table. **If** the best and second-best candidates' exact-match rates are within 10 percentage points, the sample (~108) cannot separate them: add a second public source before deciding — pick an EU plate-OCR dataset with per-image text labels from Roboflow Universe or Kaggle, write a converter to the canonical layout in the style of `convert_openalpr_benchmarks.py` (new file + test, same TDD cycle), regenerate predictions for the new dataset with Tasks 6–7's runners, and re-run Step 1. **Else** proceed.

- [ ] **Step 3: Draft `experiments/anpr_bakeoff/RESULTS.md`**

```markdown
# ANPR bake-off — results

**Date:** <fill> · **Status:** DRAFT until the Task 9/10 boxes are ticked.

## Decision rule (fixed in the spec BEFORE measuring)
Winner = most accurate on exact-match that runs on the Jetson at
< 2 s/crop and < 500 MB added RAM. ONNX feasibility order: PyPI
onnxruntime 1.9.0 (CPU) → TensorRT 8.2 trtexec → both fail ⇒ OpenALPR
wins by feasibility.

## Accuracy — eu_benchmark (public, N=<fill>)
<paste report_eu_benchmark.md table>

## Accuracy — real_crops (ours, N=<fill>, cloud reads as ground truth)
<paste report_real_crops.md table>

## On-device numbers (Jetson)
| candidate | latency/crop | peak RSS | source |
|---|---|---|---|
| openalpr_eu | <wall avg from Task 6> | <Maximum resident set size, Task 6> | measured, alpr CLI (per-process; in-process will be faster) |
| fastalpr_* | <from Task 9 spike, or "not needed"> | <idem> | spike |

## Verdict
<filled by Task 10 — winner + why, citing the rule and the tables>

## Confidence calibration notes (for Stage B's min_vehicle_confidence)
<from calib_<winner>__*.csv: at which confidence do wrong reads die out?
1-2 sentences + the threshold the STUDENT picks.>
```

Fill every `<fill>` that exists at this point (accuracy tables, Ns, OpenALPR device numbers). Leave only Task 9/10 cells open.

- [ ] **Step 4: Commit**

```bash
git add experiments/anpr_bakeoff/RESULTS.md experiments/anpr_bakeoff/predictions/report_*.md experiments/anpr_bakeoff/predictions/calib_*.csv
git commit -m "data(bakeoff): scored reports + draft RESULTS with decision rule restated"
```

- [ ] **Step 5: Branch**

If a fast-alpr variant leads on exact-match → Task 9 (spike). If OpenALPR leads outright → skip to Task 10.

---

### Task 9: ONNX-on-Jetson feasibility spike (LAPTOP writes, JETSON runs — CHECKPOINT; CONDITIONAL)

**Files:**
- Create: `experiments/anpr_bakeoff/spike_onnx_jetson.py`

**Interfaces:**
- Consumes: the laptop's cached model files (OCR: `~/.cache/fast-plate-ocr/<model>/ *.onnx` + `*_config.yaml`; detector: search `~/.cache/` for the `yolo-v9-t-384*` onnx — fast-alpr's detector package caches similarly), `onnxruntime==1.9.0` (verified cp36 aarch64 manylinux wheel on PyPI), the Jetson's system `numpy`/`cv2` (JetPack) + `pyyaml` (app venv has 5.4.1).
- Produces: PASS/FAIL feasibility verdict + measured latency and RSS for the RESULTS table. Spike code is throwaway-quality by design but still committed (it documents HOW we know).

- [ ] **Step 1: Locate and ship the model files**

Laptop — find the cached files (they were downloaded during Task 7):

```powershell
Get-ChildItem -Recurse $env:USERPROFILE\.cache -Include *.onnx, *config*.yaml | Select-Object FullName
ssh tudor@<JETSON_IP> "mkdir -p ~/anpr_spike"
scp <the .onnx and config .yaml paths found above> tudor@<JETSON_IP>:~/anpr_spike/
```

Expected: at least the winning OCR model's `.onnx` + its config yaml, and the detector `.onnx`, listed and copied. (If the cache lives elsewhere on this machine, `pip show fast-plate-ocr` + its docs name `~/.cache/fast-plate-ocr` — search `%LOCALAPPDATA%` too before concluding it's missing.)

- [ ] **Step 2: CHECKPOINT (JETSON, student) — spike venv + onnxruntime install**

```bash
python3 -m venv --system-site-packages ~/anpr_spike_venv
~/anpr_spike_venv/bin/pip install --upgrade pip
~/anpr_spike_venv/bin/pip install onnxruntime==1.9.0 pyyaml==5.4.1
~/anpr_spike_venv/bin/python -c "import onnxruntime, numpy, cv2, yaml; print(onnxruntime.__version__)"
```

Expected: `1.9.0`. (`--system-site-packages` picks up JetPack's numpy/cv2; the app venv stays untouched.) If pip cannot find the wheel, paste the exact error — do NOT compile from source; that is the trigger to move to the trtexec fallback (Step 6).

- [ ] **Step 3: Write the spike script**

`experiments/anpr_bakeoff/spike_onnx_jetson.py`:

```python
"""Feasibility spike: do the fast-plate-ocr/fast-alpr ONNX models run on
this Jetson's Python 3.6 with onnxruntime 1.9 (CPU)? JETSON only.

  ~/anpr_spike_venv/bin/python experiments/anpr_bakeoff/spike_onnx_jetson.py \\
      --ocr-model ~/anpr_spike/european_mobile_vit_v2_ocr.onnx \\
      --ocr-config ~/anpr_spike/european_mobile_vit_v2_ocr_config.yaml \\
      --detector-model ~/anpr_spike/yolo-v9-t-384-license-plate-end2end.onnx \\
      --image experiments/anpr_bakeoff/data/real_crops/images/<pick one>.jpg \\
      --expect <that crop's text from labels.csv>

PASS = OCR text matches the laptop's prediction for the same crop AND the
timed loop stays under the 2 s/crop budget. Prints everything it learns
(input names/shapes/dtypes) so a failure is still informative."""

import argparse
import resource
import time

import cv2
import numpy as np
import onnxruntime as ort
import yaml


def load_ocr(model_path, config_path):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    print("ocr config: {0}".format(cfg))
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    meta = sess.get_inputs()[0]
    print("ocr input: name={0} shape={1} type={2}".format(
        meta.name, meta.shape, meta.type))
    return sess, cfg, meta


def preprocess_plate(img_path, cfg, meta):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit("cannot read " + img_path)
    img = cv2.resize(img, (cfg["img_width"], cfg["img_height"]))
    arr = img[np.newaxis, :, :, np.newaxis]  # NHWC, batch 1, grayscale
    if meta.type == "tensor(float)":
        arr = arr.astype("float32")
    else:
        arr = arr.astype("uint8")
    return arr


def decode_ocr(outputs, cfg):
    """fast-plate-ocr heads emit per-slot char probabilities; decode =
    argmax per slot through the alphabet, pad char stripped."""
    probs = np.asarray(outputs[0])
    if probs.ndim == 3:          # (1, slots, alphabet)
        probs = probs[0]
    alphabet = cfg["alphabet"]
    pad = cfg.get("pad_char", "_")
    idxs = probs.argmax(axis=-1)
    text = "".join(alphabet[i] for i in idxs).replace(pad, "")
    conf = float(probs.max(axis=-1).mean())
    return text, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-model", required=True)
    parser.add_argument("--ocr-config", required=True)
    parser.add_argument("--detector-model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expect", required=True)
    args = parser.parse_args()

    # --- OCR stage (the semantic check) ---
    sess, cfg, meta = load_ocr(args.ocr_model, args.ocr_config)
    arr = preprocess_plate(args.image, cfg, meta)
    outputs = sess.run(None, {meta.name: arr})
    text, conf = decode_ocr(outputs, cfg)
    print("ocr read: {0!r} (conf {1:.3f}), expected {2!r}".format(
        text, conf, args.expect))

    n = 20
    t0 = time.time()
    for _ in range(n):
        sess.run(None, {meta.name: arr})
    per_crop_ms = (time.time() - t0) * 1000.0 / n
    print("ocr latency: {0:.0f} ms/crop over {1} runs".format(per_crop_ms, n))

    # --- Detector stage (load + run + shape sanity; Stage B does real decode) ---
    det = ort.InferenceSession(
        args.detector_model, providers=["CPUExecutionProvider"])
    dmeta = det.get_inputs()[0]
    print("detector input: name={0} shape={1} type={2}".format(
        dmeta.name, dmeta.shape, dmeta.type))
    side = int(dmeta.shape[-1]) if str(dmeta.shape[-1]).isdigit() else 384
    frame = cv2.imread(args.image)
    blob = cv2.resize(frame, (side, side)).astype("float32") / 255.0
    blob = blob[:, :, ::-1].transpose(2, 0, 1)[np.newaxis]  # BGR->RGB, NCHW
    t0 = time.time()
    det_out = det.run(None, {dmeta.name: np.ascontiguousarray(blob)})
    print("detector ran in {0:.0f} ms; output shapes: {1}".format(
        (time.time() - t0) * 1000.0, [np.asarray(o).shape for o in det_out]))

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print("peak RSS: {0:.0f} MB".format(rss_mb))
    ok = text.upper() == args.expect.upper() and per_crop_ms < 2000.0
    print("SPIKE {0}".format("PASS" if ok else
                             "FAIL (see numbers above)"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit and push**

```bash
git add experiments/anpr_bakeoff/spike_onnx_jetson.py
git commit -m "feat(bakeoff): onnxruntime-on-py36 feasibility spike for the fast-alpr stack"
git push
```

- [ ] **Step 5: CHECKPOINT (JETSON, student) — run the spike**

```bash
cd ~/jetson-car-logger && git pull
~/anpr_spike_venv/bin/python experiments/anpr_bakeoff/spike_onnx_jetson.py \
    --ocr-model ~/anpr_spike/european_mobile_vit_v2_ocr.onnx \
    --ocr-config ~/anpr_spike/european_mobile_vit_v2_ocr_config.yaml \
    --detector-model ~/anpr_spike/<detector file name>.onnx \
    --image experiments/anpr_bakeoff/data/real_crops/images/<one with a clear plate>.jpg \
    --expect <its labels.csv text>
```

Student pastes the FULL output (the shape/config prints matter as much as PASS/FAIL). Interpretation:
- `SPIKE PASS` → fast-alpr stack is feasible; record latency + RSS in RESULTS.
- Load error mentioning unsupported opset / `Unsupported model IR version` → Step 6.
- Text mismatch with sane latency → pre/post-processing guesswork is off; compare against the config print, adjust `preprocess_plate`/`decode_ocr` per what the config says (slots/alphabet/dims), retry once; if still off, treat as FAIL and go to Step 6.

- [ ] **Step 6 (only if Step 2 or 5 failed): TensorRT fallback**

```bash
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/anpr_spike/european_mobile_vit_v2_ocr.onnx \
    --saveEngine=$HOME/anpr_spike/ocr.trt --workspace=1024
```

- Converts and reports timings → TensorRT path is feasible: record trtexec's mean GPU time as the latency figure; RSS from `/usr/bin/time -v` around the same command. (Running the engine from Python via `tensorrt` + `pycuda` bindings becomes a Stage B task — conversion succeeding is the feasibility proof.)
- Fails on an unsupported op (attention/LayerNorm are the suspects in TRT 8.2 for MobileViTV2) → try the CCT variant (`cct_xs_v2_global.onnx`) the same way — convolutional, more likely to convert.
- Everything fails → per the decision rule, **OpenALPR wins by feasibility**. Record exactly what failed (paste trtexec's last 5 lines into RESULTS).

---

### Task 10: Verdict, final RESULTS.md, close Stage A (LAPTOP)

**Files:**
- Modify: `experiments/anpr_bakeoff/RESULTS.md` (draft → final)
- Modify: `experiments/anpr_bakeoff/README.md` (point at the verdict)

**Interfaces:**
- Consumes: everything measured in Tasks 6–9.
- Produces: the named winner + the student's chosen `min_vehicle_confidence` starting value — the two inputs Stage B's plan is written around.

- [ ] **Step 1: Fill the verdict**

Apply the decision rule to the tables — mechanically, no re-litigating: filter candidates to those with on-device proof (< 2 s/crop, < 500 MB added RSS), then pick the best exact-match among them (weigh `real_crops` as the tie-breaker — it is our camera). Name the winner in RESULTS.md's Verdict section, citing the exact rows.

- [ ] **Step 2: STUDENT DECISION — calibration read**

The student opens the winner's `calib_*` CSVs (confidence,correct), sorts by confidence, and answers in RESULTS.md's calibration section: *above which confidence are essentially all reads correct?* That number is the **starting** `min_vehicle_confidence` for Stage B (it will be re-checked live). This is the student's call, recorded with a sentence of why — same ritual as the 0.85 decision on 2026-07-08.

- [ ] **Step 3: Update README verdict pointer**

In `experiments/anpr_bakeoff/README.md`, replace the line `Verdict: see RESULTS.md.` with `Verdict (<date>): <winner> — see RESULTS.md for the numbers.`

- [ ] **Step 4: Run the full harness test suite one last time**

Run: `.venv\Scripts\pytest experiments/anpr_bakeoff -v`
Expected: **23 passed** (6 metrics + 6 datasets + 4 evaluate + 2 convert +
1 export + 2 openalpr + 2 fastalpr). The app suite needs no rerun: Stage A
never touched app code, and the Jetson confirmed "91 passed" at the Task 5
checkpoint after the harness landed.

- [ ] **Step 5: Commit**

```bash
git add experiments/anpr_bakeoff/RESULTS.md experiments/anpr_bakeoff/README.md
git commit -m "docs(bakeoff): verdict - Stage A closed, winner + starting confidence threshold recorded"
```

- [ ] **Step 6: Hand off to Stage B planning**

Stage A is done. The Stage B plan (LocalAnprClient integration of the winner, `no_plate` status, config swap, CLAUDE.md amendments, cloud-client removal as the LAST commit) is written as a separate plan **after** this verdict exists — it materially depends on which engine won. Do not start Stage B work from this document.
