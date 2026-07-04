# LPR Batch: Folder → Excel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read every image in a folder, detect + read the Romanian license plate in each, and write a spreadsheet with `filename`, `plate_text`, `confidence`.

**Architecture:** A small, isolated experiment (not the future `car_logger` app). Pure, testable helpers live in `plate_reader.py` (list images → pick best plate → write Excel); a thin CLI `read_plates.py` wires the real `fast-alpr` model to those helpers. The heavy model is injected as a `predict` callable so the logic is unit-tested without downloading or running any model.

**Tech Stack:** Python 3.10+, `fast-alpr` (bundles a YOLOv9-t ONNX plate detector + `fast-plate-ocr` CCT OCR), `openpyxl`, `pytest`.

## Global Constraints

- **Runs on an x86 laptop with Python `>=3.10`** — NOT on the Jetson. `fast-alpr`/`fast-plate-ocr` require Python 3.10; a batch accuracy test is device-independent (see `docs/research-lpr.md`).
- **Model:** `fast-alpr` with detector `yolo-v9-t-384-license-plate-end2end` and OCR `cct-s-v2-global-model` (global model reads EU/RO plates).
- **Fully offline** after the first model download; CPU only.
- **Output columns, in this exact order:** `filename`, `plate_text`, `confidence`.
- **`ocr.confidence` may be a `float` OR a `list[float]`** (one value per character) — always aggregate to a single float with the mean.
- **Keep isolated:** all files under `experiments/lpr_batch/`. Do not touch or create the `car_logger/` application package.
- **Commit style:** English messages; end each with the `Co-Authored-By` trailer used in this repo.

---

### Task 1: Scaffold + `list_images()`

**Files:**
- Create: `experiments/lpr_batch/requirements.txt`
- Create: `experiments/lpr_batch/plate_reader.py`
- Test: `experiments/lpr_batch/test_plate_reader.py`

**Interfaces:**
- Produces: `SUPPORTED_EXTENSIONS: set[str]`; `list_images(folder: str | Path) -> list[Path]` — image files directly inside `folder`, sorted by name, non-images and sub-directories excluded.

- [ ] **Step 1: Create the dependency file**

`experiments/lpr_batch/requirements.txt`:
```
fast-alpr[onnx]
openpyxl
```

- [ ] **Step 2: Write the failing test**

`experiments/lpr_batch/test_plate_reader.py`:
```python
from pathlib import Path

from plate_reader import list_images


def test_list_images_returns_only_sorted_image_files(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "sub").mkdir()

    result = list_images(tmp_path)

    assert [p.name for p in result] == ["a.png", "b.jpg"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -v`
Expected: FAIL with `ImportError` / `cannot import name 'list_images'`.

- [ ] **Step 4: Write minimal implementation**

`experiments/lpr_batch/plate_reader.py`:
```python
"""Folder -> plate rows -> Excel. Pure, testable helpers (no model here)."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder):
    """Return image files directly inside ``folder``, sorted by name."""
    folder = Path(folder)
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/lpr_batch/requirements.txt experiments/lpr_batch/plate_reader.py experiments/lpr_batch/test_plate_reader.py
git commit -m "feat(lpr): scaffold experiment and list_images helper

Co-Authored-By: Claude Fable <noreply@anthropic.com>"
```

---

### Task 2: `best_plate()` — pick the best reading, aggregate confidence

**Files:**
- Modify: `experiments/lpr_batch/plate_reader.py`
- Test: `experiments/lpr_batch/test_plate_reader.py`

**Interfaces:**
- Consumes: result objects shaped like `fast_alpr.ALPRResult` — each has an `.ocr` attribute that is either `None` or an object with `.text: str` and `.confidence: float | list[float]`.
- Produces: `best_plate(results) -> tuple[str, float]` — returns `(plate_text, confidence)` for the reading with the highest confidence; `("", 0.0)` when no plate has text. A `list` confidence is aggregated with the mean.

- [ ] **Step 1: Write the failing tests**

Append to `experiments/lpr_batch/test_plate_reader.py`:
```python
from types import SimpleNamespace

import pytest

from plate_reader import best_plate


def _result(text, confidence):
    return SimpleNamespace(ocr=SimpleNamespace(text=text, confidence=confidence))


def test_best_plate_empty_returns_blank():
    assert best_plate([]) == ("", 0.0)


def test_best_plate_picks_highest_confidence():
    results = [_result("AAA111", 0.5), _result("BBB222", 0.9)]
    assert best_plate(results) == ("BBB222", 0.9)


def test_best_plate_averages_per_character_confidence():
    results = [_result("CJ23XZI", [0.8, 1.0])]
    text, confidence = best_plate(results)
    assert text == "CJ23XZI"
    assert confidence == pytest.approx(0.9)


def test_best_plate_skips_results_without_ocr():
    results = [SimpleNamespace(ocr=None), _result("OK123", 0.7)]
    assert best_plate(results) == ("OK123", 0.7)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -k best_plate -v`
Expected: FAIL with `cannot import name 'best_plate'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/lpr_batch/plate_reader.py` (add `import statistics` at the top with the other imports):
```python
import statistics


def _aggregate_confidence(confidence):
    """Collapse a float or per-character list into one float."""
    if isinstance(confidence, list):
        return statistics.mean(confidence) if confidence else 0.0
    return float(confidence)


def best_plate(results):
    """Return (text, confidence) for the highest-confidence reading with text."""
    best_text = ""
    best_confidence = 0.0
    found = False
    for result in results:
        ocr = getattr(result, "ocr", None)
        if ocr is None or not ocr.text:
            continue
        confidence = _aggregate_confidence(ocr.confidence)
        if not found or confidence > best_confidence:
            best_text = ocr.text
            best_confidence = confidence
            found = True
    return best_text, best_confidence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -k best_plate -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/lpr_batch/plate_reader.py experiments/lpr_batch/test_plate_reader.py
git commit -m "feat(lpr): best_plate picks top reading and averages confidence

Co-Authored-By: Claude Fable <noreply@anthropic.com>"
```

---

### Task 3: `read_folder()` — build one row per image

**Files:**
- Modify: `experiments/lpr_batch/plate_reader.py`
- Test: `experiments/lpr_batch/test_plate_reader.py`

**Interfaces:**
- Consumes: `list_images`, `best_plate` (this module); a `predict` callable `predict(image_path: str) -> list` (later wired to `fast_alpr.ALPR.predict`).
- Produces: `@dataclass PlateRow(filename: str, plate_text: str, confidence: float)`; `read_folder(folder, predict) -> list[PlateRow]` — one row per image (in `list_images` order), confidence rounded to 4 decimals.

- [ ] **Step 1: Write the failing test**

Append to `experiments/lpr_batch/test_plate_reader.py`:
```python
from plate_reader import PlateRow, read_folder


def test_read_folder_builds_one_row_per_image(tmp_path):
    (tmp_path / "car1.jpg").write_bytes(b"x")
    (tmp_path / "car2.jpg").write_bytes(b"x")

    def fake_predict(image_path):
        if Path(image_path).name == "car1.jpg":
            return [SimpleNamespace(ocr=SimpleNamespace(text="CJ23XZI", confidence=0.95))]
        return []  # no plate found

    rows = read_folder(tmp_path, fake_predict)

    assert rows == [
        PlateRow(filename="car1.jpg", plate_text="CJ23XZI", confidence=0.95),
        PlateRow(filename="car2.jpg", plate_text="", confidence=0.0),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -k read_folder -v`
Expected: FAIL with `cannot import name 'PlateRow'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/lpr_batch/plate_reader.py` (add `from dataclasses import dataclass` at the top):
```python
from dataclasses import dataclass


@dataclass
class PlateRow:
    filename: str
    plate_text: str
    confidence: float


def read_folder(folder, predict):
    """Run ``predict`` on every image in ``folder`` and return one row each."""
    rows = []
    for image_path in list_images(folder):
        text, confidence = best_plate(predict(str(image_path)))
        rows.append(
            PlateRow(
                filename=image_path.name,
                plate_text=text,
                confidence=round(confidence, 4),
            )
        )
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -k read_folder -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/lpr_batch/plate_reader.py experiments/lpr_batch/test_plate_reader.py
git commit -m "feat(lpr): read_folder builds one PlateRow per image

Co-Authored-By: Claude Fable <noreply@anthropic.com>"
```

---

### Task 4: `write_excel()` — write the spreadsheet

**Files:**
- Modify: `experiments/lpr_batch/plate_reader.py`
- Test: `experiments/lpr_batch/test_plate_reader.py`

**Interfaces:**
- Consumes: `PlateRow` list.
- Produces: `write_excel(rows, out_path) -> None` — writes an `.xlsx` whose first row is the header `["filename", "plate_text", "confidence"]` followed by one row per `PlateRow`.

- [ ] **Step 1: Write the failing test**

Append to `experiments/lpr_batch/test_plate_reader.py`:
```python
from openpyxl import load_workbook

from plate_reader import write_excel


def test_write_excel_writes_header_and_rows(tmp_path):
    rows = [PlateRow(filename="car1.jpg", plate_text="CJ23XZI", confidence=0.95)]
    out_path = tmp_path / "plates.xlsx"

    write_excel(rows, out_path)

    worksheet = load_workbook(out_path).active
    assert [c.value for c in worksheet[1]] == ["filename", "plate_text", "confidence"]
    assert [c.value for c in worksheet[2]] == ["car1.jpg", "CJ23XZI", 0.95]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -k write_excel -v`
Expected: FAIL with `cannot import name 'write_excel'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/lpr_batch/plate_reader.py` (add `from openpyxl import Workbook` at the top):
```python
from openpyxl import Workbook


def write_excel(rows, out_path):
    """Write rows to an .xlsx with a header line."""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "plates"
    worksheet.append(["filename", "plate_text", "confidence"])
    for row in rows:
        worksheet.append([row.filename, row.plate_text, row.confidence])
    workbook.save(str(out_path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -k write_excel -v`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `pytest experiments/lpr_batch/test_plate_reader.py -v`
Expected: PASS (all tests from Tasks 1–4).

- [ ] **Step 6: Commit**

```bash
git add experiments/lpr_batch/plate_reader.py experiments/lpr_batch/test_plate_reader.py
git commit -m "feat(lpr): write_excel outputs the plates spreadsheet

Co-Authored-By: Claude Fable <noreply@anthropic.com>"
```

---

### Task 5: CLI `read_plates.py` + README (manual run on real photos)

**Files:**
- Create: `experiments/lpr_batch/read_plates.py`
- Create: `experiments/lpr_batch/README.md`

**Interfaces:**
- Consumes: `read_folder`, `write_excel` (this module); `fast_alpr.ALPR`.
- Produces: a command-line entry point: `python read_plates.py <folder> [--out output/plates.xlsx]`.

- [ ] **Step 1: Write the CLI**

`experiments/lpr_batch/read_plates.py`:
```python
"""CLI: read license plates from a folder of images into an Excel file."""

import argparse
from pathlib import Path

from plate_reader import read_folder, write_excel


def build_alpr():
    """Create the real fast-alpr model (downloads weights on first run)."""
    from fast_alpr import ALPR

    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v2-global-model",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Read license plates from a folder of images into an Excel file."
    )
    parser.add_argument("folder", help="Folder containing the images")
    parser.add_argument(
        "--out", default="output/plates.xlsx", help="Output .xlsx path"
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alpr = build_alpr()
    rows = read_folder(args.folder, alpr.predict)
    write_excel(rows, out_path)

    for row in rows:
        print(f"{row.filename}: {row.plate_text or '<no plate>'} ({row.confidence})")
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the README**

`experiments/lpr_batch/README.md`:
```markdown
# LPR Batch — folder → Excel

Reads every image in a folder, detects + reads the license plate, and writes
`filename, plate_text, confidence` to an `.xlsx`.

Runs on an x86 laptop with **Python 3.10+** (not the Jetson) — see
`../../docs/research-lpr.md` for why.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python read_plates.py path/to/your/photos --out output/plates.xlsx
```

The first run downloads the detector + OCR models; after that it works offline.

## Test

```bash
pytest test_plate_reader.py -v
```
```

- [ ] **Step 3: Manual smoke test on the real photos**

Put the sample photos (e.g. the `CJ 23 XZI` one) in a folder, then run:
```bash
pip install -r experiments/lpr_batch/requirements.txt
cd experiments/lpr_batch
python read_plates.py <folder-with-photos>
```
Expected: console prints each filename with a plate reading; `output/plates.xlsx` is created with the three columns. Confirm the `CJ 23 XZI` plate reads correctly (dealer-frame text is ignored).

- [ ] **Step 4: Commit**

```bash
git add experiments/lpr_batch/read_plates.py experiments/lpr_batch/README.md
git commit -m "feat(lpr): CLI to read a folder of plates into Excel

Co-Authored-By: Claude Fable <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Folder → read images: `list_images` (Task 1), `read_folder` (Task 3). ✓
- Detect plate then OCR (ignore dealer text): handled by `fast-alpr` (detector crops the plate before OCR) wired in Task 5. ✓
- Columns `filename, plate_text, confidence`: `write_excel` (Task 4), header order asserted. ✓
- Confidence may be per-character list: `_aggregate_confidence` (Task 2). ✓
- Runs on x86 / Python 3.10 / `cct-s-v2-global-model`: Global Constraints + Task 5 `build_alpr`. ✓
- Isolated from `car_logger/`: everything under `experiments/lpr_batch/`. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**3. Type consistency:** `predict(str) -> list`, results use `.ocr.text` / `.ocr.confidence`; `PlateRow(filename, plate_text, confidence)` and the `["filename","plate_text","confidence"]` header agree across Tasks 3–4. ✓

## Notes for the executor

- Tests live next to `plate_reader.py` so `import plate_reader` resolves under pytest's default import mode — run pytest from the repo root or `experiments/lpr_batch/`.
- Unit tests use fake `predict` results (`SimpleNamespace`); they never download or run the real model. Only the Task 5 manual smoke test needs `fast-alpr` installed.
 