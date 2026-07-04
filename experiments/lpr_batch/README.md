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

Drop your photos into `data/` (gitignored), then:

```bash
python read_plates.py data --out output/plates.xlsx
```

Any other folder path works too. The first run downloads the detector + OCR
models; after that it works offline.

## Test

```bash
pytest test_plate_reader.py -v
```
