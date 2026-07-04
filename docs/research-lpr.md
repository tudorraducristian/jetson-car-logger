# Research: On-device License Plate Recognition for Jetson Nano

> **Date:** 2026-07-04
> **Goal of this research:** pick ONE license plate recognition (LPR/ANPR)
> solution for a first, isolated test — read a folder of Romanian plate photos
> and produce plate text + confidence. Not the full appliance yet.
> **Method:** live web search (2024–2026 sources), verifying every compatibility
> claim against a primary source (GitHub, PyPI, NVIDIA Jetson forums).

## Constraints (recap)

- **Device:** NVIDIA Jetson Nano Developer Kit, original 2019, 4 GB RAM,
  Maxwell GPU (CUDA compute capability 5.3).
- **Stack:** JetPack 4.6.x → Ubuntu 18.04, **Python 3.6.9**, CUDA 10.2,
  cuDNN 8.2, TensorRT 8.x, OpenCV 4.1.1.
- **Must be fully offline / on-device.** No cloud ANPR API for this path.
- **Plates:** Romanian only (EU/RO format, e.g. `CJ 23 XZI`).
- **Input photos:** close-up frontal shots of a car front. The plate is large
  and clear, but the frame also contains other text (dealer frame, phone
  numbers). → The pipeline must **localize the plate first, then OCR it**
  (end-to-end), otherwise it would read the dealer text.
- **First milestone = batch** (folder → text + confidence). Speed does not
  matter. Priorities, in order: **(1) ease of install, (2) accuracy on RO/EU
  plates.** Real-time FPS is a later concern.

## The decisive finding: Python 3.6 is the real constraint

The hard limiter is not "Jetson", it is **Python 3.6**. The modern, most
accurate, easiest-to-use tools have dropped it:

- **PyTorch:** last version usable on JetPack 4.6 / Python 3.6 is **torch 1.10**;
  1.11+ require Python 3.7+.
  Sources: [Q-engineering](https://qengineering.eu/install-pytorch-on-jetson-nano.html),
  [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).
- **`fast-alpr` / `fast-plate-ocr`** (the best modern option) require
  **Python `>=3.10`** — will not install on Python 3.6.
  Sources: [fast-plate-ocr @ PyPI](https://pypi.org/project/fast-plate-ocr/),
  [fast-alpr @ PyPI](https://pypi.org/project/fast-alpr/).
- **`ultralytics` / YOLOv8+** — same story, require newer Python.

**Consequence for testing accuracy:** a given model produces the *same plate
text* on a laptop and on the Jetson — only the speed differs. So validating
"can an on-device model read my RO plates?" does **not** require the Jetson.
It can (and should) be done first on an x86 machine, where Python 3.10 is free.

## Candidate comparison

| Solution | End-to-end? | RO/EU plates | Install on Jetson (JP4.6 / Py3.6) | Offline | License | Verdict |
|---|---|---|---|---|---|---|
| **fast-alpr + fast-plate-ocr** | Yes (YOLO ONNX detector + CCT OCR) | Yes — "global" model, 65+ countries incl. EU, ~92–94% | **No** — requires Python 3.10 (not native on Py3.6) | Yes | MIT | **Best accuracy/ease — run on x86 first** |
| **OpenALPR** (`-c eu`) | Yes (detector + Tesseract OCR) | Partial — has `eu` config, but RO tuning needed; dated OCR | Medium — C++, compiles on ARM; documented on Jetson Nano | Yes | AGPLv3 | **Native Jetson fallback** |
| **winter2897 (SSD-Mobilenet + TensorRT)** | Yes | No — trained on **Vietnamese** plates | Medium-Hard, unmaintained | Yes | — | No — would need RO retraining |
| **YOLOv5 (plate) + OCR** | You assemble it | Needs RO-trained weights | Medium (torch 1.10 works on Py3.6) | Yes | GPL/varies | Too much assembly for step 1 |
| **EasyOCR** | No — OCR only, needs a detector | Generic OCR, not plate-tuned | Medium (torch 1.10; CUDA finicky) | Yes | Apache-2.0 | Generic OCR, not first choice |
| **PaddleOCR / PP-OCR** | No — OCR only (+ separate detector) | Generic | **Hard** — PaddlePaddle on JP4.6 painful (community wheels only) | Yes | Apache-2.0 | Too hard to install |

Install-verdict sources:
[ONNX Runtime on Jetson](https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/),
[ONNX on JP4.6.1 thread](https://forums.developer.nvidia.com/t/how-to-install-onnx-on-jetson-nano-jetpack-4-6-1/298032),
[Paddle on Jetson Nano](https://qengineering.eu/install-paddle-on-jetson-nano.html),
[OpenALPR on Jetson Nano](https://alannewcomer.medium.com/license-plate-recognition-with-a-jetson-nano-e94c6ff683bc),
[winter2897 repo](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano),
[fast-alpr repo](https://github.com/ankandrew/fast-alpr),
[fast-plate-ocr model zoo](https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/).

## Recommendation

**Use `fast-alpr` (with `fast-plate-ocr`, model `cct-s-v2-global-model`), and
run the first batch test (folder → Excel) on an x86 laptop, not on the Jetson.**

Why it wins for step 1:

1. **End-to-end** — ships a YOLO plate detector (ONNX) + OCR. It finds the plate
   rectangle and ignores surrounding text (the dealer frame in the sample photo).
2. **Accuracy** — the "global" CCT OCR model is trained on 65+ countries incl.
   European, so it reads the RO format; ~92–94% reported.
3. **Trivial install** — `pip install fast-alpr[onnx]`, CPU-only, offline after
   the first model download. MIT license.
4. **Speed is irrelevant** for a one-off folder batch; CPU on a laptop is plenty.

Its only blocker (Python 3.10) does not exist on the laptop, and because
accuracy is device-independent, the test is fully valid there.

**Native-on-Jetson fallback:** **OpenALPR** with `-c eu`. Runs on Python 3.6 /
CPU / ARM, offline, has a documented Jetson Nano deployment. Downsides:
abandoned since ~2018 and its Tesseract-based OCR is weaker than the modern CCT
model, so it may struggle with the current RO plate font.

## Getting-started notes

- **fast-alpr on x86:** `pip install fast-alpr[onnx]`. Detector + OCR weights
  download automatically on first run. Works on Windows/Linux, CPU.
- **Prototype on x86 before Jetson?** Yes — recommended. Validate RO-plate
  accuracy on the real photos first, then separately decide how it reaches the
  device.
- **"Native on Jetson" later, if we keep fast-alpr:** run the `.onnx` files
  directly through a Python-3.6-compatible `onnxruntime` on the device. This is
  an advanced path (need a matching onnxruntime wheel + reimplement pre/post
  processing) and is out of scope for step 1.

## Decision (to confirm with the user)

1. Model: `fast-alpr` + `fast-plate-ocr` (`cct-s-v2-global-model`).
2. Where the first folder → Excel test runs: **x86 laptop** (Python 3.10+).
3. Output columns: `filename`, `plate_text`, `confidence`.
