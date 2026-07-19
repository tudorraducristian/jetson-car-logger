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
