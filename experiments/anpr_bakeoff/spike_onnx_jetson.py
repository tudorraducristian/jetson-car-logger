"""Feasibility spike: do the fast-plate-ocr/fast-alpr ONNX models run on
this Jetson's Python 3.6 with onnxruntime 1.9 (CPU)? JETSON only.

Run ONCE PER OCR VARIANT (gate resolution 2026-07-18: both models ship,
the on-device numbers separate them):

  ~/anpr_spike_venv/bin/python experiments/anpr_bakeoff/spike_onnx_jetson.py \\
      --ocr-model ~/anpr_spike/european_mobile_vit_v2_ocr.onnx \\
      --ocr-config ~/anpr_spike/european_mobile_vit_v2_ocr_config.yaml \\
      --detector-model ~/anpr_spike/yolo-v9-t-384-license-plates-end2end.onnx \\
      --image ~/anpr_spike/event_22_plate.jpg \\
      --expect CJ45ARL

  ~/anpr_spike_venv/bin/python experiments/anpr_bakeoff/spike_onnx_jetson.py \\
      --ocr-model ~/anpr_spike/cct_xs_v2_global.onnx \\
      --ocr-config ~/anpr_spike/cct_xs_v2_global_plate_config.yaml \\
      --detector-model ~/anpr_spike/yolo-v9-t-384-license-plates-end2end.onnx \\
      --image ~/anpr_spike/event_22_plate.jpg \\
      --expect CJ45ARL

--image must be a TIGHT plate crop (event_22_plate.jpg = the plate cut
out of real_crops/images/event_22.jpg by fast-alpr's detector on the
laptop). Feeding a whole frame decodes garbage — the OCR models only
ever see detector crops in the real pipeline; verified in the laptop
dry-run 2026-07-18 (whole frame -> '722255'; tight crop -> 'CJ45ARL',
conf identical to the Task 7 CSV, which proves this decode matches
fast-plate-ocr's).

Config facts (verified 2026-07-18 from the cached yamls): european =
grayscale 140x70, 9 slots, output FLATTENED (1, 333); cct_xs_v2_global =
RGB 128x64, 10 slots, outputs (1, 10, 37) + a (1, 66) region head
(image_color_mode key; absent means grayscale). Both share the same
37-char alphabet with '_' as pad.

PASS = OCR text matches --expect AND the timed loop stays under the
2 s/crop budget. Prints everything it learns (config, input names/
shapes/dtypes, output shapes) so a failure is still informative."""

import argparse
import time

try:
    import resource          # Unix only — present on the Jetson
except ImportError:
    resource = None          # laptop dry-run: RSS reads n/a

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
    color_mode = cfg.get("image_color_mode", "grayscale")
    if color_mode == "grayscale":
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)  # BGR
    if img is None:
        raise SystemExit("cannot read " + img_path)
    if color_mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cfg["img_width"], cfg["img_height"]))
    if img.ndim == 2:
        arr = img[np.newaxis, :, :, np.newaxis]   # NHWC, batch 1, gray
    else:
        arr = img[np.newaxis, :, :, :]            # NHWC, batch 1, rgb
    if meta.type == "tensor(float)":
        arr = arr.astype("float32")
    else:
        arr = arr.astype("uint8")
    return arr


def pick_char_probs(outputs, cfg):
    """Find the output holding the per-slot char probabilities and give
    it shape (slots, alphabet). The heads may come flattened — the
    european model emits (1, 333) = 9 slots x 37 chars — and extra heads
    (plate region) may exist, so match by element count, not position."""
    slots = cfg["max_plate_slots"]
    alpha = len(cfg["alphabet"])
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.size == slots * alpha:
            return arr.reshape(slots, alpha)
    raise SystemExit("no output has {0}x{1} elements; shapes: {2}".format(
        slots, alpha, [np.asarray(o).shape for o in outputs]))


def decode_ocr(outputs, cfg):
    """fast-plate-ocr heads emit per-slot char probabilities; decode =
    argmax per slot through the alphabet, pad char stripped."""
    probs = pick_char_probs(outputs, cfg)
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
    print("ocr output shapes: {0}".format(
        [np.asarray(o).shape for o in outputs]))
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

    if resource is not None:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print("peak RSS: {0:.0f} MB".format(rss_mb))
    else:
        print("peak RSS: n/a (no resource module — laptop dry-run)")
    ok = text.upper() == args.expect.upper() and per_crop_ms < 2000.0
    print("SPIKE {0}".format("PASS" if ok else
                             "FAIL (see numbers above)"))


if __name__ == "__main__":
    main()
