"""Run the fast-alpr candidate (detector + one OCR variant). LAPTOP.

  .venv\\Scripts\\python experiments/anpr_bakeoff/run_fastalpr.py \\
      --dataset experiments/anpr_bakeoff/data/eu_benchmark \\
      --ocr european-plates-mobile-vit-v2-model \\
      --out experiments/anpr_bakeoff/predictions/fastalpr_eu__eu_benchmark.csv

Laptop latency is INDICATIVE ONLY (different CPU than the Jetson) — the
on-device number comes from the Task 9 spike. Accuracy, however, is a
property of the models, and that is what this run measures.

API note (verified 2026-07-18, fast-alpr 0.4.0): ocr.confidence is a
LIST of per-character probabilities; we reduce it to its mean, the same
convention fast-alpr's own draw code uses."""

import argparse
import os
import time

from datasets import list_plateless, load_labels, write_predictions

from fast_alpr import ALPR

DETECTOR = "yolo-v9-t-384-license-plate-end2end"


def _mean_confidence(confidence):
    """Scalar 0-1 from fast-alpr's confidence: per-char list -> mean."""
    if isinstance(confidence, list):
        return sum(confidence) / float(len(confidence))
    return float(confidence)


def _best(results):
    """Highest-mean-OCR-confidence detection, or None when nothing found."""
    best = None
    for r in results:
        if r.ocr is None:
            continue
        if best is None or (_mean_confidence(r.ocr.confidence)
                            > _mean_confidence(best.ocr.confidence)):
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
                         _mean_confidence(best.ocr.confidence), wall_ms))
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
