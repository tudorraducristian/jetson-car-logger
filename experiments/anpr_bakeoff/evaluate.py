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
