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
