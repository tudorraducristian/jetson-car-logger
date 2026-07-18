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
