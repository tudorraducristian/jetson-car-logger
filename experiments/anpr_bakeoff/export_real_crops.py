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
