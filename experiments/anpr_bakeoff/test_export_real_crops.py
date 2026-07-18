import csv
import os
import sqlite3

from export_real_crops import export


def _seed(db_path, plates_dir, rows):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, plate_text TEXT, "
        "anpr_status TEXT NOT NULL)")
    conn.executemany("INSERT INTO events VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    os.makedirs(plates_dir)


def test_export_only_successful_reads_with_existing_crop(tmp_path):
    db = str(tmp_path / "car_logger.db")
    plates = str(tmp_path / "plates")
    out = str(tmp_path / "real_crops")
    _seed(db, plates, [
        (1, "B123ABC", "success"),   # exported
        (2, None, "failed"),         # not success -> skipped
        (3, "CJ07XYZ", "success"),   # crop missing on disk -> skipped
    ])
    (tmp_path / "plates" / "1.jpg").write_bytes(b"\xff\xd8fake")

    assert export(db, plates, out) == 1

    with open(os.path.join(out, "labels.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [{"filename": "images/event_1.jpg",
                     "plate_text": "B123ABC"}]
    assert os.path.isfile(os.path.join(out, "images", "event_1.jpg"))
