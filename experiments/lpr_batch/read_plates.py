"""CLI: read license plates from a folder of images into an Excel file."""

import argparse
from pathlib import Path

from plate_reader import read_folder, write_excel


def build_alpr():
    """Create the real fast-alpr model (downloads weights on first run)."""
    from fast_alpr import ALPR

    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v2-global-model",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Read license plates from a folder of images into an Excel file."
    )
    parser.add_argument("folder", help="Folder containing the images")
    parser.add_argument(
        "--out", default="output/plates.xlsx", help="Output .xlsx path"
    )
    args = parser.parse_args()

    # Validate the input before build_alpr(): loading the model is slow and
    # a path typo should fail immediately with a clear message.
    if not Path(args.folder).is_dir():
        parser.error(f"folder not found or not a directory: {args.folder}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alpr = build_alpr()
    rows = read_folder(args.folder, alpr.predict)
    write_excel(rows, out_path)

    for row in rows:
        print(f"{row.filename}: {row.plate_text or '<no plate>'} ({row.confidence})")
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
