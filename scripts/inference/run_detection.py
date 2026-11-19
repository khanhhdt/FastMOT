"""Run YOLO26s person detection on a local video or a MOT sequence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.tracking.detection import run_detection


def main():
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / splits["dataset_root"] / splits["smoke_sequence"])
    parser.add_argument("--frames", type=int, help="Maximum frames; omit to process the full source")
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    output = args.output or ROOT / "outputs/runs" / f"detect_{args.source.name}"
    run_detection(args.source, output, args.frames)


if __name__ == "__main__":
    main()
