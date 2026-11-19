"""Capture raw detector/GMC evidence on the fixed MOT17 development windows."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.tracking.pipeline import run_tracking
from mot.evaluation.metrics import evaluate_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tracker-config", type=Path, default=ROOT / "configs/trackers/tracktrack_v1.yaml")
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/experiments/six_issues_v2.json").read_text())
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    if not json.loads(args.runtime_config.read_text())["capture_detections"]:
        parser.error("This command requires capture_detections=true")
    runs = []
    for item in protocol["selection_data"]:
        if item["sequence"] not in splits["development"]:
            raise ValueError("Diagnostic selection cannot use held-out or external data")
        output = args.output / "runs" / item["sequence"]
        run_tracking(ROOT / splits["dataset_root"] / item["sequence"], output, item["frames"],
                     args.model_config, args.tracker_config, args.runtime_config)
        runs.append(output)
    evaluate_runs(runs, args.output / "evaluation", error_clips=0)


if __name__ == "__main__":
    main()
