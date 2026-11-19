"""Run YOLO26 Detect + TrackTrack with IDs and short trajectories."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.tracking.pipeline import run_tracking


def main():
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / splits["dataset_root"] / splits["smoke_sequence"])
    parser.add_argument("--frames", type=int, help="Maximum frames; omit to process the full source")
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    parser.add_argument("--profile", choices=["fast", "quality"], default="quality", help="Preset; individual config flags override its files")
    parser.add_argument("--model-config", type=Path, help="Override the profile's model manifest")
    parser.add_argument("--tracker-config", type=Path, help="Override the profile's TrackTrack YAML")
    parser.add_argument("--runtime-config", type=Path, help="GMC guard, capture and CPU thread configuration")
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    output = args.output or ROOT / "outputs/runs" / f"track_{args.source.name}"
    profile = json.loads((ROOT / f"configs/profiles/{args.profile}.json").read_text())
    run_tracking(args.source, output, args.frames, args.model_config or ROOT / profile["model_config"],
                 args.tracker_config or ROOT / profile["tracker_config"], args.runtime_config or ROOT / profile["runtime_config"])


if __name__ == "__main__":
    main()
