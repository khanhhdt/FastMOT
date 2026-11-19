"""Prepare street/crowd videos and a local result viewer."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.app.demo import build_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--frames", type=int, default=150, help="Maximum source frames per scene")
    parser.add_argument("--model-config", type=Path, help="Model manifest; defaults to the selected configuration")
    parser.add_argument("--tracker-config", type=Path, help="TrackTrack YAML; defaults to the selected configuration")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    build_demo(args.output, args.frames, args.model_config, args.tracker_config)


if __name__ == "__main__":
    main()
