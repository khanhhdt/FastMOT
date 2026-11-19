"""Score completed MOT tracking runs and save metrics plus diagnostic clips."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.evaluation.metrics import evaluate_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=5, help="Number of error clips; 0 skips rendering")
    parser.add_argument("--benchmark", choices=["MOT17", "MOT20"], default="MOT17")
    args = parser.parse_args()
    evaluate_runs(args.runs, args.output, args.clips, args.benchmark)


if __name__ == "__main__":
    main()
