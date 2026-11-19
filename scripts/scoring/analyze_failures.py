"""Explain CLEAR false negatives/positives using captured raw detections."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.evaluation.failure_analysis import analyze_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    print(analyze_evaluation(args.evaluation)["FN_categories"])
