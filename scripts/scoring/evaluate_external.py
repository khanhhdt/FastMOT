"""Score a previously frozen configuration snapshot on the declared MOT20 windows."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mot.tracking.pipeline import run_tracking
from mot.evaluation.metrics import evaluate_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.frozen / "manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        if hashlib.sha256((args.frozen / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen file changed: {relative}")
    for relative, digest in manifest["source_hashes"].items():
        source = ROOT / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source differs from the freeze: {relative}. Use the original source or create a new freeze with scripts/experiments/freeze_profiles.py.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    protocol = json.loads((args.frozen / "protocol.json").read_text())
    comparisons = {}
    for profile in ("baseline", "quality", "fast"):
        folder = args.frozen / profile
        runs = []
        for item in protocol["external_data"]:
            source = ROOT / "data/MOT20/train" / item["sequence"]
            output = args.output / profile / "runs" / item["sequence"]
            run_tracking(source, output, item["frames"], folder / "model.json",
                         folder / "tracker.yaml", folder / "runtime.json")
            runs.append(output)
        comparisons[profile] = evaluate_runs(runs, args.output / profile / "evaluation",
                                             error_clips=0, benchmark="MOT20")["combined"]
        (args.output / "comparison.json").write_text(json.dumps({
            "frozen_at_utc": manifest["frozen_at_utc"], "frozen": str(args.frozen.resolve()),
            "scope": protocol["external_data"], "metrics": comparisons,
            "note": "No retuning after this external evaluation; checkpoint training-image overlap is not audited"
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
