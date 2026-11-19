"""Snapshot the selected profiles before inspecting external MOT20 scores."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    profiles = {name: json.loads((ROOT / f"configs/profiles/{name}.json").read_text()) for name in ("quality", "fast")}
    profiles["baseline"] = {"model_config": "configs/models/yolo26s_v1.json",
                            "tracker_config": "configs/trackers/tracktrack_v1.yaml",
                            "runtime_config": "configs/runtime/legacy.json"}
    for name, profile in profiles.items():
        directory = args.output / name
        directory.mkdir(parents=True)
        copyfile(ROOT / profile["model_config"], directory / "model.json")
        copyfile(ROOT / profile["tracker_config"], directory / "tracker.yaml")
        runtime = json.loads((ROOT / profile["runtime_config"]).read_text())
        runtime["capture_detections"] = False
        (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    copyfile(ROOT / "configs/experiments/six_issues_v2.json", args.output / "protocol.json")
    files = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(args.output.rglob("*")) if p.is_file()}
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                     ("src/mot/tracking/pipeline.py", "src/mot/tracking/motion.py", "src/mot/evaluation/metrics.py")}
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "profiles": profiles,
                "files": files, "source_hashes": source_hashes,
                "selection_scope": "MOT17 development only; no MOT20 metrics inspected",
                "training_overlap": "Checkpoint metadata refers to COCO; image overlap with MOT17/MOT20 is unaudited"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"FROZEN: {args.output}")


if __name__ == "__main__":
    main()
