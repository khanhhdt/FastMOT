"""Measure sequential end-to-end CPU/GPU throughput; do not run other inference jobs concurrently."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mot.tracking.pipeline import run_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1:
        parser.error("frames and repeats must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    profiles = {name: json.loads((ROOT / f"configs/profiles/{name}.json").read_text()) for name in ("fast", "quality")}
    profiles["baseline"] = {"model_config": "configs/models/yolo26s_v1.json", "tracker_config": "configs/trackers/tracktrack_v1.yaml",
                            "runtime_config": "configs/runtime/legacy.json"}
    for name, profile in profiles.items():
        directory = args.output / "configs" / name
        directory.mkdir(parents=True)
        copyfile(ROOT / profile["model_config"], directory / "model.json")
        copyfile(ROOT / profile["tracker_config"], directory / "tracker.yaml")
        runtime = json.loads((ROOT / profile["runtime_config"]).read_text())
        runtime["capture_detections"] = False
        (directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    report = {"source": f"MOT17-09-FRCNN frames 1-{args.frames}", "repeats": args.repeats,
              "sequential": True, "records": [],
              "note": "Baseline and fast repeated equally; quality measured once for context. No concurrent project inference jobs. Model load/warm-up and H.264 conversion excluded; all per-frame pipeline work and video finalization included."}
    for repeat in range(args.repeats + 1):
        for name in (("baseline", "fast") if repeat < args.repeats else ("quality",)):
            config = args.output / "configs" / name
            summary = run_tracking(ROOT / "data/MOT17/train/MOT17-09-FRCNN", args.output / f"{name}_{repeat}", args.frames,
                                   config / "model.json", config / "tracker.yaml", config / "runtime.json")
            report["records"].append({"profile": name, "repeat": repeat, "frames": summary["frames_processed"],
                "seconds": summary["elapsed_seconds"], "fps": summary["fps"], "hardware": summary["hardware"],
                "stage_ms_per_frame": summary["stage_ms_per_frame"], "timing_scope": summary["timing_scope"]})
            (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"BENCHMARK {name}: {summary['fps']} FPS", flush=True)


if __name__ == "__main__":
    main()
