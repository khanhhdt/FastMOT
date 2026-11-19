"""Search a declared tracker grid on development caches; no detector-free replay FPS claims."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.evaluation.metrics import evaluate_runs
from mot.tracking.replay import replay_tracking

import cv2
import torch
from ultralytics.utils import YAML


def meets_quality(metrics, baseline, acceptance):
    return (metrics["FN"] <= baseline["FN"] * (1 - acceptance["missed_people"]["FN_reduction_fraction"])
            and metrics["FP"] <= baseline["FP"] * (1 - acceptance["false_positives"]["FP_reduction_fraction"])
            and metrics["HOTA"] >= baseline["HOTA"] and metrics["IDF1"] >= baseline["IDF1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fine", action="store_true", help="Declared local refinement grid, using the same acceptance targets")
    parser.add_argument("--refine", action="store_true", help="Declared confirmation-length refinement grid")
    parser.add_argument("--buffer-study", action="store_true", help="Declared track retention study")
    parser.add_argument("--baseline-evaluation", type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/experiments/six_issues_v2.json").read_text())
    baseline_path = args.baseline_evaluation or args.captures[0] / "evaluation"
    baseline = json.loads((baseline_path / "metrics.json").read_text())["combined"]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    grid_key = ("buffer_tracker_grid" if args.buffer_study else "confirmation_tracker_grid" if args.refine
                else "fine_tracker_grid" if args.fine else "tracker_grid")
    grid = protocol[grid_key]
    keys = list(grid)
    variants = []
    for capture in args.captures:
        source_runs = [capture / "runs" / item["sequence"] for item in protocol["selection_data"]]
        for run, item in zip(source_runs, protocol["selection_data"]):
            summary = json.loads((run / "summary.json").read_text())
            if Path(summary["source"]["path"]).name != item["sequence"] or summary["frames_processed"] != item["frames"]:
                raise ValueError("Cache does not match the declared development windows")
        first = json.loads((source_runs[0] / "summary.json").read_text())
        for number, values in enumerate(product(*(grid[key] for key in keys))):
            name = f"{capture.name}_{number:03d}"
            config = YAML.load(ROOT / "configs/trackers/tracktrack_v1.yaml")
            config.update(dict(zip(keys, values)))
            config_dir = args.output / "configs"
            config_dir.mkdir(exist_ok=True)
            path = config_dir / f"{name}.yaml"
            YAML.save(path, config)
            runs = []
            for source in source_runs:
                out = args.output / "runs" / name / source.name
                replay_tracking(source, out, path)
                runs.append(out)
            evaluation = args.output / "evaluations" / name
            metrics = evaluate_runs(runs, evaluation, error_clips=0)["combined"]
            variants.append({"name": name, "capture": str(capture), "model_config": first["config_paths"]["model"],
                             "tracker_config": str(path.resolve()), "parameters": dict(zip(keys, values)),
                             "metrics": metrics, "meets_FN_FP_targets": meets_quality(metrics, baseline, protocol["acceptance"])})
            passing = [v for v in variants if v["meets_FN_FP_targets"]]
            fallback = [v for v in variants if v["metrics"]["FN"] <= baseline["FN"] and v["metrics"]["FP"] <= baseline["FP"]
                        and v["metrics"]["HOTA"] >= baseline["HOTA"] and v["metrics"]["IDF1"] >= baseline["IDF1"]]
            pool = passing or fallback
            best = (min(pool, key=lambda v: (v["metrics"]["IDSW"], -v["metrics"]["HOTA"]))
                    if passing else min(pool, key=lambda v: (v["metrics"]["FN"] / baseline["FN"] + v["metrics"]["FP"] / baseline["FP"], -v["metrics"]["HOTA"]))
                    if pool else None)
            result = {"baseline": baseline, "best": best, "passing_candidates": len(passing), "variants": variants,
                      "scope": "Development only; replay has no end-to-end FPS. Selected config requires online validation."}
            (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"GRID {name}: HOTA={metrics['HOTA']} FN={metrics['FN']} FP={metrics['FP']} IDSW={metrics['IDSW']} passes={len(passing)}", flush=True)


if __name__ == "__main__":
    main()
