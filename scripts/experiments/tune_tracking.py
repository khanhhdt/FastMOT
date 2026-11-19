"""Run a bounded, development-only TrackTrack study with one change per parent configuration."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.tracking.pipeline import run_tracking
from mot.evaluation.metrics import evaluate_runs
from ultralytics.utils import YAML


def validate_plan(plan, splits):
    names = set()
    for sequence in plan["sequences"]:
        if sequence["name"] not in splits["development"] or sequence["name"] in names:
            raise ValueError("Use distinct development sequences only; held-out sequences cannot be tuned")
        if not isinstance(sequence["frames"], int) or sequence["frames"] < 1:
            raise ValueError("Each development window needs a positive frame count")
        names.add(sequence["name"])
    names = set()
    for index, variant in enumerate(plan["variants"]):
        name = variant["name"]
        if not name.replace("_", "").isalnum() or name in names:
            raise ValueError("Variant names must be distinct simple identifiers")
        changes = variant.get("changes", {})
        if index == 0:
            if name != "baseline" or changes:
                raise ValueError("First variant must be the unchanged baseline")
        else:
            if variant["parent"] not in names | {"best"} or len(changes) != 1:
                raise ValueError("Each variant changes exactly one parameter of an earlier parent or best")
            field = next(iter(changes))
            if field not in {"inference.conf", "inference.imgsz", "tracker.new_track_thresh",
                             "tracker.track_high_thresh", "tracker.track_low_thresh", "tracker.match_thresh",
                             "tracker.track_buffer", "tracker.min_track_len", "tracker.with_reid", "tracker.gmc_method"}:
                raise ValueError(f"Unsupported study parameter: {field}")
        names.add(name)
    if not names or not plan["sequences"]:
        raise ValueError("Provide variants and development windows")


def ranking(result):
    metrics = result["metrics"]
    return metrics["HOTA"], metrics["IDF1"], metrics["fps"] or 0


def run_study(plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    validate_plan(plan, splits)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty study directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    base_model = json.loads((ROOT / plan["base_model"]).read_text())
    base_tracker = YAML.load(ROOT / plan["base_tracker"])
    results, configs, best = {}, {}, None
    for variant in plan["variants"]:
        name = variant["name"]
        parent = best if variant.get("parent") == "best" else variant.get("parent")
        model, tracker = deepcopy(configs[parent] if parent else (base_model, base_tracker))
        changes = variant.get("changes", {})
        for field, value in changes.items():
            group, key = field.split(".")
            target = model["inference"] if group == "inference" else tracker
            if target[key] == value:
                raise ValueError(f"Study change has no effect: {name} {field}")
            target[key] = value
        configs[name] = model, tracker
        config_dir = output / "configs" / name
        config_dir.mkdir(parents=True)
        model_path, tracker_path = config_dir / "model.json", config_dir / "tracker.yaml"
        model_path.write_text(json.dumps(model, indent=2) + "\n")
        YAML.save(tracker_path, tracker)
        runs = []
        print(f"STUDY {name}, parent={parent}, changes={changes}", flush=True)
        for sequence in plan["sequences"]:
            run = output / "runs" / name / sequence["name"]
            run_tracking(ROOT / splits["dataset_root"] / sequence["name"], run, sequence["frames"],
                         model_path, tracker_path)
            runs.append(run)
        evaluation = output / "evaluations" / name
        report = evaluate_runs(runs, evaluation, error_clips=0)
        results[name] = {"parent": parent, "changes": changes, "metrics": report["combined"],
                         "sequences": report["sequences"], "evaluation": str(evaluation),
                         "model_config": str(model_path), "tracker_config": str(tracker_path)}
        best = max(results, key=lambda key: ranking(results[key]))
        summary = {"selection_rule": "Highest development HOTA, then IDF1, then FPS; no held-out metrics used",
                   "best": best, "sequences": plan["sequences"], "variants": results}
        (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"STUDY best={best}, HOTA={results[best]['metrics']['HOTA']}", flush=True)
    return summary


def finalize_study(study, output):
    """Confirm on other development videos, freeze configs, then score the held-out videos once."""
    study, output = Path(study).resolve(), Path(output).resolve()
    plan = json.loads((study / "plan.json").read_text())
    comparison = json.loads((study / "comparison.json").read_text())
    if set(comparison["variants"]) != {v["name"] for v in plan["variants"]}:
        raise ValueError("Finish all planned development variants before finalizing")
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    validate_plan(plan, splits)
    protocol = json.loads((ROOT / "configs/experiments/mot17_step5_confirmation.json").read_text())
    validate_plan({"sequences": protocol["additional_development_windows"],
                   "variants": [{"name": "baseline", "changes": {}}]}, splits)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty final comparison directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    candidates = {"baseline": comparison["variants"]["baseline"],
                  "candidate": comparison["variants"][comparison["best"]]}
    development = {}
    for role, candidate in candidates.items():
        variant = "baseline" if role == "baseline" else comparison["best"]
        runs = [study / "runs" / variant / s["name"] for s in plan["sequences"]]
        for sequence in protocol["additional_development_windows"]:
            run = output / "development/runs" / role / sequence["name"]
            run_tracking(ROOT / splits["dataset_root"] / sequence["name"], run, sequence["frames"],
                         candidate["model_config"], candidate["tracker_config"])
            runs.append(run)
        report = evaluate_runs(runs, output / "development/evaluations" / role, error_clips=5 if role == "candidate" else 0)
        development[role] = {"metrics": report["combined"], "scope": report["scope"]}
    selected = max(development, key=lambda role: ranking(development[role]))
    frozen = {}
    for role, candidate in {"baseline": candidates["baseline"], "selected": candidates[selected]}.items():
        paths = {}
        for field, filename in (("model_config", "model.json"), ("tracker_config", "tracker.yaml")):
            path = output / "frozen" / role / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(candidate[field], path)
            paths[field] = str(path)
            paths[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        frozen[role] = paths
    selection = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                 "selection_rule": protocol["selection_rule"], "selected_role": selected,
                 "selected_variant": comparison["best"] if selected == "candidate" else "baseline",
                 "development": development, "configs": frozen,
                 "held_out_used_for_selection": False}
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"FROZEN: {selection['selected_variant']}; starting held-out evaluation", flush=True)
    held_out = {}
    for role, config in frozen.items():
        runs = []
        for name in splits["evaluation"]:
            for field in ("model_config", "tracker_config"):
                assert hashlib.sha256(Path(config[field]).read_bytes()).hexdigest() == config[field + "_sha256"]
            run = output / "held_out/runs" / role / name
            run_tracking(ROOT / splits["dataset_root"] / name, run, None,
                         config["model_config"], config["tracker_config"])
            runs.append(run)
        report = evaluate_runs(runs, output / "held_out/evaluations" / role, error_clips=5 if role == "selected" else 0)
        held_out[role] = {"metrics": report["combined"], "scope": report["scope"]}
    result = {"selection": selection, "held_out": held_out}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/experiments/mot17_step5.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-study", type=Path, help="Completed study: confirm, freeze, then evaluate held-out videos")
    args = parser.parse_args()
    if args.finalize_study:
        finalize_study(args.finalize_study, args.output)
    else:
        run_study(args.plan, args.output)


if __name__ == "__main__":
    main()
