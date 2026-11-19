"""Evaluate explicit MOT run windows with TrackEval's pedestrian preprocessing."""

import configparser
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import numpy as np
import trackeval

from .diagnostics import error_totals, frame_errors, render_error_clips
from ..tracking.mot_io import export_mot


def summarize_metrics(result):
    hota, clear, identity = result["HOTA"], result["CLEAR"], result["Identity"]
    return {"HOTA": round(float(np.mean(hota["HOTA"])) * 100, 3),
            "DetA": round(float(np.mean(hota["DetA"])) * 100, 3),
            "AssA": round(float(np.mean(hota["AssA"])) * 100, 3),
            "IDF1": round(float(identity["IDF1"]) * 100, 3),
            "MOTA": round(float(clear["MOTA"]) * 100, 3),
            **{key: int(clear[field]) for key, field in
               (("IDSW", "IDSW"), ("TP", "CLR_TP"), ("FN", "CLR_FN"), ("FP", "CLR_FP"))}}


def evaluate_runs(runs, output, error_clips=5, benchmark="MOT17"):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new or empty evaluation directory: {output}")
    if not runs or error_clips < 0:
        raise ValueError("Provide at least one run and a nonnegative clip count")
    if benchmark not in ("MOT17", "MOT20"):
        raise ValueError("Supported evaluation benchmarks: MOT17, MOT20")
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    items, scope, signature = {}, [], None
    for run in map(Path, runs):
        summary = json.loads((run / "summary.json").read_text())
        if summary.get("mode") != "track" or summary["source"]["kind"] != "mot_sequence":
            raise ValueError("Evaluation requires tracking runs from MOT sequences with ground truth")
        source = Path(summary["source"]["path"])
        name, count = source.name, summary["frames_processed"]
        if name in items:
            raise ValueError(f"Duplicate sequence in one tracker evaluation: {name}")
        current = {key: summary[key] for key in ("model_sha256", "inference_options", "tracker_config", "ultralytics", "torch")}
        current["tracking_policy"] = summary.get("tracking_policy", {"robust_gmc": False})
        if signature is not None and current != signature:
            raise ValueError("Combined runs must use the same model, inference and tracker configuration")
        signature = current
        config = configparser.ConfigParser()
        config.read(source / "seqinfo.ini")
        full_length = int(config["Sequence"]["seqLength"])
        if not 1 <= count <= full_length:
            raise ValueError(f"Invalid evaluation window: {name}")
        gt_dir = output / "inputs/gt" / name
        (gt_dir / "gt").mkdir(parents=True)
        original_gt = source / "gt/gt.txt"
        with original_gt.open() as src, (gt_dir / "gt/gt.txt").open("w") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerows(row for row in csv.reader(src) if 1 <= int(row[0]) <= count)
        config["Sequence"]["seqLength"] = str(count)
        with (gt_dir / "seqinfo.ini").open("w") as handle:
            config.write(handle)
        predictions = output / "inputs/trackers/YOLO26_TrackTrack/data"
        predictions.mkdir(parents=True, exist_ok=True)
        prediction = predictions / f"{name}.txt"
        export_mot(run / "tracks.jsonl", prediction, count)
        if (run / "tracks.txt").exists() and (run / "tracks.txt").read_bytes() != prediction.read_bytes():
            raise ValueError(f"MOT TXT disagrees with JSONL in {run}")
        split = (next((s for s in ("development", "evaluation") if name in splits[s]), "unlisted")
                 if benchmark == "MOT17" else "external")
        scope.append({"sequence": name, "frames": [1, count], "original_length": full_length,
                      "partial": count != full_length, "project_split": split,
                      "run": str(run.resolve()), "source_gt_sha256": hashlib.sha256(original_gt.read_bytes()).hexdigest(),
                      "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()})
        items[name] = {"summary": summary}

    dataset_config = {"GT_FOLDER": str(output / "inputs/gt"),
                      "TRACKERS_FOLDER": str(output / "inputs/trackers"),
                      "OUTPUT_FOLDER": str(output / "trackeval"), "TRACKERS_TO_EVAL": ["YOLO26_TrackTrack"],
                      "BENCHMARK": benchmark, "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True,
                      "SEQ_INFO": {s["sequence"]: s["frames"][1] for s in scope},
                      "DO_PREPROC": True, "PRINT_CONFIG": False}
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    evaluator = trackeval.Evaluator({"PRINT_CONFIG": False, "PRINT_RESULTS": False, "TIME_PROGRESS": False,
                                     "USE_PARALLEL": False, "PLOT_CURVES": False, "LOG_ON_ERROR": None})
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
               trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    results, messages = evaluator.evaluate([dataset], metrics)
    key = dataset.get_name()
    if messages[key]["YOLO26_TrackTrack"] != "Success":
        raise RuntimeError(messages[key]["YOLO26_TrackTrack"])
    results = results[key]["YOLO26_TrackTrack"]
    report = {"evaluator": {"package": "trackeval", "version": importlib.metadata.version("trackeval"),
                             "distribution_repo": "https://github.com/kovalp/TrackEval",
                             "upstream_repo": "https://github.com/JonathonLuiten/TrackEval"},
              "protocol": {"benchmark": benchmark, "class": "pedestrian", "DO_PREPROC": True,
                           "MOT_origin": 1, "internal_origin": 0,
                           "metric_scale": "HOTA/DetA/AssA/IDF1/MOTA in percent; MOTA may be negative",
                           "scope": "Local run windows, not an official benchmark submission"},
              "scope": scope, "sequences": {}, "combined": summarize_metrics(results["COMBINED_SEQ"]["pedestrian"])}
    for name, item in items.items():
        item["data"] = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data("YOLO26_TrackTrack", name), "pedestrian")
        item["events"] = frame_errors(item["data"])
        values = summarize_metrics(results[name]["pedestrian"])
        totals = error_totals(item["events"])
        if any(totals[k] != values[k] for k in totals):
            raise RuntimeError(f"Diagnostic association disagrees with TrackEval CLEAR: {name}")
        summary = item["summary"]
        values["runtime"] = {"fps": summary.get("fps") if "timing_protocol" in summary else None,
                             "frame_latency_ms": summary.get("frame_latency_ms"),
                             "hardware": summary.get("hardware"), "scope": summary["timing_scope"]}
        report["sequences"][name] = values
        (output / f"{name}_errors.json").write_text(json.dumps(item["events"], indent=2) + "\n")
    report["error_clips"] = render_error_clips(items, output / "error_clips", error_clips)
    report["combined"]["fps"] = (round(sum(v["summary"]["frames_processed"] for v in items.values()) /
                                              sum(v["summary"]["elapsed_seconds"] for v in items.values()), 3)
                                          if all("timing_protocol" in v["summary"] and
                                                 v["summary"].get("hardware") == next(iter(items.values()))["summary"].get("hardware")
                                                 for v in items.values()) else None)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"combined": report["combined"], "output": str(output)}, indent=2), flush=True)
    return report
