"""Account for every CLEAR miss using detector evidence, without treating overlap as proof of causality."""

from collections import Counter
import json
from pathlib import Path

from .metrics import ROOT
from .diagnostics import frame_errors
from ..tracking.mot_io import write_mot_frame

import numpy as np
from scipy.optimize import linear_sum_assignment
import trackeval


def frame_matches(similarity):
    if not similarity.shape[0] or not similarity.shape[1]:
        return set()
    scores = np.where(similarity >= 0.5 - np.finfo(float).eps, similarity, 0)
    rows, cols = linear_sum_assignment(-scores)
    return {int(r) for r, c in zip(rows, cols) if scores[r, c] > 0}


def classify_misses(similarity, missed):
    matched = frame_matches(similarity)
    counts = Counter()
    for g in missed:
        best = float(similarity[g].max()) if similarity.shape[1] else 0.0
        if g in matched:
            counts["raw_match_but_no_final_track_match"] += 1
        elif best >= 0.5 - np.finfo(float).eps:
            counts["crowd_competition_for_raw_box"] += 1
        elif best >= 0.1:
            counts["raw_box_localization_below_0_5"] += 1
        else:
            counts["no_raw_box_overlap_above_0_1"] += 1
    return dict(counts)


def analyze_evaluation(evaluation, source_runs=None):
    evaluation = Path(evaluation).resolve()
    report = json.loads((evaluation / "metrics.json").read_text())
    raw_root = evaluation / "diagnostic_inputs/raw/data"
    raw_root.mkdir(parents=True, exist_ok=True)
    runs = {Path(r).name: Path(r) for r in source_runs} if source_runs else {
        s["sequence"]: Path(s["run"]) for s in report["scope"]}
    for scope in report["scope"]:
        run = runs[scope["sequence"]]
        identity = count = 0
        with (run / "detections.jsonl").open() as source, (raw_root / f"{scope['sequence']}.txt").open("w") as target:
            for row in map(json.loads, source):
                count += 1
                if row["frame"] != count:
                    raise ValueError("Raw cache frame sequence mismatch")
                tracks = []
                for values in row.get("detector_tight", row["tight"]):
                    identity += 1
                    tracks.append({"id": identity, "xyxy": values[:4], "confidence": values[4], "class_id": 0})
                write_mot_frame(target, count, tracks)
        if count != scope["frames"][1]:
            raise ValueError("Raw cache length differs from evaluation window")
    common = {"GT_FOLDER": str(evaluation / "inputs/gt"), "BENCHMARK": report["protocol"]["benchmark"],
              "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True, "DO_PREPROC": True,
              "PRINT_CONFIG": False, "SEQ_INFO": {s["sequence"]: s["frames"][1] for s in report["scope"]}}
    raw_dataset = trackeval.datasets.MotChallenge2DBox({**common, "TRACKERS_FOLDER": str(raw_root.parent.parent),
                                                      "TRACKERS_TO_EVAL": ["raw"]})
    tracked_dataset = trackeval.datasets.MotChallenge2DBox({**common,
        "TRACKERS_FOLDER": str(evaluation / "inputs/trackers"), "TRACKERS_TO_EVAL": ["YOLO26_TrackTrack"]})
    totals, size_bins, sequences, fp_types = Counter(), {}, {}, Counter()
    for scope in report["scope"]:
        name = scope["sequence"]
        raw = raw_dataset.get_preprocessed_seq_data(raw_dataset.get_raw_seq_data("raw", name), "pedestrian")
        tracked = tracked_dataset.get_preprocessed_seq_data(tracked_dataset.get_raw_seq_data("YOLO26_TrackTrack", name), "pedestrian")
        errors = frame_errors(tracked)
        causes, detector = Counter(), Counter()
        for t, event in enumerate(errors):
            if not np.array_equal(raw["gt_dets"][t], tracked["gt_dets"][t]):
                raise ValueError("Detector and tracker GT preprocessing differs")
            similarity = raw["similarity_scores"][t]
            matched = frame_matches(similarity)
            detector.update(TP=len(matched), FN=len(raw["gt_ids"][t]) - len(matched),
                            FP=len(raw["tracker_ids"][t]) - len(matched))
            causes.update(classify_misses(similarity, event["fn"]))
            for g, box in enumerate(raw["gt_dets"][t]):
                key = "height_lt_50px" if box[3] < 50 else "height_50_to_99px" if box[3] < 100 else "height_ge_100px"
                values = size_bins.setdefault(key, {"GT": 0, "detector_matches": 0, "tracked_matches": 0})
                values["GT"] += 1
                values["detector_matches"] += int(g in matched)
                values["tracked_matches"] += int(g not in event["fn"])
            for p in event["fp"]:
                column = tracked["similarity_scores"][t][:, p]
                overlap = float(column.max()) if len(column) else 0.0
                key = "assignment_or_duplicate_overlap" if overlap >= 0.5 else "localization_overlap_0_1_to_0_5" if overlap >= 0.1 else "no_GT_overlap_above_0_1"
                fp_types[key] += 1
        if sum(causes.values()) != report["sequences"][name]["FN"]:
            raise ValueError(f"Unaccounted FN: {name}")
        totals.update(causes)
        sequences[name] = {"FN_categories": dict(causes), "detector_IoU_0_5": dict(detector)}
    if sum(totals.values()) != report["combined"]["FN"] or sum(fp_types.values()) != report["combined"]["FP"]:
        raise ValueError("Diagnostic totals disagree with CLEAR")
    result = {"scope": report["scope"], "FN_categories": dict(totals), "FP_categories": dict(fp_types),
              "height_bins": size_bins, "sequences": sequences,
              "interpretation": "IoU=0.5 after MOT pedestrian preprocessing. Raw means the tight-NMS detector output; loose-NMS recovery is separately cached for replay. Raw matches are per-frame Hungarian matches. A raw match lost at the output may involve filtering, confirmation, association or Kalman box alignment; these categories do not identify a unique causal component.",
              "coverage": {"FN_accounted": sum(totals.values()), "FP_accounted": sum(fp_types.values())}}
    (evaluation / "failure_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
