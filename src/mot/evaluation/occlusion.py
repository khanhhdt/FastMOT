"""Measure whether an ID survives a low-visibility interval annotated in MOT ground truth."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_ground_truth(path, count):
    frames = [dict() for _ in range(count + 1)]
    tracks = defaultdict(dict)
    with path.open(newline="") as source:
        for frame, person, x, y, w, h, mark, cls, visibility in csv.reader(source):
            frame, person = int(frame), int(person)
            if 1 <= frame <= count and int(mark) and int(cls) == 1:
                x, y, w, h = map(float, (x, y, w, h))
                frames[frame][person] = (x - 1, y - 1, x - 1 + w, y - 1 + h)
                tracks[person][frame] = float(visibility)
    return frames, tracks


def find_events(tracks, visible_min, severely_occluded_max, min_gap_frames, anchor_window_frames):
    """Require contiguous GT labels, visible endpoints and at least one severely occluded frame."""
    events = []
    for person, visibility in sorted(tracks.items()):
        before, previous, severe = None, None, False
        for frame, value in sorted(visibility.items()):
            if previous != frame - 1:
                before, severe = None, False
            previous = frame
            if value >= visible_min:
                if before is not None and severe and frame - before - 1 >= min_gap_frames:
                    before_frames, after_frames = [], []
                    for step in range(anchor_window_frames):
                        candidate = before - step
                        if visibility.get(candidate, -1) < visible_min:
                            break
                        before_frames.append(candidate)
                    for step in range(anchor_window_frames):
                        candidate = frame + step
                        if visibility.get(candidate, -1) < visible_min:
                            break
                        after_frames.append(candidate)
                    events.append({"person": person, "before": before_frames, "after": after_frames,
                                   "gap_frames": frame - before - 1})
                before, severe = frame, False
            elif value <= severely_occluded_max:
                severe = True
    return events


def match_frame(ground_truth, tracks, threshold):
    if not ground_truth or not tracks:
        return {}
    people = list(ground_truth)
    gt = np.asarray(list(ground_truth.values()), dtype=float)
    predictions = np.asarray([track["xyxy"] for track in tracks], dtype=float)
    left_top = np.maximum(gt[:, None, :2], predictions[None, :, :2])
    right_bottom = np.minimum(gt[:, None, 2:], predictions[None, :, 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=2)
    union = ((gt[:, 2:] - gt[:, :2]).prod(axis=1)[:, None]
             + (predictions[:, 2:] - predictions[:, :2]).prod(axis=1) - intersection)
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    # Maximize the number of valid matches first, then their total IoU.
    scores = np.where(iou >= threshold, min(len(gt), len(predictions)) + 1 + iou, 0)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    return {people[row]: tracks[col]["id"] for row, col in zip(rows, cols) if scores[row, col] > 0}


def classify_event(event, matches):
    person = event["person"]
    before_frame = next((t for t in event["before"] if person in matches[t]), None)
    after_frame = next((t for t in event["after"] if person in matches[t]), None)
    before = matches[before_frame][person] if before_frame is not None else None
    after = matches[after_frame][person] if after_frame is not None else None
    if before is None:
        status = "missing_before"
    elif after is None:
        status = "missing_after"
    else:
        status = "same_id" if before == after else "different_id"
    conflicting_id = status == "different_id" and (
        any(other != person and track == before for other, track in matches[after_frame].items())
        or any(other != person and track == after for other, track in matches[before_frame].items())
    )
    return {"status": status, "id_before": before, "id_after": after,
            "matched_before_frame": before_frame, "matched_after_frame": after_frame,
            "cross_person_id_at_anchors": conflicting_id}


def summarize(events, variant):
    counts = Counter(event["results"][variant]["status"] for event in events)
    total = len(events)
    matched = counts["same_id"] + counts["different_id"]
    return {"events": total, "same_id": counts["same_id"], "different_id": counts["different_id"],
            "missing_before": counts["missing_before"], "missing_after": counts["missing_after"],
            "cross_person_id_at_anchors": sum(
                event["results"][variant]["cross_person_id_at_anchors"] for event in events),
            "retention_pct": 100 * counts["same_id"] / total if total else None,
            "retention_when_matched_pct": 100 * counts["same_id"] / matched if matched else None}


def analyze_study(directory):
    plan = json.loads((directory / "protocol.json").read_text())
    settings = plan["event_protocol"]
    variants, events, run_metrics = plan["variants"], [], {}
    for item in plan["sequences"]:
        source, count = Path(item["source"]), item["frames"]
        gt_frames, gt_tracks = read_ground_truth(source / "gt/gt.txt", count)
        found = find_events(gt_tracks, **{k: v for k, v in settings.items() if k != "iou_threshold"})
        for event in found:
            event.update(sequence=source.name, results={})
        run_metrics[source.name] = {}
        for variant in variants:
            run = directory / variant / source.name
            summary = json.loads((run / "summary.json").read_text())
            records = [json.loads(line) for line in (run / "tracks.jsonl").read_text().splitlines()]
            if summary["frames_processed"] != count or [row["frame"] for row in records] != list(range(1, count + 1)):
                raise ValueError(f"Unexpected frame coverage: {run}")
            matches = {t: match_frame(gt_frames[t], records[t - 1]["tracks"], settings["iou_threshold"])
                       for t in {t for event in found for t in event["before"] + event["after"]}}
            for event in found:
                event["results"][variant] = classify_event(event, matches)
            metrics = json.loads((directory / variant / "scoring" / source.name / "metrics.json").read_text())
            run_metrics[source.name][variant] = {"trackeval": metrics["combined"], "reid": summary["reid"],
                                                  "frames": count, "seconds": summary["elapsed_seconds"]}
        events.extend(found)
    common = [e for e in events if all(r["status"] in {"same_id", "different_id"} for r in e["results"].values())]
    subsets = {"all": events, "matched_by_all_variants": common}
    for low, high in ((3, 15), (16, 30), (31, 60), (61, None)):
        subsets[f"{low}-{high or 'plus'}"] = [e for e in events if e["gap_frames"] >= low
                                             and (high is None or e["gap_frames"] <= high)]
    cpu_fps = {v: sum(m[v]["frames"] for m in run_metrics.values()) /
                  sum(m[v]["seconds"] for m in run_metrics.values()) for v in variants}
    paired = {}
    for base, candidate in (("quality_v2", "buffer60"), ("buffer60", "reid60"), ("quality_v2", "reid60")):
        improvement = [(e["results"][candidate]["status"] == "same_id")
                       - (e["results"][base]["status"] == "same_id") for e in events]
        paired[f"{base}_to_{candidate}"] = {"gained": improvement.count(1), "lost": improvement.count(-1)}
    return {"protocol": plan, "summary": {group: {v: summarize(items, v) for v in variants}
                                         for group, items in subsets.items()},
            "by_sequence": {s: {v: summarize([e for e in events if e["sequence"] == s], v) for v in variants}
                            for s in run_metrics},
            "run_metrics": run_metrics, "pipeline_fps": cpu_fps, "paired_retention": paired, "events": events}
