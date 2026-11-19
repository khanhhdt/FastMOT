"""Locate CLEAR errors after TrackEval preprocessing and render review clips."""

import configparser
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def frame_errors(data):
    """Follow CLEAR's IoU=0.5 association, including its previous-match priority."""
    last = np.full(data["num_gt_ids"], np.nan)
    previous = last.copy()
    events = []
    epsilon = np.finfo(float).eps
    for t, (gt, pred) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        matched_gt, matched_pred, switches = [], [], []
        if len(gt) and len(pred):
            similarity = data["similarity_scores"][t]
            scores = 1000 * (pred[None, :] == previous[gt[:, None]]) + similarity
            scores[similarity < 0.5 - epsilon] = 0
            rows, cols = linear_sum_assignment(-scores)
            keep = scores[rows, cols] > epsilon
            matched_gt, matched_pred = rows[keep], cols[keep]
            for g, p in zip(matched_gt, matched_pred):
                if not np.isnan(last[gt[g]]) and last[gt[g]] != pred[p]:
                    switches.append({"gt_index": int(g), "tracker_index": int(p),
                                     "previous_id": int(last[gt[g]]), "current_id": int(pred[p])})
            last[gt[matched_gt]] = pred[matched_pred]
            previous[:] = np.nan
            previous[gt[matched_gt]] = pred[matched_pred]
        events.append({"frame": t + 1,
                       "fn": sorted(set(range(len(gt))) - set(matched_gt)),
                       "fp": sorted(set(range(len(pred))) - set(matched_pred)),
                       "switches": switches})
    return events


def error_totals(events):
    return {"FN": sum(len(e["fn"]) for e in events), "FP": sum(len(e["fp"]) for e in events),
            "IDSW": sum(len(e["switches"]) for e in events)}


def draw_errors(image, data, event):
    annotated = image.copy()
    t = event["frame"] - 1
    switched = {e["tracker_index"] for e in event["switches"]}
    for kind, boxes, ids in (("G", data["gt_dets"][t], data["gt_ids"][t]),
                             ("T", data["tracker_dets"][t], data["tracker_ids"][t])):
        for index, (box, identity) in enumerate(zip(boxes, ids)):
            x, y, width, height = box
            # Evaluation boxes are one-origin MOT xywh; OpenCV drawing is zero-origin.
            x1, y1, x2, y2 = map(round, (x - 1, y - 1, x - 1 + width, y - 1 + height))
            color = (0, 180, 0) if kind == "G" else (255, 200, 0)
            if kind == "G" and index in event["fn"]:
                color = (0, 0, 255)
            if kind == "T" and index in event["fp"]:
                color = (0, 255, 255)
            if kind == "T" and index in switched:
                color = (255, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{kind}{int(identity) + 1}", (x1, max(42, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    title = f"Frame {t + 1} | FN {len(event['fn'])} | FP {len(event['fp'])} | IDSW {len(switched)}"
    cv2.rectangle(annotated, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(annotated, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return annotated


def render_error_clips(items, output, limit=5):
    """Prefer a mix of ID switches, missed GT and false positives at separated times."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = [(name, event) for name, item in items.items() for event in item["events"]
                  if event["fn"] or event["fp"] or event["switches"]]
    selected = []
    for field, quota in (("switches", 1), ("fn", 2), ("fp", 2), (None, limit)):
        pool = sorted(candidates, key=lambda pair: len(pair[1][field]) if field else
                      sum(len(pair[1][f]) for f in ("fn", "fp", "switches")), reverse=True)
        added = 0
        for name, event in pool:
            if len(selected) >= limit or added >= quota:
                break
            if field and not event[field]:
                continue
            spacing = round(items[name]["summary"]["source"]["fps"])
            if any(n == name and abs(e["frame"] - event["frame"]) <= spacing for n, e in selected):
                continue
            selected.append((name, event))
            added += 1
    clips = []
    for number, (name, event) in enumerate(selected, 1):
        item = items[name]
        source = Path(item["summary"]["source"]["path"])
        config = configparser.ConfigParser()
        config.read(source / "seqinfo.ini")
        sequence = config["Sequence"]
        fps = float(sequence["frameRate"])
        start = max(1, event["frame"] - round(fps / 2))
        end = min(item["summary"]["frames_processed"], event["frame"] + round(fps / 2))
        filename = f"{number:02d}_{name}_frame_{event['frame']:06d}"
        width, height = int(sequence["imWidth"]), int(sequence["imHeight"])
        writer = cv2.VideoWriter(str(output / f"{filename}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (width + width % 2, height + height % 2))
        if not writer.isOpened():
            raise RuntimeError("Cannot create diagnostic clip")
        try:
            for frame in range(start, end + 1):
                image = cv2.imread(str(source / sequence["imDir"] / f"{frame:06d}{sequence['imExt']}"))
                if image is None:
                    raise ValueError(f"Cannot read diagnostic frame {name}/{frame}")
                annotated = draw_errors(image, item["data"], item["events"][frame - 1])
                annotated = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                writer.write(annotated)
                if frame == event["frame"] and not cv2.imwrite(str(output / f"{filename}.jpg"), annotated):
                    raise RuntimeError("Cannot save diagnostic image")
        finally:
            writer.release()
        clips.append({"sequence": name, "frame": event["frame"], "range": [start, end],
                      "FN": len(event["fn"]), "FP": len(event["fp"]), "IDSW": len(event["switches"]),
                      "video": f"error_clips/{filename}.mp4", "image": f"error_clips/{filename}.jpg"})
    (output / "index.json").write_text(json.dumps({
        "legend": "G=GT, T=tracker; IDs normalized by TrackEval. Red=FN, yellow=FP, magenta=IDSW.",
        "clips": clips}, indent=2) + "\n")
    return clips
