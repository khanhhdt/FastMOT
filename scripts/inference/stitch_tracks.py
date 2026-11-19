"""Repair moving pedestrian IDs in an existing fixed-camera ReID run."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import cv2
import torch
from ultralytics.trackers.utils.reid import ReID
from ultralytics.utils.ops import xyxy2xywh

from mot.tracking.mot_io import write_mot_frame
from mot.tracking.sources import open_source
from mot.tracking.stitching import POLICY, motion_candidates, rank_links, relabel, sample_embeddings, select_links
from mot.tracking.visualization import TrackVisualizer


def render(records, source, output, trail_length, scene_cuts):
    visualizer = TrackVisualizer(trail_length=trail_length)
    identities = set()
    with open_source(source, len(records)) as (info, frames):
        width, height = info["width"], info["height"]
        size = width + width % 2, height + height % 2
        writer = cv2.VideoWriter(str(output / "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], size)
        try:
            if not writer.isOpened():
                raise RuntimeError("Cannot create the stitched video")
            with (output / "tracks.jsonl").open("w") as jsonl, (output / "tracks.txt").open("w") as mot:
                for (frame, image), record in zip(frames, records, strict=True):
                    tracks = record["tracks"]
                    if frame in scene_cuts:
                        visualizer.history.clear()
                    identities.update(track["id"] for track in tracks)
                    write_mot_frame(mot, frame, tracks)
                    jsonl.write(json.dumps(record, allow_nan=False) + "\n")
                    annotated = visualizer.draw(image, frame, tracks)
                    writer.write(cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT))
                    if frame in (1, len(records)) and not cv2.imwrite(str(output / f"frame_{frame:06d}.jpg"), annotated):
                        raise RuntimeError("Cannot save the stitched preview")
        finally:
            writer.release()
    return sorted(identities)


def stitch(input_run, output):
    input_run, output = input_run.resolve(), output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {output}")
    summary = json.loads((input_run / "summary.json").read_text())
    reid, policy = summary.get("reid", {}), summary.get("tracking_policy", {})
    if summary.get("mode") != "track" or not reid.get("enabled") or reid.get("native_features"):
        raise ValueError("Expected a tracking run with an external ReID model")
    if {"stationary_anchor_id", "stationary_anchor_ids", "moving_track_stitch"} & policy.keys():
        raise ValueError("Use the original ReID run before ID relabeling")
    model = ROOT / reid["model"]
    if not model.is_file():
        raise FileNotFoundError(f"Missing ReID model: {model}")
    with (input_run / "tracks.jsonl").open() as handle:
        records = [json.loads(line) for line in handle]
    if not records or len(records) != summary["frames_processed"] or any(
        frame != row["frame"] for frame, row in enumerate(records, 1)
    ):
        raise ValueError("Input tracks must cover every exported frame in order")

    info = summary["source"]
    cuts = sorted(event["frame"] for event in summary["motion"]["events"] if event["type"] == "scene_cut")
    candidates = motion_candidates(records, info, cuts)
    print(f"Screened {len(candidates)} pairs by motion; sampling appearance.", flush=True)
    if candidates:
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        encoder = ReID(str(model), device="cpu")
        with open_source(info["path"], len(records)) as (_, frames):
            features = sample_embeddings(candidates, frames, lambda image, boxes: encoder(image, xyxy2xywh(boxes)))
        ranked = rank_links(candidates, features)
    else:
        ranked = []
    links = select_links(ranked)
    stitched = list(relabel(records, links))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        target = Path(work) / "run"
        target.mkdir()
        summary["unique_track_ids"] = render(stitched, info["path"], target, summary["trail_length_frames"], cuts)
        summary["derived_from"] = str(input_run)
        summary["tracking_policy"] = {**policy, "moving_track_stitch": POLICY}
        summary["stitching"] = {"links": links, "motion_candidates": len(candidates), "scored_candidates": len(ranked)}
        for key in ("fps", "elapsed_seconds", "frame_latency_ms", "stage_seconds", "stage_ms_per_frame",
                    "stage_timing_note", "timing_protocol"):
            summary.pop(key, None)
        summary["timing_scope"] = "Offline relabeling of an existing ReID run; no end-to-end inference timing"
        (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (target / "link_candidates.json").write_text(json.dumps(ranked, indent=2) + "\n")
        target.replace(output)
    print(f"PASS: {len(records)} frames, {len(links)} ID links. Output: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Fixed-camera ReID run directory")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args()
    stitch(args.input, args.output)


if __name__ == "__main__":
    main()
