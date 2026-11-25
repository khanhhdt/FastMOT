"""Measure sparse CPU detection plus per-frame optical flow on a MOT sequence."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import torch
import ultralytics

from mot.evaluation.metrics import evaluate_runs
from mot.tracking.hybrid import SparseFlow
from mot.tracking.mot_io import write_mot_frame
from mot.tracking.pipeline import FrameTracker
from mot.tracking.sources import open_source


def write_frame(handles, frame, tracks):
    jsonl, mot = handles
    jsonl.write(json.dumps({"frame": frame, "tracks": tracks}, allow_nan=False) + "\n")
    write_mot_frame(mot, frame, tracks)


def benchmark(source, output, frames, interval, reference=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_config = ROOT / "configs/models/yolo26s_1280_openvino_fp32.json"
    tracker_config = ROOT / "configs/trackers/tracktrack_camera_4fps.yaml"
    runtime = {"robust_gmc": True, "torch_threads": 4, "capture_detections": False}
    tracker = FrameTracker(model_config, tracker_config, runtime, cpu_only=True)
    flow = SparseFlow()
    runs = {name: output / name for name in ("flow", "hold")}
    for run in runs.values():
        run.mkdir()
    handles = {name: ((run / "tracks.jsonl").open("w"), (run / "tracks.txt").open("w"))
               for name, run in runs.items()}
    stage = {name: 0.0 for name in ("decode", "detector_and_tracktrack", "flow", "export")}
    count = detections = 0
    held = []
    try:
        with open_source(source, frames) as (info, images):
            tracker.warmup(info["width"], info["height"])
            started = time.perf_counter()
            previous = started
            for frame, image in images:
                decoded = time.perf_counter()
                stage["decode"] += decoded - previous
                keyframe = (frame - 1) % interval == 0
                if keyframe:
                    held = tracker.step(image)
                    tracks = flow.correct(image, held)
                    detections += 1
                    stage["detector_and_tracktrack"] += time.perf_counter() - decoded
                else:
                    tracks = flow.advance(image)
                    stage["flow"] += time.perf_counter() - decoded
                exporting = time.perf_counter()
                write_frame(handles["flow"], frame, tracks)
                write_frame(handles["hold"], frame, held)
                stage["export"] += time.perf_counter() - exporting
                count = frame
                previous = time.perf_counter()
                if frame % 100 == 0:
                    print(f"{frame} frames, {detections} detector calls", flush=True)
            elapsed = time.perf_counter() - started
    finally:
        for pair in handles.values():
            for handle in pair:
                handle.close()
    summary = {
        "mode": "track", "source": info, "frames_processed": count,
        "model_sha256": tracker.digest, "inference_options": tracker.options,
        "tracker_config": tracker_config.read_text(),
        "tracking_policy": {"robust_gmc": True, "sparse_interval_frames": interval},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "timing_scope": "Sequential CPU run after detector warmup; source decode, sparse detector, flow and MOT export; excludes model loading, web transport and display",
        "timing_protocol": {"detector_warmup_frames": 1, "sequential": True},
        "hardware": {"device": "cpu"},
        "elapsed_seconds": round(elapsed, 3), "fps": round(count / elapsed, 3),
        "detector_calls": detections, "detector_fps": round(detections / elapsed, 3),
        "stage_seconds": {key: round(value, 3) for key, value in stage.items()},
        "note": "Flow and hold frames predict positions between detector calls; TrackTrack assigns IDs only on detector frames",
    }
    for name, run in runs.items():
        (run / "summary.json").write_text(json.dumps({**summary, "interpolation": name}, indent=2) + "\n")
    scores = {}
    for name, run in runs.items():
        scores[name] = evaluate_runs([run], output / f"score_{name}", error_clips=0)["combined"]
    if reference:
        run = output / "reference_full_rate"
        run.mkdir()
        original = json.loads((reference / "summary.json").read_text())
        original["source"] = info
        original["frames_processed"] = count
        original.pop("timing_protocol", None)
        original["timing_scope"] = "First matching source frames from existing full-rate run"
        (run / "summary.json").write_text(json.dumps(original, indent=2) + "\n")
        with (reference / "tracks.jsonl").open() as src, (run / "tracks.jsonl").open("w") as dst:
            for _ in range(count):
                line = src.readline()
                if not line:
                    raise ValueError("Reference has fewer frames than benchmark")
                dst.write(line)
        scores["reference_full_rate"] = evaluate_runs(
            [run], output / "score_reference_full_rate", error_clips=0)["combined"]
    report = {"source": str(source), "frames": count, "interval": interval,
              "detector_calls": detections, "sequential_fps": summary["fps"],
              "detector_fps": summary["detector_fps"], "stage_seconds": summary["stage_seconds"],
              "scores": scores, "output": str(output)}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/MOT17/train/MOT17-04-FRCNN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--interval", type=int, default=7)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.frames < 1 or args.interval < 1:
        parser.error("--frames and --interval must be positive")
    cv2.setNumThreads(1)
    benchmark(args.source.resolve(), args.output.resolve(), args.frames, args.interval, args.reference)
