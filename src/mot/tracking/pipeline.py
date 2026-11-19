"""Run YOLO26s + TrackTrack on a local video or a MOT image sequence."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from .sources import open_source
from .visualization import TrackVisualizer
from .mot_io import write_mot_frame
from .motion import ObservedGMC, SceneObserver, reset_scene
from .stationary import StationaryAnchor, StationaryAnchors


def extract_tracks(result, width, height):
    """Ignore unassigned detections; clip exported boxes without changing tracker state."""
    if result.boxes.id is None:
        return []
    boxes = result.boxes.data.cpu().numpy()
    if not np.isfinite(boxes).all() or (boxes[:, -1] != 0).any():
        raise ValueError("Expected finite person tracks")
    ids = boxes[:, 4].astype(int)
    if (ids <= 0).any() or (ids != boxes[:, 4]).any() or len(set(ids)) != len(ids):
        raise ValueError("Track IDs must be positive, integer and unique within each frame")
    tracks = []
    for row, track_id in zip(boxes, ids):
        xyxy = row[:4].copy()
        xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, width)
        xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, height)
        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
            tracks.append({"id": int(track_id), "xyxy": xyxy.tolist(),
                           "confidence": float(row[5]), "class_id": 0})
    return tracks


def run_tracking(source, output, max_frames=None, model_config=None, tracker_config=None,
                 runtime_config=None, progress_callback=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose another --output directory: {output}")
    model_config = Path(model_config or ROOT / "configs/models/yolo26s.json").resolve()
    manifest = json.loads(model_config.read_text())
    weights = ROOT / manifest["weights"]
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from configs/models/yolo26s.json")
    tracker_config = Path(tracker_config or ROOT / "configs/trackers/tracktrack.yaml").resolve()
    tracker_text = tracker_config.read_text()
    runtime = {"robust_gmc": True, "torch_threads": 4, "capture_detections": False}
    if runtime_config is not None:
        runtime.update(json.loads(Path(runtime_config).read_text()))
    if not isinstance(runtime["torch_threads"], int) or not 1 <= runtime["torch_threads"] <= 16:
        raise ValueError("torch_threads must be between 1 and 16")
    anchor_id = runtime.get("stationary_anchor_id")
    if anchor_id is not None and (type(anchor_id) is not int or anchor_id < 1):
        raise ValueError("stationary_anchor_id must be a positive track ID")
    anchor_ids = runtime.get("stationary_anchor_ids")
    if anchor_ids is not None and (
        anchor_id is not None or not isinstance(anchor_ids, list) or not anchor_ids
        or any(type(identity) is not int or identity < 1 for identity in anchor_ids)
        or len(set(anchor_ids)) != len(anchor_ids)
    ):
        raise ValueError("stationary_anchor_ids must be a list of unique positive IDs, supplied without stationary_anchor_id")
    seed_frames = runtime.get("stationary_anchor_seed_frames")
    if seed_frames is not None and (
        anchor_ids is None or not isinstance(seed_frames, dict)
        or any(
            identity not in {str(i) for i in anchor_ids}
            or not isinstance(window, list) or len(window) != 2
            or any(type(frame) is not int for frame in window)
            or window[0] < 1 or not 5 <= window[1] - window[0] + 1 <= StationaryAnchor.LEARN_FRAMES
            for identity, window in seed_frames.items()
        )
    ):
        raise ValueError("stationary_anchor_seed_frames needs selected IDs and 5–30 consecutive frames per window")
    hold_seconds = runtime.get("stationary_anchor_hold_seconds", 4)
    if type(hold_seconds) not in (int, float) or not math.isfinite(hold_seconds) or hold_seconds <= 0:
        raise ValueError("stationary_anchor_hold_seconds must be positive and finite")
    options = {**manifest["inference"], "classes": [0],
               "device": 0 if torch.cuda.is_available() else "cpu", "verbose": False}
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(1)
    # Each call owns its model, tracker and drawing history: IDs never leak into the next source.
    model = YOLO(str(weights))
    if model.task != "detect" or model.names[0] != "person":
        raise ValueError("Expected a Detect checkpoint with COCO person class 0")
    visualizer = TrackVisualizer(trail_length=30)
    seen_ids = set()
    count = observations = frames_with_tracks = 0
    frames_with_features = 0
    frame_times = []
    stages = {key: 0.0 for key in ("decode", "detection", "tracking", "drawing", "export")}
    observer, motion_events, context = SceneObserver(), [], {}

    def before_tracking(predictor):
        tracker = predictor.trackers[0]
        if runtime["robust_gmc"] and context["event"] == "scene_cut":
            reset_scene(tracker)
            visualizer.history.clear()
            motion_events.append({"frame": context["index"], "type": "scene_cut"})
        if not isinstance(tracker.gmc, ObservedGMC):
            tracker.gmc = ObservedGMC(tracker.gmc, guarded=runtime["robust_gmc"])
        tracker.gmc.event = context["event"]
        if runtime["capture_detections"]:
            result = predictor.results[0]
            alternative = getattr(result, "_loose", None)
            context["raw"] = {"frame": context["index"],
                              "tight": result.boxes.data.cpu().numpy().tolist(),
                              "loose": alternative[0].boxes.data.cpu().numpy().tolist() if alternative else None,
                              "scene_event": context["event"]}
        context["captured_at"] = time.perf_counter()

    with open_source(source, max_frames) as (info, frames):
        max_gap = max(1, round(info["fps"] * hold_seconds))
        anchor = None
        if anchor_ids is not None:
            anchor = StationaryAnchors(anchor_ids, max_gap, {int(i): window for i, window in (seed_frames or {}).items()})
        elif anchor_id is not None:
            anchor = StationaryAnchor(anchor_id, max_gap)
        width, height = info["width"], info["height"]
        video_size = (width + width % 2, height + height % 2)
        total = min(info["frames"], max_frames) if max_frames else info["frames"]
        if progress_callback:
            progress_callback({"phase": "warmup", "frames": 0, "total": total})
        # Warm the detector without creating tracker state or consuming a source frame.
        model.predict(np.zeros((height, width, 3), dtype=np.uint8), **options)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.set_num_threads(runtime["torch_threads"])
        # Switching from predict warm-up to track rebuilds the native predictor once.
        # Its CPU setup resets Torch threads, so apply the chosen limit after setup.
        model.add_callback("on_predict_start", lambda predictor: torch.set_num_threads(runtime["torch_threads"]))
        model.add_callback("on_predict_postprocess_end", before_tracking)
        output.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output / "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info["fps"], video_size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot create output video: {output}")
        started = time.perf_counter()
        frame_started = started
        try:
            with (output / "tracks.jsonl").open("w") as handle, (output / "tracks.txt").open("w") as mot, \
                 (output / "detections.jsonl").open("w") if runtime["capture_detections"] else open(os.devnull, "w") as raw:
                for index, image in frames:
                    if image.shape[:2] != (height, width):
                        raise ValueError(f"Frame {index} dimensions differ from source metadata")
                    model_started = time.perf_counter()
                    stages["decode"] += model_started - frame_started
                    context.update(index=index, event=observer.observe(image) if runtime["robust_gmc"] or anchor else "normal")
                    result = model.track(image, persist=True, tracker=str(tracker_config), **options)[0]
                    tracked_at = time.perf_counter()
                    stages["detection"] += context["captured_at"] - model_started
                    stages["tracking"] += tracked_at - context["captured_at"]
                    tracker = model.predictor.trackers[0]
                    if tracker.frame_id != index:
                        raise RuntimeError(f"Tracker did not advance exactly once at frame {index}")
                    frames_with_features += int(any(getattr(t, "curr_feat", None) is not None
                                                    for t in tracker.tracked_stracks))
                    tracks = extract_tracks(result, width, height)
                    if anchor is not None:
                        tracks = anchor.update(index, tracks, scene_cut=context["event"] == "scene_cut")
                    handle.write(json.dumps({"frame": index, "tracks": tracks}, allow_nan=False) + "\n")
                    write_mot_frame(mot, index, tracks)
                    if runtime["capture_detections"]:
                        context["raw"]["gmc_warp"] = tracker.gmc.last_warp.tolist()
                        context["raw"]["gmc_dtype"] = str(tracker.gmc.last_warp.dtype)
                        raw.write(json.dumps(context["raw"], allow_nan=False) + "\n")
                    seen_ids.update(track["id"] for track in tracks)
                    observations += len(tracks)
                    frames_with_tracks += int(bool(tracks))
                    draw_started = time.perf_counter()
                    stages["export"] += draw_started - tracked_at
                    annotated = visualizer.draw(image, index, tracks)
                    annotated = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                    export_started = time.perf_counter()
                    stages["drawing"] += export_started - draw_started
                    writer.write(annotated)
                    if index == 1 and not cv2.imwrite(str(output / "frame_000001.jpg"), annotated):
                        raise RuntimeError("Cannot save first preview image")
                    count += 1
                    if count % 50 == 0:
                        print(f"Processed {count} frames; {len(seen_ids)} track IDs", flush=True)
                    if progress_callback:
                        progress_callback({"phase": "tracking", "frames": count, "total": total})
                    now = time.perf_counter()
                    stages["export"] += now - export_started
                    frame_times.append(now - frame_started)
                    frame_started = now
            if not count:
                raise ValueError("Source contains no readable frames")
            if not cv2.imwrite(str(output / f"frame_{count:06d}.jpg"), annotated):
                raise RuntimeError("Cannot save last preview image")
        finally:
            writer.release()
        elapsed = time.perf_counter() - started
        stages["finalization"] = max(0.0, elapsed - sum(stages.values()))
    cpu_model = platform.processor() or platform.machine()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    summary = {
        "mode": "track", "source": info, "frames_processed": count,
        "frames_with_tracks": frames_with_tracks, "empty_track_frames": count - frames_with_tracks,
        "track_observations": observations, "unique_track_ids": sorted(seen_ids),
        "tracker_backend": type(tracker).__name__, "tracker_frames_processed": tracker.frame_id,
        "tracker_config": tracker_text, "inference_options": options,
        "runtime_options": runtime, "tracking_policy": {"robust_gmc": runtime["robust_gmc"]},
        "stage_seconds": {key: round(value, 6) for key, value in stages.items()},
        "stage_ms_per_frame": {key: round(value * 1000 / count, 3) for key, value in stages.items()},
        "stage_timing_note": "Detection interval ends at the pre-tracker callback and includes first-call initialization and capture; export includes progress callbacks; finalization includes remaining overhead",
        "motion": {"events": motion_events, "blank_frames_skipped": tracker.gmc.blank_frames,
                   "rejected_warps": tracker.gmc.rejected_warps,
                   "failed_estimates": tracker.gmc.failed_estimates},
        "config_paths": {"model": str(model_config), "tracker": str(tracker_config)},
        "reid": {"enabled": bool(tracker.args.with_reid), "model": tracker.args.model,
                 "native_features": bool(tracker.args.with_reid and tracker.args.model == "auto"),
                 "frames_with_track_features": frames_with_features},
        "model_sha256": digest, "model_source": manifest["source"],
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "video_size": list(video_size), "trail_length_frames": visualizer.trail_length,
        "coordinates": "xyxy, clipped to original image, pixels with origin (0, 0); frames start at 1",
        "output_policy": "Only tracks assigned an ID in the current frame; unassigned detections and unobserved positions are not drawn",
        "elapsed_seconds": round(elapsed, 3),
        "fps": round(count / elapsed, 3),
        "frame_latency_ms": {"median": round(float(np.median(frame_times)) * 1000, 3),
                             "p95": round(float(np.percentile(frame_times, 95)) * 1000, 3)},
        "timing_protocol": {"detector_warmup_frames": 1, "tracker_initialization_included": True,
                            "video_finalization_included_in_fps": True},
        "hardware": {"device": str(options["device"]), "cpu_model": cpu_model,
                     "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "platform": platform.platform()},
        "timing_scope": "After detector warm-up: source decode, tracking initialization/inference, drawing, JSONL/MOT export and video finalization; excludes model loading and warm-up",
        "note": "Unique track IDs are not a ground-truth person count or an accuracy metric",
    }
    if anchor_ids is not None:
        summary["tracking_policy"]["stationary_anchor_ids"] = anchor_ids
        if seed_frames:
            summary["tracking_policy"]["stationary_anchor_seed_frames"] = seed_frames
        summary["stationary_anchors"] = {
            "max_gap_frames": anchor.max_gap_frames,
            "by_id": {str(identity): item.events for identity, item in anchor.anchors.items()},
        }
    elif anchor_id is not None:
        summary["tracking_policy"]["stationary_anchor_id"] = anchor_id
        summary["stationary_anchor"] = {"id": anchor_id, "max_gap_frames": anchor.max_gap_frames, "events": anchor.events}
    if anchor is not None and hold_seconds != 4:
        summary["tracking_policy"]["stationary_anchor_hold_seconds"] = hold_seconds
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {count} frames, {observations} track observations. Output: {output}", flush=True)
    return summary
