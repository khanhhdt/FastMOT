"""Replay TrackTrack from verified raw detections and sparse-flow warps; never report end-to-end FPS."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import tempfile
from types import SimpleNamespace

from .pipeline import extract_tracks
from .motion import reset_scene
from .mot_io import write_mot_frame

import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.trackers.track_tracker import TRACKTRACK, compute_dets_del
from ultralytics.utils import IterableSimpleNamespace, YAML


class CachedGMC:
    def __init__(self, method):
        self.method = None if method == "none" else method
        self.warp = np.eye(2, 3)

    def apply(self, image, detections=None):
        return self.warp

    def reset_params(self):
        pass


def replay_tracking(source_run, output, tracker_config):
    source_run, output = Path(source_run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    summary = deepcopy(json.loads((source_run / "summary.json").read_text()))
    if {"stationary_anchor_id", "stationary_anchor_ids"} & summary.get("tracking_policy", {}).keys():
        raise ValueError("Replay does not apply stationary-anchor ID recovery")
    config = YAML.load(tracker_config)
    with tempfile.TemporaryDirectory(prefix="mot-replay-config-") as scratch:
        path = Path(scratch) / "original.yaml"
        path.write_text(summary["tracker_config"])
        original = YAML.load(path)
    if config["with_reid"] or config["gmc_method"] != original["gmc_method"]:
        raise ValueError("Replay supports ReID-off and the same camera-motion method as its cache")
    if config["track_low_thresh"] < summary["inference_options"]["conf"]:
        raise ValueError("Replay threshold cannot request detections absent from the cache")
    tracker = TRACKTRACK(IterableSimpleNamespace(**config))
    tracker.gmc = CachedGMC(config["gmc_method"])
    info = summary["source"]
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    output.mkdir(parents=True, exist_ok=True)
    seen, observations, with_tracks, count = set(), 0, 0, 0
    started = time.perf_counter()
    with (source_run / "detections.jsonl").open() as raw, \
         (output / "tracks.jsonl").open("w") as handle, (output / "tracks.txt").open("w") as mot:
        for record in map(json.loads, raw):
            count += 1
            if record["frame"] != count:
                raise ValueError("Cache frame order is invalid")
            if summary.get("tracking_policy", {}).get("robust_gmc") and record["scene_event"] == "scene_cut":
                reset_scene(tracker)
            tracker.gmc.warp = np.asarray(record["gmc_warp"], dtype=record.get("gmc_dtype", "float64"))
            tight = Results(dummy, path="cache.jpg", names={0: "person"},
                            boxes=torch.tensor(record["tight"], dtype=torch.float32).reshape(-1, 6))
            if record["loose"] is not None:
                alternative = Results(dummy, path="cache.jpg", names={0: "person"},
                                      boxes=torch.tensor(record["loose"], dtype=torch.float32).reshape(-1, 6))
                tight._loose = (alternative, 0.95)
            deleted = compute_dets_del(SimpleNamespace(results=[tight], args=SimpleNamespace(task="detect")))
            rows = tracker.update(tight.boxes.cpu().numpy(), dummy, dets_del=deleted[0] if deleted else None)
            if len(rows):
                result = Results(dummy, path="cache.jpg", names={0: "person"}, boxes=torch.from_numpy(rows[:, :-1]))
                tracks = extract_tracks(result, info["width"], info["height"])
            else:
                tracks = []
            handle.write(json.dumps({"frame": count, "tracks": tracks}) + "\n")
            write_mot_frame(mot, count, tracks)
            seen.update(t["id"] for t in tracks)
            observations += len(tracks)
            with_tracks += int(bool(tracks))
    if count != summary["frames_processed"]:
        raise ValueError("Incomplete detection cache")
    for key in ("timing_protocol", "fps", "frame_latency_ms", "stage_seconds", "stage_ms_per_frame"):
        summary.pop(key, None)
    summary.update(tracker_config=Path(tracker_config).read_text(), unique_track_ids=sorted(seen),
                   track_observations=observations, frames_with_tracks=with_tracks,
                   empty_track_frames=count - with_tracks, tracker_frames_processed=count,
                   elapsed_seconds=round(time.perf_counter() - started, 3),
                   timing_scope="Cached replay only; no detector, image decoding, drawing or video encoding; not end-to-end FPS",
                   replay={"source_run": str(source_run.resolve()),
                           "cache_sha256": hashlib.sha256((source_run / "detections.jsonl").read_bytes()).hexdigest()})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
