"""Run YOLO26s + TrackTrack on a short MOT17 development clip."""

import argparse
import configparser
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    sequence = ROOT / splits["dataset_root"] / splits["smoke_sequence"]
    metadata = configparser.ConfigParser()
    metadata.read(sequence / "seqinfo.ini")
    info = metadata["Sequence"]
    if not 1 <= args.frames <= int(info["seqLength"]):
        parser.error("--frames must be between 1 and the sequence length")

    weights = ROOT / "weights/yolo26s.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Download the checkpoint as described in README.md: {weights}")
    output = ROOT / "outputs/runs/smoke_test"
    output.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(1)
    model = YOLO(str(weights))
    if model.task != "detect" or model.names[0] != "person":
        raise ValueError("Expected a YOLO26 Detect checkpoint with COCO person class 0.")
    options = {"classes": [0], "imgsz": 640, "conf": 0.05,
               "max_det": 1000, "device": device, "verbose": False}
    first = sequence / info["imDir"] / f"000001{info['imExt']}"
    model.predict(str(first), **options)  # Warm up without creating tracker state.

    video_path = output / "preview.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             int(info["frameRate"]),
                             (int(info["imWidth"]), int(info["imHeight"])))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")
    seen_ids = set()
    tracked_frames = 0
    observations = 0
    started = time.perf_counter()
    try:
        for index in range(1, args.frames + 1):
            path = sequence / info["imDir"] / f"{index:06d}{info['imExt']}"
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            result = model.track(image, persist=True,
                                 tracker=str(ROOT / "configs/trackers/tracktrack.yaml"), **options)[0]
            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                if len(ids) != len(set(ids)):
                    raise RuntimeError(f"Duplicate track IDs at frame {index}")
                tracked_frames += 1
                observations += len(ids)
                seen_ids.update(ids)
            annotated = result.plot(line_width=2, conf=False, color_mode="instance")
            writer.write(annotated)
            if index in (1, args.frames):
                if not cv2.imwrite(str(output / f"frame_{index:06d}.jpg"), annotated):
                    raise RuntimeError("Cannot save preview image")
            if index % 20 == 0 or index == args.frames:
                print(f"Processed {index}/{args.frames} frames; {len(seen_ids)} track IDs", flush=True)
    finally:
        writer.release()
    elapsed = time.perf_counter() - started
    if not observations:
        raise RuntimeError("The model ran, but no track IDs were produced.")

    report = {
        "sequence": sequence.name, "frames_processed": args.frames,
        "frames_with_tracks": tracked_frames, "track_observations": observations,
        "unique_track_ids": sorted(seen_ids),
        "note": "Unique track IDs are not a ground-truth person count or an accuracy metric.",
        "model": weights.name, "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "model_source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
        "checkpoint_training_data": (model.ckpt or {}).get("train_args", {}).get("data"),
        "tracker_backend": type(model.predictor.trackers[0]).__name__,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "torch": torch.__version__, "device": str(device), "torch_threads": torch.get_num_threads(),
        "inference_options": options,
        "tracker_config": (ROOT / "configs/trackers/tracktrack.yaml").read_text(),
        "elapsed_seconds": round(elapsed, 3),
        "preview": str(video_path.relative_to(ROOT)),
        "timing_scope": "Smoke test loop, including decode, inference, tracker, drawing and writing; not a formal benchmark.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.frames} frames, {tracked_frames} frames with tracks, device={device}")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
