"""Run person detection and save original-image coordinates for later tracking."""

import hashlib
import json
import os
from pathlib import Path
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


def run_detection(source, output, max_frames=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose another --output directory: {output}")
    manifest = json.loads((ROOT / "configs/models/yolo26s.json").read_text())
    weights = ROOT / manifest["weights"]
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from configs/models/yolo26s.json")
    display_confidence = manifest["display_confidence"]
    options = {**manifest["inference"], "classes": [0],
               "device": 0 if torch.cuda.is_available() else "cpu", "verbose": False}
    if not 0 <= options["conf"] <= display_confidence <= 1:
        raise ValueError("Expected 0 <= inference confidence <= display confidence <= 1")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(1)
    model = YOLO(str(weights))
    if model.task != "detect" or model.names[0] != "person":
        raise ValueError("Expected a Detect checkpoint with COCO person class 0")
    count = total = displayed = maximum = frames_at_limit = empty_frames = 0
    below_display = below_quarter = 0
    with open_source(source, max_frames) as (info, frames):
        width, height = info["width"], info["height"]
        # MP4 encoders require even dimensions; only pad the rendered output, never the detection input.
        video_size = (width + width % 2, height + height % 2)
        output.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output / "detections.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info["fps"], video_size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot create output video: {output}")
        started = time.perf_counter()
        try:
            with (output / "detections.jsonl").open("w") as handle:
                for index, image in frames:
                    if image.shape[:2] != (height, width):
                        raise ValueError(f"Frame {index} dimensions differ from source metadata")
                    result = model.predict(image, **options)[0]
                    boxes = result.boxes.data.cpu().numpy()
                    if len(boxes) and (
                        not np.isfinite(boxes).all() or (boxes[:, 5] != 0).any()
                        or (boxes[:, [0, 1]] < 0).any()
                        or (boxes[:, 2] > width).any() or (boxes[:, 3] > height).any()
                        or (boxes[:, 2:4] <= boxes[:, :2]).any()
                        or (boxes[:, 4] < options["conf"]).any() or (boxes[:, 4] > 1).any()
                    ):
                        raise ValueError(f"Invalid person detection or original-image coordinates at frame {index}")
                    detections = [{"xyxy": row[:4].tolist(), "confidence": float(row[4]), "class_id": 0}
                                  for row in boxes]
                    handle.write(json.dumps({"frame": index, "detections": detections}, allow_nan=False) + "\n")
                    visible = result[result.boxes.conf >= display_confidence]
                    annotated = visible.plot(line_width=2, conf=True)
                    annotated = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                    writer.write(annotated)
                    if index == 1 and not cv2.imwrite(str(output / "frame_000001.jpg"), annotated):
                        raise RuntimeError("Cannot save first preview image")
                    count += 1
                    total += len(boxes)
                    displayed += len(visible.boxes)
                    below_display += int((boxes[:, 4] < display_confidence).sum())
                    below_quarter += int((boxes[:, 4] < 0.25).sum())
                    maximum = max(maximum, len(boxes))
                    frames_at_limit += int(len(boxes) >= options["max_det"])
                    empty_frames += int(not len(boxes))
                    if count % 50 == 0:
                        print(f"Processed {count} frames; {total} person detections", flush=True)
            if not count:
                raise ValueError("Source contains no readable frames")
            if not cv2.imwrite(str(output / f"frame_{count:06d}.jpg"), annotated):
                raise RuntimeError("Cannot save last preview image")
        finally:
            writer.release()
        elapsed = time.perf_counter() - started
    summary = {
        "mode": "detect", "source": info, "frames_processed": count,
        "detections": total, "displayed_detections": displayed, "empty_frames": empty_frames,
        "detections_below_display_threshold": below_display,
        "detections_below_0_25": below_quarter,
        "max_people_in_frame": maximum, "frames_at_max_det": frames_at_limit,
        "coordinates": "xyxy, pixels in original image, origin (0, 0); frame numbers start at 1",
        "video_size": list(video_size), "display_confidence": display_confidence,
        "inference_options": options, "postprocessing": "nms_free" if model.predictor.model.end2end else "nms",
        "model_sha256": digest, "model_source": manifest["source"],
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "elapsed_seconds": round(elapsed, 3),
        "timing_scope": "Decode, first-inference setup, detection, drawing and export; not a formal benchmark",
        "note": "Detection counts are observations per frame, not unique people or tracking accuracy",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {count} frames, {total} detections. Output: {output}", flush=True)
    return summary
