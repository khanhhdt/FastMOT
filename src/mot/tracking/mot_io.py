"""Export zero-origin JSONL tracks as one-origin MOTChallenge rows."""

import json
import math
from pathlib import Path


def write_mot_frame(handle, frame, tracks):
    if not isinstance(frame, int) or frame < 1:
        raise ValueError("MOT frame indices must be positive integers")
    seen = set()
    for track in tracks:
        if track["class_id"] != 0:
            raise ValueError("Only person tracks may be exported")
        track_id = track["id"]
        if not isinstance(track_id, int) or track_id < 1 or track_id in seen:
            raise ValueError("Track IDs must be positive integers and unique per frame")
        seen.add(track_id)
        x1, y1, x2, y2 = track["xyxy"]
        score = track["confidence"]
        if (not all(math.isfinite(v) for v in (x1, y1, x2, y2, score))
                or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or not 0 <= score <= 1):
            raise ValueError("Invalid box or confidence in tracking result")
        handle.write(f"{frame},{track_id},{x1 + 1:.6f},{y1 + 1:.6f},"
                     f"{x2 - x1:.6f},{y2 - y1:.6f},{score:.6f},-1,-1,-1\n")


def export_mot(jsonl, destination, expected_frames):
    """Require a record for every frame, including frames with no assigned tracks."""
    count = 0
    with Path(jsonl).open() as source, Path(destination).open("x") as target:
        for count, line in enumerate(source, 1):
            row = json.loads(line)
            if row["frame"] != count:
                raise ValueError("JSONL frame indices must be consecutive and start at 1")
            write_mot_frame(target, count, row["tracks"])
    if count != expected_frames:
        raise ValueError(f"Expected {expected_frames} JSONL frames, found {count}")
