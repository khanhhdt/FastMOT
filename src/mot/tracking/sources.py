"""Read local video files or MOT image sequences without loading them into RAM."""

import configparser
from contextlib import contextmanager
import math
from pathlib import Path

import cv2


@contextmanager
def open_source(source, max_frames=None):
    source = Path(source).resolve()
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if source.is_dir():
        config = configparser.ConfigParser()
        if not config.read(source / "seqinfo.ini"):
            raise FileNotFoundError(f"Expected a MOT sequence with seqinfo.ini: {source}")
        sequence = config["Sequence"]
        info = {
            "kind": "mot_sequence", "path": str(source),
            "fps": float(sequence["frameRate"]), "frames": int(sequence["seqLength"]),
            "width": int(sequence["imWidth"]), "height": int(sequence["imHeight"]),
        }

        def frames():
            count = min(info["frames"], max_frames) if max_frames else info["frames"]
            for index in range(1, count + 1):
                path = source / sequence["imDir"] / f"{index:06d}{sequence['imExt']}"
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot read frame: {path}")
                yield index, image

        capture = None
    else:
        if not source.is_file():
            raise FileNotFoundError(source)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot open video: {source}")
        info = {
            "kind": "video", "path": str(source),
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        def frames():
            index = 0
            while max_frames is None or index < max_frames:
                ok, image = capture.read()
                if not ok:
                    if info["frames"] > index:
                        raise ValueError(f"Video ended before expected frame {index + 1}: {source}")
                    break
                index += 1
                yield index, image

    try:
        if not math.isfinite(info["fps"]) or info["fps"] <= 0 or min(info["width"], info["height"]) <= 0:
            raise ValueError(f"Invalid FPS or image dimensions: {source}")
        yield info, frames()
    finally:
        if capture is not None:
            capture.release()
