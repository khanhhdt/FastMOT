"""Check the installed runtime and the selected MOT17 sequences."""

import configparser
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import cv2
import torch
import ultralytics
from ultralytics.trackers.track import TRACKER_MAP


def main():
    splits = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    development = {name.rsplit("-", 1)[0] for name in splits["development"]}
    evaluation = {name.rsplit("-", 1)[0] for name in splits["evaluation"]}
    if development & evaluation:
        raise ValueError("The development and evaluation sets share original videos.")
    if splits["smoke_sequence"] not in splits["development"]:
        raise ValueError("The smoke test must use a development sequence.")

    tracker_config = Path(ultralytics.__file__).parent / "cfg/trackers/tracktrack.yaml"
    if "tracktrack" not in TRACKER_MAP or not tracker_config.is_file():
        raise RuntimeError("This Ultralytics installation does not include TrackTrack.")

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "ultralytics", "opencv-python", "numpy", "lap")
        },
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "tracker_config": str(tracker_config),
        "image_checks": "All filenames; first and last image decoded per sequence.",
        "sequences": [],
    }
    for split in ("development", "evaluation"):
        for name in splits[split]:
            sequence = ROOT / splits["dataset_root"] / name
            metadata = configparser.ConfigParser()
            if not metadata.read(sequence / "seqinfo.ini"):
                raise FileNotFoundError(sequence / "seqinfo.ini")
            info = metadata["Sequence"]
            count = int(info["seqLength"])
            image_dir = sequence / info["imDir"]
            frames = sorted(image_dir.glob("*" + info["imExt"]))
            expected = [f"{index:06d}{info['imExt']}" for index in range(1, count + 1)]
            if [frame.name for frame in frames] != expected:
                raise ValueError(f"Missing or unexpected frame filenames: {name}")
            width, height = int(info["imWidth"]), int(info["imHeight"])
            for frame in (frames[0], frames[-1]):
                image = cv2.imread(str(frame))
                if image is None or image.shape[:2] != (height, width):
                    raise ValueError(f"Unreadable image or incorrect dimensions: {frame}")
            with (sequence / "gt/gt.txt").open() as handle:
                gt_rows = 0
                for row in csv.reader(handle):
                    if len(row) != 9 or not 1 <= int(row[0]) <= count:
                        raise ValueError(f"Invalid ground-truth row in {name}: {row}")
                    gt_rows += 1
            if not gt_rows:
                raise ValueError(f"Empty ground truth: {name}")
            report["sequences"].append({
                "name": name, "split": split, "frames": count,
                "fps": int(info["frameRate"]), "width": width, "height": height,
                "gt_rows": gt_rows,
            })
            print(f"OK {name}: {count} frames, {gt_rows} GT rows ({split})", flush=True)

    output = ROOT / "outputs/checks/setup_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Runtime: {report['device']}; Ultralytics {ultralytics.__version__}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
