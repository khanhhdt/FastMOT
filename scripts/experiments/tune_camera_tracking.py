"""Tune the live tracking profile on sampled MOT17. Keep MOT17-04 for validation."""

import argparse
import configparser
import csv
from itertools import product
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.evaluation.metrics import evaluate_runs
from mot.tracking.pipeline import run_tracking
from ultralytics.utils import YAML

MODEL = ROOT / "configs/models/yolo26s_1280_openvino_fp32.json"
TRACKER = ROOT / "configs/trackers/tracktrack_quality_reid60.yaml"
RUNTIME = ROOT / "configs/runtime/quality.json"
DEVELOPMENT = ("MOT17-02-FRCNN", "MOT17-09-FRCNN")
VALIDATION = "MOT17-04-FRCNN"


def sample(source, target, fps):
    # Use the first source frame at or after each sample timestamp.
    config = configparser.ConfigParser()
    config.read(source / "seqinfo.ini")
    sequence = config["Sequence"]
    source_fps, count = sequence.getint("frameRate"), sequence.getint("seqLength")
    selected = [1 + (tick * source_fps + fps - 1) // fps for tick in range((count * fps + source_fps - 1) // source_fps)]
    selected = [frame for frame in selected if frame <= count]
    mapping = {frame: index for index, frame in enumerate(selected, 1)}
    images = sequence["imDir"]
    ext = sequence["imExt"]
    (target / images).mkdir(parents=True)
    (target / "gt").mkdir()
    for old, new in mapping.items():
        (target / images / f"{new:06}{ext}").symlink_to(source / images / f"{old:06}{ext}")
    with (source / "gt/gt.txt").open() as original, (target / "gt/gt.txt").open("w") as filtered:
        writer = csv.writer(filtered, lineterminator="\n")
        for frame, *row in csv.reader(original):
            if int(frame) in mapping:
                writer.writerow([mapping[int(frame)], *row])
    sequence.update(frameRate=str(fps), seqLength=str(len(mapping)))
    with (target / "seqinfo.ini").open("w") as handle:
        config.write(handle)
    (target / "source_frames.json").write_text(json.dumps(selected) + "\n")
    return mapping


def sample_reference(original, run, target, mapping, fps):
    run.mkdir(parents=True)
    with (original / "tracks.jsonl").open() as source, (run / "tracks.jsonl").open("w") as dest:
        for line in source:
            frame = json.loads(line)
            if frame["frame"] in mapping:
                frame["frame"] = mapping[frame["frame"]]
                dest.write(json.dumps(frame) + "\n")
    summary = json.loads((original / "summary.json").read_text())
    summary["source"].update(path=str(target), fps=fps, frames=len(mapping))
    summary["frames_processed"] = len(mapping)
    summary.pop("timing_protocol", None)
    summary["timing_scope"] = "Sampled predictions from the complete online OpenVINO run"
    (run / "summary.json").write_text(json.dumps(summary) + "\n")
    return run


def score_candidate(folder, sources, tracker):
    runs = []
    for source in sources:
        run = folder / source.name
        run.mkdir(parents=True)
        with TemporaryDirectory() as scratch:
            temporary_run = Path(scratch) / "run"
            run_tracking(source, temporary_run, None, MODEL, tracker, RUNTIME)
            for name in ("tracks.jsonl", "tracks.txt", "summary.json"):
                shutil.copyfile(temporary_run / name, run / name)
        runs.append(run)
    return evaluate_runs(runs, folder / "score", error_clips=0)["combined"]


def tune(output, reference, fps):
    folder = output / str(fps)
    sources, baselines = {}, {}
    for name in (*DEVELOPMENT, VALIDATION):
        source = ROOT / "data/MOT17/train" / name
        target = sources[name] = folder / "sampled" / name
        mapping = sample(source, target, fps)
        baselines[name] = sample_reference(reference / name, folder / "reference" / name, target, mapping, fps)
    baseline = evaluate_runs([baselines[VALIDATION]], folder / "reference/score", error_clips=0)["combined"]

    variants = []
    for seconds, confirmations in product((2, 4), (2, 3)):
        candidate = folder / f"buffer_{seconds}s_confirm_{confirmations}"
        candidate.mkdir()
        tracker = candidate / "tracktrack.yaml"
        config = YAML.load(TRACKER)
        config.update(track_buffer=fps * seconds, min_track_len=confirmations)
        YAML.save(tracker, config)
        score = score_candidate(candidate / "development", [sources[name] for name in DEVELOPMENT], tracker)
        variants.append({"track_buffer": fps * seconds, "min_track_len": confirmations,
                         "development": score, "tracker": str(tracker)})
    # Stable grid order breaks exact ties in favor of shorter buffers and earlier confirmation.
    selected = max(variants, key=lambda choice: (choice["development"]["IDF1"], choice["development"]["HOTA"]))
    tracker = Path(selected["tracker"])
    validation = score_candidate(tracker.parent / "validation", [sources[VALIDATION]], tracker)
    delta = {metric: round(validation[metric] - baseline[metric], 3) for metric in ("HOTA", "IDF1")}
    result = {"fps": fps, "candidates": variants, "selected": selected, "reference": baseline,
              "validation": validation, "delta": delta,
              "quality_pass": delta["HOTA"] >= -2 and delta["IDF1"] >= -3}
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"FPS {fps} quality gate: {result['quality_pass']}, delta {delta}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Saved full-rate OpenVINO online/train directory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    results = [tune(output, args.reference.resolve(), rate) for rate in (3, 4)]
    (output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
