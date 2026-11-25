"""Compare OpenVINO detector precisions with the live camera tracker on MOT17."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.utils import YAML

from mot.evaluation.metrics import evaluate_runs
from mot.tracking.pipeline import checkpoint_sha256, run_tracking
from tune_camera_tracking import DEVELOPMENT, VALIDATION, sample


def make_int8(fp32, data, output):
    import nncf
    import openvino as ov

    paths = []
    for name in DEVELOPMENT:
        images = sorted((data / name / "img1").glob("*.jpg"))
        paths.extend(images[i] for i in np.linspace(0, len(images) - 1, 48, dtype=int))
    letterbox = LetterBox(new_shape=(1280, 1280), auto=True, stride=32)

    def preprocess(path):
        image = letterbox(image=cv2.imread(str(path)))
        return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255

    print(f"Calibrating INT8 on {len(paths)} images from {', '.join(DEVELOPMENT)}", flush=True)
    xml = next(fp32.glob("*.xml"))
    model = ov.Core().read_model(xml)
    quantized = nncf.quantize(model, nncf.Dataset(paths, preprocess), subset_size=len(paths),
                             preset=nncf.QuantizationPreset.MIXED)
    output.mkdir()
    ov.save_model(quantized, output / xml.name, compress_to_fp16=False)
    metadata = YAML.load(fp32 / "metadata.yaml")
    metadata["args"]["quantize"] = 8
    YAML.save(output / "metadata.yaml", metadata)
    return {"sequences": list(DEVELOPMENT), "images": len(paths), "nncf": nncf.__version__,
            "openvino": ov.__version__, "preset": "mixed"}


def benchmark(output):
    profile = json.loads((ROOT / "configs/profiles/camera_live.json").read_text())
    fp32_config = ROOT / profile["model_config"]
    fp32 = json.loads(fp32_config.read_text())
    data = ROOT / "data/MOT17/train"
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    sequence = output / "sampled" / VALIDATION
    mapping = sample(data / VALIDATION, sequence, profile["fps"])
    int8_weights = output / "yolo26s_int8_openvino_model"
    calibration = make_int8(ROOT / fp32["weights"], data, int8_weights)
    int8_config = output / "int8.json"
    int8 = {**fp32, "model": "YOLO26s OpenVINO INT8", "weights": str(int8_weights),
            "sha256": checkpoint_sha256(int8_weights), "export": {**fp32["export"], "precision": "INT8"}}
    int8_config.write_text(json.dumps(int8, indent=2) + "\n")

    configs = {
        "FP32": fp32_config,
        "FP16": ROOT / "configs/models/yolo26s_1280_openvino_fp16.json",
        "INT8": int8_config,
    }
    results = {}
    for precision, config in configs.items():
        print(f"Benchmarking {precision} on {VALIDATION}", flush=True)
        run = output / precision / sequence.name
        summary = run_tracking(sequence, run, None, config, ROOT / profile["tracker_config"],
                               ROOT / profile["runtime_config"])
        score = evaluate_runs([run], output / precision / "score", error_clips=0)["combined"]
        weights = ROOT / json.loads(config.read_text())["weights"]
        results[precision] = {**score, "stages_ms": summary["stage_ms_per_frame"],
                              "weights_bytes": sum(path.stat().st_size for path in weights.glob("*.bin"))}
    report = {"imgsz": fp32["inference"]["imgsz"], "sample_fps": profile["fps"],
              "evaluation": {"sequence": VALIDATION, "frames": len(mapping)},
              "calibration": calibration, "cpu": summary["hardware"]["cpu_model"],
              "timing": summary["timing_scope"], "results": results}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    benchmark(parser.parse_args().output.resolve())
