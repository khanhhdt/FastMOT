"""Compare the V2 tracker, a longer buffer, and a dedicated ReID encoder."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.evaluation.metrics import evaluate_runs
from mot.evaluation.occlusion import analyze_study
from mot.tracking.pipeline import run_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-only", action="store_true", help="Measure ID retention from existing runs")
    args = parser.parse_args()
    if not args.score_only:
        protocol = ROOT / "configs/experiments/reid_occlusion.json"
        plan = json.loads(protocol.read_text())
        encoder = plan["encoder"]
        if hashlib.sha256(Path(encoder["path"]).read_bytes()).hexdigest() != encoder["sha256"]:
            raise ValueError("ReID encoder checksum differs from the experiment protocol")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "protocol.json").write_bytes(protocol.read_bytes())
        for item in plan["sequences"]:
            source = Path(item["source"])
            for name, tracker in plan["variants"].items():
                target = args.output / name / source.name
                print(f"RUN {name} {source.name}", flush=True)
                summary = run_tracking(source, target, item["frames"], plan["model_config"], tracker,
                                       plan["runtime_config"])
                if name == "reid60" and not summary["reid"]["frames_with_track_features"]:
                    raise RuntimeError(f"No ReID features reached the tracker: {source}")
                print(f"RESULT {name} {source.name}: {summary['fps']} FPS", flush=True)
            for name in plan["variants"]:
                evaluate_runs([args.output / name / source.name], args.output / name / "scoring" / source.name,
                              error_clips=0, benchmark=source.parts[1])
    report = analyze_study(args.output)
    (args.output / "occlusion_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]["all"], indent=2))


if __name__ == "__main__":
    main()
