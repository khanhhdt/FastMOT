"""Recheck the six V2 acceptance criteria from saved, measured evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/improvements")
    args = parser.parse_args()
    folder = args.directory
    protocol = read(ROOT / "configs/experiments/six_issues_v2.json")
    targets = protocol["acceptance"]
    baseline = read(folder / "baseline/evaluation/metrics.json")
    quality = read(folder / "quality/evaluation/metrics.json")
    selection = read(folder / "selection.json")
    fast = read(ROOT / selection["fast_evaluation"])
    b, q, f = baseline["combined"], quality["combined"], fast["combined"]
    diagnostic = read(folder / "quality/evaluation/failure_analysis.json")
    baseline_diagnostic = read(folder / "baseline/evaluation/failure_analysis.json")
    regression = read(folder / "motion/regression.json")
    replay = read(folder / "replay_validation.json")
    benchmark = read(folder / "final_speed/benchmark.json")
    external = read(folder / "external/comparison.json")
    frozen = folder / "frozen"
    freeze = read(frozen / "manifest.json")
    api = read(folder / "app_api_final/api_validation.json")
    browser = read(folder / "app_browser_validation.json")
    test_log = (folder / "tests.log").read_text()
    tests_passed = "\nOK\n" in test_log and "FAILED" not in test_log
    test_count = int(re.search(r"Ran (\d+) tests", test_log).group(1))

    def scope(report):
        return [(item["sequence"], item["frames"][1]) for item in report["scope"]]

    expected_development = [(item["sequence"], item["frames"]) for item in protocol["selection_data"]]
    assert scope(baseline) == scope(quality) == scope(fast) == expected_development
    assert replay["passed"] and tests_passed, "Prediction parity or automated regression suite failed"
    fn_reduction = 1 - q["FN"] / b["FN"]
    fp_reduction = 1 - q["FP"] / b["FP"]
    recall_change = (q["TP"] / (q["TP"] + q["FN"]) - b["TP"] / (b["TP"] + b["FN"])) * 100
    speed = {}
    for name in ("baseline", "fast"):
        records = [r for r in benchmark["records"] if r["profile"] == name]
        assert len(records) == benchmark["repeats"]
        speed[name] = sum(r["frames"] for r in records) / sum(r["seconds"] for r in records)
    speedup = speed["fast"] / speed["baseline"]
    cpu_measurements = all(r["hardware"]["device"] == "cpu" for r in benchmark["records"])
    frozen_ok = all(hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest
                    for name, digest in freeze["files"].items())
    for name, digest in freeze["source_hashes"].items():
        source = ROOT / name
        if not source.is_file():
            # Historical evidence can retain the original source after a file move.
            source = frozen / "source_snapshot" / name
        frozen_ok &= source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
    expected_external = [(i["sequence"], i["frames"]) for i in protocol["external_data"]]
    external_ok = frozen_ok
    freeze_time = datetime.fromisoformat(freeze["frozen_at_utc"]).timestamp()
    for name in ("baseline", "quality", "fast"):
        report = read(folder / "external" / name / "evaluation/metrics.json")
        external_ok &= scope(report) == expected_external and report["protocol"]["benchmark"] == "MOT20"
        for item in report["scope"]:
            summary_path = Path(item["run"]) / "summary.json"
            external_ok &= summary_path.stat().st_mtime >= freeze_time
            summary = read(summary_path)
            external_ok &= summary["model_sha256"] == read(frozen / name / "model.json")["sha256"]
            external_ok &= summary["tracker_config"] == (frozen / name / "tracker.yaml").read_text()
    ext_b, ext_q = external["metrics"]["baseline"], external["metrics"]["quality"]
    ext_f = external["metrics"]["fast"]
    fast_model = read(frozen / "fast/model.json")
    app_summary = api["actual_summary"]
    app_matches = app_summary["model_sha256"] == fast_model["sha256"]
    app_matches &= all(app_summary["inference_options"][key] == value for key, value in fast_model["inference"].items())
    app_matches &= app_summary["tracker_config"] == (frozen / "fast/tracker.yaml").read_text()
    app_matches &= app_summary["runtime_options"] == read(frozen / "fast/runtime.json")

    checks = [
        ("missed_people", fn_reduction >= targets["missed_people"]["FN_reduction_fraction"]
         and q["HOTA"] >= b["HOTA"] and diagnostic["coverage"]["FN_accounted"] == q["FN"],
         {"FN_before": b["FN"], "FN_after": q["FN"], "reduction_percent": round(fn_reduction * 100, 3),
          "HOTA_before": b["HOTA"], "HOTA_after": q["HOTA"], "accounted_FN": diagnostic["coverage"]["FN_accounted"]}),
        ("false_positives", fp_reduction >= targets["false_positives"]["FP_reduction_fraction"]
         and recall_change >= -targets["false_positives"]["recall_drop_limit_percentage_points"],
         {"FP_before": b["FP"], "FP_after": q["FP"], "reduction_percent": round(fp_reduction * 100, 3),
          "recall_change_points": round(recall_change, 3)}),
        ("identity", q["IDSW"] <= b["IDSW"] and q["IDF1"] >= b["IDF1"] and regression["passed"]
         and "test_hard_scene_cut_creates_new_ids_without_resetting_frame_numbers" in test_log,
         {"IDSW_before": b["IDSW"], "IDSW_after": q["IDSW"], "IDF1_before": b["IDF1"], "IDF1_after": q["IDF1"],
          "black_frame_regression": regression, "hard_cut_native_tracker_regression": tests_passed}),
        ("speed", cpu_measurements and speed["fast"] >= targets["speed"]["fast_min_cpu_fps"]
         and speedup >= targets["speed"]["speedup_over_v1_quality"]
         and q["HOTA"] - f["HOTA"] <= targets["speed"]["fast_HOTA_drop_limit_points"],
         {"baseline_fps": round(speed["baseline"], 3), "fast_fps": round(speed["fast"], 3),
          "speedup": round(speedup, 3), "fast_HOTA": f["HOTA"], "quality_HOTA": q["HOTA"],
          "CPU_measurements": cpu_measurements,
          "HOTA_gate_scope": "MOT17 development windows used during profile selection",
          "external_fast_HOTA_drop_points": round(ext_q["HOTA"] - ext_f["HOTA"], 3),
          "30_fps_reached": speed["fast"] >= 30, "scope": benchmark["source"]}),
        ("generalization", external_ok and ext_q["HOTA"] >= ext_b["HOTA"] - targets["generalization"]["external_HOTA_drop_limit_points"],
         {"baseline": ext_b, "quality": ext_q, "scope": protocol["external_data"],
          "frozen_checksums_and_execution_order_valid": bool(external_ok), "training_overlap_audited": False}),
        ("application", api["passed"] and all(api["checks"].values()) and browser["passed"] and app_matches,
         {"api_checks": api["checks"], "browser_passed": browser["passed"], "final_fast_model_matches": bool(app_matches),
          "mobile_width": browser["mobile"]["width"], "mobile_document_width": browser["mobile"]["documentWidth"]}),
    ]
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "passed": sum(bool(passed) for _, passed, _ in checks), "total": 6,
              "automated_tests_passed": tests_passed, "automated_test_count": test_count, "online_replay_parity": replay,
              "results": [{"criterion": name, "status": "pass" if passed else "partial", "evidence": evidence}
                          for name, passed, evidence in checks],
              "remaining_limitations": {
                  "external_FP_before_after": [ext_b["FP"], ext_q["FP"]],
                  "external_IDSW_before_after": [ext_b["IDSW"], ext_q["IDSW"]],
                  "external_fast_quality_HOTA_gap_exceeds_8_points": ext_q["HOTA"] - ext_f["HOTA"] > 8,
                  "small_people_lt_50px": {"baseline": baseline_diagnostic["height_bins"]["height_lt_50px"],
                                             "quality": diagnostic["height_bins"]["height_lt_50px"]},
                  "CPU_30_fps_reached": speed["fast"] >= 30},
              "scope_note": "Acceptance on declared windows and hardware; not zero errors, SOTA rank, full-MOT20 or universal real-time performance"}
    (folder / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "total": 6, "results": [(r["criterion"], r["status"]) for r in report["results"]]}, indent=2))


if __name__ == "__main__":
    main()
