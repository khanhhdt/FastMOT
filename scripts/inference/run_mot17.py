"""Produce full train/test MOT17 or MOT20 tracking outputs."""

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
from shutil import copyfile, copytree
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.tracking.pipeline import run_tracking
from mot.app.demo import browser_video
from mot.evaluation.metrics import evaluate_runs

import cv2
import torch
import ultralytics


def discover_sequences(dataset, benchmark="MOT17"):
    sequences = []
    for split in ("train", "test"):
        pattern = "MOT*-FRCNN" if benchmark == "MOT17" else "MOT20-*"
        for source in sorted((Path(dataset) / split).glob(pattern)):
            signatures, latest_change = {}, 0
            metadata = None
            variants = ([source.with_name(source.name.replace("FRCNN", suffix))
                         for suffix in ("DPM", "FRCNN", "SDP")] if benchmark == "MOT17" else [source])
            for variant in variants:
                config = configparser.ConfigParser()
                if not config.read(variant / "seqinfo.ini"):
                    raise FileNotFoundError(variant / "seqinfo.ini")
                info = config["Sequence"]
                values = {k: info[k] for k in ("seqLength", "frameRate", "imWidth", "imHeight", "imExt")}
                if metadata is not None and values != metadata:
                    raise ValueError(f"Variant metadata differs: {variant}")
                metadata = values
                image_dir = variant / info["imDir"]
                images = sorted(image_dir.glob("*" + info["imExt"]))
                expected = [f"{i:06d}{info['imExt']}" for i in range(1, int(info['seqLength']) + 1)]
                if [p.name for p in images] != expected:
                    raise ValueError(f"Missing or unexpected frames: {variant}")
                digest = hashlib.sha256()
                for image in images:
                    digest.update(image.name.encode())
                    digest.update(hashlib.sha256(image.read_bytes()).digest())
                    if variant == source:
                        latest_change = max(latest_change, image.stat().st_ctime_ns)
                signatures[variant.name] = digest.hexdigest()
            if len(set(signatures.values())) != 1:
                raise ValueError(f"Images differ across variants; cannot deduplicate: {source}")
            sequences.append({"split": split, "name": source.name, "source": str(source.resolve()),
                              "frames": int(metadata["seqLength"]), "variants": signatures,
                              "latest_image_change_ns": latest_change, "status": "pending"})
            print(f"INPUT OK {split}/{source.name}: {metadata['seqLength']} frames ({len(variants)} matching input names)", flush=True)
    if not sequences or not all(any(s["split"] == split for s in sequences) for split in ("train", "test")):
        raise ValueError(f"Both {benchmark} train and test sequences are required")
    return sequences


def matching_run(path, item, model, tracker_text, tracking_policy=None):
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    return (summary.get("mode") == "track" and summary["source"]["kind"] == "mot_sequence"
            and Path(summary["source"]["path"]).resolve() == Path(item["source"])
            and summary["frames_processed"] == summary["tracker_frames_processed"] == item["frames"]
            and summary["model_sha256"] == model["sha256"] and summary["tracker_config"] == tracker_text
            and summary.get("tracking_policy", {"robust_gmc": False}) == (tracking_policy or {"robust_gmc": False})
            and summary["ultralytics"] == ultralytics.__version__ and summary["torch"] == torch.__version__
            and all(summary["inference_options"][k] == v for k, v in model["inference"].items())
            and item["latest_image_change_ns"] <= summary_path.stat().st_mtime_ns)


def verify_run(path, count):
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "tracks.jsonl").read_text().splitlines()]
    if [r["frame"] for r in rows] != list(range(1, count + 1)):
        raise ValueError(f"Incomplete JSONL: {path}")
    observations = sum(len(r["tracks"]) for r in rows)
    if observations != summary["track_observations"] or len((path / "tracks.txt").read_text().splitlines()) != observations:
        raise ValueError(f"Tracking export count mismatch: {path}")
    cap = cv2.VideoCapture(str(path / "tracking.mp4"))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != count:
        raise ValueError(f"Incomplete video {path}: {decoded}/{count}")
    return summary


def write_index(output, manifest):
    benchmark = manifest.get("benchmark", "MOT17")
    model = json.loads((output / "configs/model.json").read_text())
    model_label = escape(model["model"])
    profile = escape(manifest.get("profile", "V1"))
    sections = []
    for split in ("train", "test"):
        cards = []
        for item in manifest["sequences"]:
            if item["split"] != split or item["status"] != "complete":
                continue
            name = escape(item["name"])
            folder = f"{split}/{name}"
            cards.append(f'''<article><h3>{name}</h3><p>{item['frames']} frame · {item['fps']:.2f} FPS xử lý</p>
              <video controls muted playsinline preload="none" poster="{folder}/frame_000001.jpg" src="{folder}/preview.mp4"></video>
              <p class="links"><a href="{folder}/preview.mp4" download>Video</a>
              <a href="{folder}/tracks.txt" download>MOT TXT</a><a href="{folder}/tracks.jsonl" download>JSONL</a>
              <a href="{folder}/summary.json">Thống kê</a></p></article>''')
        if cards:
            sections.append(f'<section id="{split}"><h2>{split.upper()}</h2><div class="grid">{"".join(cards)}</div></section>')
    complete = sum(i["status"] == "complete" for i in manifest["sequences"])
    metrics = '<p><a href="evaluation_train/metrics.json">Metric toàn bộ train</a></p>' if (output / "evaluation_train/metrics.json").exists() else ''
    exports = '<p><a href="motchallenge/train.zip">Tải MOT TXT train</a> · <a href="motchallenge/test.zip">Tải MOT TXT test</a></p>' if (output / "motchallenge/test.zip").exists() else ''
    dedup = ("Các ảnh DPM/FRCNN/SDP đã được đối chiếu SHA-256. Mỗi video xử lý một lần, sau đó xuất TXT theo cả ba tên."
             if benchmark == "MOT17" else "Mỗi sequence có một video theo dõi và một tệp MOT TXT.")
    page = output / "index.tmp"
    page.write_text(f'''<!doctype html><html lang="vi"><meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1"><title>Toàn bộ {benchmark} · YOLO26 + TrackTrack</title>
      <style>body{{font:16px system-ui;background:#f6f7f3;color:#203127;margin:0}}main{{max-width:1280px;margin:auto;padding:32px}}
      h1{{font-size:36px;letter-spacing:-.03em}}h2{{margin-top:40px}}h3{{font-size:19px;margin:0}}p{{line-height:1.6}}
      a{{color:#286242;text-underline-offset:4px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}
      article{{border-top:1px solid #cfd7cd;padding-top:20px}}video{{width:100%;aspect-ratio:16/9;background:#111}}
      .links{{display:flex;gap:18px;flex-wrap:wrap}}aside{{margin-top:36px;padding-top:20px;border-top:1px solid #cfd7cd}}
      @media(max-width:760px){{main{{padding:20px}}.grid{{grid-template-columns:1fr}}h1{{font-size:28px}}}}</style>
      <main><p>{model_label} · {model["inference"]["imgsz"]} px · TrackTrack · {profile}</p><h1>Kết quả toàn bộ {benchmark}</h1>
      <p>Đã hoàn tất {complete}/{len(manifest['sequences'])} video khác nhau · {manifest['unique_frames']} frame nguồn.</p>
      <nav><a href="#train">Train</a> · <a href="#test">Test</a> · <a href="manifest.json">Thông tin lần chạy</a></nav>
      {metrics}{exports}{''.join(sections)}<aside><p>{dedup} ID có ý nghĩa trong từng video. FPS xử lý khác tốc độ phát.</p>
      <p>Điểm train được chấm bằng GT cục bộ. Tập test chỉ có kết quả dự đoán.</p></aside></main></html>''', encoding="utf-8")
    page.replace(output / "index.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["MOT17", "MOT20"], default="MOT17")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/mot17_full or outputs/mot20_full")
    parser.add_argument("--resume", action="store_true", help="Continue this batch, retaining completed sequences")
    parser.add_argument("--profile", choices=["quality", "fast"], default="quality", help="New batches use this profile; resume uses its saved configuration")
    args = parser.parse_args()
    benchmark = args.benchmark
    output = (args.output or ROOT / f"outputs/{benchmark.lower()}_full").resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose an empty output directory or use --resume")
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("benchmark", "MOT17") != benchmark:
            raise ValueError(f"Resume with --benchmark {manifest.get('benchmark', 'MOT17')}")
    else:
        items = discover_sequences(ROOT / "data" / benchmark, benchmark)
        (output / "configs").mkdir(parents=True)
        profile = json.loads((ROOT / f"configs/profiles/{args.profile}.json").read_text())
        copyfile(ROOT / profile["model_config"], output / "configs/model.json")
        copyfile(ROOT / profile["tracker_config"], output / "configs/tracktrack.yaml")
        copyfile(ROOT / profile["runtime_config"], output / "configs/runtime.json")
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                    "unique_frames": sum(i["frames"] for i in items), "sequences": items,
                    "benchmark": benchmark,
                    "deduplication": ("SHA-256 of every image checked across DPM/FRCNN/SDP"
                                      if benchmark == "MOT17" else None),
                    "profile": args.profile,
                    "configs": {"model": "configs/model.json", "tracker": "configs/tracktrack.yaml", "runtime": "configs/runtime.json"}}
    model_path, tracker_path = output / "configs/model.json", output / "configs/tracktrack.yaml"
    runtime_path = output / "configs/runtime.json"
    if not runtime_path.exists():
        # Resuming a V1 archive must keep its original, unguarded camera motion.
        runtime_path.write_text(json.dumps({"robust_gmc": False, "torch_threads": 8, "capture_detections": False}) + "\n")
    policy = {"robust_gmc": json.loads(runtime_path.read_text())["robust_gmc"]}
    model = json.loads(model_path.read_text())
    tracker_text = tracker_path.read_text()

    def save():
        temp = output / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(output / "manifest.json")
        write_index(output, manifest)

    save()
    for item in manifest["sequences"]:
        target = output / item["split"] / item["name"]
        if item["status"] == "complete":
            if not matching_run(target, item, model, tracker_text, policy):
                raise ValueError(f"Completed output differs from the saved configuration: {target}")
            verify_run(target, item["frames"])
            continue
        # A process can stop after the directory rename but before the manifest is saved.
        if target.exists():
            if not matching_run(target, item, model, tracker_text, policy):
                raise ValueError(f"Existing output does not match the batch configuration: {target}")
            summary = verify_run(target, item["frames"])
            if not (target / "preview.mp4").exists():
                browser_video(target / "tracking.mp4", target / "preview.mp4")
            item.update(status="complete", output=str(target.relative_to(output)), fps=summary["fps"],
                        track_observations=summary["track_observations"],
                        prediction_sha256=hashlib.sha256((target / "tracks.txt").read_bytes()).hexdigest())
            save()
            continue
        partial = target.with_name(target.name + ".incomplete")
        if partial.exists() and not matching_run(partial, item, model, tracker_text, policy):
            archive = output / "incomplete" / f"{item['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial.rename(archive)
        target.parent.mkdir(parents=True, exist_ok=True)
        candidates = [ROOT / "outputs/improvements/quality/runs" / item["name"],
                      ROOT / "outputs/improvements/external/quality/runs" / item["name"],
                      ROOT / "outputs/experiments/mot17_step5_final/held_out/runs/selected" / item["name"],
                      ROOT / "outputs/experiments/mot17_step5/runs/imgsz_960" / item["name"]]
        candidate = next((p for p in candidates if matching_run(p, item, model, tracker_text, policy)), None)
        item["status"] = "running"
        save()
        if partial.exists():
            print(f"RESUME EXPORT {item['split']}/{item['name']}", flush=True)
        elif candidate:
            verify_run(candidate, item["frames"])
            copytree(candidate, partial)
            item["reused_from"] = str(candidate)
            print(f"REUSE {item['split']}/{item['name']}: {item['frames']} verified frames", flush=True)
        else:
            print(f"RUN {item['split']}/{item['name']}: {item['frames']} frames", flush=True)
            run_tracking(item["source"], partial, None, model_path, tracker_path, runtime_path)
        summary = verify_run(partial, item["frames"])
        if not (partial / "preview.mp4").exists():
            browser_video(partial / "tracking.mp4", partial / "preview.mp4")
        partial.rename(target)
        item.update(status="complete", output=str(target.relative_to(output)), fps=summary["fps"],
                    track_observations=summary["track_observations"],
                    prediction_sha256=hashlib.sha256((target / "tracks.txt").read_bytes()).hexdigest())
        save()
        print(f"COMPLETE {item['split']}/{item['name']}", flush=True)
    for split in ("train", "test"):
        destination = output / "motchallenge" / split
        destination.mkdir(parents=True, exist_ok=True)
        for item in manifest["sequences"]:
            if item["split"] == split:
                for alias in item["variants"]:
                    copyfile(output / item["output"] / "tracks.txt", destination / f"{alias}.txt")
        with zipfile.ZipFile(destination.with_suffix('.zip'), "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(destination.glob("*.txt")):
                archive.write(path, path.name)
    evaluation = output / "evaluation_train"
    if not (evaluation / "metrics.json").exists():
        if evaluation.exists():
            archive = output / "incomplete" / f"evaluation_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            evaluation.rename(archive)
        runs = [output / item["output"] for item in manifest["sequences"] if item["split"] == "train"]
        evaluate_runs(runs, evaluation, error_clips=0, benchmark=benchmark)
    manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(f"ALL COMPLETE: {len(manifest['sequences'])} videos, {manifest['unique_frames']} frames -> {output}", flush=True)


if __name__ == "__main__":
    main()
