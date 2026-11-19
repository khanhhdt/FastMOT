"""Build two local, browser-playable demonstrations using the tracking pipeline."""

from html import escape
import json
from pathlib import Path
from shutil import copyfile, which
import subprocess
from concurrent.futures import CancelledError

import cv2

from ..tracking.pipeline import ROOT, run_tracking
from ..tracking.sources import open_source

CASES = (
    ("street", "Cảnh đường phố", "MOT17-09-FRCNN", "Người đi gần camera, cắt nhau và ra khỏi khung hình."),
    ("crowd", "Cảnh đông người", "MOT17-04-FRCNN", "Góc nhìn từ trên cao, nhiều người nhỏ và bị che khuất."),
)


def browser_video(source, destination, cancelled=None):
    """Keep the original tracking export and make an H.264 preview for browsers."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                             "-i", str(source), "-an", "-c:v", "libx264", "-threads", "2",
                             "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(destination)]
    if cancelled is not None:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        while True:
            if cancelled():
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                raise CancelledError()
            try:
                _, error = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(f"FFmpeg could not create browser video: {error.strip()}")
        return
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFmpeg could not create browser video: {result.stderr.strip()}")


def make_input(source, directory, max_frames):
    raw = directory / "input_raw.mp4"
    with open_source(source, max_frames) as (info, frames):
        width, height = info["width"], info["height"]
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                                 (width + width % 2, height + height % 2))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Cannot create demo input video")
        try:
            for index, image in frames:
                padded = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                writer.write(padded)
                if index == 1 and not cv2.imwrite(str(directory / "input.jpg"), padded):
                    raise RuntimeError("Cannot save input poster")
        finally:
            writer.release()
    browser_video(raw, directory / "input.mp4")
    raw.unlink()


def build_demo(output, max_frames=150, model_config=None, tracker_config=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty demo directory")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if which("ffmpeg") is None:
        raise RuntimeError("Install FFmpeg with libx264 support before preparing the demo")
    config_dir = output / "configs"
    config_dir.mkdir(parents=True)
    model_path, tracker_path = config_dir / "model.json", config_dir / "tracktrack.yaml"
    copyfile(model_config or ROOT / "configs/models/yolo26s.json", model_path)
    copyfile(tracker_config or ROOT / "configs/trackers/tracktrack.yaml", tracker_path)
    dataset = json.loads((ROOT / "configs/datasets/mot17.json").read_text())
    cases, sections = [], []
    for number, (name, title, sequence, description) in enumerate(CASES, 1):
        directory = output / name
        directory.mkdir()
        make_input(ROOT / dataset["dataset_root"] / sequence, directory, max_frames)
        # This exercises the same video-file input path a user runs from the CLI.
        summary = run_tracking(directory / "input.mp4", directory / "result", None, model_path, tracker_path)
        browser_video(directory / "result/tracking.mp4", directory / "result/preview.mp4")
        frames, fps = summary["frames_processed"], summary["source"]["fps"]
        cases.append({"name": name, "title": title, "original_sequence": sequence,
                      "original_frame_window": [1, frames], "frames": frames, "source_fps": fps,
                      "processing_fps": summary["fps"], "input": f"{name}/input.mp4",
                      "preview": f"{name}/result/preview.mp4", "summary": f"{name}/result/summary.json"})
        sections.append(f'''<section id="{name}">
          <div class="section-heading"><span class="number">0{number}</span><div>
            <h2>{escape(title)}</h2><p>{escape(description)} <span class="sequence">{sequence}</span></p>
          </div></div>
          <p class="stats">{frames} frame <span> / </span> {frames / fps:.1f} giây
            <span> / </span> {summary['fps']:.2f} FPS xử lý</p>
          <div class="videos">
            <figure><figcaption>VIDEO ĐẦU VÀO</figcaption>
              <video controls muted playsinline preload="metadata" poster="{name}/input.jpg" src="{name}/input.mp4">
              <a href="{name}/input.mp4">Tải video đầu vào</a></video></figure>
            <figure><figcaption>KẾT QUẢ · KHUNG BAO, ID VÀ QUỸ ĐẠO</figcaption>
              <video controls muted playsinline preload="metadata" poster="{name}/result/frame_000001.jpg"
                src="{name}/result/preview.mp4"><a href="{name}/result/preview.mp4">Tải kết quả</a></video></figure>
          </div>
          <div class="downloads"><a href="{name}/result/preview.mp4" download>Tải video kết quả</a>
            <a href="{name}/result/tracks.txt" download>Tracking MOT</a>
            <a href="{name}/result/tracks.jsonl" download>Tracking JSONL</a>
            <a href="{name}/result/summary.json">Thống kê lần chạy</a></div>
        </section>''')
    manifest = {"pipeline": "YOLO26s Detect + TrackTrack", "cases": cases,
                "configs": {"model": "configs/model.json", "tracker": "configs/tracktrack.yaml"},
                "timing_note": "Processing FPS excludes input preparation and browser-video conversion",
                "evaluation_note": "Video demos are not GT-scored benchmark results"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    template = Path(__file__).with_name("demo.html").read_text()
    (output / "index.html").write_text(template.replace("<!--CASES-->", "\n".join(sections)))
    print(f"Demo ready: {output / 'index.html'}", flush=True)
    return manifest
