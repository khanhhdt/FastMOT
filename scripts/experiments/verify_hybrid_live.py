"""Measure WebRTC display and asynchronous CPU tracking on the same RTSP replay."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from mot.app.camera import open_rtsp
from mot.app.hybrid_live import HybridLive
from mot.tracking.pipeline import FrameTracker
from scripts.web_tools.verify_camera import child, wait_for


def percentile(values, fraction):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))], 2)


def verify(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = output / "mediamtx.yml"
    config.write_text("""rtspAddress: 127.0.0.1:8554
rtspTransports: [tcp]
rtmp: false
hls: false
srt: false
moq: false
webrtc: true
webrtcAddress: 127.0.0.1:8889
webrtcLocalUDPAddress: 127.0.0.1:8189
webrtcIPsFromInterfaces: false
webrtcAdditionalHosts: [127.0.0.1]
paths:
  camera:
""")
    uri = "rtsp://127.0.0.1:8554/camera"
    publisher = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re",
                 "-stream_loop", "-1", "-i", str(args.fixture.resolve()), "-an", "-c", "copy",
                 "-f", "rtsp", "-rtsp_transport", "tcp", uri]
    tracker = FrameTracker(ROOT / "configs/models/yolo26s_1280_openvino_fp32.json",
                           ROOT / "configs/trackers/tracktrack_camera_4fps.yaml",
                           {"robust_gmc": True, "torch_threads": 4, "capture_detections": False}, cpu_only=True)
    raw_video = output / "tracking_raw.mp4"
    video = output / "tracking.mp4"
    live = HybridLive(lambda: open_rtsp(uri), tracker, interval=args.interval, record_path=raw_video)
    chrome_port = 9222
    chrome = ["google-chrome", "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
              "--autoplay-policy=no-user-gesture-required", "--remote-allow-origins=*",
              f"--remote-debugging-port={chrome_port}",
              f"--user-data-dir={output / 'chrome-profile'}", "http://127.0.0.1:8889/camera"]
    with child([str(args.mediamtx.resolve()), str(config)], output / "mediamtx.log") as relay:
        wait_for(lambda: socket.create_connection(("127.0.0.1", 8554), timeout=.5), 10)
        if relay.poll() is not None:
            raise RuntimeError("MediaMTX exited; see mediamtx.log")
        with child(publisher, output / "publisher.log"):
            time.sleep(1)
            live.start()
            try:
                def ready():
                    if live.error is not None:
                        raise live.error
                    return live.snapshot()["applied_corrections"] >= 3

                wait_for(ready, 30)
                with child(chrome, output / "chrome.log") as browser:
                    def browser_ready():
                        if browser.poll() is not None:
                            raise RuntimeError("Chrome exited; see chrome.log")
                        with socket.create_connection(("127.0.0.1", chrome_port), timeout=.5):
                            return True

                    wait_for(browser_ready, 10)
                    measured = subprocess.run(["node", "scripts/experiments/measure_webrtc.mjs",
                                               str(chrome_port), str(args.seconds)], cwd=ROOT,
                                              capture_output=True, text=True, timeout=args.seconds + 40,
                                              check=True)
                    browser_stats = json.loads(measured.stdout)
                    if live.error is not None:
                        raise live.error
            finally:
                live.close()
                tracking = live.snapshot()
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(raw_video),
                    "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
    raw_video.unlink()
    numbers = tracking["detector_source_numbers"]
    report = {"source": str(args.fixture.resolve()), "interval": args.interval,
              "model_device": "CPU", "webrtc": browser_stats,
              "tracking_video": str(video),
              "tracking": {key: value for key, value in tracking.items()
                           if key not in ("latencies_ms", "flow_gaps_ms", "detector_lag_ms",
                                          "detector_source_numbers")},
              "tracking_latency_p50_ms": percentile(tracking["latencies_ms"], .5),
              "tracking_latency_p95_ms": percentile(tracking["latencies_ms"], .95),
              "tracking_gap_p95_ms": percentile(tracking["flow_gaps_ms"], .95),
              "tracking_gap_max_ms": round(max(tracking["flow_gaps_ms"]), 2),
              "detector_lag_p95_ms": percentile(tracking["detector_lag_ms"], .95),
              "detector_source_gap_p95_frames": percentile(
                  [right - left for left, right in zip(numbers, numbers[1:])], .95),
              "display_25fps": browser_stats["presented_fps"] >= 25,
              "tracking_25fps": tracking["tracking_fps_10s"] >= 25,
              "display_cadence_25fps": browser_stats["display_gap_p95_ms"] <= 40,
              "tracking_cadence_25fps": percentile(tracking["flow_gaps_ms"], .95) <= 40,
              "note": "Chrome measures raw 1080p WebRTC; tracking.mp4 contains per-frame flow boxes and TrackTrack IDs at 960 px width; H.264 conversion happens after the FPS measurement"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", type=Path, default=ROOT / ".runtime/camera-test/mediamtx")
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "outputs/camera_smooth_4fps_resized_20261123/fixture.mp4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--interval", type=int, default=6)
    args = parser.parse_args()
    if args.seconds < 5 or args.interval < 1:
        parser.error("--seconds must be at least 5 and --interval must be positive")
    verify(args)
