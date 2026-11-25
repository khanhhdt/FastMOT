"""Serve live tracking from an RTSP URL stored in a local credential file."""

import argparse
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
from threading import Event
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
# Backend diagnostics can include the full RTSP URL.
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from mot.app.camera import Camera, open_rtsp
from mot.app.camera_web import camera_server
from mot.tracking.pipeline import FrameTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtsp-url-file", type=Path, required=True)
    parser.add_argument("--fps", type=int, choices=(3, 4), default=3)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--pilot", action="store_true", help="Allow a profile pending acceptance")
    args = parser.parse_args()
    profile = json.loads((ROOT / "configs/profiles/camera_live.json").read_text())
    rate = profile["rates"].get(str(args.fps))
    if rate is None:
        parser.error("No evaluated preset exists for the requested FPS")
    if not rate["quality_pass"] and not args.pilot:
        parser.error("The profile has not passed the quality gate. Use --pilot for a controlled test.")
    url = args.rtsp_url_file.read_text().strip()
    try:
        address = urlsplit(url)
        valid = address.scheme == "rtsp" and bool(address.hostname) and not any(c.isspace() for c in url)
        address.port  # Validate the port even when OpenCV would accept it.
    except ValueError:
        valid = False
    if not valid:
        parser.error("The credential file must contain one RTSP URL")

    os.chdir(ROOT)  # The locked TrackTrack YAML references the local ReID weights.
    reid = ROOT / profile["reid_weights"]
    if hashlib.sha256(reid.read_bytes()).hexdigest() != profile["reid_sha256"]:
        raise ValueError("ReID model checksum differs from the camera profile")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = json.loads((ROOT / profile["runtime_config"]).read_text())
    tracker = FrameTracker(ROOT / profile["model_config"], ROOT / rate["tracker_config"], runtime, cpu_only=True)
    camera = Camera(partial(open_rtsp, url), tracker, args.fps)
    stop = Event()

    def request_stop(*_):
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    with camera_server(camera, args.port) as server:
        server.timeout = 0.5
        camera.start()
        try:
            logging.info("Camera viewer listening on http://127.0.0.1:%s", server.server_port)
            while not stop.is_set() and not camera.failed.is_set():
                server.handle_request()
        finally:
            camera.close()
    return int(camera.failed.is_set())


if __name__ == "__main__":
    raise SystemExit(main())
