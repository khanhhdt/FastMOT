"""Exercise the live service with a localhost-only H.264 RTSP publisher."""

import argparse
from contextlib import contextmanager
import http.client
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def child(command, log):
    with log.open("ab") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def request(port, path, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.getheader("ETag"), response.read()
    finally:
        connection.close()


def wait_for(predicate, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.2)
    raise TimeoutError("The expected camera state was not reached")


def verify(args):
    output = args.output.resolve()
    output.mkdir(parents=True)
    clip = args.fixture.resolve() if args.fixture else output / "fixture.mp4"
    if not args.fixture:
        source = ROOT / "data/MOT17/train/MOT17-04-FRCNN/img1/%06d.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-framerate", "30",
                        "-i", str(source), "-c:v", "libx264", "-threads", "4", "-preset", "veryfast",
                        "-g", "30", "-bf", "0", "-pix_fmt", "yuv420p", str(clip)], check=True, cwd=ROOT)
    config = output / "mediamtx.yml"
    config.write_text(f"""rtspAddress: 127.0.0.1:{args.rtsp_port}
rtspTransports: [tcp]
rtmp: false
hls: false
srt: false
moq: false
webrtc: {str(args.webrtc).lower()}
webrtcAddress: 127.0.0.1:8889
webrtcLocalUDPAddress: 127.0.0.1:8189
webrtcIPsFromInterfaces: false
webrtcAdditionalHosts: [127.0.0.1]
paths:
  camera:
""")
    uri = f"rtsp://127.0.0.1:{args.rtsp_port}/camera"
    credential = output / "rtsp-url"
    fd = os.open(credential, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(uri + "\n")
    publish = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
               "-i", str(clip), "-an", "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", uri]
    serve = [sys.executable, "scripts/web_tools/serve_camera.py", "--rtsp-url-file", str(credential),
             "--fps", str(args.fps), "--port", str(args.http_port), "--pilot"]

    def status():
        code, _, payload = request(args.http_port, "/api/camera/status")
        if code != 200:
            raise RuntimeError(f"Camera status returned HTTP {code}")
        return json.loads(payload)

    def live():
        state = status()
        return state if state["state"] == "live" else None

    def no_frame():
        return request(args.http_port, "/api/camera/frame.jpg")[0] == 503 and status()["state"] != "live"

    with child([str(args.mediamtx.resolve()), str(config)], output / "rtsp.log") as relay:
        def relay_ready():
            if relay.poll() is not None:
                raise RuntimeError("MediaMTX exited; see rtsp.log")
            with socket.create_connection(("127.0.0.1", args.rtsp_port), timeout=0.5):
                return True

        wait_for(relay_ready, 10)
        with child(serve, output / "service.log") as service:
            def viewer_ready():
                if service.poll() is not None:
                    raise RuntimeError("Camera service exited; see service.log")
                return no_frame()

            wait_for(viewer_ready, 30)
            with child(publish, output / "publisher.log"):
                initial = wait_for(live, 30)
                code, etag, jpeg = request(args.http_port, "/api/camera/frame.jpg")
                assert code == 200 and etag and jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
                assert request(args.http_port, "/api/camera/frame.jpg", {"If-None-Match": etag})[0] in (200, 304)
                code, _, live_jpeg = request(args.http_port, "/api/camera/live.jpg")
                assert code == 200 and live_jpeg.startswith(b"\xff\xd8")
                assert request(args.http_port, "/", {"Host": "outside.invalid"})[0] == 403
                print(f"Viewer is live at http://127.0.0.1:{args.http_port}", flush=True)
                if args.webrtc:
                    print("30 FPS source over WebRTC at http://127.0.0.1:8889/camera", flush=True)
            wait_for(no_frame, 8)
            resumed_at = time.monotonic()
            with child(publish, output / "publisher.log"):
                recovered = wait_for(live, 20)
                reconnect = round(time.monotonic() - resumed_at, 3)
                assert recovered["session"] != initial["session"]
                print(f"RTSP recovered after {reconnect:.2f} s, with fresh tracking IDs", flush=True)
                time.sleep(3)
                started = window_start = time.monotonic()
                deadline = started + args.seconds
                window_deadline = started + 60
                before = status()["processed_frames"]
                windows, ages, memory, stages = [], [], [], {}
                next_sample = started
                while True:
                    assert service.poll() is None, "Camera service exited; see service.log"
                    now = time.monotonic()
                    state = status()
                    assert state["state"] == "live", state
                    ages.append(state["frame_age_ms"])
                    stages = state["stage_mean_ms"]
                    hwm = next(line for line in Path(f"/proc/{service.pid}/status").read_text().splitlines()
                               if line.startswith("VmHWM:"))
                    memory.append(int(hwm.split()[1]))
                    if window_deadline <= deadline and now >= window_deadline:
                        rate = (state["processed_frames"] - before) / (now - window_start)
                        windows.append(round(rate, 3))
                        print(f"Minute {len(windows)}: {rate:.3f} FPS", flush=True)
                        window_start, before = now, state["processed_frames"]
                        window_deadline += 60
                    if now >= deadline:
                        break
                    next_sample += 0.25
                    time.sleep(max(0, next_sample - time.monotonic()))
                if args.seconds < 60:
                    windows.append(round((state["processed_frames"] - before) / (now - window_start), 3))
    threshold = {3: 2.9, 4: 3.8}[args.fps]
    expected_windows = max(1, args.seconds // 60)
    report = {"source": "Simulated localhost RTSP, replay of MOT17-04 H.264 at 30 FPS",
              "fps": args.fps, "seconds": args.seconds, "throughput_windows_fps": windows,
              "webrtc_source_enabled": args.webrtc,
              "display_fps_last": state["display_fps"],
              "display_pass": state["display_fps"] >= 12,
              "throughput_windows_expected": expected_windows,
              "throughput_pass": sum(fps >= threshold for fps in windows) / expected_windows >= 0.95,
              "service_frame_age_p95_ms": sorted(ages)[math.ceil(len(ages) * 0.95) - 1],
              "service_peak_rss_mib": round(max(memory) / 1024, 1), "stage_mean_ms": stages,
              "reconnect_seconds": reconnect, "reconnect_pass": reconnect <= 15,
              "memory_pass": max(memory) <= 1.5 * 1024 * 1024}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, choices=(3, 4), default=3)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--http-port", type=int, default=8003)
    parser.add_argument("--rtsp-port", type=int, default=8554)
    parser.add_argument("--webrtc", action="store_true", help="Also expose the unannotated 30 FPS source on localhost")
    parser.add_argument("--fixture", type=Path, help="Reuse an existing H.264 fixture instead of encoding MOT17-04 again")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    verify(args)


if __name__ == "__main__":
    main()
