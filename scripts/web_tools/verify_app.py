"""Exercise the local HTTP API with actual YOLO/TrackTrack inference and downloads."""

import argparse
import json
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.app.jobs import JobManager, TERMINAL
from mot.app.web import make_server
import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/demo/street/input.mp4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    manager = JobManager(args.output / "jobs")
    server = make_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    checks, progress = {}, []
    content = args.video.read_bytes()

    def request(path, body=None, headers=None, method=None):
        req = Request(base + path, data=body, headers=headers or {}, method=method)
        try:
            response = urlopen(req, timeout=30)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, dict(response.headers), response.read()

    def get(job_id):
        status, _, body = request(f"/api/jobs/{job_id}")
        assert status == 200
        return json.loads(body)

    def upload(frames=60):
        status, _, body = request("/api/jobs?" + urlencode({"filename": "api-check.mp4", "profile": "fast", "frames": frames}), content)
        assert status == 202, (status, body)
        return json.loads(body)["id"]

    def wait(job_id, expected):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            job = get(job_id)
            progress.append({key: job[key] for key in ("id", "state", "phase", "frames", "total")})
            if job["state"] in TERMINAL:
                assert job["state"] == expected, job
                return job
            time.sleep(0.1)
        raise TimeoutError(job_id)

    try:
        assert request("/")[0] == 200
        assert request("/api/profiles")[0] == 200
        checks["app_and_profiles"] = True
        assert request("/api/jobs", headers={"Host": "attacker.example"})[0] == 403
        assert request("/api/jobs", headers={"Origin": "https://attacker.example"})[0] == 403
        assert request("/../../README.md")[0] == 404
        checks["host_origin_and_path_checks"] = True
        assert request("/api/jobs?filename=bad.mp4&profile=fast", b"not a video")[0] == 400
        assert request("/api/jobs?filename=bad.py&profile=fast", b"x")[0] == 400
        assert request("/api/jobs?filename=bad.mp4&profile=unknown", b"x")[0] == 400
        checks["invalid_uploads"] = True
        job_id = upload()
        done = wait(job_id, "done")
        assert done["frames"] == done["total"] == 60
        assert any(p["id"] == job_id and 0 < p["frames"] < 60 for p in progress)
        checks["actual_inference_and_intermediate_progress"] = True
        artifacts = {}
        for name in ("preview.mp4", "tracking.mp4", "tracks.txt", "tracks.jsonl", "summary.json"):
            status, headers, body = request(f"/api/jobs/{job_id}/files/{name}?download=1")
            assert status == 200 and len(body) == int(headers["Content-Length"])
            assert "attachment" in headers["Content-Disposition"]
            artifacts[name] = body
        rows = [json.loads(line) for line in artifacts["tracks.jsonl"].splitlines()]
        assert [row["frame"] for row in rows] == list(range(1, 61))
        assert sum(len(row["tracks"]) for row in rows) == done["observations"]
        summary = json.loads(artifacts["summary.json"])
        assert summary["hardware"]["torch_threads"] == summary["runtime_options"]["torch_threads"]
        assert len(artifacts["tracks.txt"].splitlines()) == done["observations"]
        preview = manager.directory / job_id / "result/preview.mp4"
        capture, count = cv2.VideoCapture(str(preview)), 0
        while capture.read()[0]:
            count += 1
        capture.release()
        assert count == 60
        checks["complete_downloads_and_decodable_preview"] = True
        path = f"/api/jobs/{job_id}/files/preview.mp4"
        status, headers, body = request(path, headers={"Range": "bytes=0-99"})
        assert status == 206 and body == artifacts["preview.mp4"][:100]
        assert headers["Content-Range"].startswith("bytes 0-99/")
        status, headers, body = request(path, method="HEAD")
        assert status == 200 and not body and int(headers["Content-Length"]) == preview.stat().st_size
        assert request(path, headers={"Range": "bytes=999999999999-"})[0] == 416
        checks["range_head_and_unsatisfiable_range"] = True
        running, queued, third = upload(150), upload(150), upload(150)
        assert get(queued)["state"] == get(third)["state"] == "queued"
        status, _, _ = request("/api/jobs?filename=four.mp4&profile=fast", content)
        assert status == 429
        assert request(f"/api/jobs/{running}/files/tracks.txt")[0] == 404
        for key in (third, queued, running):
            assert request(f"/api/jobs/{key}/cancel", b"")[0] == 200
            wait(key, "cancelled")
        checks["single_worker_queue_capacity_and_cancellation"] = True
        interrupted = manager.create("interrupted.mp4", "fast")
        manager.close()
        recovered = JobManager(manager.directory)
        try:
            assert recovered.get(job_id)["state"] == "done"
            assert recovered.get(interrupted["id"])["state"] == "failed"
            assert not recovered.futures
        finally:
            recovered.close()
        checks["restart_preserves_results_and_marks_interrupted_job"] = True
        report = {"passed": all(checks.values()), "checks": checks, "completed_job": done,
                  "progress_samples": progress, "actual_summary": summary,
                  "scope": "Local HTTP integration with actual inference, separate validation job storage"}
        (args.output / "api_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "checks": checks, "output": str(args.output)}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager.close()


if __name__ == "__main__":
    main()
