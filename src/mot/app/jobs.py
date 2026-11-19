"""Persist local video jobs and run exactly one inference worker."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copyfile
import threading
import time
from uuid import uuid4

from ..tracking.pipeline import ROOT, run_tracking
from .demo import browser_video
from ..tracking.sources import open_source

TERMINAL = {"done", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_profiles():
    result = {}
    for name in ("quality", "fast"):
        profile = json.loads((ROOT / f"configs/profiles/{name}.json").read_text())
        model = json.loads((ROOT / profile["model_config"]).read_text())
        result[name] = {**profile, "name": name, "model": model["model"], "imgsz": model["inference"]["imgsz"]}
    return result


class JobManager:
    def __init__(self, directory, profiles=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.profiles = profiles or load_profiles()
        self.lock = threading.RLock()
        self.jobs, self.cancel_flags, self.futures = {}, {}, {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mot-inference")
        for path in self.directory.glob("*/job.json"):
            job = json.loads(path.read_text())
            self.jobs[job["id"]] = job
            if job["state"] not in TERMINAL:
                self.update(job["id"], state="failed", message="Máy chủ đã khởi động lại. Hãy gửi video để chạy một job mới.")

    def update(self, job_id, **values):
        with self.lock:
            job = self.jobs[job_id]
            job.update(values, updated_at=now())
            directory = self.directory / job_id
            temp = directory / "job.tmp"
            temp.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n")
            temp.replace(directory / "job.json")

    def create(self, filename, profile, frames=None):
        if profile not in self.profiles:
            raise ValueError("Chế độ xử lý không hợp lệ")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
            raise ValueError("Chọn tệp video MP4, MOV, MKV, AVI hoặc WebM")
        if frames is not None and not 1 <= frames <= 100000:
            raise ValueError("Giới hạn frame phải từ 1 đến 100000")
        with self.lock:
            if sum(j["state"] not in TERMINAL for j in self.jobs.values()) >= 3:
                raise RuntimeError("Hàng đợi đã đầy; chờ một video hoàn tất rồi thử lại")
            job_id = uuid4().hex
            directory = self.directory / job_id
            (directory / "configs").mkdir(parents=True)
            config = self.profiles[profile]
            for field, name in (("model_config", "model.json"), ("tracker_config", "tracker.yaml"), ("runtime_config", "runtime.json")):
                copyfile(ROOT / config[field], directory / "configs" / name)
            self.jobs[job_id] = {"id": job_id, "filename": Path(filename).name, "profile": profile,
                "state": "uploading", "phase": "upload", "frames": 0, "total": 0,
                "created_at": now(), "updated_at": now(), "message": "Đang nhận video", "frame_limit": frames,
                "input": "input" + suffix}
            self.cancel_flags[job_id] = threading.Event()
            self.update(job_id)
            return self.get(job_id)

    def get(self, job_id):
        with self.lock:
            return json.loads(json.dumps(self.jobs[job_id]))

    def list(self):
        with self.lock:
            return [self.get(key) for key in sorted(self.jobs, key=lambda k: self.jobs[k]["created_at"], reverse=True)[:20]]

    def enqueue(self, job_id):
        job = self.get(job_id)
        source = self.directory / job_id / job["input"]
        with open_source(source, job["frame_limit"]) as (info, frames):
            next(frames)  # Reject empty/corrupt uploads before starting an inference job.
        total = min(info["frames"], job["frame_limit"]) if job["frame_limit"] else info["frames"]
        self.update(job_id, state="queued", phase="queued", total=total, message="Đang chờ xử lý")
        with self.lock:
            self.futures[job_id] = self.executor.submit(self._run, job_id)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["state"] in TERMINAL:
                return self.get(job_id)
            self.cancel_flags[job_id].set()
            future = self.futures.get(job_id)
            if future and future.cancel():
                self.update(job_id, state="cancelled", message="Đã hủy job đang chờ")
            else:
                self.update(job_id, message="Đang hủy…")
            return self.get(job_id)

    def _run(self, job_id):
        directory = self.directory / job_id
        job = self.get(job_id)
        stop = self.cancel_flags[job_id]
        last_save = 0.0

        def progress(values):
            nonlocal last_save
            if stop.is_set():
                raise CancelledError()
            stamp = time.monotonic()
            if stamp - last_save >= 0.2 or values["frames"] == values["total"]:
                self.update(job_id, state="running", message="Đang theo dõi người", **values)
                last_save = stamp

        try:
            if stop.is_set():
                raise CancelledError()
            self.update(job_id, state="running", phase="loading", message="Đang nạp mô hình")
            summary = run_tracking(directory / job["input"], directory / "result", job["frame_limit"],
                                   directory / "configs/model.json", directory / "configs/tracker.yaml",
                                   directory / "configs/runtime.json", progress)
            if stop.is_set():
                raise CancelledError()
            self.update(job_id, state="running", phase="preview", message="Đang chuẩn bị video để xem")
            browser_video(directory / "result/tracking.mp4", directory / "result/preview.mp4", cancelled=stop.is_set)
            if stop.is_set():
                raise CancelledError()
            self.update(job_id, state="done", phase="done", frames=summary["frames_processed"],
                        total=summary["frames_processed"], message="Đã xử lý xong", fps=summary["fps"],
                        observations=summary["track_observations"], device=summary["hardware"]["device"])
        except CancelledError:
            self.update(job_id, state="cancelled", phase="cancelled", message="Đã hủy xử lý")
        except Exception as exc:
            self.update(job_id, state="failed", phase="failed", message=str(exc))

    def close(self):
        for flag in self.cancel_flags.values():
            flag.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
