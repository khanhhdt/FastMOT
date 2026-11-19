"""Loopback-only HTTP API for local video upload and tracking jobs."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit

from .jobs import JobManager

MAX_UPLOAD = 1024 * 1024 * 1024
ASSETS = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
FILES = {"preview.mp4", "tracking.mp4", "tracks.txt", "tracks.jsonl", "summary.json"}
JOB = re.compile(r"/api/jobs/([0-9a-f]{32})(?:/(cancel|files/[^/]+))?$")


def make_server(host, port, manager):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("This app is local-only; bind to 127.0.0.1 or localhost")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def discard_body(self, remaining):
            # Finish bounded, valid-size requests before an early 400/429 response.
            # Otherwise clients still sending a video can see BrokenPipe instead.
            while remaining:
                block = self.rfile.read(min(1024 * 1024, remaining))
                if not block:
                    break
                remaining -= len(block)

        def json(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def permitted(self):
            actual_port = self.server.server_port
            hosts = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}"}
            origins = {"http://" + value for value in hosts}
            if self.headers.get("Host") not in hosts or self.headers.get("Origin") not in origins | {None}:
                self.json(403, {"error": "Ứng dụng chỉ nhận yêu cầu từ trang localhost của chính nó"})
                return False
            return True

        def file(self, path, download=False):
            if not path.is_file():
                self.json(404, {"error": "Không tìm thấy tệp"})
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not size:
                    self.send_error(416)
                    return
                a, b = match.groups()
                if a:
                    start, end = int(a), min(int(b), end) if b else end
                elif b:
                    start = max(0, size - int(b))
                else:
                    self.send_error(416)
                    return
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            length = max(0, end - start + 1)
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with path.open("rb") as handle:
                    handle.seek(start)
                    while length:
                        block = handle.read(min(65536, length))
                        if not block:
                            break
                        self.wfile.write(block)
                        length -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A browser can stop a range request after seeking.

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self.permitted():
                return
            url = urlsplit(self.path)
            path = unquote(url.path)
            if path in ASSETS:
                return self.file(Path(__file__).with_name("webui") / ASSETS[path])
            if path == "/api/profiles":
                profiles = [{k: p[k] for k in ("name", "label", "model", "imgsz")} for p in manager.profiles.values()]
                return self.json(200, {"profiles": profiles, "max_upload_bytes": MAX_UPLOAD})
            if path == "/api/jobs":
                return self.json(200, {"jobs": manager.list()})
            match = JOB.fullmatch(path)
            if not match:
                return self.json(404, {"error": "Không tìm thấy đường dẫn"})
            job_id, action = match.groups()
            try:
                job = manager.get(job_id)
            except KeyError:
                return self.json(404, {"error": "Không tìm thấy job"})
            if action is None:
                return self.json(200, job)
            if action.startswith("files/") and job["state"] == "done":
                name = action.removeprefix("files/")
                if name in FILES:
                    return self.file(manager.directory / job_id / "result" / name, "download" in parse_qs(url.query))
            return self.json(404, {"error": "Tệp chưa sẵn sàng hoặc không được phép truy cập"})

        def do_POST(self):
            if not self.permitted():
                return
            url = urlsplit(self.path)
            match = JOB.fullmatch(unquote(url.path))
            if match and match.group(2) == "cancel":
                try:
                    return self.json(200, manager.cancel(match.group(1)))
                except KeyError:
                    return self.json(404, {"error": "Không tìm thấy job"})
            if url.path != "/api/jobs":
                return self.json(404, {"error": "Không tìm thấy đường dẫn"})
            job_id = None
            remaining = 0
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= MAX_UPLOAD:
                    return self.json(413, {"error": "Video phải lớn hơn 0 byte và không quá 1 GiB"})
                remaining = length
                query = parse_qs(url.query)
                frames = int(query["frames"][0]) if query.get("frames") else None
                job = manager.create(query.get("filename", [""])[0], query.get("profile", ["quality"])[0], frames)
                job_id = job["id"]
                destination = manager.directory / job_id / job["input"]
                remaining = length
                with destination.open("wb") as target:
                    while remaining:
                        block = self.rfile.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError("Tải video chưa hoàn tất")
                        target.write(block)
                        remaining -= len(block)
                manager.enqueue(job_id)
                return self.json(202, manager.get(job_id))
            except (ValueError, FileNotFoundError, StopIteration) as exc:
                self.discard_body(remaining)
                if job_id:
                    manager.update(job_id, state="failed", message=str(exc) or "Video không có frame đọc được")
                return self.json(400, {"error": str(exc) or "Video không đọc được"})
            except RuntimeError as exc:
                self.discard_body(remaining)
                if job_id:
                    manager.update(job_id, state="failed", message=str(exc))
                return self.json(429 if job_id is None else 500, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                if job_id:
                    manager.update(job_id, state="failed", message="Kết nối tải video đã bị đóng")

    return ThreadingHTTPServer((host, port), Handler)
