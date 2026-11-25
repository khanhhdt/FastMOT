"""Read-only camera API behind the loopback reverse proxy."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from struct import pack

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/camera.js": ("camera.js", "text/javascript; charset=utf-8"),
    "/camera.css": ("camera.css", "text/css; charset=utf-8"),
}


def camera_server(camera, port):
    root = Path(__file__).with_name("camera_ui")
    assets = {path: ((root / name).read_bytes(), kind) for path, (name, kind) in ASSETS.items()}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def send_data(self, status, data=b"", kind="text/plain", *, etag=None):
            self.send_response(status)
            if status != 304:
                self.send_header("Content-Type", kind)
                if data is not None:
                    self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'")
            if etag:
                self.send_header("ETag", etag)
            self.end_headers()
            if self.command == "GET" and data:
                self.wfile.write(data)

        def do_GET(self):
            try:
                self.read_camera()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        do_HEAD = do_GET

        def stream_camera(self):
            preview = camera.latest_display()
            if preview is None:
                return self.send_data(503)
            self.send_data(200, None, "application/octet-stream")
            if self.command == "HEAD":
                return
            while preview is not None and not camera.stopping.is_set():
                # Each JPEG is prefixed by its byte length as a big-endian uint32.
                self.wfile.write(pack("!I", len(preview.jpeg)) + preview.jpeg)
                preview = camera.wait_display(preview)

        def read_camera(self):
            port = self.server.server_port
            if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                return self.send_data(403)
            if self.path in assets:
                image, kind = assets[self.path]
                return self.send_data(200, image, kind)
            if self.path == "/api/camera/status":
                return self.send_data(200, json.dumps(camera.status()).encode(), "application/json")
            if self.path == "/api/camera/stream":
                return self.stream_camera()
            if self.path not in ("/api/camera/frame.jpg", "/api/camera/live.jpg"):
                return self.send_data(404)
            preview = camera.latest_display() if self.path == "/api/camera/live.jpg" else camera.latest()
            if preview is None:
                return self.send_data(503)
            etag = f'"{preview.session}-{preview.sequence}"'
            if self.headers.get("If-None-Match") == etag:
                return self.send_data(304, etag=etag)
            return self.send_data(200, preview.jpeg, "image/jpeg", etag=etag)

        def log_message(self, *args):
            pass  # The dashboard polls several times per second.

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
