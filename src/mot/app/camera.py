"""Bounded live capture and tracking for one camera."""

from collections import deque
from dataclasses import dataclass
import logging
from threading import Condition, Event, Lock, Thread
import time
from uuid import uuid4

import cv2
import numpy as np

from mot.tracking.hybrid import SparseFlow
from mot.tracking.visualization import TrackVisualizer

LOG = logging.getLogger(__name__)
STALE_SECONDS = 2
DISPLAY_FPS = 30
DISPLAY_WIDTH = 1280
FLOW_MAX_AGE = 0.8


@dataclass(frozen=True)
class InputFrame:
    image: np.ndarray
    received_at: float
    read_ms: float
    generation: int


@dataclass(frozen=True)
class Preview:
    jpeg: bytes
    received_at: float
    sequence: int
    session: str


@dataclass(frozen=True)
class FlowCorrection:
    flow: SparseFlow
    detected_at: float
    projected_at: float


def open_rtsp(url):
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
    ])


def _resize_for_display(image):
    height, width = image.shape[:2]
    if width <= DISPLAY_WIDTH:
        return image
    return cv2.resize(image, (DISPLAY_WIDTH, round(height * DISPLAY_WIDTH / width)),
                      interpolation=cv2.INTER_AREA)


class FlowPreview:
    """Project a detector result to the JPEG frame using recent display frames."""

    def __init__(self):
        self.segment = None
        self.history = deque(maxlen=round(FLOW_MAX_AGE * DISPLAY_FPS) + 1)
        self.history_lock = Lock()
        self.flow = SparseFlow()
        self.visualizer = TrackVisualizer(trail_length=1)
        self.corrected_at = None

    def prepare(self, frame, tracks, segment):
        # Replay most of the delayed correction in the detector thread.
        with self.history_lock:
            history = tuple(self.history) if self.segment == segment else ()
        key_image = _resize_for_display(frame.image)
        height, width = frame.image.shape[:2]
        scale_x, scale_y = key_image.shape[1] / width, key_image.shape[0] / height
        scaled = [{**track, "xyxy": [value * scale for value, scale in
                                    zip(track["xyxy"], (scale_x, scale_y, scale_x, scale_y))]}
                  for track in tracks]
        flow = SparseFlow()
        flow.correct(key_image, scaled)
        projected_at = frame.received_at
        for stamp, previous in history:
            if stamp > projected_at:
                flow.advance(previous)
                projected_at = stamp
        return FlowCorrection(flow, frame.received_at, projected_at)

    def draw(self, image, now, sequence, segment, correction):
        with self.history_lock:
            if self.segment != segment:
                self.segment = segment
                self.history.clear()
                self.flow = SparseFlow()
                self.visualizer.history.clear()
                self.corrected_at = None
            self.history.append((now, image))
        if correction is not None and now - correction.detected_at <= FLOW_MAX_AGE:
            self.flow = correction.flow
            self.corrected_at = correction.detected_at
            tracks = self.flow.tracks
            for stamp, previous in self.history:
                if stamp > correction.projected_at:
                    tracks = self.flow.advance(previous)
        elif self.corrected_at is not None and now - self.corrected_at <= FLOW_MAX_AGE:
            tracks = self.flow.advance(image)
        else:
            return image, False
        return self.visualizer.draw(image, sequence, tracks), True


class Camera:
    def __init__(self, connect, tracker, fps):
        self.connect = connect
        self.tracker = tracker
        self.fps = fps
        self.condition = Condition()
        self.stopping = Event()
        self.failed = Event()
        self.pending = self.preview = self.display = None
        self.overlay = None
        self.flow_preview = FlowPreview()
        self.correction = None
        self.last_receive = None
        self.shape = None
        self.generation = self.segment = 0
        self.boot = uuid4().hex
        self.processed = self.replaced = self.reconnects = self.displayed = 0
        self.state = "connecting"
        self.timings = deque(maxlen=256)
        self.display_times = deque(maxlen=256)
        self.flow_times = deque(maxlen=256)
        self.next_display = 0.0
        self.threads = [
            Thread(target=self._read, name="camera-capture", daemon=True),
            Thread(target=self._track, name="camera-tracking", daemon=True),
        ]

    def start(self):
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=6)

    def _invalidate(self, state):
        self.state = state
        self.pending = self.preview = self.display = None
        self.overlay = None
        self.correction = None
        self.timings.clear()
        self.display_times.clear()
        self.flow_times.clear()
        self.generation += 1
        self.segment += 1
        self.condition.notify_all()

    def _receive(self, image, now, read_ms):
        interval = 1 / DISPLAY_FPS
        # Capture at the source cadence even when a frame arrives slightly ahead of its slot.
        with self.condition:
            if self.last_receive is not None and (
                now - self.last_receive > STALE_SECONDS or image.shape != self.shape
            ):
                self._invalidate("connecting")
            if self.pending is not None:
                self.replaced += 1
            self.last_receive, self.shape = now, image.shape
            self.pending = InputFrame(image, now, read_ms, self.generation)
            self.condition.notify_all()
            if now + interval / 2 < self.next_display:
                return
            segment, correction = self.segment, self.correction
            self.correction = None

        image = _resize_for_display(image)
        image, flowed = self.flow_preview.draw(image, now, self.displayed + 1, segment, correction)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        self.next_display = (self.next_display or now) + interval
        if self.next_display < now - interval:
            self.next_display = now + interval
        with self.condition:
            if segment == self.segment:
                self.displayed += 1
                self.display = Preview(encoded.tobytes(), now, self.displayed, f"{self.boot}-{segment}")
                self.display_times.append(now)
                if flowed:
                    self.flow_times.append(now)
                self.condition.notify_all()

    def _read(self):
        delay = 1
        while not self.stopping.is_set():
            capture = None
            try:
                capture = self.connect()
                if capture.isOpened():
                    while not self.stopping.is_set():
                        started = time.monotonic()
                        ok, image = capture.read()
                        now = time.monotonic()
                        if not ok or image is None or not image.size:
                            break
                        delay = 1
                        self._receive(image, now, 1000 * (now - started))
            except (OSError, cv2.error):
                # Some backends include the URL in exceptions.
                pass
            finally:
                if capture is not None:
                    capture.release()
            if self.stopping.is_set():
                return
            with self.condition:
                self._invalidate("reconnecting")
                self.last_receive = self.shape = None
                self.reconnects += 1
            LOG.warning("RTSP unavailable, retrying in %s s", delay)
            self.stopping.wait(delay)
            delay = min(8, 2 * delay)

    def _next(self, after):
        with self.condition:
            while not self.stopping.is_set():
                now = time.monotonic()
                if self.pending is not None and now >= after:
                    frame, self.pending = self.pending, None
                    return frame
                self.condition.wait(timeout=after - now if self.pending is not None else None)
        return None

    def _cut(self, generation):
        with self.condition:
            if generation == self.generation:
                self.preview = self.display = None
                self.overlay = None
                self.correction = None
                self.flow_times.clear()
                self.state = "connecting"
                self.segment += 1
                self.timings.clear()
                self.condition.notify_all()

    def _track(self):
        generation, deadline = None, 0.0
        try:
            while (frame := self._next(deadline)) is not None:
                started = time.monotonic()
                if started - frame.received_at > STALE_SECONDS:
                    continue
                if generation is None:
                    height, width = frame.image.shape[:2]
                    self.tracker.warmup(width, height)
                elif generation != frame.generation:
                    self.tracker.reset()
                generation = frame.generation
                infer_start = time.perf_counter()
                tracks = self.tracker.step(frame.image, on_scene_cut=lambda: self._cut(generation))
                jpeg_start = time.perf_counter()
                annotated = self.tracker.visualizer.draw(frame.image, self.tracker.index, tracks)
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("Could not encode the camera preview")
                completed = time.monotonic()
                jpeg = encoded.tobytes()
                flow_segment = None
                with self.condition:
                    if generation == self.generation and completed - frame.received_at <= STALE_SECONDS:
                        first = self.state != "live"
                        self.processed += 1
                        session = f"{self.boot}-{self.segment}"
                        self.preview = Preview(jpeg, frame.received_at, self.processed, session)
                        self.overlay = (session, tracks, frame.received_at, frame.image.shape[1], frame.image.shape[0])
                        flow_segment = self.segment
                        self.state = "live"
                        self.timings.append((completed, frame.read_ms,
                                             1000 * (self.tracker.captured_at - infer_start),
                                             1000 * (self.tracker.tracked_at - self.tracker.captured_at),
                                             1000 * (time.perf_counter() - jpeg_start)))
                        if first:
                            LOG.info("Camera preview available at target %s FPS", self.fps)
                if flow_segment is not None:
                    correction = self.flow_preview.prepare(frame, tracks, flow_segment)
                    with self.condition:
                        if flow_segment == self.segment:
                            self.correction = correction
                deadline = started + 1 / self.fps
        except Exception:
            LOG.exception("Camera inference exited")
            with self.condition:
                self._invalidate("failed")
            self.failed.set()

    def latest(self):
        with self.condition:
            preview = self.preview
            if preview is not None and time.monotonic() - preview.received_at <= STALE_SECONDS:
                return preview
            return None

    def latest_display(self):
        with self.condition:
            display = self.display
            if (self.state == "live" and display is not None
                    and time.monotonic() - display.received_at <= STALE_SECONDS):
                return display
            return None

    def wait_display(self, previous):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stopping.is_set() or self.display is not previous, timeout=STALE_SECONDS)
            latest = self.latest_display()
            if latest is not None and latest is not previous and latest.session == previous.session:
                return latest
            return None

    def status(self):
        with self.condition:
            now = time.monotonic()
            preview, state = self.preview, self.state
            age = now - preview.received_at if preview else None
            if age is not None and age > STALE_SECONDS:
                state = "stale"
            recent = [sample for sample in self.timings if now - sample[0] <= 10]
            fps = (len(recent) - 1) / (recent[-1][0] - recent[0][0]) if len(recent) > 1 else 0
            display_times = [stamp for stamp in self.display_times if now - stamp <= 10]
            display_fps = ((len(display_times) - 1) / (display_times[-1] - display_times[0])
                           if len(display_times) > 1 else 0)
            flow_times = [stamp for stamp in self.flow_times if now - stamp <= 10]
            flow_fps = ((len(flow_times) - 1) / (flow_times[-1] - flow_times[0])
                        if len(flow_times) > 1 else 0)
            overlay = self.overlay if state == "live" else None
            average = {key: round(sum(row[index] for row in recent) / len(recent), 1)
                       for index, key in enumerate(("read", "detector", "tracking", "preview"), 1)} if recent else {}
            return {"state": state, "target_fps": self.fps, "effective_fps": round(fps, 2),
                    "display_target_fps": DISPLAY_FPS, "display_fps": round(display_fps, 2),
                    "flow_fps": round(flow_fps, 2),
                    "overlay": {"session": overlay[0], "tracks": overlay[1],
                                "age_ms": round((now - overlay[2]) * 1000),
                                "width": overlay[3], "height": overlay[4]} if overlay else None,
                    "frame_age_ms": round(age * 1000) if age is not None else None,
                    "session": preview.session if state == "live" else None,
                    "processed_frames": self.processed, "replaced_frames": self.replaced,
                    "reconnect_attempts": self.reconnects, "stage_mean_ms": average}
