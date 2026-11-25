"""Experimental CPU camera pipeline with an independent sparse detector worker."""

from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
import time

import cv2

from mot.tracking.hybrid import SparseFlow
from mot.tracking.visualization import TrackVisualizer


@dataclass(frozen=True)
class Frame:
    number: int
    image: object
    received_at: float


class HybridLive:
    def __init__(self, connect, tracker, interval=6, record_path=None):
        self.connect = connect
        self.tracker = tracker
        self.interval = interval
        self.record_path = record_path
        self.flow_queue = Queue(maxsize=32)
        self.detector_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.stopping = False
        self.error = None
        self.correction = None
        self.source_frames = self.flow_frames = self.detector_frames = 0
        self.recorded_frames = 0
        self.flow_drops = self.stale_corrections = self.applied_corrections = 0
        self.detector_queue_replacements = 0
        self.capture_times = deque(maxlen=300)
        self.flow_times = deque(maxlen=300)
        self.detector_times = deque(maxlen=300)
        self.detector_source_numbers = deque(maxlen=300)
        self.detector_lag_ms = deque(maxlen=300)
        self.latencies = deque(maxlen=300)
        self.latest_tracks = []
        self.threads = [Thread(target=self._capture, daemon=True),
                        Thread(target=self._detect, daemon=True),
                        Thread(target=self._flow, daemon=True)]

    def start(self):
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stopping = True
        for thread in self.threads:
            thread.join(timeout=6)

    @staticmethod
    def _offer(queue, frame):
        try:
            queue.put_nowait(frame)
            return False
        except Full:
            queue.get_nowait()
            queue.put_nowait(frame)
            return True

    def _capture(self):
        capture = None
        try:
            capture = self.connect()
            if not capture.isOpened():
                raise RuntimeError("Cannot open simulated RTSP stream")
            while not self.stopping:
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("Simulated RTSP stream ended")
                now = time.monotonic()
                with self.lock:
                    self.source_frames += 1
                    frame = Frame(self.source_frames, image, now)
                    self.capture_times.append(now)
                    self.flow_drops += self._offer(self.flow_queue, frame)
                    if (frame.number - 1) % self.interval == 0:
                        self.detector_queue_replacements += self._offer(self.detector_queue, frame)
        except Exception as error:
            self.error = error
            self.stopping = True
        finally:
            if capture is not None:
                capture.release()

    def _detect(self):
        warmed = False
        try:
            while not self.stopping:
                try:
                    frame = self.detector_queue.get(timeout=.1)
                except Empty:
                    continue
                if not warmed:
                    height, width = frame.image.shape[:2]
                    self.tracker.warmup(width, height)
                    warmed = True
                tracks = self.tracker.step(frame.image)
                now = time.monotonic()
                with self.lock:
                    self.correction = (frame, tracks)
                    self.detector_frames += 1
                    self.detector_times.append(now)
                    self.detector_source_numbers.append(frame.number)
                    self.detector_lag_ms.append(1000 * (now - frame.received_at))
        except Exception as error:
            self.error = error
            self.stopping = True

    def _flow(self):
        flow = SparseFlow()
        visualizer = TrackVisualizer(trail_length=30)
        history = deque(maxlen=32)
        writer = None
        try:
            while not self.stopping:
                try:
                    frame = self.flow_queue.get(timeout=.1)
                except Empty:
                    continue
                history.append(frame)
                with self.lock:
                    correction = self.correction
                    if correction is not None and correction[0].number <= frame.number:
                        self.correction = None
                    else:
                        correction = None
                if correction is None:
                    tracks = flow.advance(frame.image)
                else:
                    keyframe, identities = correction
                    past = next((item for item in history if item.number == keyframe.number), None)
                    if past is None:
                        self.stale_corrections += 1
                        tracks = flow.advance(frame.image)
                    else:
                        tracks = flow.correct(past.image, identities)
                        for item in history:
                            if keyframe.number < item.number <= frame.number:
                                tracks = flow.advance(item.image)
                        self.applied_corrections += 1
                if self.record_path is not None and self.applied_corrections:
                    if writer is None:
                        height, width = frame.image.shape[:2]
                        size = (960, round(height * 960 / width / 2) * 2)
                        writer = cv2.VideoWriter(str(self.record_path),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
                        if not writer.isOpened():
                            raise RuntimeError("Could not create tracking video")
                    preview = cv2.resize(frame.image, size, interpolation=cv2.INTER_AREA)
                    scale_x, scale_y = size[0] / frame.image.shape[1], size[1] / frame.image.shape[0]
                    scaled = [{**track, "xyxy": [track["xyxy"][0] * scale_x,
                                                   track["xyxy"][1] * scale_y,
                                                   track["xyxy"][2] * scale_x,
                                                   track["xyxy"][3] * scale_y]}
                              for track in tracks]
                    preview = visualizer.draw(preview, frame.number, scaled)
                    writer.write(preview)
                now = time.monotonic()
                with self.lock:
                    self.flow_frames += 1
                    if writer is not None:
                        self.recorded_frames += 1
                    self.flow_times.append(now)
                    self.latencies.append(1000 * (now - frame.received_at))
                    self.latest_tracks = tracks
        except Exception as error:
            self.error = error
            self.stopping = True
        finally:
            if writer is not None:
                writer.release()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()

            def rate(times):
                recent = [stamp for stamp in times if now - stamp <= 10]
                return (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0

            recent_flow = [stamp for stamp in self.flow_times if now - stamp <= 10]
            flow_gaps = [1000 * (right - left) for left, right in zip(recent_flow, recent_flow[1:])]

            return {"source_frames": self.source_frames, "flow_frames": self.flow_frames,
                    "recorded_frames": self.recorded_frames,
                    "detector_frames": self.detector_frames, "source_fps_10s": round(rate(self.capture_times), 2),
                    "tracking_fps_10s": round(rate(self.flow_times), 2),
                    "detector_fps_10s": round(rate(self.detector_times), 2),
                    "flow_drops": self.flow_drops, "stale_corrections": self.stale_corrections,
                    "detector_queue_replacements": self.detector_queue_replacements,
                    "applied_corrections": self.applied_corrections,
                    "latest_track_count": len(self.latest_tracks),
                    "latencies_ms": list(self.latencies),
                    "flow_gaps_ms": flow_gaps,
                    "detector_lag_ms": list(self.detector_lag_ms),
                    "detector_source_numbers": list(self.detector_source_numbers)}
