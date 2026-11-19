"""Guard camera-motion estimates and distinguish blank frames from hard scene changes."""

import cv2
import numpy as np


def valid_warp(warp, width, height):
    warp = np.asarray(warp)
    if warp.shape != (2, 3) or not np.isfinite(warp).all():
        return False
    linear = warp[:, :2]
    scales = np.linalg.svd(linear, compute_uv=False)
    angle = abs(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    return bool(np.linalg.det(linear) > 0 and scales.min() >= 0.8 and scales.max() <= 1.25
                and angle <= 15 and abs(warp[0, 2]) <= width * 0.2 and abs(warp[1, 2]) <= height * 0.2)


class SceneObserver:
    def __init__(self):
        self.previous = None

    def observe(self, image):
        gray = cv2.cvtColor(cv2.resize(image, (64, 36)), cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 2:
            return "blank"
        current = gray.astype(np.float32) / 255
        cut = False
        if self.previous is not None:
            difference = float(np.abs(current - self.previous).mean())
            hist = np.histogram(current, bins=16, range=(0, 1))[0] / current.size
            previous_hist = np.histogram(self.previous, bins=16, range=(0, 1))[0] / current.size
            cut = difference > 0.5 and np.abs(hist - previous_hist).sum() > 0.7
        self.previous = current
        return "scene_cut" if cut else "normal"


class ObservedGMC:
    def __init__(self, backend, guarded=False):
        self.backend = backend
        self.guarded = guarded
        self.event = "normal"
        self.last_warp = np.eye(2, 3, dtype=np.float32)
        self.blank_frames = 0
        self.rejected_warps = 0
        self.failed_estimates = 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def apply(self, image, detections=None):
        identity = np.eye(2, 3, dtype=np.float32)
        if self.guarded and self.event == "blank":
            # Do not replace useful optical-flow history with a dropout frame.
            self.blank_frames += 1
            self.last_warp = identity
            return identity
        try:
            warp = self.backend.apply(image, detections)
        except Exception:
            self.failed_estimates += 1
            self.last_warp = identity
            return identity
        if self.guarded and not valid_warp(warp, image.shape[1], image.shape[0]):
            self.rejected_warps += 1
            self.backend.reset_params()
            try:
                self.backend.apply(image, detections)  # Seed from this valid image.
            except Exception:
                self.failed_estimates += 1
            warp = identity
        self.last_warp = np.asarray(warp).copy()
        return self.last_warp


def reset_scene(tracker):
    """Reset motion/track pools at a cut while keeping source frame count and unique IDs."""
    counter, ids = tracker.frame_id, tracker._ids
    tracker.reset()
    tracker.frame_id, tracker._ids = counter, ids
