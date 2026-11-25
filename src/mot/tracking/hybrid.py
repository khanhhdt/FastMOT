"""Propagate TrackTrack IDs between sparse detector frames on a fixed camera."""

import cv2
import numpy as np


class SparseFlow:
    """Move current boxes with local optical flow; detector frames remain authoritative."""

    def __init__(self, scale=0.5):
        self.scale = scale
        self.gray = None
        self.tracks = []

    def _gray(self, image):
        height, width = image.shape[:2]
        small = cv2.resize(image, (round(width * self.scale), round(height * self.scale)),
                           interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def correct(self, image, tracks):
        self.gray = self._gray(image)
        self.tracks = [dict(track) for track in tracks]
        return self.tracks

    def advance(self, image):
        gray = self._gray(image)
        if self.gray is None or not self.tracks:
            self.gray = gray
            return []
        points = []
        for track in self.tracks:
            x1, y1, x2, y2 = np.asarray(track["xyxy"], dtype=np.float32) * self.scale
            xs = np.linspace(x1 + (x2 - x1) * .2, x2 - (x2 - x1) * .2, 3)
            ys = np.linspace(y1 + (y2 - y1) * .2, y2 - (y2 - y1) * .2, 3)
            points.extend((x, y) for y in ys for x in xs)
        previous = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, previous, None, winSize=(15, 15), maxLevel=2)
        height, width = image.shape[:2]
        updated = []
        if next_points is not None:
            delta = (next_points - previous).reshape(-1, 9, 2)
            valid = status.reshape(-1, 9).astype(bool)
            for track, movement, good in zip(self.tracks, delta, valid):
                if good.sum() < 3:
                    continue
                shift = np.median(movement[good], axis=0) / self.scale
                x1, y1, x2, y2 = np.asarray(track["xyxy"]) + np.tile(shift, 2)
                x1, x2 = np.clip((x1, x2), 0, width)
                y1, y2 = np.clip((y1, y2), 0, height)
                if x2 > x1 and y2 > y1:
                    updated.append({**track, "xyxy": [float(x1), float(y1), float(x2), float(y2)]})
        self.gray, self.tracks = gray, updated
        return updated
