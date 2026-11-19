"""Draw current track IDs and short, bounded trajectories in image coordinates."""

from collections import deque

import cv2
from ultralytics.utils.plotting import colors


class TrackVisualizer:
    def __init__(self, trail_length=30):
        self.trail_length = trail_length
        self.history = {}

    def draw(self, image, frame_index, tracks):
        annotated = image.copy()
        for track_id in list(self.history):
            points = self.history[track_id]
            while points and frame_index - points[0][0] >= self.trail_length:
                points.popleft()
            if not points:
                del self.history[track_id]

        height, width = image.shape[:2]
        for track in tracks:
            track_id = track["id"]
            x1, y1, x2, y2 = map(round, track["xyxy"])
            point = (frame_index, min(width - 1, (x1 + x2) // 2), min(height - 1, y2))
            points = self.history.setdefault(track_id, deque(maxlen=self.trail_length))
            points.append(point)
            color = colors(track_id, True)
            for previous, current in zip(points, list(points)[1:]):
                # A gap is an unobserved position; do not invent a line through it.
                if current[0] == previous[0] + 1:
                    cv2.line(annotated, previous[1:], current[1:], color, 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}"
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            top = max(0, y1 - 22)
            cv2.rectangle(annotated, (x1, top), (x1 + text_width + 6, top + 22), color, -1)
            cv2.putText(annotated, label, (x1 + 3, top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated
