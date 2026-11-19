"""Recover the exported ID of a selected stationary track on a fixed camera."""

from collections import deque
from statistics import median


def intersection_area(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(a, b):
    intersection = intersection_area(a, b)
    return intersection / (box_area(a) + box_area(b) - intersection)


class StationaryAnchor:
    """Hold an ID through weak observations, then confirm a new track at the anchor."""

    LEARN_FRAMES = 30
    CONFIRM_FRAMES = 5
    ACQUIRE_CONFIDENCE = 0.45
    KEEP_CONFIDENCE = 0.10
    MIN_IOU = 0.65

    def __init__(self, identity, max_gap_frames):
        self.identity = self.owner = identity
        self.max_gap_frames = max_gap_frames
        self.samples = deque(maxlen=self.LEARN_FRAMES)
        self.box = None
        self.last_seen = 0
        self.retired = False
        self.seen = set()
        self.bound = {identity}
        self.pending = None
        self.streak = 0
        self.events = []

    def ready(self, frame, scene_cut):
        if self.retired:
            return False
        if scene_cut or (self.box is not None and frame - self.last_seen > self.max_gap_frames):
            if self.box is not None:
                self.events.append({"frame": frame, "type": "anchor_ended", "reason": "scene_cut" if scene_cut else "timeout"})
            self.retired = True
            return False
        return True

    def learn(self, frame, owner, sample_count=LEARN_FRAMES):
        if owner is None or owner["confidence"] < self.ACQUIRE_CONFIDENCE:
            self.samples.clear()
        else:
            self.samples.append(owner["xyxy"])
            if len(self.samples) == sample_count:
                box = [median(edge) for edge in zip(*self.samples)]
                if all(box_iou(box, sample) >= 0.8 for sample in self.samples):
                    self.box = box
                    self.last_seen = frame
                    self.events.append({"frame": frame, "type": "anchor_learned", "xyxy": box})

    def recover(self, frame, identity):
        self.events.append({"frame": frame, "type": "anchor_recovered", "from_id": identity, "to_id": self.identity})
        self.owner = identity
        self.last_seen = frame

    def update(self, frame, tracks, scene_cut=False):
        if not self.ready(frame, scene_cut):
            return tracks

        indexed = {track["id"]: track for track in tracks}
        new_ids = indexed.keys() - self.seen
        self.seen.update(indexed)
        owner = indexed.get(self.owner)
        if self.box is None:
            self.learn(frame, owner)
            return tracks

        result = [track for track in tracks if track["id"] not in self.bound]
        if owner is not None and owner["confidence"] >= self.KEEP_CONFIDENCE and box_iou(self.box, owner["xyxy"]) >= self.MIN_IOU:
            self.pending, self.streak = None, 0
            self.last_seen = frame
            return [*result, {**owner, "id": self.identity}]

        # An ID already seen elsewhere cannot be borrowed when it walks past the anchor.
        fitting = [track for track in result if track["confidence"] >= self.ACQUIRE_CONFIDENCE
                   and box_iou(self.box, track["xyxy"]) >= self.MIN_IOU]
        if len(fitting) != 1 or (fitting[0]["id"] not in new_ids and fitting[0]["id"] != self.pending):
            self.pending, self.streak = None, 0
            return result

        candidate = fitting[0]
        identity = candidate["id"]
        self.streak = self.streak + 1 if self.pending == identity else 1
        self.pending = identity
        # Wait for confirmation without displaying a provisional ID in this region.
        result = [track for track in result if track["id"] != identity]
        if self.streak == self.CONFIRM_FRAMES:
            self.recover(frame, identity)
            self.bound.add(identity)
            self.pending, self.streak = None, 0
            result.append({**candidate, "id": self.identity})
        return result


class StationaryAnchors:
    """Assign observations to selected anchors jointly, even when a native ID changes seats."""

    def __init__(self, identities, max_gap_frames, seed_frames=None):
        self.anchors = {identity: StationaryAnchor(identity, max_gap_frames) for identity in identities}
        self.seed_frames = seed_frames or {}
        self.max_gap_frames = max_gap_frames
        self.origins = {}
        self.pending = {}
        self.references = {}

    def similarity(self, identity, box):
        initial = box_iou(box, self.anchors[identity].box)
        if initial < 0.45 or identity not in self.references:
            return initial
        return max(initial, box_iou(box, self.references[identity]))

    def update(self, frame, tracks, scene_cut=False):
        indexed = {track["id"]: track for track in tracks}
        active, learning = {}, set()
        for identity, anchor in self.anchors.items():
            if not anchor.ready(frame, scene_cut):
                continue
            if anchor.box is None:
                if identity not in self.seed_frames:
                    anchor.learn(frame, indexed.get(identity))
                else:
                    first, last = self.seed_frames[identity]
                    if first <= frame <= last:
                        anchor.learn(frame, indexed.get(identity), sample_count=last - first + 1)
            if anchor.box is None:
                learning.add(identity)
            else:
                active[identity] = anchor

        candidates = {identity: [] for identity in active}
        for identity, track in indexed.items():
            box = track["xyxy"]
            last, origin = self.origins.get(identity, (frame, box))
            # A native ID observed again after a gap can belong to a different seated person.
            if frame - last > 3:
                origin = box
            self.origins[identity] = (frame, origin)
            if identity in learning or track["confidence"] < StationaryAnchor.KEEP_CONFIDENCE:
                continue
            matches = sorted(((self.similarity(k, box), k) for k in active), reverse=True)
            if matches and matches[0][0] >= StationaryAnchor.MIN_IOU:
                if len(matches) == 1 or matches[0][0] - matches[1][0] >= 0.10:
                    candidates[matches[0][1]].append(track)
                elif identity in active and active[identity].owner == identity:
                    if self.similarity(identity, box) >= StationaryAnchor.MIN_IOU:
                        candidates[identity].append(track)

        mapped, withheld, pending = {}, set(), {}
        for identity, anchor in active.items():
            fitting = candidates[identity]
            owner = next((track for track in fitting if track["id"] == anchor.owner), None)
            if owner is not None:
                mapped[owner["id"]] = identity
                anchor.last_seen = frame
                continue
            strong = [track for track in fitting if track["confidence"] >= StationaryAnchor.ACQUIRE_CONFIDENCE]
            if len(strong) != 1:
                continue
            candidate = strong[0]
            native_id, box = candidate["id"], candidate["xyxy"]
            origin = self.origins[native_id][1]
            # A partly visible person can first appear as a small box inside the anchor.
            if box_iou(origin, anchor.box) < 0.40 and intersection_area(origin, anchor.box) < 0.90 * box_area(origin):
                continue
            earlier_id, samples = self.pending.get(identity, (native_id, []))
            if earlier_id != native_id or (samples and box_iou(samples[0], box) < 0.8):
                samples = []
            samples.append(box)
            if len(samples) == StationaryAnchor.CONFIRM_FRAMES:
                anchor.recover(frame, native_id)
                # Keep the original region as a limit, even after a clean reappearance shifts the box.
                self.references[identity] = [median(edge) for edge in zip(*samples)]
                mapped[native_id] = identity
            else:
                pending[identity] = (native_id, samples)
                withheld.add(native_id)

        self.pending = pending
        return [{**track, "id": mapped[identity]} if identity in mapped else track
                for identity, track in indexed.items()
                if identity in mapped or (identity not in active and identity not in withheld)]
