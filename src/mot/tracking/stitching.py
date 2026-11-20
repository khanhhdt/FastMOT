"""Join interrupted tracks in recorded footage using appearance and motion."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .motion import valid_warp


POLICY = {
    "max_gap_seconds": 3.0,
    "endpoint_window_seconds": 0.60,
    "extended_endpoint_window_seconds": 1.50,
    "minimum_similarity": 0.42,
    "minimum_within_track_similarity": 0.30,
    "minimum_link_score": 0.33,
    "minimum_margin": 0.04,
    "assignment": "hungarian",
}


class CameraMotion:
    """Map nearby frames into a shared image coordinate system."""

    def __init__(self, warps, width, height):
        self.steps = [np.eye(3)]
        self.inverses = [np.eye(3)]
        self.segments = [0]
        for frame, warp in enumerate(warps, 1):
            step = np.eye(3)
            usable = frame > 1 and warp is not None and valid_warp(warp, width, height)
            # Sparse optical flow also returns exact identity when estimation fails.
            usable = usable and not np.array_equal(warp, step[:2])
            if usable:
                step[:2] = warp
            self.steps.append(step)
            self.inverses.append(np.linalg.inv(step))
            self.segments.append(self.segments[-1] if usable else frame)

    def relative(self, frames, reference):
        if any(self.segments[frame] != self.segments[reference] for frame in frames):
            return None
        matrices = {reference: np.eye(3)}
        for frame in range(reference - 1, min(frames) - 1, -1):
            matrices[frame] = matrices[frame + 1] @ self.steps[frame + 1]
        for frame in range(reference + 1, max(frames) + 1):
            matrices[frame] = matrices[frame - 1] @ self.inverses[frame]
        return np.stack([matrices[frame][:2] for frame in frames])


@dataclass
class Endpoint:
    identity: int
    boundary: int
    scene: int
    frames: list
    boxes: np.ndarray
    velocity: np.ndarray
    height: float
    position_error: float
    apply_from_frame: int
    opposite_samples: dict = field(default_factory=dict)

    @property
    def foot(self):
        box = self.boxes[0]
        return np.array([(box[0] + box[2]) / 2, box[3]])


def clear_observations(tracks, width, height):
    """Keep reliable crops, including high-confidence boxes near a side edge."""
    if not tracks:
        return []
    boxes = np.array([track["xyxy"] for track in tracks])
    sizes = boxes[:, 2:] - boxes[:, :2]
    overlap = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    overlap -= np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    intersection = np.maximum(overlap, 0).prod(axis=2)
    np.fill_diagonal(intersection, 0)
    covered = intersection.max(axis=1) / sizes.prod(axis=1)
    vertical_inside = (boxes[:, 1] > 2) & (boxes[:, 3] < height - 2)
    horizontal_inside = (boxes[:, 0] > 2) & (boxes[:, 2] < width - 2)
    return [track for index, track in enumerate(tracks)
            if vertical_inside[index] and sizes[index, 1] >= 45 and covered[index] <= 0.35
            and (track["confidence"] >= 0.40 if horizontal_inside[index] else
                 track["confidence"] >= 0.65 and sizes[index, 0] >= 0.22 * sizes[index, 1])]


def fit_endpoint(identity, scene, observations, clear, departure, fps, camera_motion=None):
    ordered = observations[::-1] if departure else observations
    boundary = ordered[0][0]
    window = round(fps * POLICY["endpoint_window_seconds"])
    extended = round(fps * POLICY["extended_endpoint_window_seconds"])
    nearby = []
    for frame, box in ordered:
        if abs(frame - boundary) > extended:
            break
        if frame in clear and (camera_motion is None or
                               camera_motion.segments[frame] == camera_motion.segments[boundary]):
            nearby.append((frame, box))
    if len(nearby) < 3:
        return None
    min_span = max(2, round(fps * 0.10))
    nearest = nearby[0][0]
    short = [(frame, box) for frame, box in nearby if abs(frame - boundary) <= window]
    if len(short) >= 3 and abs(nearest - boundary) <= 0.75 * window and abs(short[-1][0] - nearest) >= min_span:
        nearby = short
    else:
        # Estimate direction near the nearest clear frame when the boundary is obscured.
        nearby = [(frame, box) for frame, box in nearby if abs(frame - nearest) <= window]
    frames, boxes = zip(*nearby)
    if len(frames) < 3 or abs(frames[-1] - frames[0]) < min_span:
        return None
    boxes = np.array(boxes)
    points = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
    heights = boxes[:, 3] - boxes[:, 1]
    if camera_motion is not None:
        warps = camera_motion.relative(frames, nearest)
        points = np.einsum("nij,nj->ni", warps[:, :, :2], points) + warps[:, :, 2]
        heights = heights * np.linalg.norm(warps[:, :, 1], axis=1)
    seconds = np.array(frames) / fps
    seconds -= seconds.mean()
    velocity = (seconds @ points) / (seconds @ seconds)
    height = float(np.median(heights))
    if np.linalg.norm(velocity) / height > 3.5:
        return None
    residuals = points - points.mean(axis=0) - seconds[:, None] * velocity
    position_error = float(np.median(np.linalg.norm(residuals, axis=1)))
    sample = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
    # Avoid relabeling a long, poorly observed prefix of a returning track.
    apply_from = nearest if not departure and nearest - boundary > window else boundary
    return Endpoint(identity, boundary, scene, [frames[i] for i in sample], boxes[sample], velocity, height,
                    position_error, apply_from)


def motion_candidates(records, info, scene_cuts, camera_motion=None):
    """Screen track endpoints by time, movement and perspective-scaled distance."""
    observed, clear = defaultdict(list), defaultdict(set)
    for record in records:
        frame, tracks = record["frame"], record["tracks"]
        scene = bisect_right(scene_cuts, frame)
        for track in tracks:
            observed[scene, track["id"]].append((frame, track["xyxy"]))
        for track in clear_observations(tracks, info["width"], info["height"]):
            clear[scene, track["id"]].add(frame)

    fps = info["fps"]
    starts, ends = [], []
    for (scene, identity), observations in observed.items():
        if len(observations) < max(5, round(fps * 0.20)):
            continue
        reliable = clear[scene, identity]
        start = fit_endpoint(identity, scene, observations, reliable, False, fps, camera_motion)
        end = fit_endpoint(identity, scene, observations, reliable, True, fps, camera_motion)
        if start is not None and end is not None and end.frames[0] - start.frames[0] >= fps:
            start.opposite_samples = dict(zip(end.frames, end.boxes))
            end.opposite_samples = dict(zip(start.frames, start.boxes))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)

    starts.sort(key=lambda point: point.boundary)
    start_frames = [point.boundary for point in starts]
    candidates = []
    for end in ends:
        first = bisect_left(start_frames, end.boundary + 2)
        stop = bisect_right(start_frames, end.boundary + round(fps * POLICY["max_gap_seconds"]) + 1)
        for start in starts[first:stop]:
            if end.scene != start.scene or end.identity == start.identity:
                continue
            error = motion_error(end, start, fps, camera_motion)
            if error is not None:
                candidates.append((end, start, error))
    return candidates


def motion_error(end, start, fps, camera_motion=None):
    start_foot, start_velocity = start.foot, start.velocity
    start_height, start_error = start.height, start.position_error
    if camera_motion is not None:
        warps = camera_motion.relative([start.frames[0]], end.frames[0])
        if warps is None:
            return None
        warp = warps[0]
        start_foot = warp[:, :2] @ start_foot + warp[:, 2]
        start_velocity = warp[:, :2] @ start_velocity
        camera_scale = np.linalg.norm(warp[:, 1])
        start_height *= camera_scale
        start_error *= camera_scale
    height = math.sqrt(end.height * start_height)
    if min(end.height, start_height) / max(end.height, start_height) < 0.45:
        return None
    seconds = (start.frames[0] - end.frames[0]) / fps
    displacement = start_foot - end.foot
    distance = np.linalg.norm(displacement) / height
    slowest = min(np.linalg.norm(end.velocity) / end.height, np.linalg.norm(start_velocity) / start_height)
    if distance > 0.25 + 2.5 * seconds or (distance < 0.20 and slowest < 0.10):
        return None
    # Compare predictions halfway through the missing interval. Position uncertainty
    # grows with elapsed time and with the residuals of the endpoint fits.
    forward = end.foot + 0.5 * seconds * end.velocity
    backward = start_foot - 0.5 * seconds * start_velocity
    scale = height * (0.25 + 0.45 * seconds) + end.position_error + start_error
    error = float(np.linalg.norm(forward - backward) / scale)
    return error if error <= 2 else None


def sample_embeddings(candidates, frames, encode):
    requested = defaultdict(dict)
    for end, start, _ in candidates:
        for endpoint in (end, start):
            for frame, box in zip(endpoint.frames, endpoint.boxes):
                requested[frame][endpoint.identity] = box
            for frame, box in endpoint.opposite_samples.items():
                requested[frame][endpoint.identity] = box
    features = {}
    if not requested:
        return features
    last_sample = max(requested)
    for frame, image in frames:
        if frame in requested:
            samples = requested[frame]
            for identity, feature in zip(samples, encode(image, np.array(list(samples.values())))):
                if feature is not None and np.isfinite(feature).all():
                    norm = np.linalg.norm(feature)
                    if norm > 0:
                        features[frame, identity] = feature / norm
        if frame == last_sample:
            break
    return features


def endpoint_gallery(endpoint, features):
    gallery = [features[frame, endpoint.identity] for frame in endpoint.frames
               if (frame, endpoint.identity) in features]
    if len(gallery) < 2:
        return None, False
    gallery = np.stack(gallery)
    opposite = [features[frame, endpoint.identity] for frame in endpoint.opposite_samples
                if (frame, endpoint.identity) in features]
    # An existing tracker ID can change people. Strong appearance disagreement
    # between its own clear endpoints makes relabeling the entire fragment unsafe.
    consistent = len(opposite) < 2 or (gallery @ np.stack(opposite).T).max() >= POLICY["minimum_within_track_similarity"]
    return gallery, bool(consistent)


def rank_links(candidates, features):
    links = []
    for end, start, error in candidates:
        before, consistent_before = endpoint_gallery(end, features)
        after, consistent_after = endpoint_gallery(start, features)
        if before is None or after is None:
            continue
        similarity = float(np.median(before @ after.T))
        links.append({"from_id": start.identity, "to_id": end.identity,
                      "last_frame": end.boundary, "first_frame": start.boundary,
                      "apply_from_frame": start.apply_from_frame,
                      "gap_frames": start.boundary - end.boundary - 1,
                      "samples_before": end.frames, "samples_after": start.frames,
                      "consistent_before": consistent_before, "consistent_after": consistent_after,
                      "similarity": similarity, "motion_error": error,
                      "score": similarity - 0.15 * error})
    return sorted(links, key=lambda link: -link["score"])


def select_links(ranked):
    """Maximize one-to-one link scores, leaving ambiguous tracks unpaired."""
    eligible = [link for link in ranked if link["similarity"] >= POLICY["minimum_similarity"]
                and link["score"] > POLICY["minimum_link_score"]
                and link.get("consistent_before", True) and link.get("consistent_after", True)]
    if not eligible:
        return []
    departures = {identity: index for index, identity in enumerate(sorted({link["to_id"] for link in eligible}))}
    arrivals = {identity: index for index, identity in enumerate(sorted({link["from_id"] for link in eligible}))}
    weights = np.full((len(departures), len(arrivals) + len(departures)), -np.inf)
    weights[:, len(arrivals):] = 0  # Each departure may remain unmatched.
    edges = {}
    for link in eligible:
        edge = departures[link["to_id"]], arrivals[link["from_id"]]
        edges[edge] = link
        weights[edge] = link["score"] - POLICY["minimum_link_score"]
    rows, columns = linear_sum_assignment(weights, maximize=True)
    best_score = weights[rows, columns].sum()
    real_links = np.isfinite(weights[:, :len(arrivals)])
    row_counts, column_counts = real_links.sum(axis=1), real_links.sum(axis=0)
    accepted = []
    for edge in zip(rows, columns):
        if edge not in edges:
            continue
        weight = weights[edge]
        margin = weight
        if row_counts[edge[0]] > 1 or column_counts[edge[1]] > 1:
            weights[edge] = -np.inf
            alternative = linear_sum_assignment(weights, maximize=True)
            weights[edge] = weight
            # Measure the best complete assignment that excludes this link.
            margin = best_score - weights[alternative].sum()
        if margin >= POLICY["minimum_margin"]:
            accepted.append(edges[edge])
    return sorted(accepted, key=lambda link: link["first_frame"])


def relabel(records, links):
    """Preserve every observed box and trace a chain back to its earliest ID."""
    aliases = {}
    for link in links:
        prior, _ = aliases.get(link["to_id"], (link["to_id"], 1))
        aliases[link["from_id"]] = prior, link.get("apply_from_frame", 1)
    for record in records:
        tracks = []
        for track in record["tracks"]:
            identity, apply_from = aliases.get(track["id"], (track["id"], 1))
            tracks.append({**track, "id": identity} if record["frame"] >= apply_from and identity != track["id"] else track)
        yield {**record, "tracks": tracks}
