"""Join interrupted tracks in recorded footage from a fixed camera."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np


POLICY = {
    "max_gap_seconds": 3.0,
    "endpoint_window_seconds": 0.60,
    "minimum_similarity": 0.42,
    "minimum_margin": 0.06,
}


@dataclass
class Endpoint:
    identity: int
    boundary: int
    scene: int
    frames: list
    boxes: np.ndarray
    velocity: np.ndarray
    height: float

    @property
    def foot(self):
        box = self.boxes[0]
        return np.array([(box[0] + box[2]) / 2, box[3]])


def clear_observations(tracks, width, height):
    """Reject clipped, weak or heavily overlapped boxes before sampling appearance."""
    if not tracks:
        return []
    boxes = np.array([track["xyxy"] for track in tracks])
    sizes = boxes[:, 2:] - boxes[:, :2]
    overlap = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    overlap -= np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    intersection = np.maximum(overlap, 0).prod(axis=2)
    np.fill_diagonal(intersection, 0)
    covered = intersection.max(axis=1) / sizes.prod(axis=1)
    inside = (boxes[:, :2] > 2).all(axis=1) & (boxes[:, 2:] < [width - 2, height - 2]).all(axis=1)
    return [track for index, track in enumerate(tracks)
            if inside[index] and sizes[index, 1] >= 45 and covered[index] <= 0.35
            and track["confidence"] >= 0.40]


def fit_endpoint(identity, scene, observations, clear, departure, fps):
    ordered = observations[::-1] if departure else observations
    boundary = ordered[0][0]
    window = round(fps * POLICY["endpoint_window_seconds"])
    nearby = [(frame, box) for frame, box in ordered if abs(frame - boundary) <= window
              and frame in clear]
    if len(nearby) < 3 or abs(nearby[0][0] - boundary) > window * 0.75:
        return None
    frames, boxes = zip(*nearby)
    if abs(frames[-1] - frames[0]) < max(2, round(fps * 0.10)):
        return None
    boxes = np.array(boxes)
    points = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
    seconds = np.array(frames) / fps
    seconds -= seconds.mean()
    velocity = (seconds @ points) / (seconds @ seconds)
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))
    if not 0.22 <= np.linalg.norm(velocity) / height <= 3.5:
        return None
    sample = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
    return Endpoint(identity, boundary, scene, [frames[i] for i in sample], boxes[sample], velocity, height)


def motion_candidates(records, info, scene_cuts):
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
        for departure, group in ((False, starts), (True, ends)):
            endpoint = fit_endpoint(identity, scene, observations, clear[scene, identity], departure, fps)
            if endpoint is not None:
                group.append(endpoint)

    starts.sort(key=lambda point: point.boundary)
    start_frames = [point.boundary for point in starts]
    candidates = []
    for end in ends:
        first = bisect_left(start_frames, end.boundary + 2)
        stop = bisect_right(start_frames, end.boundary + round(fps * POLICY["max_gap_seconds"]) + 1)
        for start in starts[first:stop]:
            if end.scene != start.scene or end.identity == start.identity:
                continue
            error = motion_error(end, start, fps)
            if error is not None:
                candidates.append((end, start, error))
    return candidates


def motion_error(end, start, fps):
    height = math.sqrt(end.height * start.height)
    if min(end.height, start.height) / max(end.height, start.height) < 0.55:
        return None
    direction = np.dot(end.velocity, start.velocity) / (np.linalg.norm(end.velocity) * np.linalg.norm(start.velocity))
    if direction < -0.25:
        return None
    seconds = (start.frames[0] - end.frames[0]) / fps
    displacement = start.foot - end.foot
    if np.linalg.norm(displacement) / height > 0.25 + 2.5 * seconds:
        return None
    error = max(np.linalg.norm(displacement - endpoint.velocity * seconds)
                for endpoint in (end, start)) / height
    limit = 0.30 + 0.55 * seconds
    return float(error / limit) if error <= limit else None


def sample_embeddings(candidates, frames, encode):
    requested = defaultdict(dict)
    for end, start, _ in candidates:
        for endpoint in (end, start):
            for frame, box in zip(endpoint.frames, endpoint.boxes):
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


def rank_links(candidates, features):
    links = []
    for end, start, error in candidates:
        galleries = [[features[frame, endpoint.identity] for frame in endpoint.frames
                      if (frame, endpoint.identity) in features] for endpoint in (end, start)]
        if min(map(len, galleries)) < 2:
            continue
        before, after = map(np.stack, galleries)
        similarity = float(np.median(before @ after.T))
        links.append({"from_id": start.identity, "to_id": end.identity,
                      "last_frame": end.boundary, "first_frame": start.boundary,
                      "gap_frames": start.boundary - end.boundary - 1,
                      "samples_before": end.frames, "samples_after": start.frames,
                      "similarity": similarity, "motion_error": error,
                      "score": similarity - 0.15 * error})
    return sorted(links, key=lambda link: -link["score"])


def select_links(ranked):
    """Require a clear mutual match, avoiding competition for either endpoint."""
    departures, arrivals = defaultdict(list), defaultdict(list)
    for link in ranked:
        departures[link["to_id"]].append(link)
        arrivals[link["from_id"]].append(link)

    accepted = []
    for link in ranked:
        if link["similarity"] < POLICY["minimum_similarity"]:
            continue
        choices = (departures[link["to_id"]], arrivals[link["from_id"]])
        if all(group[0] is link and (len(group) == 1 or
               link["score"] - group[1]["score"] >= POLICY["minimum_margin"]) for group in choices):
            accepted.append(link)
    return sorted(accepted, key=lambda link: link["first_frame"])


def relabel(records, links):
    """Preserve every observed box and trace a chain back to its earliest ID."""
    aliases = {}
    for link in links:
        aliases[link["from_id"]] = aliases.get(link["to_id"], link["to_id"])
    for record in records:
        tracks = [{**track, "id": aliases[track["id"]]} if track["id"] in aliases else track
                  for track in record["tracks"]]
        yield {**record, "tracks": tracks}
