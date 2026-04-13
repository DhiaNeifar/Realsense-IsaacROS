from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from floor_object_detection.detector import DetectionCandidate


@dataclass
class TrackedDetection:
    track_id: int
    bbox: tuple[float, float, float, float]
    center_pixel: tuple[float, float]
    center_xyz: tuple[float, float, float] | None
    distance_m: float
    median_height_m: float
    width_m: float | None
    confidence: float
    stale: bool
    confirmed_frames: int
    missed_frames: int


class TemporalDetectionTracker:
    """Stabilize detections and keep the track alive through short dropouts."""

    def __init__(
        self,
        *,
        min_confirmed_frames: int,
        max_missed_frames: int,
        bbox_smoothing_alpha: float,
        center_smoothing_alpha: float,
        max_center_jump_px: float,
        max_depth_jump_m: float,
        min_iou: float = 0.10,
    ) -> None:
        self.min_confirmed_frames = max(1, int(min_confirmed_frames))
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.bbox_smoothing_alpha = float(np.clip(bbox_smoothing_alpha, 0.0, 1.0))
        self.center_smoothing_alpha = float(np.clip(center_smoothing_alpha, 0.0, 1.0))
        self.max_center_jump_px = float(max_center_jump_px)
        self.max_depth_jump_m = float(max_depth_jump_m)
        self.min_iou = float(min_iou)

        self._track: TrackedDetection | None = None
        self._next_track_id = 1

    def update(self, candidate: "DetectionCandidate" | None) -> TrackedDetection | None:
        if candidate is None:
            return self.handle_missed_detection()

        if self._track is None or not self.is_compatible(self._track, candidate):
            self._track = self.create_track(candidate)
            return self._track if self._track.confirmed_frames >= self.min_confirmed_frames else None

        self._track = self.update_track(self._track, candidate)
        return self._track if self._track.confirmed_frames >= self.min_confirmed_frames else None

    def handle_missed_detection(self) -> TrackedDetection | None:
        if self._track is None:
            return None

        self._track.missed_frames += 1
        if self._track.missed_frames > self.max_missed_frames:
            self._track = None
            return None

        self._track.stale = True
        self._track.confidence *= 0.95
        return self._track if self._track.confirmed_frames >= self.min_confirmed_frames else None

    def create_track(self, candidate: "DetectionCandidate") -> TrackedDetection:
        track = TrackedDetection(
            track_id=self._next_track_id,
            bbox=tuple(float(v) for v in candidate.bbox),
            center_pixel=tuple(float(v) for v in candidate.center_pixel),
            center_xyz=candidate.center_xyz,
            distance_m=float(candidate.distance_m),
            median_height_m=float(candidate.median_height_m),
            width_m=candidate.width_m,
            confidence=float(candidate.confidence),
            stale=False,
            confirmed_frames=1,
            missed_frames=0,
        )
        self._next_track_id += 1
        return track

    def update_track(self, track: TrackedDetection, candidate: "DetectionCandidate") -> TrackedDetection:
        track.bbox = tuple(
            self.smooth_scalar(old, new, self.bbox_smoothing_alpha)
            for old, new in zip(track.bbox, candidate.bbox)
        )
        track.center_pixel = tuple(
            self.smooth_scalar(old, new, self.center_smoothing_alpha)
            for old, new in zip(track.center_pixel, candidate.center_pixel)
        )

        if track.center_xyz is None:
            track.center_xyz = candidate.center_xyz
        elif candidate.center_xyz is not None:
            track.center_xyz = tuple(
                self.smooth_scalar(old, new, self.center_smoothing_alpha)
                for old, new in zip(track.center_xyz, candidate.center_xyz)
            )

        track.distance_m = self.smooth_scalar(track.distance_m, candidate.distance_m, self.center_smoothing_alpha)
        track.median_height_m = self.smooth_scalar(
            track.median_height_m,
            candidate.median_height_m,
            self.center_smoothing_alpha,
        )

        if candidate.width_m is not None:
            if track.width_m is None:
                track.width_m = candidate.width_m
            else:
                track.width_m = self.smooth_scalar(track.width_m, candidate.width_m, self.center_smoothing_alpha)

        track.confidence = max(track.confidence * 0.70, candidate.confidence)
        track.stale = False
        track.missed_frames = 0
        track.confirmed_frames += 1
        return track

    def is_compatible(self, track: TrackedDetection, candidate: "DetectionCandidate") -> bool:
        iou = self.compute_iou(track.bbox, candidate.bbox)
        if iou >= self.min_iou and abs(track.distance_m - candidate.distance_m) <= self.max_depth_jump_m:
            return True

        center_distance = float(np.linalg.norm(np.array(track.center_pixel) - np.array(candidate.center_pixel)))
        depth_distance = abs(track.distance_m - candidate.distance_m)
        return center_distance <= self.max_center_jump_px and depth_distance <= self.max_depth_jump_m

    @staticmethod
    def compute_iou(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b

        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        union = aw * ah + bw * bh - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    @staticmethod
    def smooth_scalar(old_value: float, new_value: float, alpha: float) -> float:
        return (1.0 - alpha) * float(old_value) + alpha * float(new_value)
