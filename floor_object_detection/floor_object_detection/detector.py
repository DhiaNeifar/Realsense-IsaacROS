from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class DetectionCandidate:
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    center_pixel: tuple[int, int]
    center_xyz: tuple[float, float, float] | None
    distance_m: float
    median_height_m: float
    max_height_m: float
    width_m: float | None
    confidence: float
    score: float
    support_pixels: int


@dataclass
class DetectionFrame:
    candidate: DetectionCandidate | None
    foreground_mask: np.ndarray
    floor_mask: np.ndarray
    plane: np.ndarray | None
    used_fallback: bool
    status_text: str


class DepthBasedFloorObjectDetector:
    """Detect floor objects using depth-first segmentation and geometry filters."""

    def __init__(
        self,
        *,
        min_depth_m: float,
        max_depth_m: float,
        depth_scale: float,
        plane_ransac_iterations: int,
        plane_inlier_threshold_m: float,
        min_floor_points: int,
        min_plane_y_component: float,
        plane_sample_stride: int,
        min_height_above_floor_m: float,
        max_height_above_floor_m: float,
        fallback_foreground_margin_m: float,
        local_background_sigma: float,
        search_top_ignore_ratio: float,
        min_contour_area: int,
        min_bbox_size_px: int,
        max_bbox_aspect_ratio: float,
        min_extent: float,
        min_solidity: float,
        open_kernel_size: int,
        close_kernel_size: int,
        point_depth_window: int,
    ) -> None:
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_scale = float(depth_scale)
        self.plane_ransac_iterations = int(plane_ransac_iterations)
        self.plane_inlier_threshold_m = float(plane_inlier_threshold_m)
        self.min_floor_points = int(min_floor_points)
        self.min_plane_y_component = float(min_plane_y_component)
        self.plane_sample_stride = max(1, int(plane_sample_stride))
        self.min_height_above_floor_m = float(min_height_above_floor_m)
        self.max_height_above_floor_m = float(max_height_above_floor_m)
        self.fallback_foreground_margin_m = float(fallback_foreground_margin_m)
        self.local_background_sigma = float(local_background_sigma)
        self.search_top_ignore_ratio = float(np.clip(search_top_ignore_ratio, 0.0, 0.8))
        self.min_contour_area = int(min_contour_area)
        self.min_bbox_size_px = int(min_bbox_size_px)
        self.max_bbox_aspect_ratio = float(max_bbox_aspect_ratio)
        self.min_extent = float(min_extent)
        self.min_solidity = float(min_solidity)
        self.open_kernel_size = max(1, int(open_kernel_size))
        self.close_kernel_size = max(1, int(close_kernel_size))
        self.point_depth_window = max(1, int(point_depth_window))

        self._rng = np.random.default_rng(7)
        self._cached_shape: tuple[int, int] | None = None
        self._cached_intrinsics: CameraIntrinsics | None = None
        self._ray_x: np.ndarray | None = None
        self._ray_y: np.ndarray | None = None

    def detect(
        self,
        *,
        color_shape: tuple[int, int],
        depth_raw: np.ndarray,
        intrinsics: CameraIntrinsics | None,
    ) -> DetectionFrame:
        depth_m = self.depth_to_meters(depth_raw)
        depth_m = self.match_depth_to_color(depth_m, color_shape)
        valid_depth = np.isfinite(depth_m) & (depth_m > self.min_depth_m) & (depth_m < self.max_depth_m)

        empty_mask = np.zeros(color_shape, dtype=np.uint8)
        if not np.any(valid_depth):
            return DetectionFrame(
                candidate=None,
                foreground_mask=empty_mask,
                floor_mask=empty_mask,
                plane=None,
                used_fallback=True,
                status_text="no valid depth",
            )

        plane = None
        floor_mask = np.zeros_like(empty_mask)
        used_fallback = True
        status_text = "depth fallback"

        if intrinsics is not None:
            plane = self.estimate_floor_plane(depth_m, intrinsics, valid_depth)

        if plane is not None and intrinsics is not None:
            height_map = self.compute_signed_distance_map(depth_m, intrinsics, plane)
            foreground_mask = (
                valid_depth
                & (height_map > self.min_height_above_floor_m)
                & (height_map < self.max_height_above_floor_m)
            )
            floor_mask = (valid_depth & (np.abs(height_map) <= self.plane_inlier_threshold_m)).astype(np.uint8) * 255
            used_fallback = False
            status_text = "plane segmentation"
        else:
            height_map = self.compute_local_foreground_height(depth_m, valid_depth)
            foreground_mask = (
                valid_depth
                & (height_map > self.fallback_foreground_margin_m)
                & (height_map < self.max_height_above_floor_m)
            )
            status_text = "depth fallback" if intrinsics is not None else "waiting for camera_info"

        foreground_mask_u8 = self.clean_foreground_mask(foreground_mask.astype(np.uint8) * 255)
        candidate = self.find_best_candidate(
            foreground_mask_u8=foreground_mask_u8,
            depth_m=depth_m,
            height_map=height_map,
            intrinsics=intrinsics,
        )

        return DetectionFrame(
            candidate=candidate,
            foreground_mask=foreground_mask_u8,
            floor_mask=floor_mask,
            plane=plane,
            used_fallback=used_fallback,
            status_text=status_text,
        )

    def depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        if depth_raw.dtype == np.uint16:
            return depth_raw.astype(np.float32) * self.depth_scale
        return depth_raw.astype(np.float32)

    def match_depth_to_color(self, depth_m: np.ndarray, color_shape: tuple[int, int]) -> np.ndarray:
        color_h, color_w = color_shape
        if depth_m.shape[:2] == (color_h, color_w):
            return depth_m
        return cv2.resize(depth_m, (color_w, color_h), interpolation=cv2.INTER_NEAREST)

    def clean_foreground_mask(self, foreground_mask_u8: np.ndarray) -> np.ndarray:
        cleaned = foreground_mask_u8.copy()
        height, _ = cleaned.shape[:2]
        cleaned[: int(height * self.search_top_ignore_ratio), :] = 0

        if self.open_kernel_size > 1:
            kernel = np.ones((self.open_kernel_size, self.open_kernel_size), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        if self.close_kernel_size > 1:
            kernel = np.ones((self.close_kernel_size, self.close_kernel_size), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        return cleaned

    def compute_local_foreground_height(self, depth_m: np.ndarray, valid_depth: np.ndarray) -> np.ndarray:
        depth_fill = np.where(valid_depth, depth_m, self.max_depth_m).astype(np.float32)
        background = cv2.GaussianBlur(
            depth_fill,
            (0, 0),
            sigmaX=self.local_background_sigma,
            sigmaY=self.local_background_sigma,
        )
        return np.clip(background - depth_m, 0.0, None)

    def ensure_projection_cache(
        self,
        shape: tuple[int, int],
        intrinsics: CameraIntrinsics,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._cached_shape == shape
            and self._cached_intrinsics == intrinsics
            and self._ray_x is not None
            and self._ray_y is not None
        ):
            return self._ray_x, self._ray_y

        height, width = shape
        xs = (np.arange(width, dtype=np.float32) - intrinsics.cx) / intrinsics.fx
        ys = (np.arange(height, dtype=np.float32) - intrinsics.cy) / intrinsics.fy

        self._ray_x = np.tile(xs[np.newaxis, :], (height, 1))
        self._ray_y = np.tile(ys[:, np.newaxis], (1, width))
        self._cached_shape = shape
        self._cached_intrinsics = intrinsics
        return self._ray_x, self._ray_y

    def compute_signed_distance_map(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        plane: np.ndarray,
    ) -> np.ndarray:
        ray_x, ray_y = self.ensure_projection_cache(depth_m.shape[:2], intrinsics)
        x_m = ray_x * depth_m
        y_m = ray_y * depth_m
        return plane[0] * x_m + plane[1] * y_m + plane[2] * depth_m + plane[3]

    def estimate_floor_plane(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        valid_depth: np.ndarray,
    ) -> np.ndarray | None:
        ray_x, ray_y = self.ensure_projection_cache(depth_m.shape[:2], intrinsics)
        height, width = depth_m.shape[:2]
        sample_mask = np.zeros((height, width), dtype=bool)
        sample_mask[int(height * 0.55) :: self.plane_sample_stride, :: self.plane_sample_stride] = True
        sample_mask &= valid_depth

        if np.count_nonzero(sample_mask) < self.min_floor_points:
            return None

        z = depth_m[sample_mask]
        points = np.column_stack((ray_x[sample_mask] * z, ray_y[sample_mask] * z, z))
        if len(points) < self.min_floor_points:
            return None

        best_plane = None
        best_inlier_count = 0

        for _ in range(self.plane_ransac_iterations):
            sample_indices = self._rng.choice(len(points), size=3, replace=False)
            plane = self.fit_plane(points[sample_indices])
            if plane is None or abs(plane[1]) < self.min_plane_y_component:
                continue

            distances = np.abs(points @ plane[:3] + plane[3])
            inliers = distances < self.plane_inlier_threshold_m
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count <= best_inlier_count:
                continue

            refined_plane = self.fit_plane(points[inliers])
            if refined_plane is None or abs(refined_plane[1]) < self.min_plane_y_component:
                continue

            best_plane = refined_plane
            best_inlier_count = inlier_count

        if best_plane is None or best_inlier_count < self.min_floor_points:
            return None

        if best_plane[3] < 0.0:
            best_plane = -best_plane

        return best_plane

    def fit_plane(self, points: np.ndarray) -> np.ndarray | None:
        if len(points) < 3:
            return None

        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None

        normal = normal / norm
        d = -float(np.dot(normal, centroid))
        plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
        if plane[3] < 0.0:
            plane = -plane
        return plane

    def find_best_candidate(
        self,
        *,
        foreground_mask_u8: np.ndarray,
        depth_m: np.ndarray,
        height_map: np.ndarray,
        intrinsics: CameraIntrinsics | None,
    ) -> DetectionCandidate | None:
        contours, _ = cv2.findContours(foreground_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        working_mask = foreground_mask_u8 > 0
        best_candidate = None
        best_score = -math.inf

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_bbox_size_px or h < self.min_bbox_size_px:
                continue

            aspect = w / float(max(h, 1))
            if aspect > self.max_bbox_aspect_ratio or aspect < (1.0 / self.max_bbox_aspect_ratio):
                continue

            contour_mask = np.zeros_like(foreground_mask_u8, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            support_mask = (contour_mask > 0) & working_mask
            support_pixels = int(np.count_nonzero(support_mask))
            if support_pixels < self.min_contour_area:
                continue

            support_depth = depth_m[support_mask]
            support_height = height_map[support_mask]
            valid_support = np.isfinite(support_depth) & np.isfinite(support_height)
            if not np.any(valid_support):
                continue

            support_depth = support_depth[valid_support]
            support_height = support_height[valid_support]

            hull = cv2.convexHull(contour)
            hull_area = float(max(cv2.contourArea(hull), 1.0))
            solidity = area / hull_area
            extent = support_pixels / float(max(w * h, 1))
            if solidity < self.min_solidity or extent < self.min_extent:
                continue

            median_depth = float(np.median(support_depth))
            median_height = float(np.median(support_height))
            max_height = float(np.percentile(support_height, 90))
            center_pixel = (x + w // 2, y + h // 2)
            center_xyz = self.pixel_to_xyz(center_pixel, depth_m, intrinsics)

            left_xyz = self.pixel_to_xyz((x, center_pixel[1]), depth_m, intrinsics)
            right_xyz = self.pixel_to_xyz((x + w - 1, center_pixel[1]), depth_m, intrinsics)
            width_m = None
            if left_xyz is not None and right_xyz is not None:
                width_m = float(np.linalg.norm(np.array(right_xyz) - np.array(left_xyz)))

            confidence = float(np.clip(
                0.20
                + 0.35 * min(1.0, support_pixels / 5000.0)
                + 0.20 * min(1.0, median_height / max(self.min_height_above_floor_m * 4.0, 1e-3))
                + 0.15 * extent
                + 0.10 * solidity,
                0.0,
                1.0,
            ))
            score = (
                support_pixels
                * max(median_height, 0.01)
                * max(extent, 0.10)
                * max(solidity, 0.10)
                / max(median_depth, 0.20)
            )

            if score <= best_score:
                continue

            best_score = score
            best_candidate = DetectionCandidate(
                bbox=(int(x), int(y), int(w), int(h)),
                contour=contour,
                mask=contour_mask,
                center_pixel=center_pixel,
                center_xyz=center_xyz,
                distance_m=median_depth,
                median_height_m=median_height,
                max_height_m=max_height,
                width_m=width_m,
                confidence=confidence,
                score=float(score),
                support_pixels=support_pixels,
            )

        return best_candidate

    def pixel_to_xyz(
        self,
        pixel: tuple[int, int],
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics | None,
    ) -> tuple[float, float, float] | None:
        if intrinsics is None:
            return None

        u, v = int(pixel[0]), int(pixel[1])
        height, width = depth_m.shape[:2]
        if u < 0 or u >= width or v < 0 or v >= height:
            return None

        half_window = self.point_depth_window // 2
        u0 = max(0, u - half_window)
        u1 = min(width, u + half_window + 1)
        v0 = max(0, v - half_window)
        v1 = min(height, v + half_window + 1)

        local_depth = depth_m[v0:v1, u0:u1]
        valid = np.isfinite(local_depth) & (local_depth > self.min_depth_m) & (local_depth < self.max_depth_m)
        if not np.any(valid):
            return None

        z = float(np.median(local_depth[valid]))
        x = (float(u) - intrinsics.cx) * z / intrinsics.fx
        y = (float(v) - intrinsics.cy) * z / intrinsics.fy
        return (x, y, z)
