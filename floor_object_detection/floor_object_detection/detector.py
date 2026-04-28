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
    left_xyz: tuple[float, float, float] | None
    right_xyz: tuple[float, float, float] | None
    distance_m: float
    median_height_m: float
    max_height_m: float
    width_m: float | None
    confidence: float
    score: float
    support_pixels: int
    score_components: dict[str, float] | None = None


@dataclass
class DetectionFrame:
    candidate: DetectionCandidate | None
    foreground_mask: np.ndarray
    floor_mask: np.ndarray
    edge_mask: np.ndarray
    valid_depth_mask: np.ndarray
    ignore_mask: np.ndarray
    plane: np.ndarray | None
    used_fallback: bool
    status_text: str
    candidate_scores: list[dict[str, float | str | tuple[int, int, int, int]]]


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
        use_ignore_mask: bool = True,
        ignore_regions_normalized: list[float] | None = None,
        center_prior_weight: float = 0.12,
        color_score_weight: float = 0.22,
        edge_score_weight: float = 0.16,
        geometry_score_weight: float = 0.42,
        temporal_score_weight: float = 0.08,
        auto_canny_sigma: float = 0.33,
        min_candidate_area_ratio: float = 0.0008,
        max_candidate_area_ratio: float = 0.35,
        min_depth_support_ratio: float = 0.35,
        debug_candidate_scores: bool = False,
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
        self.use_ignore_mask = bool(use_ignore_mask)
        self.ignore_regions_normalized = self.normalize_ignore_regions(ignore_regions_normalized)
        self.center_prior_weight = max(0.0, float(center_prior_weight))
        self.color_score_weight = max(0.0, float(color_score_weight))
        self.edge_score_weight = max(0.0, float(edge_score_weight))
        self.geometry_score_weight = max(0.0, float(geometry_score_weight))
        self.temporal_score_weight = max(0.0, float(temporal_score_weight))
        self.auto_canny_sigma = float(np.clip(auto_canny_sigma, 0.05, 0.95))
        self.min_candidate_area_ratio = float(np.clip(min_candidate_area_ratio, 1e-6, 0.20))
        self.max_candidate_area_ratio = float(np.clip(max_candidate_area_ratio, self.min_candidate_area_ratio, 0.95))
        self.min_depth_support_ratio = float(np.clip(min_depth_support_ratio, 0.01, 1.0))
        self.debug_candidate_scores = bool(debug_candidate_scores)

        self._rng = np.random.default_rng(7)
        self._cached_shape: tuple[int, int] | None = None
        self._cached_intrinsics: CameraIntrinsics | None = None
        self._ray_x: np.ndarray | None = None
        self._ray_y: np.ndarray | None = None
        self._last_edge_bbox: tuple[int, int, int, int] | None = None

    def detect(
        self,
        *,
        color_bgr: np.ndarray,
        color_shape: tuple[int, int],
        depth_raw: np.ndarray,
        intrinsics: CameraIntrinsics | None,
    ) -> DetectionFrame:
        depth_m = self.depth_to_meters(depth_raw)
        depth_m = self.match_depth_to_color(depth_m, color_shape)
        valid_depth = np.isfinite(depth_m) & (depth_m > self.min_depth_m) & (depth_m < self.max_depth_m)

        empty_mask = np.zeros(color_shape, dtype=np.uint8)
        valid_depth_mask = valid_depth.astype(np.uint8) * 255
        ignore_mask = self.build_ignore_mask(color_shape)
        if not np.any(valid_depth):
            return DetectionFrame(
                candidate=None,
                foreground_mask=empty_mask,
                floor_mask=empty_mask,
                edge_mask=empty_mask,
                valid_depth_mask=valid_depth_mask,
                ignore_mask=ignore_mask,
                plane=None,
                used_fallback=True,
                status_text="no valid depth",
                candidate_scores=[],
            )

        plane = None
        floor_mask = np.zeros_like(empty_mask)
        used_fallback = True
        status_text = "depth fallback"
        floor_color_stats = None

        if intrinsics is not None:
            plane = self.estimate_floor_plane(depth_m, intrinsics, valid_depth)

        if plane is not None and intrinsics is not None:
            height_map = self.compute_signed_distance_map(depth_m, intrinsics, plane)
            floor_inliers = valid_depth & (np.abs(height_map) <= self.plane_inlier_threshold_m)
            floor_color_stats = self.compute_color_stats(color_bgr, floor_inliers)
            positive_heights = height_map[valid_depth & (height_map > 0.0)]
            threshold = self.adaptive_height_threshold(
                positive_heights,
                fallback=self.min_height_above_floor_m,
            )
            foreground_mask = (
                valid_depth
                & (height_map > threshold)
                & (height_map < self.max_height_above_floor_m)
            )
            floor_mask = floor_inliers.astype(np.uint8) * 255
            used_fallback = False
            status_text = f"plane segmentation h>{threshold:.3f}m"
        else:
            height_map = self.compute_local_foreground_height(depth_m, valid_depth)
            positive_heights = height_map[valid_depth & (height_map > 0.0)]
            threshold = self.adaptive_height_threshold(
                positive_heights,
                fallback=self.fallback_foreground_margin_m,
            )
            foreground_mask = (
                valid_depth
                & (height_map > threshold)
                & (height_map < self.max_height_above_floor_m)
            )
            floor_color_stats = self.compute_color_stats(color_bgr, valid_depth & ~foreground_mask)
            if intrinsics is not None:
                status_text = f"depth fallback h>{threshold:.3f}m"
            else:
                status_text = f"waiting for camera_info h>{threshold:.3f}m"

        foreground_mask_u8 = self.clean_foreground_mask(foreground_mask.astype(np.uint8) * 255)
        foreground_mask_u8[ignore_mask > 0] = 0
        edge_mask = self.compute_auto_canny(color_bgr)
        candidate, candidate_scores = self.find_best_candidate(
            foreground_mask_u8=foreground_mask_u8,
            depth_m=depth_m,
            height_map=height_map,
            valid_depth=valid_depth,
            color_bgr=color_bgr,
            edge_mask=edge_mask,
            floor_color_stats=floor_color_stats,
            ignore_mask=ignore_mask,
            intrinsics=intrinsics,
            relaxed=False,
        )
        if candidate is not None:
            self._last_edge_bbox = candidate.bbox

        return DetectionFrame(
            candidate=candidate,
            foreground_mask=foreground_mask_u8,
            floor_mask=floor_mask,
            edge_mask=edge_mask,
            valid_depth_mask=valid_depth_mask,
            ignore_mask=ignore_mask,
            plane=plane,
            used_fallback=used_fallback,
            status_text=status_text,
            candidate_scores=candidate_scores,
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

    @staticmethod
    def normalize_ignore_regions(regions: list[float] | None) -> list[tuple[float, float, float, float]]:
        if regions is None:
            regions = [
                0.00, 0.55, 0.16, 1.00,
                0.84, 0.55, 1.00, 1.00,
            ]
        normalized = []
        for index in range(0, len(regions) - 3, 4):
            x0, y0, x1, y1 = [float(v) for v in regions[index:index + 4]]
            x0, x1 = sorted((float(np.clip(x0, 0.0, 1.0)), float(np.clip(x1, 0.0, 1.0))))
            y0, y1 = sorted((float(np.clip(y0, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))))
            if x1 > x0 and y1 > y0:
                normalized.append((x0, y0, x1, y1))
        return normalized

    def build_ignore_mask(self, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if not self.use_ignore_mask:
            return mask
        height, width = shape
        for x0_n, y0_n, x1_n, y1_n in self.ignore_regions_normalized:
            x0 = int(round(x0_n * width))
            x1 = int(round(x1_n * width))
            y0 = int(round(y0_n * height))
            y1 = int(round(y1_n * height))
            mask[y0:y1, x0:x1] = 255
        return mask

    def adaptive_height_threshold(self, values: np.ndarray, *, fallback: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size < 32:
            return float(fallback)
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        robust_sigma = 1.4826 * mad
        lower_object_percentile = float(np.percentile(finite, 35))
        robust_noise_gate = median + 2.5 * robust_sigma
        threshold = max(float(fallback), min(robust_noise_gate, lower_object_percentile))
        return float(np.clip(threshold, fallback, self.max_height_above_floor_m * 0.75))

    def compute_auto_canny(self, color_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        median_gray = float(np.median(gray_blur))
        lower = int(max(0.0, (1.0 - self.auto_canny_sigma) * median_gray))
        upper = int(min(255.0, (1.0 + self.auto_canny_sigma) * median_gray))
        if upper <= lower:
            upper = min(255, lower + 30)
        return cv2.Canny(gray_blur, lower, upper)

    @staticmethod
    def compute_color_stats(color_bgr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray] | None:
        if mask is None or np.count_nonzero(mask) < 64:
            return None
        lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab[mask > 0]
        if pixels.shape[0] < 64:
            return None
        median = np.median(pixels, axis=0)
        mad = np.median(np.abs(pixels - median), axis=0)
        return {
            "median": median,
            "mad": np.maximum(mad, 3.0),
        }

    @staticmethod
    def robust_unit(value: float, *, midpoint: float = 1.0, slope: float = 1.0) -> float:
        return float(1.0 / (1.0 + math.exp(-slope * (value - midpoint))))

    @staticmethod
    def clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def compute_local_foreground_height(self, depth_m: np.ndarray, valid_depth: np.ndarray) -> np.ndarray:
        depth_fill = np.where(valid_depth, depth_m, self.max_depth_m).astype(np.float32)
        background = cv2.GaussianBlur(
            depth_fill,
            (0, 0),
            sigmaX=self.local_background_sigma,
            sigmaY=self.local_background_sigma,
        )
        return np.clip(background - depth_m, 0.0, None)

    def detect_with_edge_fallback(
        self,
        *,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        valid_depth: np.ndarray,
        intrinsics: CameraIntrinsics | None,
    ) -> tuple[DetectionCandidate | None, np.ndarray]:
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        depth_fill = np.where(valid_depth, depth_m, self.max_depth_m).astype(np.float32)
        local_bg = cv2.GaussianBlur(
            depth_fill,
            (0, 0),
            sigmaX=max(21.0, self.local_background_sigma * 0.7),
            sigmaY=max(21.0, self.local_background_sigma * 0.7),
        )
        foreground_depth = valid_depth & ((local_bg - depth_m) > self.fallback_foreground_margin_m)

        depth_mask = np.zeros_like(gray, dtype=np.uint8)
        depth_mask[foreground_depth] = 255

        kernel = np.ones((5, 5), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=1)

        edges = cv2.Canny(gray_blur, 50, 120)
        search_region = cv2.dilate(depth_mask, kernel, iterations=3)
        combined = cv2.bitwise_or(depth_mask, cv2.bitwise_and(edges, search_region))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        candidate = self.find_edge_assisted_candidate(
            combined_mask=combined,
            depth_m=depth_m,
            valid_depth=valid_depth,
            intrinsics=intrinsics,
        )
        return candidate, combined

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
        valid_depth: np.ndarray,
        color_bgr: np.ndarray,
        edge_mask: np.ndarray,
        floor_color_stats: dict[str, np.ndarray] | None,
        ignore_mask: np.ndarray,
        intrinsics: CameraIntrinsics | None,
        relaxed: bool,
    ) -> tuple[DetectionCandidate | None, list[dict[str, float | str | tuple[int, int, int, int]]]]:
        contours, _ = cv2.findContours(foreground_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, []

        working_mask = foreground_mask_u8 > 0
        best_candidate = None
        best_score = -math.inf
        candidate_scores: list[dict[str, float | str | tuple[int, int, int, int]]] = []
        image_h, image_w = foreground_mask_u8.shape[:2]
        image_area = float(max(image_h * image_w, 1))
        min_area = image_area * self.min_candidate_area_ratio
        max_area = image_area * self.max_candidate_area_ratio
        center = np.array([image_w * 0.5, image_h * 0.5], dtype=np.float32)
        center_scale = max(float(np.linalg.norm(center)), 1.0)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = float(max(w * h, 1))
            if bbox_area > max_area * 1.5:
                continue

            bbox_slice = np.s_[y:y + h, x:x + w]
            ignored_pixels = int(np.count_nonzero(ignore_mask[bbox_slice]))
            ignore_overlap = ignored_pixels / bbox_area
            if ignore_overlap > 0.20:
                continue

            contour_mask = np.zeros_like(foreground_mask_u8, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            support_mask = (contour_mask > 0) & working_mask & valid_depth & (ignore_mask == 0)
            support_pixels = int(np.count_nonzero(support_mask))
            depth_support_ratio = support_pixels / bbox_area
            if depth_support_ratio < self.min_depth_support_ratio:
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
            solidity = self.clamp01(area / hull_area)
            extent = self.clamp01(support_pixels / bbox_area)

            median_depth = float(np.median(support_depth))
            median_height = float(np.median(support_height))
            max_height = float(np.percentile(support_height, 90))
            depth_spread = float(np.percentile(support_depth, 90) - np.percentile(support_depth, 10))
            if median_depth <= 0.0 or median_height > self.max_height_above_floor_m:
                continue
            center_pixel = (x + w // 2, y + h // 2)
            center_xyz = self.pixel_to_xyz(center_pixel, depth_m, intrinsics)

            left_xyz = self.pixel_to_xyz((x, center_pixel[1]), depth_m, intrinsics)
            right_xyz = self.pixel_to_xyz((x + w - 1, center_pixel[1]), depth_m, intrinsics)
            width_m = None
            if left_xyz is not None and right_xyz is not None:
                width_m = float(np.linalg.norm(np.array(right_xyz) - np.array(left_xyz)))

            perimeter = float(cv2.arcLength(contour, True))
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            min_rect_area = float(max(rect_w * rect_h, 1.0))
            rectangularity = self.clamp01(area / min_rect_area)
            approx = cv2.approxPolyDP(contour, 0.04 * max(perimeter, 1.0), True)
            corner_score = self.clamp01(1.0 - abs(len(approx) - 4) / 8.0)

            color_score, color_contrast, color_consistency = self.compute_candidate_color_score(
                color_bgr,
                support_mask,
                floor_color_stats,
            )
            edge_score = self.compute_candidate_edge_score(edge_mask, contour_mask, contour)

            normalized_center_distance = float(
                np.linalg.norm(np.array(center_pixel, dtype=np.float32) - center) / center_scale
            )
            center_score = math.exp(-2.5 * normalized_center_distance * normalized_center_distance)

            temporal_score = 0.0
            if self._last_edge_bbox is not None:
                temporal_score = self.compute_iou(self._last_edge_bbox, (x, y, w, h))

            height_score = self.clamp01(median_height / max(self.max_height_above_floor_m * 0.25, 1e-3))
            depth_consistency = math.exp(-depth_spread / max(0.06, median_depth * 0.08))
            area_score = self.clamp01(
                math.log1p(area / max(min_area, 1.0))
                / math.log1p(max_area / max(min_area, 1.0))
            )
            aspect = w / float(max(h, 1))
            aspect_score = math.exp(-abs(math.log(max(aspect, 1e-3))) / math.log(4.0))
            geometry_score = self.clamp01(
                0.18 * area_score
                + 0.18 * depth_support_ratio
                + 0.18 * height_score
                + 0.16 * depth_consistency
                + 0.12 * solidity
                + 0.10 * extent
                + 0.05 * rectangularity
                + 0.03 * aspect_score
            )
            total_weight = max(
                self.geometry_score_weight
                + self.color_score_weight
                + self.edge_score_weight
                + self.center_prior_weight
                + self.temporal_score_weight,
                1e-6,
            )
            normalized_score = (
                self.geometry_score_weight * geometry_score
                + self.color_score_weight * color_score
                + self.edge_score_weight * edge_score
                + self.center_prior_weight * center_score
                + self.temporal_score_weight * temporal_score
            ) / total_weight
            score = float(normalized_score * 100.0)
            confidence = float(np.clip(
                0.15 + 0.80 * normalized_score,
                0.0,
                1.0,
            ))

            score_components = {
                "geometry": float(geometry_score),
                "color": float(color_score),
                "edge": float(edge_score),
                "center": float(center_score),
                "temporal": float(temporal_score),
                "contrast": float(color_contrast),
                "color_consistency": float(color_consistency),
                "depth_support": float(depth_support_ratio),
                "height": float(median_height),
                "depth_spread": float(depth_spread),
                "solidity": float(solidity),
                "extent": float(extent),
                "rectangularity": float(rectangularity),
                "corners": float(corner_score),
            }
            candidate_scores.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "score": float(score),
                "confidence": float(confidence),
                **score_components,
            })

            if score <= best_score:
                continue

            best_score = score
            best_candidate = DetectionCandidate(
                bbox=(int(x), int(y), int(w), int(h)),
                contour=contour,
                mask=contour_mask,
                center_pixel=center_pixel,
                center_xyz=center_xyz,
                left_xyz=left_xyz,
                right_xyz=right_xyz,
                distance_m=median_depth,
                median_height_m=median_height,
                max_height_m=max_height,
                width_m=width_m,
                confidence=confidence,
                score=float(score),
                support_pixels=support_pixels,
                score_components=score_components,
            )

        candidate_scores.sort(key=lambda item: float(item["score"]), reverse=True)
        return best_candidate, candidate_scores[:8]

    def compute_candidate_color_score(
        self,
        color_bgr: np.ndarray,
        support_mask: np.ndarray,
        floor_color_stats: dict[str, np.ndarray] | None,
    ) -> tuple[float, float, float]:
        stats = self.compute_color_stats(color_bgr, support_mask)
        if stats is None:
            return 0.35, 0.0, 0.0

        candidate_median = stats["median"]
        candidate_mad = stats["mad"]
        consistency = float(np.mean(np.exp(-candidate_mad / np.array([18.0, 14.0, 14.0], dtype=np.float32))))
        if floor_color_stats is None:
            return 0.45 * consistency, 0.0, consistency

        floor_median = floor_color_stats["median"]
        floor_mad = floor_color_stats["mad"]
        pooled_scale = np.maximum(floor_mad + candidate_mad, 6.0)
        lab_distance = float(np.linalg.norm((candidate_median - floor_median) / pooled_scale))
        contrast = self.robust_unit(lab_distance, midpoint=1.25, slope=1.4)
        score = self.clamp01(0.68 * contrast + 0.32 * consistency)
        return score, contrast, consistency

    def compute_candidate_edge_score(
        self,
        edge_mask: np.ndarray,
        contour_mask: np.ndarray,
        contour: np.ndarray,
    ) -> float:
        boundary = np.zeros_like(edge_mask, dtype=np.uint8)
        cv2.drawContours(boundary, [contour], -1, 255, thickness=2)
        boundary_pixels = int(np.count_nonzero(boundary))
        if boundary_pixels == 0:
            return 0.0
        boundary_support = np.count_nonzero((edge_mask > 0) & (boundary > 0)) / float(boundary_pixels)
        inside_edges = np.count_nonzero((edge_mask > 0) & (contour_mask > 0))
        inside_area = max(np.count_nonzero(contour_mask), 1)
        interior_texture = min(1.0, inside_edges / float(inside_area) * 12.0)
        return self.clamp01(0.75 * min(1.0, boundary_support * 4.0) + 0.25 * interior_texture)

    def find_edge_assisted_candidate(
        self,
        *,
        combined_mask: np.ndarray,
        depth_m: np.ndarray,
        valid_depth: np.ndarray,
        intrinsics: CameraIntrinsics | None,
    ) -> DetectionCandidate | None:
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_candidate = None
        best_score = -math.inf
        image_h, image_w = combined_mask.shape[:2]
        image_area = float(image_h * image_w)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(250.0, self.min_contour_area):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < max(30, self.min_bbox_size_px) or h < max(30, self.min_bbox_size_px):
                continue

            bbox_area = float(w * h)
            if bbox_area > image_area * 0.35:
                continue
            if w > image_w * 0.85 or h > image_h * 0.85:
                continue
            if x <= 2 or y <= 2 or (x + w) >= image_w - 2:
                continue

            aspect = w / float(max(h, 1))
            if aspect < 0.3 or aspect > max(3.5, self.max_bbox_aspect_ratio):
                continue

            contour_mask = np.zeros_like(combined_mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            support_mask = (contour_mask > 0) & valid_depth
            if not np.any(support_mask):
                continue

            support_depth = depth_m[support_mask]
            support_depth = support_depth[np.isfinite(support_depth)]
            if support_depth.size == 0:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(max(cv2.contourArea(hull), 1.0))
            solidity = area / hull_area
            if solidity < 0.20:
                continue

            extent = area / max(bbox_area, 1.0)
            if extent < 0.35:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, 0.04 * max(perimeter, 1.0), True)
            corner_score = max(0.0, 1.0 - abs(len(approx) - 4) / 6.0)

            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            min_rect_area = float(max(rect_w * rect_h, 1.0))
            rectangularity = area / min_rect_area
            if rectangularity < 0.45:
                continue

            median_depth = float(np.median(support_depth))
            depth_spread = float(np.percentile(support_depth, 90) - np.percentile(support_depth, 10))
            if depth_spread > 0.20:
                continue
            center_pixel = (x + w // 2, y + h // 2)
            center_xyz = self.pixel_to_xyz(center_pixel, depth_m, intrinsics)

            left_xyz = self.pixel_to_xyz((x, center_pixel[1]), depth_m, intrinsics)
            right_xyz = self.pixel_to_xyz((x + w - 1, center_pixel[1]), depth_m, intrinsics)
            width_m = None
            if left_xyz is not None and right_xyz is not None:
                width_m = float(np.linalg.norm(np.array(right_xyz) - np.array(left_xyz)))

            support_pixels = int(np.count_nonzero(support_mask))
            temporal_bonus = 1.0
            if self._last_edge_bbox is not None:
                temporal_bonus += 1.5 * self.compute_iou(self._last_edge_bbox, (x, y, w, h))

            confidence = float(np.clip(
                0.25
                + 0.30 * min(1.0, support_pixels / 4000.0)
                + 0.20 * min(1.0, area / 6000.0)
                + 0.10 * solidity,
                0.0,
                0.95,
            ))
            score = (
                area
                * max(solidity, 0.1)
                * max(extent, 0.1)
                * max(rectangularity, 0.1)
                * max(corner_score, 0.2)
                * temporal_bonus
                / max(median_depth, 1e-3)
                / max(0.03, depth_spread + 0.03)
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
                left_xyz=left_xyz,
                right_xyz=right_xyz,
                distance_m=median_depth,
                median_height_m=0.0,
                max_height_m=0.0,
                width_m=width_m,
                confidence=confidence,
                score=float(score),
                support_pixels=support_pixels,
            )

        return best_candidate

    @staticmethod
    def compute_iou(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
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

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = float(inter_w * inter_h)
        union = float(aw * ah + bw * bh) - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def get_candidate_thresholds(self, *, relaxed: bool) -> dict[str, float]:
        if not relaxed:
            return {
                "min_contour_area": float(self.min_contour_area),
                "min_support_pixels": float(self.min_contour_area),
                "min_bbox_size_px": float(self.min_bbox_size_px),
                "max_bbox_aspect_ratio": float(self.max_bbox_aspect_ratio),
                "min_extent": float(self.min_extent),
                "min_solidity": float(self.min_solidity),
            }

        return {
            "min_contour_area": float(max(250, int(self.min_contour_area * 0.30))),
            "min_support_pixels": float(max(250, int(self.min_contour_area * 0.30))),
            "min_bbox_size_px": float(max(14, int(self.min_bbox_size_px * 0.60))),
            "max_bbox_aspect_ratio": float(max(self.max_bbox_aspect_ratio, self.max_bbox_aspect_ratio * 1.6)),
            "min_extent": float(max(0.08, self.min_extent * 0.50)),
            "min_solidity": float(max(0.20, self.min_solidity * 0.55)),
        }

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
