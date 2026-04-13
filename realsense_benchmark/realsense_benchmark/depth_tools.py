from __future__ import annotations

import time

import cv2
import numpy as np


def depth_to_meters(depth_raw: np.ndarray, depth_scale: float = 0.001) -> np.ndarray:
    """Convert a raw depth frame to meters."""

    if depth_raw.dtype == np.uint16:
        return depth_raw.astype(np.float32) * depth_scale

    return depth_raw.astype(np.float32)


def render_depth_band_overlay(
    depth_raw: np.ndarray,
    *,
    band_min_m: float,
    band_max_m: float,
    cpu_loops: int = 0,
    depth_scale: float = 0.001,
) -> tuple[np.ndarray, float]:
    """Render a colorized depth image with a highlighted depth band."""

    start_time = time.perf_counter()

    depth_m = depth_to_meters(depth_raw, depth_scale=depth_scale)
    valid_depth = np.isfinite(depth_m)

    mask = np.zeros(depth_m.shape, dtype=np.uint8)
    mask[(valid_depth) & (depth_m >= band_min_m) & (depth_m <= band_max_m)] = 255

    for _ in range(cpu_loops):
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.Sobel(mask, cv2.CV_8U, 1, 0, ksize=3)

    depth_vis = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    overlay = np.zeros_like(depth_vis)
    overlay[:, :, 1] = mask
    combined = cv2.addWeighted(depth_vis, 1.0, overlay, 0.45, 0.0)

    processing_time_ms = (time.perf_counter() - start_time) * 1000.0
    return combined, processing_time_ms
