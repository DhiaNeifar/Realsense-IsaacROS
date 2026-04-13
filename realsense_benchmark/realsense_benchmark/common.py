from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import cv2
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


DEFAULT_RESULTS_DIR = Path.home() / "ros2_ws" / "src" / "realsense_benchmark" / "results"


class RollingFPS:
    """Track a rolling frames-per-second estimate over recent timestamps."""

    def __init__(self, window_size: int = 60) -> None:
        self.timestamps = deque(maxlen=window_size)

    def tick(self) -> None:
        self.timestamps.append(time.perf_counter())

    def get(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0

        dt = self.timestamps[-1] - self.timestamps[0]
        if dt <= 0:
            return 0.0

        return (len(self.timestamps) - 1) / dt


class RollingMean:
    """Track a rolling mean over recent scalar values."""

    def __init__(self, window_size: int = 60) -> None:
        self.values = deque(maxlen=window_size)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def get(self) -> float:
        if not self.values:
            return 0.0

        return sum(self.values) / len(self.values)


def image_qos_profile(
    *,
    depth: int = 10,
    reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE,
) -> QoSProfile:
    """Create a standard image-stream QoS profile for camera topics."""

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


def ensure_directory(path: str | Path) -> str:
    """Create the directory if needed and return it as a string."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def probe_display_available(show_window: bool) -> bool:
    """Return True when an OpenCV window can likely be opened."""

    if not show_window:
        return False

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False

    try:
        cv2.namedWindow("_probe", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("_probe", 1, 1)
        cv2.waitKey(1)
        cv2.destroyWindow("_probe")
        return True
    except Exception:
        return False


def destroy_opencv_windows(show_window: bool, display_available: bool) -> None:
    """Close any OpenCV windows only when the node actually used them."""

    if show_window and display_available:
        cv2.destroyAllWindows()
