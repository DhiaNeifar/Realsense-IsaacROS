from __future__ import annotations

from builtin_interfaces.msg import Duration
import cv2
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray

from floor_object_detection.detector import DetectionFrame
from floor_object_detection.tracking import TrackedDetection


def draw_debug_image(
    color_image: np.ndarray,
    detection_frame: DetectionFrame,
    tracked_detection: TrackedDetection | None,
) -> np.ndarray:
    debug = color_image.copy()

    if detection_frame.floor_mask.size > 0:
        floor_overlay = np.zeros_like(debug)
        floor_overlay[:, :, 0] = detection_frame.floor_mask
        debug = cv2.addWeighted(debug, 1.0, floor_overlay, 0.15, 0.0)

    if detection_frame.foreground_mask.size > 0:
        foreground_overlay = np.zeros_like(debug)
        foreground_overlay[:, :, 1] = detection_frame.foreground_mask
        debug = cv2.addWeighted(debug, 1.0, foreground_overlay, 0.30, 0.0)

    if tracked_detection is not None:
        x, y, w, h = [int(round(v)) for v in tracked_detection.bbox]
        color = (0, 255, 0) if not tracked_detection.stale else (0, 200, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)

        cx, cy = [int(round(v)) for v in tracked_detection.center_pixel]
        cv2.circle(debug, (cx, cy), 5, color, -1)

        status = "stable" if not tracked_detection.stale else "stale"
        line_1 = (
            f"id={tracked_detection.track_id} "
            f"z={tracked_detection.distance_m:.2f}m "
            f"h={tracked_detection.median_height_m:.2f}m "
            f"conf={tracked_detection.confidence:.2f}"
        )
        line_2 = f"{detection_frame.status_text} | {status}"
        cv2.putText(debug, line_1, (x, max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(debug, line_2, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
    elif detection_frame.candidate is not None:
        x, y, w, h = detection_frame.candidate.bbox
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.putText(
            debug,
            f"tentative z={detection_frame.candidate.distance_m:.2f}m",
            (x, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    header_lines = [
        "Floor Object Detection",
        f"mode: {detection_frame.status_text}",
        f"plane: {'yes' if detection_frame.plane is not None else 'no'}",
    ]
    y = 28
    for line in header_lines:
        cv2.putText(debug, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24

    return debug


def create_marker_array(
    *,
    header,
    tracked_detection: TrackedDetection | None,
    marker_lifetime_sec: float,
) -> MarkerArray:
    marker_array = MarkerArray()

    if tracked_detection is None or tracked_detection.center_xyz is None:
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        return marker_array

    center_x, center_y, center_z = tracked_detection.center_xyz
    color_rgb = (0.15, 0.90, 0.20) if not tracked_detection.stale else (0.95, 0.70, 0.10)
    lifetime = duration_from_seconds(marker_lifetime_sec)

    center_marker = Marker()
    center_marker.header = header
    center_marker.ns = "floor_object"
    center_marker.id = 0
    center_marker.type = Marker.SPHERE
    center_marker.action = Marker.ADD
    center_marker.pose.position.x = float(center_x)
    center_marker.pose.position.y = float(center_y)
    center_marker.pose.position.z = float(center_z)
    center_marker.pose.orientation.w = 1.0
    center_marker.scale.x = 0.06
    center_marker.scale.y = 0.06
    center_marker.scale.z = 0.06
    center_marker.color.r = color_rgb[0]
    center_marker.color.g = color_rgb[1]
    center_marker.color.b = color_rgb[2]
    center_marker.color.a = 0.95
    center_marker.lifetime = lifetime
    marker_array.markers.append(center_marker)

    text_marker = Marker()
    text_marker.header = header
    text_marker.ns = "floor_object"
    text_marker.id = 1
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = float(center_x)
    text_marker.pose.position.y = float(center_y)
    text_marker.pose.position.z = float(center_z + 0.10)
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.z = 0.05
    text_marker.color.r = 1.0
    text_marker.color.g = 1.0
    text_marker.color.b = 1.0
    text_marker.color.a = 0.95
    text_marker.text = (
        f"id={tracked_detection.track_id} "
        f"z={tracked_detection.distance_m:.2f}m "
        f"conf={tracked_detection.confidence:.2f}"
    )
    text_marker.lifetime = lifetime
    marker_array.markers.append(text_marker)

    return marker_array


def duration_from_seconds(seconds: float) -> Duration:
    clamped = max(0.0, float(seconds))
    sec = int(clamped)
    nanosec = int((clamped - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)
