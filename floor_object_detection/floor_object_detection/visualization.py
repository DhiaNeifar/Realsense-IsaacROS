from __future__ import annotations

from builtin_interfaces.msg import Duration
import cv2
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray

from floor_object_detection.detector import DetectionFrame
from floor_object_detection.tracking import TrackedDetection


def draw_debug_image(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    detection_frame: DetectionFrame,
    tracked_detection: TrackedDetection | None,
    rgb_tracked_detection: TrackedDetection | None = None,
) -> np.ndarray:
    rgb_panel = color_image.copy()
    depth_panel = colorize_depth(depth_image)
    edge_panel = cv2.cvtColor(detection_frame.edge_mask, cv2.COLOR_GRAY2BGR)
    rgb_detection = rgb_tracked_detection if rgb_tracked_detection is not None else tracked_detection

    if detection_frame.floor_mask.size > 0:
        floor_overlay = np.zeros_like(rgb_panel)
        floor_overlay[:, :, 0] = detection_frame.floor_mask
        rgb_panel = cv2.addWeighted(rgb_panel, 1.0, floor_overlay, 0.15, 0.0)

    if detection_frame.ignore_mask.size > 0 and np.count_nonzero(detection_frame.ignore_mask) > 0:
        ignore_overlay = np.zeros_like(rgb_panel)
        ignore_overlay[:, :, 2] = detection_frame.ignore_mask
        rgb_panel = cv2.addWeighted(rgb_panel, 1.0, ignore_overlay, 0.25, 0.0)
        edge_panel = cv2.addWeighted(edge_panel, 1.0, ignore_overlay, 0.20, 0.0)

    if detection_frame.foreground_mask.size > 0:
        foreground_overlay = np.zeros_like(rgb_panel)
        foreground_overlay[:, :, 1] = detection_frame.foreground_mask
        rgb_panel = cv2.addWeighted(rgb_panel, 1.0, foreground_overlay, 0.30, 0.0)
        edge_overlay = np.zeros_like(edge_panel)
        edge_overlay[:, :, 1] = detection_frame.foreground_mask
        edge_panel = cv2.addWeighted(edge_panel, 1.0, edge_overlay, 0.35, 0.0)

    for index, item in enumerate(detection_frame.candidate_scores[:5]):
        x, y, w, h = item["bbox"]
        color = (255, 180, 0) if index > 0 else (0, 255, 255)
        thickness = 2 if index == 0 else 1
        cv2.rectangle(edge_panel, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            edge_panel,
            f"{float(item['score']):.0f}",
            (x, max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if tracked_detection is not None:
        x, y, w, h = [int(round(v)) for v in tracked_detection.bbox]
        color = (0, 255, 0) if not tracked_detection.stale else (0, 200, 255)
        if detection_frame.candidate is not None:
            cv2.drawContours(rgb_panel, [detection_frame.candidate.contour], -1, color, 2)
            cv2.drawContours(depth_panel, [detection_frame.candidate.contour], -1, color, 2)
            cv2.drawContours(edge_panel, [detection_frame.candidate.contour], -1, color, 2)
        if rgb_detection is not None:
            rx, ry, rw, rh = [int(round(v)) for v in rgb_detection.bbox]
            cv2.rectangle(rgb_panel, (rx, ry), (rx + rw, ry + rh), color, 2)
            rcx, rcy = [int(round(v)) for v in rgb_detection.center_pixel]
            cv2.circle(rgb_panel, (rcx, rcy), 5, color, -1)
        cv2.rectangle(rgb_panel, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(depth_panel, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(edge_panel, (x, y), (x + w, y + h), color, 2)

        cx, cy = [int(round(v)) for v in tracked_detection.center_pixel]
        cv2.circle(depth_panel, (cx, cy), 5, color, -1)
        cv2.circle(edge_panel, (cx, cy), 5, color, -1)

        status = "stable" if not tracked_detection.stale else "stale"
        line_1 = (
            f"id={tracked_detection.track_id} "
            f"z={tracked_detection.distance_m:.2f}m "
            f"h={tracked_detection.median_height_m:.2f}m "
            f"conf={tracked_detection.confidence:.2f}"
        )
        line_2 = f"{detection_frame.status_text} | {status}"
        line_3 = score_summary(detection_frame)
        if rgb_detection is not None:
            cv2.putText(
                rgb_panel,
                line_1,
                (rx, max(24, ry - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb_panel,
                line_2,
                (rx, ry + rh + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
                cv2.LINE_AA,
            )
            if line_3:
                cv2.putText(
                    rgb_panel,
                    line_3,
                    (rx, ry + rh + 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(depth_panel, line_1, (x, max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(depth_panel, line_2, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        if line_3:
            cv2.putText(depth_panel, line_3, (x, y + h + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(edge_panel, line_1, (x, max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(edge_panel, line_2, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        if line_3:
            cv2.putText(edge_panel, line_3, (x, y + h + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    elif detection_frame.candidate is not None:
        x, y, w, h = detection_frame.candidate.bbox
        cv2.drawContours(rgb_panel, [detection_frame.candidate.contour], -1, (0, 200, 255), 2)
        cv2.drawContours(depth_panel, [detection_frame.candidate.contour], -1, (0, 200, 255), 2)
        cv2.drawContours(edge_panel, [detection_frame.candidate.contour], -1, (0, 200, 255), 2)
        cv2.rectangle(rgb_panel, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.rectangle(depth_panel, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.rectangle(edge_panel, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.putText(
            rgb_panel,
            f"tentative z={detection_frame.candidate.distance_m:.2f}m",
            (x, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            depth_panel,
            f"tentative z={detection_frame.candidate.distance_m:.2f}m",
            (x, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            edge_panel,
            f"tentative z={detection_frame.candidate.distance_m:.2f}m",
            (x, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        line_3 = score_summary(detection_frame)
        if line_3:
            draw_small_text(rgb_panel, line_3, (x, y + h + 22), (0, 200, 255))
            draw_small_text(depth_panel, line_3, (x, y + h + 22), (0, 200, 255))
            draw_small_text(edge_panel, line_3, (x, y + h + 22), (0, 200, 255))

    annotate_panel(
        rgb_panel,
        [
            "RGB",
            f"mode: {detection_frame.status_text}",
            f"plane: {'yes' if detection_frame.plane is not None else 'no'}",
        ],
    )
    annotate_panel(
        depth_panel,
        [
            "Depth",
            "colormap: TURBO",
            f"valid px: {int(np.count_nonzero(detection_frame.valid_depth_mask))}",
            f"foreground px: {int(np.count_nonzero(detection_frame.foreground_mask))}",
        ],
    )
    annotate_panel(
        edge_panel,
        [
            "Edges",
            f"edge px: {int(np.count_nonzero(detection_frame.edge_mask))}",
            f"ignore px: {int(np.count_nonzero(detection_frame.ignore_mask))}",
            f"candidate: {'yes' if detection_frame.candidate is not None else 'no'}",
        ],
    )

    return np.hstack((rgb_panel, depth_panel, edge_panel))


def draw_preview_image(
    color_image: np.ndarray,
    depth_image: np.ndarray | None,
    status_lines: list[str],
) -> np.ndarray:
    rgb_panel = color_image.copy()

    if depth_image is None:
        depth_panel = np.zeros_like(rgb_panel)
        edge_panel = np.zeros_like(rgb_panel)
    else:
        depth_panel = colorize_depth(depth_image)
        edge_panel = np.zeros_like(depth_panel)

    annotate_panel(rgb_panel, ["RGB"] + status_lines[:2])
    annotate_panel(depth_panel, ["Depth"] + status_lines[:2])
    annotate_panel(edge_panel, ["Edges"] + status_lines[:2])
    return np.hstack((rgb_panel, depth_panel, edge_panel))


def colorize_depth(depth_image: np.ndarray) -> np.ndarray:
    depth = depth_image.astype(np.float32, copy=False)
    finite_mask = np.isfinite(depth) & (depth > 0.0)

    if not np.any(finite_mask):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    finite_values = depth[finite_mask]
    near = float(np.percentile(finite_values, 5))
    far = float(np.percentile(finite_values, 95))
    if far - near < 1e-6:
        far = near + 1.0

    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[finite_mask] = (depth[finite_mask] - near) / (far - near)
    normalized = np.clip(1.0 - normalized, 0.0, 1.0)
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    colorized = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colorized[~finite_mask] = 0
    return colorized


def annotate_panel(image: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24


def draw_small_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def score_summary(detection_frame: DetectionFrame) -> str:
    candidate = detection_frame.candidate
    if candidate is None or not candidate.score_components:
        return ""
    components = candidate.score_components
    return (
        f"score={candidate.score:.1f} "
        f"g={components['geometry']:.2f} "
        f"c={components['color']:.2f} "
        f"e={components['edge']:.2f} "
        f"ctr={components['center']:.2f}"
    )


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
