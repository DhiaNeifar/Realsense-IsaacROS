#!/usr/bin/env python3

from __future__ import annotations

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray
import numpy as np

from floor_object_detection.detector import CameraIntrinsics, DepthBasedFloorObjectDetector
from floor_object_detection.tracking import TrackedDetection
from floor_object_detection.tracking import TemporalDetectionTracker
from floor_object_detection.visualization import create_marker_array, draw_debug_image, draw_preview_image


class FloorObjectDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("floor_object_detector")

        self.bridge = CvBridge()
        self._color_camera_info: CameraInfo | None = None
        self._depth_camera_info: CameraInfo | None = None
        self._window_disabled = False
        self._last_log_ns = 0
        self._last_candidate_score_log_ns = 0
        self._last_published_log_ns = 0
        self._last_pipeline_error_ns = 0
        self._latest_display_image: np.ndarray | None = None
        self._raw_color_image: np.ndarray | None = None
        self._latest_depth_msg: Image | None = None
        self._preview_frame_count = 0
        self._processed_frame_count = 0
        self._last_waiting_log_ns = 0

        self.declare_parameter("color_topic", "/d435i/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/d435i/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/d435i/camera/color/camera_info")
        self.declare_parameter("depth_camera_info_topic", "/d435i/camera/depth/camera_info")
        self.declare_parameter("sync_slop_sec", 0.08)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 2.00)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("plane_ransac_iterations", 80)
        self.declare_parameter("plane_inlier_threshold_m", 0.02)
        self.declare_parameter("min_floor_points", 180)
        self.declare_parameter("min_plane_y_component", 0.35)
        self.declare_parameter("plane_sample_stride", 4)
        self.declare_parameter("min_height_above_floor_m", 0.015)
        self.declare_parameter("max_height_above_floor_m", 0.60)
        self.declare_parameter("fallback_foreground_margin_m", 0.025)
        self.declare_parameter("local_background_sigma", 31.0)
        self.declare_parameter("search_top_ignore_ratio", 0.10)
        self.declare_parameter("min_contour_area", 600)
        self.declare_parameter("min_bbox_size_px", 18)
        self.declare_parameter("max_bbox_aspect_ratio", 4.0)
        self.declare_parameter("min_extent", 0.10)
        self.declare_parameter("min_solidity", 0.30)
        self.declare_parameter("open_kernel_size", 3)
        self.declare_parameter("close_kernel_size", 5)
        self.declare_parameter("point_depth_window", 5)
        self.declare_parameter("use_ignore_mask", True)
        self.declare_parameter(
            "ignore_regions_normalized",
            [
                0.00, 0.55, 0.16, 1.00,
                0.84, 0.55, 1.00, 1.00,
            ],
        )
        self.declare_parameter("center_prior_weight", 0.12)
        self.declare_parameter("color_score_weight", 0.22)
        self.declare_parameter("edge_score_weight", 0.16)
        self.declare_parameter("geometry_score_weight", 0.42)
        self.declare_parameter("temporal_score_weight", 0.08)
        self.declare_parameter("auto_canny_sigma", 0.33)
        self.declare_parameter("min_candidate_area_ratio", 0.0008)
        self.declare_parameter("max_candidate_area_ratio", 0.35)
        self.declare_parameter("min_depth_support_ratio", 0.35)
        self.declare_parameter("debug_candidate_scores", False)
        self.declare_parameter("min_confirmed_frames", 1)
        self.declare_parameter("max_missed_frames", 4)
        self.declare_parameter("bbox_smoothing_alpha", 0.18)
        self.declare_parameter("center_smoothing_alpha", 0.15)
        self.declare_parameter("max_center_jump_px", 140.0)
        self.declare_parameter("max_depth_jump_m", 0.30)
        self.declare_parameter("reinit_after_incompatible_frames", 8)
        self.declare_parameter("marker_lifetime_sec", 0.25)
        self.declare_parameter("show_debug_window", True)
        self.declare_parameter("window_name", "floor_object_detector")
        self.declare_parameter("log_detection_period_sec", 1.0)
        self.declare_parameter("display_fps", 30.0)
        self.declare_parameter("max_color_depth_age_sec", 1.0)
        self.declare_parameter("depth_to_color_translation", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "depth_to_color_rotation",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )

        color_topic = str(self.get_parameter("color_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        depth_camera_info_topic = str(self.get_parameter("depth_camera_info_topic").value)
        sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)

        self.log_detection_period_ns = int(float(self.get_parameter("log_detection_period_sec").value) * 1e9)
        self.marker_lifetime_sec = float(self.get_parameter("marker_lifetime_sec").value)
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_period_sec = 1.0 / max(1.0, float(self.get_parameter("display_fps").value))
        self.max_color_depth_age_ns = int(float(self.get_parameter("max_color_depth_age_sec").value) * 1e9)
        self.depth_to_color_translation = np.array(
            list(self.get_parameter("depth_to_color_translation").value),
            dtype=np.float32,
        ).reshape(3)
        self.depth_to_color_rotation = np.array(
            list(self.get_parameter("depth_to_color_rotation").value),
            dtype=np.float32,
        ).reshape(3, 3)

        self.detector = DepthBasedFloorObjectDetector(
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_depth_m=float(self.get_parameter("max_depth_m").value),
            depth_scale=float(self.get_parameter("depth_scale").value),
            plane_ransac_iterations=int(self.get_parameter("plane_ransac_iterations").value),
            plane_inlier_threshold_m=float(self.get_parameter("plane_inlier_threshold_m").value),
            min_floor_points=int(self.get_parameter("min_floor_points").value),
            min_plane_y_component=float(self.get_parameter("min_plane_y_component").value),
            plane_sample_stride=int(self.get_parameter("plane_sample_stride").value),
            min_height_above_floor_m=float(self.get_parameter("min_height_above_floor_m").value),
            max_height_above_floor_m=float(self.get_parameter("max_height_above_floor_m").value),
            fallback_foreground_margin_m=float(self.get_parameter("fallback_foreground_margin_m").value),
            local_background_sigma=float(self.get_parameter("local_background_sigma").value),
            search_top_ignore_ratio=float(self.get_parameter("search_top_ignore_ratio").value),
            min_contour_area=int(self.get_parameter("min_contour_area").value),
            min_bbox_size_px=int(self.get_parameter("min_bbox_size_px").value),
            max_bbox_aspect_ratio=float(self.get_parameter("max_bbox_aspect_ratio").value),
            min_extent=float(self.get_parameter("min_extent").value),
            min_solidity=float(self.get_parameter("min_solidity").value),
            open_kernel_size=int(self.get_parameter("open_kernel_size").value),
            close_kernel_size=int(self.get_parameter("close_kernel_size").value),
            point_depth_window=int(self.get_parameter("point_depth_window").value),
            use_ignore_mask=bool(self.get_parameter("use_ignore_mask").value),
            ignore_regions_normalized=list(self.get_parameter("ignore_regions_normalized").value),
            center_prior_weight=float(self.get_parameter("center_prior_weight").value),
            color_score_weight=float(self.get_parameter("color_score_weight").value),
            edge_score_weight=float(self.get_parameter("edge_score_weight").value),
            geometry_score_weight=float(self.get_parameter("geometry_score_weight").value),
            temporal_score_weight=float(self.get_parameter("temporal_score_weight").value),
            auto_canny_sigma=float(self.get_parameter("auto_canny_sigma").value),
            min_candidate_area_ratio=float(self.get_parameter("min_candidate_area_ratio").value),
            max_candidate_area_ratio=float(self.get_parameter("max_candidate_area_ratio").value),
            min_depth_support_ratio=float(self.get_parameter("min_depth_support_ratio").value),
            debug_candidate_scores=bool(self.get_parameter("debug_candidate_scores").value),
        )

        self.tracker = TemporalDetectionTracker(
            min_confirmed_frames=int(self.get_parameter("min_confirmed_frames").value),
            max_missed_frames=int(self.get_parameter("max_missed_frames").value),
            bbox_smoothing_alpha=float(self.get_parameter("bbox_smoothing_alpha").value),
            center_smoothing_alpha=float(self.get_parameter("center_smoothing_alpha").value),
            max_center_jump_px=float(self.get_parameter("max_center_jump_px").value),
            max_depth_jump_m=float(self.get_parameter("max_depth_jump_m").value),
            reinit_after_incompatible_frames=int(self.get_parameter("reinit_after_incompatible_frames").value),
        )

        self.debug_pub = self.create_publisher(Image, "~/debug_image", 10)
        self.foreground_mask_pub = self.create_publisher(Image, "~/foreground_mask", 10)
        self.floor_mask_pub = self.create_publisher(Image, "~/floor_mask", 10)
        self.distance_pub = self.create_publisher(Float32, "~/distance", 10)
        self.detections_pub = self.create_publisher(Detection2DArray, "~/detections_2d", 10)
        self.center_pub = self.create_publisher(PointStamped, "~/box_center", 10)
        self.left_pub = self.create_publisher(PointStamped, "~/left_point", 10)
        self.right_pub = self.create_publisher(PointStamped, "~/right_point", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "~/box_pose", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/marker_array", 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.color_camera_info_callback,
            qos_profile_sensor_data,
        )
        self.depth_camera_info_sub = self.create_subscription(
            CameraInfo,
            depth_camera_info_topic,
            self.depth_camera_info_callback,
            qos_profile_sensor_data,
        )
        self.preview_sub = self.create_subscription(
            Image,
            color_topic,
            self.preview_callback,
            qos_profile_sensor_data,
        )
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.display_timer = None

        if self.show_debug_window:
            self.initialize_debug_window()

        self.get_logger().info(f"Subscribing to color: {color_topic}")
        self.get_logger().info(f"Subscribing to depth: {depth_topic}")
        self.get_logger().info(f"Subscribing to camera info: {camera_info_topic}")
        self.get_logger().info(f"Subscribing to depth camera info: {depth_camera_info_topic}")

    def color_camera_info_callback(self, msg: CameraInfo) -> None:
        self._color_camera_info = msg

    def depth_camera_info_callback(self, msg: CameraInfo) -> None:
        self._depth_camera_info = msg

    def preview_callback(self, color_msg: Image) -> None:
        try:
            preview_image = self.ros_image_to_bgr(color_msg)
        except Exception as exc:
            self.get_logger().error(f"Preview image conversion failed: {exc}")
            return

        self._raw_color_image = preview_image
        self._latest_display_image = self.build_preview_display(preview_image)
        self._preview_frame_count += 1

        if self._preview_frame_count == 1:
            self.get_logger().info(
                f"Preview stream active: {color_msg.width}x{color_msg.height} encoding={color_msg.encoding}"
            )

        self.process_with_latest_depth(color_msg, preview_image)

    def depth_callback(self, depth_msg: Image) -> None:
        self._latest_depth_msg = depth_msg

    def process_with_latest_depth(self, color_msg: Image, color_image: np.ndarray) -> None:
        if self._latest_depth_msg is None:
            self.log_waiting_state("waiting for depth")
            return

        color_stamp_ns = self.stamp_to_nanoseconds(color_msg.header.stamp)
        depth_stamp_ns = self.stamp_to_nanoseconds(self._latest_depth_msg.header.stamp)
        if color_stamp_ns != 0 and depth_stamp_ns != 0:
            age_ns = abs(color_stamp_ns - depth_stamp_ns)
            if age_ns > self.max_color_depth_age_ns:
                self.log_waiting_state(
                    f"waiting for closer depth ({age_ns / 1e9:.3f}s > {self.max_color_depth_age_ns / 1e9:.3f}s)"
                )
                self._latest_display_image = self.build_preview_display(color_image, self._latest_depth_msg)
                return

        self.process_frame(color_msg, color_image, self._latest_depth_msg)

    def initialize_debug_window(self) -> None:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.display_timer = self.create_timer(self.display_period_sec, self.display_latest_image)
        except cv2.error as exc:
            self.get_logger().warn(f"Disabling OpenCV debug window: {exc}")
            self._window_disabled = True

    def process_frame(self, color_msg: Image, color: np.ndarray, depth_msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        self._processed_frame_count += 1
        if self._processed_frame_count == 1:
            self.get_logger().info(
                f"Synchronized stream active: color={color_msg.width}x{color_msg.height} {color_msg.encoding}, "
                f"depth={depth_msg.width}x{depth_msg.height} {depth_msg.encoding}"
            )

        self._latest_display_image = self.build_preview_display(color, depth_msg)

        try:
            intrinsics = self.camera_info_to_intrinsics(self._color_camera_info)
            detection_frame = self.detector.detect(
                color_bgr=color,
                color_shape=color.shape[:2],
                depth_raw=depth,
                intrinsics=intrinsics,
            )
            tracked_detection = self.tracker.update(detection_frame.candidate)
            rgb_tracked_detection = self.reproject_tracked_detection_to_color(
                tracked_detection=tracked_detection,
                depth_raw=depth,
                color_shape=color.shape[:2],
            )

            debug_image = draw_debug_image(
                color,
                depth,
                detection_frame,
                tracked_detection,
                rgb_tracked_detection=rgb_tracked_detection,
            )
            self._latest_display_image = debug_image
            self.publish_debug_outputs(debug_image, detection_frame, color_msg.header)
            output_detection = rgb_tracked_detection if rgb_tracked_detection is not None else tracked_detection
            self.publish_detection_outputs(output_detection, color_msg.header)
            self.log_candidate_scores(detection_frame)
            self.log_detection(tracked_detection)
        except Exception as exc:
            self.publish_pipeline_error(color, color_msg.header, exc)

    def ros_image_to_bgr(self, image_msg: Image) -> np.ndarray:
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

        if image is None:
            raise ValueError("cv_bridge returned no image data")

        if image.ndim == 2:
            return self.normalize_single_channel_to_bgr(image)

        if image.ndim != 3:
            raise ValueError(f"unsupported image shape: {image.shape}")

        channels = image.shape[2]
        encoding = image_msg.encoding.lower()

        if channels == 3:
            if encoding == "rgb8":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if image.dtype != np.uint8:
                image = self.normalize_to_uint8(image)
            return image

        if channels == 4:
            if encoding == "rgba8":
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if encoding == "bgra8":
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if image.dtype != np.uint8:
                image = self.normalize_to_uint8(image)
            return image[:, :, :3]

        raise ValueError(f"unsupported channel count: {channels}")

    def stamp_to_nanoseconds(self, stamp) -> int:
        if stamp is None:
            return 0
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def build_preview_display(self, color_image: np.ndarray, depth_msg: Image | None = None) -> np.ndarray:
        depth_image = None
        if depth_msg is not None:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            except Exception:
                depth_image = None

        status_lines = [
            "preview only" if depth_image is None else "preview + depth",
            "no detection yet" if self._processed_frame_count == 0 else "processing active",
        ]
        return draw_preview_image(color_image, depth_image, status_lines)

    def log_waiting_state(self, message: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_waiting_log_ns < int(1e9):
            return
        self._last_waiting_log_ns = now_ns
        self.get_logger().warn(message)

    def normalize_single_channel_to_bgr(self, image: np.ndarray) -> np.ndarray:
        normalized = self.normalize_to_uint8(image)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    def normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image

        finite_mask = np.isfinite(image)
        if not np.any(finite_mask):
            return np.zeros_like(image, dtype=np.uint8)

        finite_values = image[finite_mask].astype(np.float32)
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        if max_value - min_value < 1e-6:
            return np.zeros_like(image, dtype=np.uint8)

        scaled = np.zeros_like(image, dtype=np.float32)
        scaled[finite_mask] = (image[finite_mask].astype(np.float32) - min_value) / (max_value - min_value)
        return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)

    def display_latest_image(self) -> None:
        if self._window_disabled:
            return

        if self._latest_display_image is None and self._raw_color_image is not None:
            self._latest_display_image = self._raw_color_image.copy()

        if self._latest_display_image is None:
            return

        try:
            cv2.imshow(self.window_name, self._latest_display_image)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.get_logger().warn(f"Disabling OpenCV debug window: {exc}")
            self._window_disabled = True

    def publish_pipeline_error(self, color_image: np.ndarray, header, exc: Exception) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_pipeline_error_ns >= int(1e9):
            self.get_logger().error(f"Detection pipeline failed: {exc}")
            self._last_pipeline_error_ns = now_ns

        fallback_image = color_image.copy()
        self.draw_status_banner(fallback_image, f"pipeline error: {type(exc).__name__}")
        self._latest_display_image = fallback_image

        debug_msg = self.bridge.cv2_to_imgmsg(fallback_image, encoding="bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

        self.publish_debug_masks_empty(color_image.shape[:2], header)
        self.publish_detection_outputs(None, header)

    def publish_debug_masks_empty(self, image_shape: tuple[int, int], header) -> None:
        empty_mask = np.zeros(image_shape, dtype=np.uint8)
        foreground_msg = self.bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
        foreground_msg.header = header
        self.foreground_mask_pub.publish(foreground_msg)

        floor_msg = self.bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
        floor_msg.header = header
        self.floor_mask_pub.publish(floor_msg)

    def draw_status_banner(self, image: np.ndarray, text: str) -> None:
        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(image, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 255), 2, cv2.LINE_AA)

    def camera_info_to_intrinsics(self, camera_info: CameraInfo | None) -> CameraIntrinsics | None:
        if camera_info is None:
            return None
        return CameraIntrinsics(
            fx=float(camera_info.k[0]),
            fy=float(camera_info.k[4]),
            cx=float(camera_info.k[2]),
            cy=float(camera_info.k[5]),
        )

    def reproject_tracked_detection_to_color(
        self,
        *,
        tracked_detection: TrackedDetection | None,
        depth_raw: np.ndarray,
        color_shape: tuple[int, int],
    ) -> TrackedDetection | None:
        if tracked_detection is None:
            return None

        color_intrinsics = self.camera_info_to_intrinsics(self._color_camera_info)
        depth_intrinsics = self.camera_info_to_intrinsics(self._depth_camera_info)
        if color_intrinsics is None or depth_intrinsics is None:
            return tracked_detection

        projected_pixels: list[tuple[float, float]] = []
        x, y, w, h = tracked_detection.bbox
        sample_points = [
            (x, y),
            (x + w, y),
            (x, y + h),
            (x + w, y + h),
            (x + w * 0.5, y + h * 0.5),
            (x + w * 0.5, y),
            (x + w * 0.5, y + h),
            (x, y + h * 0.5),
            (x + w, y + h * 0.5),
        ]

        for point in sample_points:
            depth_xyz = self.depth_pixel_to_xyz(point, depth_raw, depth_intrinsics, tracked_detection.distance_m)
            if depth_xyz is None:
                continue
            color_xyz = self.transform_depth_point_to_color(depth_xyz)
            projected_pixel = self.project_color_point(color_xyz, color_intrinsics, color_shape)
            if projected_pixel is not None:
                projected_pixels.append(projected_pixel)

        if not projected_pixels:
            return tracked_detection

        projected_array = np.array(projected_pixels, dtype=np.float32)
        min_xy = np.min(projected_array, axis=0)
        max_xy = np.max(projected_array, axis=0)
        bbox_x = float(np.clip(min_xy[0], 0, color_shape[1] - 1))
        bbox_y = float(np.clip(min_xy[1], 0, color_shape[0] - 1))
        bbox_w = float(np.clip(max_xy[0] - min_xy[0], 1, color_shape[1]))
        bbox_h = float(np.clip(max_xy[1] - min_xy[1], 1, color_shape[0]))

        center_xyz = tracked_detection.center_xyz
        if center_xyz is not None:
            center_xyz = self.transform_depth_point_to_color(center_xyz)
        left_xyz = tracked_detection.left_xyz
        if left_xyz is not None:
            left_xyz = self.transform_depth_point_to_color(left_xyz)
        right_xyz = tracked_detection.right_xyz
        if right_xyz is not None:
            right_xyz = self.transform_depth_point_to_color(right_xyz)
        projected_center = (
            self.project_color_point(center_xyz, color_intrinsics, color_shape)
            if center_xyz is not None
            else None
        )
        if projected_center is None:
            projected_center = (bbox_x + bbox_w * 0.5, bbox_y + bbox_h * 0.5)

        return TrackedDetection(
            track_id=tracked_detection.track_id,
            bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
            center_pixel=(float(projected_center[0]), float(projected_center[1])),
            center_xyz=center_xyz,
            left_xyz=left_xyz,
            right_xyz=right_xyz,
            distance_m=float(center_xyz[2]) if center_xyz is not None else tracked_detection.distance_m,
            median_height_m=tracked_detection.median_height_m,
            width_m=tracked_detection.width_m,
            confidence=tracked_detection.confidence,
            stale=tracked_detection.stale,
            confirmed_frames=tracked_detection.confirmed_frames,
            missed_frames=tracked_detection.missed_frames,
            incompatible_frames=tracked_detection.incompatible_frames,
        )

    def depth_pixel_to_xyz(
        self,
        pixel: tuple[float, float],
        depth_raw: np.ndarray,
        intrinsics: CameraIntrinsics,
        fallback_depth_m: float,
    ) -> tuple[float, float, float] | None:
        depth_m = self.detector.depth_to_meters(depth_raw)
        u = int(round(np.clip(pixel[0], 0, depth_m.shape[1] - 1)))
        v = int(round(np.clip(pixel[1], 0, depth_m.shape[0] - 1)))
        local_depth = depth_m[max(0, v - 1):min(depth_m.shape[0], v + 2), max(0, u - 1):min(depth_m.shape[1], u + 2)]
        valid = (
            np.isfinite(local_depth)
            & (local_depth > self.detector.min_depth_m)
            & (local_depth < self.detector.max_depth_m)
        )
        z = float(np.median(local_depth[valid])) if np.any(valid) else float(fallback_depth_m)
        if not np.isfinite(z) or z <= 0.0:
            return None
        x = (float(u) - intrinsics.cx) * z / intrinsics.fx
        y = (float(v) - intrinsics.cy) * z / intrinsics.fy
        return (x, y, z)

    def transform_depth_point_to_color(
        self,
        point_depth: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        point = np.array(point_depth, dtype=np.float32)
        transformed = self.depth_to_color_rotation @ point + self.depth_to_color_translation
        return (float(transformed[0]), float(transformed[1]), float(transformed[2]))

    def project_color_point(
        self,
        point_color: tuple[float, float, float] | None,
        intrinsics: CameraIntrinsics,
        color_shape: tuple[int, int],
    ) -> tuple[float, float] | None:
        if point_color is None:
            return None
        x, y, z = point_color
        if not np.isfinite(z) or z <= 1e-6:
            return None
        u = (x * intrinsics.fx / z) + intrinsics.cx
        v = (y * intrinsics.fy / z) + intrinsics.cy
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return (
            float(np.clip(u, 0, color_shape[1] - 1)),
            float(np.clip(v, 0, color_shape[0] - 1)),
        )

    def publish_debug_outputs(self, debug_image, detection_frame, header) -> None:
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

        self.publish_debug_masks(
            detection_frame.foreground_mask,
            detection_frame.floor_mask,
            header,
        )

    def publish_debug_masks(self, foreground_mask: np.ndarray, floor_mask: np.ndarray, header) -> None:
        foreground_msg = self.bridge.cv2_to_imgmsg(foreground_mask, encoding="mono8")
        foreground_msg.header = header
        self.foreground_mask_pub.publish(foreground_msg)

        floor_msg = self.bridge.cv2_to_imgmsg(floor_mask, encoding="mono8")
        floor_msg.header = header
        self.floor_mask_pub.publish(floor_msg)

    def publish_detection_outputs(self, tracked_detection, header) -> None:
        self.publish_detection_array(tracked_detection, header)
        self.marker_pub.publish(
            create_marker_array(
                header=header,
                tracked_detection=tracked_detection,
                marker_lifetime_sec=self.marker_lifetime_sec,
            )
        )

        if tracked_detection is None:
            return

        distance_msg = Float32()
        distance_msg.data = float(tracked_detection.distance_m)
        self.distance_pub.publish(distance_msg)

        center_str = "(nan, nan, nan)"
        if tracked_detection.center_xyz is None:
            self.log_published_outputs(tracked_detection, center_str)
            return

        center_x, center_y, center_z = tracked_detection.center_xyz
        center_str = f"({center_x:.3f}, {center_y:.3f}, {center_z:.3f})"

        center_msg = PointStamped()
        center_msg.header = header
        center_msg.point.x = float(center_x)
        center_msg.point.y = float(center_y)
        center_msg.point.z = float(center_z)
        self.center_pub.publish(center_msg)

        if tracked_detection.left_xyz is not None:
            left_msg = PointStamped()
            left_msg.header = header
            left_msg.point.x = float(tracked_detection.left_xyz[0])
            left_msg.point.y = float(tracked_detection.left_xyz[1])
            left_msg.point.z = float(tracked_detection.left_xyz[2])
            self.left_pub.publish(left_msg)

        if tracked_detection.right_xyz is not None:
            right_msg = PointStamped()
            right_msg.header = header
            right_msg.point.x = float(tracked_detection.right_xyz[0])
            right_msg.point.y = float(tracked_detection.right_xyz[1])
            right_msg.point.z = float(tracked_detection.right_xyz[2])
            self.right_pub.publish(right_msg)

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = float(center_x)
        pose_msg.pose.position.y = float(center_y)
        pose_msg.pose.position.z = float(center_z)
        pose_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)
        self.log_published_outputs(tracked_detection, center_str)

    def publish_detection_array(self, tracked_detection, header) -> None:
        detection_array = Detection2DArray()
        detection_array.header = header

        if tracked_detection is None:
            self.detections_pub.publish(detection_array)
            return

        detection = Detection2D()
        detection.header = header
        detection.id = str(tracked_detection.track_id)
        detection.bbox.center.position.x = float(tracked_detection.center_pixel[0])
        detection.bbox.center.position.y = float(tracked_detection.center_pixel[1])
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = float(tracked_detection.bbox[2])
        detection.bbox.size_y = float(tracked_detection.bbox[3])

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = "floor_object"
        hypothesis.hypothesis.score = float(tracked_detection.confidence)
        hypothesis.pose.pose.orientation.w = 1.0

        if tracked_detection.center_xyz is not None:
            hypothesis.pose.pose.position.x = float(tracked_detection.center_xyz[0])
            hypothesis.pose.pose.position.y = float(tracked_detection.center_xyz[1])
            hypothesis.pose.pose.position.z = float(tracked_detection.center_xyz[2])

        detection.results.append(hypothesis)
        detection_array.detections.append(detection)
        self.detections_pub.publish(detection_array)

    def log_detection(self, tracked_detection) -> None:
        if tracked_detection is None:
            return

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_log_ns < self.log_detection_period_ns:
            return

        self._last_log_ns = now_ns
        center_str = "(nan, nan, nan)"
        if tracked_detection.center_xyz is not None:
            center_x, center_y, center_z = tracked_detection.center_xyz
            center_str = f"({center_x:.3f}, {center_y:.3f}, {center_z:.3f})"

        status = "stable" if not tracked_detection.stale else "stale"
        self.get_logger().info(
            f"track_id={tracked_detection.track_id} "
            f"status={status} "
            f"distance_m={tracked_detection.distance_m:.3f} "
            f"height_m={tracked_detection.median_height_m:.3f} "
            f"confidence={tracked_detection.confidence:.2f} "
            f"center={center_str}"
        )

    def log_candidate_scores(self, detection_frame) -> None:
        if not bool(self.get_parameter("debug_candidate_scores").value):
            return
        if not detection_frame.candidate_scores:
            return
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_candidate_score_log_ns < self.log_detection_period_ns:
            return
        self._last_candidate_score_log_ns = now_ns
        parts = []
        for index, item in enumerate(detection_frame.candidate_scores[:3], start=1):
            bbox = item["bbox"]
            parts.append(
                f"#{index} bbox={bbox} total={float(item['score']):.1f} "
                f"geom={float(item['geometry']):.2f} color={float(item['color']):.2f} "
                f"edge={float(item['edge']):.2f} center={float(item['center']):.2f} "
                f"temp={float(item['temporal']):.2f}"
            )
        self.get_logger().info("candidate_scores " + " | ".join(parts))

    def log_published_outputs(self, tracked_detection, center_str: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_published_log_ns < self.log_detection_period_ns:
            return
        self._last_published_log_ns = now_ns

        bbox_center_x = tracked_detection.center_pixel[0]
        bbox_center_y = tracked_detection.center_pixel[1]
        bbox_width = tracked_detection.bbox[2]
        bbox_height = tracked_detection.bbox[3]
        self.get_logger().info(
            "published "
            f"detections_2d=id={tracked_detection.track_id},"
            f"bbox_center=({bbox_center_x:.1f},{bbox_center_y:.1f}),"
            f"bbox_size=({bbox_width:.1f},{bbox_height:.1f}) "
            f"distance={tracked_detection.distance_m:.3f}m "
            f"box_center={center_str} "
            f"left_point={self.format_xyz(tracked_detection.left_xyz)} "
            f"right_point={self.format_xyz(tracked_detection.right_xyz)} "
            f"box_pose=pos{center_str},quat=(0.000,0.000,0.000,1.000)"
        )

    def format_xyz(self, xyz: tuple[float, float, float] | None) -> str:
        if xyz is None:
            return "(nan, nan, nan)"
        return f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"

    def destroy_node(self) -> None:
        if self.show_debug_window and not self._window_disabled:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FloorObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
