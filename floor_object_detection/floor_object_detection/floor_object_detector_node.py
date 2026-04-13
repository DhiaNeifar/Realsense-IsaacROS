#!/usr/bin/env python3

from __future__ import annotations

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray

from floor_object_detection.detector import CameraIntrinsics, DepthBasedFloorObjectDetector
from floor_object_detection.tracking import TemporalDetectionTracker
from floor_object_detection.visualization import create_marker_array, draw_debug_image


class FloorObjectDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("floor_object_detector")

        self.bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._window_disabled = False
        self._last_log_ns = 0

        self.declare_parameter("color_topic", "/d435i/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/d435i/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/d435i/camera/color/camera_info")
        self.declare_parameter("sync_slop_sec", 0.08)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 2.00)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("plane_ransac_iterations", 80)
        self.declare_parameter("plane_inlier_threshold_m", 0.02)
        self.declare_parameter("min_floor_points", 180)
        self.declare_parameter("min_plane_y_component", 0.35)
        self.declare_parameter("plane_sample_stride", 4)
        self.declare_parameter("min_height_above_floor_m", 0.03)
        self.declare_parameter("max_height_above_floor_m", 0.60)
        self.declare_parameter("fallback_foreground_margin_m", 0.05)
        self.declare_parameter("local_background_sigma", 31.0)
        self.declare_parameter("search_top_ignore_ratio", 0.10)
        self.declare_parameter("min_contour_area", 1800)
        self.declare_parameter("min_bbox_size_px", 28)
        self.declare_parameter("max_bbox_aspect_ratio", 4.0)
        self.declare_parameter("min_extent", 0.20)
        self.declare_parameter("min_solidity", 0.45)
        self.declare_parameter("open_kernel_size", 5)
        self.declare_parameter("close_kernel_size", 7)
        self.declare_parameter("point_depth_window", 5)
        self.declare_parameter("min_confirmed_frames", 2)
        self.declare_parameter("max_missed_frames", 4)
        self.declare_parameter("bbox_smoothing_alpha", 0.35)
        self.declare_parameter("center_smoothing_alpha", 0.25)
        self.declare_parameter("max_center_jump_px", 140.0)
        self.declare_parameter("max_depth_jump_m", 0.30)
        self.declare_parameter("marker_lifetime_sec", 0.25)
        self.declare_parameter("show_debug_window", True)
        self.declare_parameter("window_name", "floor_object_detector")
        self.declare_parameter("log_detection_period_sec", 1.0)

        color_topic = str(self.get_parameter("color_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)

        self.log_detection_period_ns = int(float(self.get_parameter("log_detection_period_sec").value) * 1e9)
        self.marker_lifetime_sec = float(self.get_parameter("marker_lifetime_sec").value)
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)
        self.window_name = str(self.get_parameter("window_name").value)

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
        )

        self.tracker = TemporalDetectionTracker(
            min_confirmed_frames=int(self.get_parameter("min_confirmed_frames").value),
            max_missed_frames=int(self.get_parameter("max_missed_frames").value),
            bbox_smoothing_alpha=float(self.get_parameter("bbox_smoothing_alpha").value),
            center_smoothing_alpha=float(self.get_parameter("center_smoothing_alpha").value),
            max_center_jump_px=float(self.get_parameter("max_center_jump_px").value),
            max_depth_jump_m=float(self.get_parameter("max_depth_jump_m").value),
        )

        self.debug_pub = self.create_publisher(Image, "~/debug_image", 10)
        self.foreground_mask_pub = self.create_publisher(Image, "~/foreground_mask", 10)
        self.floor_mask_pub = self.create_publisher(Image, "~/floor_mask", 10)
        self.distance_pub = self.create_publisher(Float32, "~/distance", 10)
        self.detections_pub = self.create_publisher(Detection2DArray, "~/detections_2d", 10)
        self.center_pub = self.create_publisher(PointStamped, "~/box_center", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "~/box_pose", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/marker_array", 10)

        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.color_sub = Subscriber(self, Image, color_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=sync_slop_sec)
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(f"Subscribing to color: {color_topic}")
        self.get_logger().info(f"Subscribing to depth: {depth_topic}")
        self.get_logger().info(f"Subscribing to camera info: {camera_info_topic}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def synced_callback(self, color_msg: Image, depth_msg: Image) -> None:
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        intrinsics = self.camera_info_to_intrinsics(self._camera_info)
        detection_frame = self.detector.detect(
            color_shape=color.shape[:2],
            depth_raw=depth,
            intrinsics=intrinsics,
        )
        tracked_detection = self.tracker.update(detection_frame.candidate)

        debug_image = draw_debug_image(color, detection_frame, tracked_detection)
        self.publish_debug_outputs(debug_image, detection_frame, color_msg.header)
        self.publish_detection_outputs(tracked_detection, color_msg.header)
        self.log_detection(tracked_detection)

        if self.show_debug_window and not self._window_disabled:
            try:
                cv2.imshow(self.window_name, debug_image)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.get_logger().warn(f"Disabling OpenCV debug window: {exc}")
                self._window_disabled = True

    def camera_info_to_intrinsics(self, camera_info: CameraInfo | None) -> CameraIntrinsics | None:
        if camera_info is None:
            return None
        return CameraIntrinsics(
            fx=float(camera_info.k[0]),
            fy=float(camera_info.k[4]),
            cx=float(camera_info.k[2]),
            cy=float(camera_info.k[5]),
        )

    def publish_debug_outputs(self, debug_image, detection_frame, header) -> None:
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

        foreground_msg = self.bridge.cv2_to_imgmsg(detection_frame.foreground_mask, encoding="mono8")
        foreground_msg.header = header
        self.foreground_mask_pub.publish(foreground_msg)

        floor_msg = self.bridge.cv2_to_imgmsg(detection_frame.floor_mask, encoding="mono8")
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

        if tracked_detection.center_xyz is None:
            return

        center_x, center_y, center_z = tracked_detection.center_xyz

        center_msg = PointStamped()
        center_msg.header = header
        center_msg.point.x = float(center_x)
        center_msg.point.y = float(center_y)
        center_msg.point.z = float(center_z)
        self.center_pub.publish(center_msg)

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = float(center_x)
        pose_msg.pose.position.y = float(center_y)
        pose_msg.pose.position.z = float(center_z)
        pose_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)

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
        rclpy.shutdown()
