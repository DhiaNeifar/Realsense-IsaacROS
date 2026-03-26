#!/usr/bin/env python3

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class BoxFloorDetector(Node):
    def __init__(self):
        super().__init__('box_floor_detector')

        self.bridge = CvBridge()
        self._shape_warning_emitted = False
        self._window_disabled = False
        self._camera_info = None
        self._smoothed_bbox = None
        self._last_points_log_ns = 0

        # Parameters
        self.declare_parameter('color_topic', '/d435i/color/image_raw')
        self.declare_parameter('depth_topic', '/d435i/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/d435i/color/camera_info')
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 2.00)
        self.declare_parameter('min_contour_area', 2500)
        self.declare_parameter('depth_scale', 0.001)  # D435i depth usually in mm -> meters
        self.declare_parameter('foreground_margin_m', 0.05)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('window_name', 'floor_box_detector')
        self.declare_parameter('smoothing_alpha', 0.30)
        self.declare_parameter('point_depth_window', 5)
        self.declare_parameter('log_points_period_sec', 0.50)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.min_contour_area = int(self.get_parameter('min_contour_area').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.foreground_margin_m = float(self.get_parameter('foreground_margin_m').value)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        self.point_depth_window = int(self.get_parameter('point_depth_window').value)
        self.log_points_period_ns = int(float(self.get_parameter('log_points_period_sec').value) * 1e9)

        # Publishers
        self.debug_pub = self.create_publisher(Image, '/box_detection/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/box_detection/mask', 10)
        self.distance_pub = self.create_publisher(Float32, '/box_detection/distance', 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10,
        )

        # Subscribers with synchronization
        self.color_sub = Subscriber(self, Image, color_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.10
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(f'Subscribing to color: {color_topic}')
        self.get_logger().info(f'Subscribing to depth: {depth_topic}')
        self.get_logger().info(f'Subscribing to camera info: {camera_info_topic}')

    def camera_info_callback(self, msg: CameraInfo):
        self._camera_info = msg

    def synced_callback(self, color_msg: Image, depth_msg: Image):
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert color image: {e}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return

        if depth is None or color is None:
            return

        # Convert depth to meters
        depth_m = self.depth_to_meters(depth)
        depth_m = self.match_depth_to_color(depth_m, color.shape[:2])

        # Build valid depth mask
        valid_depth = np.isfinite(depth_m) & (depth_m > self.min_depth_m) & (depth_m < self.max_depth_m)

        if not np.any(valid_depth):
            self.publish_outputs(color, None, None, None, color_msg.header)
            return

        # Classical RGB cue: edge-enhanced grayscale
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use a smoothed local background depth estimate to isolate objects that rise off the floor.
        depth_fill = np.where(valid_depth, depth_m, self.max_depth_m).astype(np.float32)
        local_bg = cv2.GaussianBlur(depth_fill, (0, 0), sigmaX=21, sigmaY=21)
        foreground_depth = valid_depth & ((local_bg - depth_m) > self.foreground_margin_m)

        depth_mask = np.zeros_like(gray, dtype=np.uint8)
        depth_mask[foreground_depth] = 255

        # Clean depth mask
        kernel = np.ones((5, 5), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=1)

        # RGB edges
        edges = cv2.Canny(gray_blur, 50, 120)

        # Restrict edge growth to the neighborhood of the depth foreground.
        search_region = cv2.dilate(depth_mask, kernel, iterations=3)
        combined = cv2.bitwise_or(depth_mask, cv2.bitwise_and(edges, search_region))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = -math.inf

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Basic shape filtering
            if w < 30 or h < 30:
                continue

            aspect = w / float(h)
            if aspect < 0.3 or aspect > 3.5:
                continue

            cnt_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            cnt_depth_mask = (cnt_mask > 0) & valid_depth
            if not np.any(cnt_depth_mask):
                continue

            median_depth = float(np.median(depth_m[cnt_depth_mask]))
            score = area / max(median_depth, 1e-3)
            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            self.publish_outputs(color, None, combined, None, color_msg.header)
            return

        # Build object mask from best contour
        obj_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(obj_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

        # Use only valid depth inside contour
        obj_depth_mask = (obj_mask > 0) & valid_depth
        if not np.any(obj_depth_mask):
            self.publish_outputs(color, best_contour, obj_mask, None, color_msg.header)
            return

        obj_depth_values = depth_m[obj_depth_mask]
        distance_m = float(np.median(obj_depth_values))

        x, y, w, h = cv2.boundingRect(best_contour)
        smooth_bbox = self.smooth_bbox((x, y, w, h))
        x, y, w, h = [int(round(v)) for v in smooth_bbox]

        left_pt = (x, y + h // 2)
        right_pt = (x + w, y + h // 2)
        center_pt = (x + w // 2, y + h // 2)

        left_xyz = self.pixel_to_xyz(left_pt, depth_m)
        right_xyz = self.pixel_to_xyz(right_pt, depth_m)
        center_xyz = self.pixel_to_xyz(center_pt, depth_m)

        # Debug image
        debug = color.copy()
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(debug, left_pt, 6, (0, 0, 255), -1)
        cv2.circle(debug, right_pt, 6, (0, 255, 255), -1)
        cv2.putText(
            debug,
            f'dist={distance_m:.2f} m',
            (center_pt[0] - 60, max(20, center_pt[1] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        self.log_points(left_xyz, right_xyz, center_xyz)
        self.publish_outputs(debug, best_contour, obj_mask, distance_m, color_msg.header)

    def depth_to_meters(self, depth):
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * self.depth_scale
        elif depth.dtype == np.float32 or depth.dtype == np.float64:
            return depth.astype(np.float32)
        else:
            return depth.astype(np.float32)

    def match_depth_to_color(self, depth_m, color_shape):
        color_h, color_w = color_shape
        if depth_m.shape[:2] == (color_h, color_w):
            return depth_m

        if not self._shape_warning_emitted:
            self.get_logger().warn(
                f'Depth/color shape mismatch: depth={depth_m.shape[:2]} color={(color_h, color_w)}. '
                'Resizing depth to color for robustness.'
            )
            self._shape_warning_emitted = True

        return cv2.resize(depth_m, (color_w, color_h), interpolation=cv2.INTER_NEAREST)

    def smooth_bbox(self, bbox):
        bbox_array = np.array(bbox, dtype=np.float32)
        if self._smoothed_bbox is None or self._smoothed_bbox.shape != bbox_array.shape:
            self._smoothed_bbox = bbox_array.copy()
            return self._smoothed_bbox

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        self._smoothed_bbox = (1.0 - alpha) * self._smoothed_bbox + alpha * bbox_array
        return self._smoothed_bbox

    def pixel_to_xyz(self, pixel, depth_m):
        if self._camera_info is None:
            return None

        u, v = int(pixel[0]), int(pixel[1])
        h, w = depth_m.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return None

        half = max(0, self.point_depth_window // 2)
        u0 = max(0, u - half)
        u1 = min(w, u + half + 1)
        v0 = max(0, v - half)
        v1 = min(h, v + half + 1)

        local_depth = depth_m[v0:v1, u0:u1]
        valid = np.isfinite(local_depth) & (local_depth > self.min_depth_m) & (local_depth < self.max_depth_m)
        if not np.any(valid):
            return None

        z = float(np.median(local_depth[valid]))
        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])

        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return (x, y, z)

    def log_points(self, left_xyz, right_xyz, center_xyz):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_points_log_ns < self.log_points_period_ns:
            return

        self._last_points_log_ns = now_ns
        self.get_logger().info(
            'camera_frame_points '
            f'left={self.format_xyz(left_xyz)} '
            f'right={self.format_xyz(right_xyz)} '
            f'center={self.format_xyz(center_xyz)}'
        )

    def format_xyz(self, xyz):
        if xyz is None:
            return '(nan, nan, nan)'
        return f'({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})'

    def publish_outputs(self, debug_img, contour, mask_img, distance_m, header):
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

        if mask_img is None:
            mask_img = np.zeros(debug_img.shape[:2], dtype=np.uint8)

        mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding='mono8')
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)

        if distance_m is not None:
            dist_msg = Float32()
            dist_msg.data = float(distance_m)
            self.distance_pub.publish(dist_msg)

        if self.show_debug_window and not self._window_disabled:
            try:
                cv2.imshow(self.window_name, debug_img)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.get_logger().warn(f'Disabling OpenCV debug window: {exc}')
                self._window_disabled = True

    def destroy_node(self):
        if self.show_debug_window and not self._window_disabled:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BoxFloorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
