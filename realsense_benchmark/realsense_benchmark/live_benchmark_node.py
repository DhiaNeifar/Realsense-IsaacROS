import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class RollingFPS:
    def __init__(self, window_size=60):
        self.timestamps = deque(maxlen=window_size)

    def tick(self):
        self.timestamps.append(time.perf_counter())

    def get(self):
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / dt


class RollingMean:
    def __init__(self, window_size=60):
        self.values = deque(maxlen=window_size)

    def add(self, value):
        self.values.append(float(value))

    def get(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class LiveBenchmarkNode(Node):
    def __init__(self):
        super().__init__("live_benchmark_node")

        self.declare_parameter("color_topic", "/d435i/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/d435i/camera/depth/image_rect_raw")
        self.declare_parameter("cpu_loops", 0)
        self.declare_parameter("band_min_m", 0.4)
        self.declare_parameter("band_max_m", 1.5)
        self.declare_parameter("report_period_sec", 1.0)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.cpu_loops = int(self.get_parameter("cpu_loops").value)
        self.band_min_m = float(self.get_parameter("band_min_m").value)
        self.band_max_m = float(self.get_parameter("band_max_m").value)
        self.report_period_sec = float(self.get_parameter("report_period_sec").value)

        self.bridge = CvBridge()

        self.color_fps = RollingFPS()
        self.depth_fps = RollingFPS()
        self.display_fps = RollingFPS()
        self.proc_ms = RollingMean()

        self.last_report = time.perf_counter()

        self.latest_color = None
        self.latest_depth = None

        self.color_sub = self.create_subscription(
            Image,
            self.color_topic,
            self.color_callback,
            qos_profile_sensor_data,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"Subscribed color: {self.color_topic}")
        self.get_logger().info(f"Subscribed depth: {self.depth_topic}")
        self.get_logger().info(f"cpu_loops={self.cpu_loops}")

    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.color_fps.tick()
            self.try_display()
        except Exception as e:
            self.get_logger().error(f"Color callback error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_fps.tick()
            self.try_display()
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def process_depth(self, depth_raw):
        t0 = time.perf_counter()

        depth = depth_raw.astype(np.float32) / 1000.0  # mm -> meters
        valid = np.isfinite(depth)

        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[(valid) & (depth >= self.band_min_m) & (depth <= self.band_max_m)] = 255

        for _ in range(self.cpu_loops):
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = cv2.medianBlur(mask, 5)
            mask = cv2.Sobel(mask, cv2.CV_8U, 1, 0, ksize=3)

        depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        overlay = np.zeros_like(depth_vis)
        overlay[:, :, 1] = mask
        out = cv2.addWeighted(depth_vis, 1.0, overlay, 0.45, 0.0)

        t1 = time.perf_counter()
        self.proc_ms.add((t1 - t0) * 1000.0)
        return out

    def draw_overlay(self, image):
        lines = [
            f"Color FPS   : {self.color_fps.get():.2f}",
            f"Depth FPS   : {self.depth_fps.get():.2f}",
            f"Display FPS : {self.display_fps.get():.2f}",
            f"Proc        : {self.proc_ms.get():.2f} ms",
            f"CPU loops   : {self.cpu_loops}",
        ]

        y = 30
        for line in lines:
            cv2.putText(
                image,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

    def try_display(self):
        if self.latest_color is None or self.latest_depth is None:
            return

        color = self.latest_color.copy()
        depth_vis = self.process_depth(self.latest_depth)

        h, w = color.shape[:2]
        depth_vis = cv2.resize(depth_vis, (w, h))

        combined = np.hstack([color, depth_vis])
        self.draw_overlay(combined)

        cv2.imshow("RealSense Benchmark", combined)
        cv2.waitKey(1)

        self.display_fps.tick()

        now = time.perf_counter()
        if now - self.last_report >= self.report_period_sec:
            self.get_logger().info(
                f"color_fps={self.color_fps.get():.2f} "
                f"depth_fps={self.depth_fps.get():.2f} "
                f"display_fps={self.display_fps.get():.2f} "
                f"proc_ms={self.proc_ms.get():.2f}"
            )
            self.last_report = now


def main(args=None):
    rclpy.init(args=args)
    node = LiveBenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
