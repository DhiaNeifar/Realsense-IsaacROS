import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image

from realsense_benchmark.common import (
    RollingFPS,
    RollingMean,
    destroy_opencv_windows,
    image_qos_profile,
    probe_display_available,
)
from realsense_benchmark.depth_tools import render_depth_band_overlay


class LiveBenchmarkNode(Node):
    def __init__(self):
        super().__init__("live_benchmark_node")

        self.declare_parameter("color_topic", "/d435i/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/d435i/camera/depth/image_rect_raw")
        self.declare_parameter("cpu_loops", 0)
        self.declare_parameter("band_min_m", 0.4)
        self.declare_parameter("band_max_m", 1.5)
        self.declare_parameter("report_period_sec", 1.0)
        self.declare_parameter("show_window", True)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.cpu_loops = int(self.get_parameter("cpu_loops").value)
        self.band_min_m = float(self.get_parameter("band_min_m").value)
        self.band_max_m = float(self.get_parameter("band_max_m").value)
        self.report_period_sec = float(self.get_parameter("report_period_sec").value)
        self.show_window = bool(self.get_parameter("show_window").value)

        self.bridge = CvBridge()

        self.color_fps = RollingFPS()
        self.depth_fps = RollingFPS()
        self.display_fps = RollingFPS()
        self.proc_ms = RollingMean()

        self.last_report = time.perf_counter()

        self.latest_color = None
        self.latest_depth = None

        self._display_available = probe_display_available(self.show_window)

        self.color_sub = self.create_subscription(
            Image,
            self.color_topic,
            self.color_callback,
            image_qos_profile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            image_qos_profile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.get_logger().info(f"Subscribed color: {self.color_topic}")
        self.get_logger().info(f"Subscribed depth: {self.depth_topic}")
        self.get_logger().info(f"cpu_loops={self.cpu_loops}")
        if self.show_window and not self._display_available:
            self.get_logger().warn(
                "show_window=True but no display found. Live window disabled."
            )

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
        depth_vis, proc_ms = render_depth_band_overlay(
            self.latest_depth,
            band_min_m=self.band_min_m,
            band_max_m=self.band_max_m,
            cpu_loops=self.cpu_loops,
        )
        self.proc_ms.add(proc_ms)

        h, w = color.shape[:2]
        depth_vis = cv2.resize(depth_vis, (w, h))

        combined = np.hstack([color, depth_vis])
        self.draw_overlay(combined)

        if self.show_window and self._display_available:
            try:
                cv2.imshow("RealSense Benchmark", combined)
                cv2.waitKey(1)
            except Exception as exc:
                self.get_logger().error(f"Display error: {exc}")
                self._display_available = False

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
        destroy_opencv_windows(node.show_window, node._display_available)
        node.destroy_node()
        rclpy.shutdown()
