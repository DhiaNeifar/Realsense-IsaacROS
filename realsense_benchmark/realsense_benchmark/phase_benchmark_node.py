import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from realsense_benchmark.common import (
    DEFAULT_RESULTS_DIR,
    RollingFPS,
    destroy_opencv_windows,
    ensure_directory,
    image_qos_profile,
    probe_display_available,
)
from realsense_benchmark.depth_tools import render_depth_band_overlay


class PhaseBenchmarkNode(Node):
    def __init__(self):
        super().__init__("phase_benchmark_node")

        self.declare_parameter("color_topic", "/d435i/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/d435i/camera/depth/image_rect_raw")
        self.declare_parameter("baseline_duration_sec", 30.0)
        self.declare_parameter("stress_duration_sec", 30.0)
        self.declare_parameter("stress_cpu_loops", 8)
        self.declare_parameter("band_min_m", 0.4)
        self.declare_parameter("band_max_m", 1.5)
        self.declare_parameter("sample_hz", 5.0)
        self.declare_parameter("render_hz", 30.0)
        self.declare_parameter(
            "output_dir",
            str(DEFAULT_RESULTS_DIR),
        )
        self.declare_parameter("show_window", True)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.baseline_duration_sec = float(self.get_parameter("baseline_duration_sec").value)
        self.stress_duration_sec = float(self.get_parameter("stress_duration_sec").value)
        self.stress_cpu_loops = int(self.get_parameter("stress_cpu_loops").value)
        self.band_min_m = float(self.get_parameter("band_min_m").value)
        self.band_max_m = float(self.get_parameter("band_max_m").value)
        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.render_hz = float(self.get_parameter("render_hz").value)
        self.output_dir = self.get_parameter("output_dir").value
        self.show_window = bool(self.get_parameter("show_window").value)

        self.output_dir = ensure_directory(self.output_dir)

        self.bridge = CvBridge()

        self.color_fps = RollingFPS()
        self.depth_fps = RollingFPS()
        self.display_fps = RollingFPS()

        self.latest_color = None
        self.latest_depth = None

        self.first_color_logged = False
        self.first_depth_logged = False
        self.first_render_logged = False

        self.start_time = time.perf_counter()
        self.done = False

        self.records_t = []
        self.records_color_fps = []
        self.records_depth_fps = []
        self.records_display_fps = []
        self.records_phase = []

        # Use the same explicit image QoS across the benchmark nodes so the
        # behavior is predictable and easy to reason about.
        sensor_qos = image_qos_profile()

        self.color_sub = self.create_subscription(
            Image, self.color_topic, self.color_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sensor_qos
        )

        self.render_timer = self.create_timer(1.0 / self.render_hz, self.render_loop)
        self.sample_timer = self.create_timer(1.0 / self.sample_hz, self.sample_metrics)

        # ---------------------------------------------------------------
        # FIX 2: Detect whether a display is actually available so we
        # don't silently swallow cv2.imshow errors for the entire run.
        # ---------------------------------------------------------------
        self._display_available = probe_display_available(self.show_window)
        if self.show_window and not self._display_available:
            self.get_logger().warn(
                "show_window=True but no display found (DISPLAY env var not set or "
                "cv2 headless build). Live window disabled — results will still be saved."
            )

        self.get_logger().info(f"Subscribed color: {self.color_topic}")
        self.get_logger().info(f"Subscribed depth: {self.depth_topic}")
        self.get_logger().info(
            f"Baseline={self.baseline_duration_sec}s, Stress={self.stress_duration_sec}s, "
            f"stress_cpu_loops={self.stress_cpu_loops}"
        )
        self.get_logger().info(f"Saving results to: {self.output_dir}")
        self.get_logger().info(
            f"QoS: RELIABLE/VOLATILE/KEEP_LAST(10) — "
            "change to BEST_EFFORT if your driver publishes with that policy"
        )

    def current_elapsed(self):
        return time.perf_counter() - self.start_time

    def current_phase(self):
        t = self.current_elapsed()
        if t < self.baseline_duration_sec:
            return "baseline"
        if t < self.baseline_duration_sec + self.stress_duration_sec:
            return "stress"
        return "done"

    def current_cpu_loops(self):
        return 0 if self.current_phase() == "baseline" else self.stress_cpu_loops

    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.color_fps.tick()
            if not self.first_color_logged:
                self.get_logger().info(f"First color frame received: {msg.width}x{msg.height}")
                self.first_color_logged = True
        except Exception as e:
            self.get_logger().error(f"Color callback error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_fps.tick()
            if not self.first_depth_logged:
                self.get_logger().info(f"First depth frame received: {msg.width}x{msg.height}")
                self.first_depth_logged = True
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def draw_overlay(self, image, phase, cpu_loops):
        lines = [
            f"Phase       : {phase}",
            f"CPU loops   : {cpu_loops}",
            f"Color FPS   : {self.color_fps.get():.2f}",
            f"Depth FPS   : {self.depth_fps.get():.2f}",
            f"Display FPS : {self.display_fps.get():.2f}",
            f"Elapsed     : {self.current_elapsed():.1f}s",
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

    def render_loop(self):
        if self.done:
            return

        phase = self.current_phase()
        if phase == "done":
            self.finish_and_shutdown()
            return

        if self.latest_color is None or self.latest_depth is None:
            return

        cpu_loops = self.current_cpu_loops()

        color = self.latest_color.copy()
        depth_vis, _ = render_depth_band_overlay(
            self.latest_depth,
            band_min_m=self.band_min_m,
            band_max_m=self.band_max_m,
            cpu_loops=cpu_loops,
        )

        h, w = color.shape[:2]
        depth_vis = cv2.resize(depth_vis, (w, h))
        combined = np.hstack([color, depth_vis])

        self.draw_overlay(combined, phase, cpu_loops)

        if not self.first_render_logged:
            self.get_logger().info(f"First render built: {combined.shape[1]}x{combined.shape[0]}")
            self.first_render_logged = True

        # ---------------------------------------------------------------
        # FIX 2 (continued): Only call imshow when display is confirmed
        # available; avoids silent exception swallowing masking real bugs.
        # ---------------------------------------------------------------
        if self.show_window and self._display_available:
            try:
                cv2.imshow("Phase Benchmark", combined)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().error(f"Display error: {e}")
                self._display_available = False  # stop trying after first real failure

        self.display_fps.tick()

    def sample_metrics(self):
        if self.done:
            return

        phase = self.current_phase()
        if phase == "done":
            self.finish_and_shutdown()
            return

        t = self.current_elapsed()
        local_t = t if phase == "baseline" else t - self.baseline_duration_sec

        self.records_t.append(local_t)
        self.records_color_fps.append(self.color_fps.get())
        self.records_depth_fps.append(self.depth_fps.get())
        self.records_display_fps.append(self.display_fps.get())
        self.records_phase.append(phase)

        self.get_logger().info(
            f"phase={phase} t={local_t:.1f}s "
            f"color_fps={self.color_fps.get():.2f} "
            f"depth_fps={self.depth_fps.get():.2f} "
            f"display_fps={self.display_fps.get():.2f}"
        )

    def finish_and_shutdown(self):
        if self.done:
            return
        self.done = True

        destroy_opencv_windows(self.show_window, self._display_available)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.output_dir, f"phase_compare_{ts}.png")
        raw_path = os.path.join(self.output_dir, f"phase_compare_{ts}.npz")

        t = np.array(self.records_t)
        color_fps = np.array(self.records_color_fps)
        depth_fps = np.array(self.records_depth_fps)
        display_fps = np.array(self.records_display_fps)
        phase = np.array(self.records_phase, dtype=object)

        np.savez(
            raw_path,
            t=t,
            color_fps=color_fps,
            depth_fps=depth_fps,
            display_fps=display_fps,
            phase=phase,
        )

        base_mask = phase == "baseline"
        stress_mask = phase == "stress"

        t_base, t_stress = t[base_mask], t[stress_mask]
        color_base, color_stress = color_fps[base_mask], color_fps[stress_mask]
        depth_base, depth_stress = depth_fps[base_mask], depth_fps[stress_mask]
        display_base, display_stress = display_fps[base_mask], display_fps[stress_mask]

        def avg(x):
            return float(np.mean(x)) if len(x) > 0 else 0.0

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=170, sharex=True)

        axes[0].plot(t_base, color_base, linewidth=2, label=f"Baseline (avg={avg(color_base):.2f})")
        axes[0].plot(t_stress, color_stress, linewidth=2, label=f"Stress (avg={avg(color_stress):.2f})")
        axes[0].set_ylabel("Color FPS")
        axes[0].set_title("Color FPS: Baseline vs Stress", fontsize=13, weight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t_base, depth_base, linewidth=2, label=f"Baseline (avg={avg(depth_base):.2f})")
        axes[1].plot(t_stress, depth_stress, linewidth=2, label=f"Stress (avg={avg(depth_stress):.2f})")
        axes[1].set_ylabel("Depth FPS")
        axes[1].set_title("Depth FPS: Baseline vs Stress", fontsize=13, weight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(t_base, display_base, linewidth=2, label=f"Baseline (avg={avg(display_base):.2f})")
        axes[2].plot(t_stress, display_stress, linewidth=2, label=f"Stress (avg={avg(display_stress):.2f})")
        axes[2].set_ylabel("Display FPS")
        axes[2].set_xlabel("Time within phase (s)")
        axes[2].set_title("Display FPS: Baseline vs Stress", fontsize=13, weight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.suptitle("RealSense Benchmark Comparison", fontsize=16, weight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        self.get_logger().info(f"Saved figure: {fig_path}")
        self.get_logger().info(f"Saved raw data: {raw_path}")

        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PhaseBenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            destroy_opencv_windows(node.show_window, node._display_available)
            node.destroy_node()
            rclpy.shutdown()
