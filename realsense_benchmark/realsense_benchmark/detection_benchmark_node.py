"""
detection_benchmark_node.py
============================
ROS 2 node that benchmarks RealSense camera FPS under two phases:

  Baseline  – raw color + depth stream, zero inference
  Stress    – simultaneous MediaPipe Face Detection + Face Mesh
              (468 landmarks → expression tags) + Hand tracking

All three detectors run on every color frame in the stress phase,
which is intentionally heavy enough to show a clear FPS drop on a
Jetson Orin Super.

Dependencies
------------
    pip install mediapipe --break-system-packages

Launch example
--------------
    ros2 run realsense_benchmark detection_benchmark_node \
        --ros-args \
        -p color_topic:=/d435i/camera/color/image_raw \
        -p depth_topic:=/d435i/camera/depth/image_rect_raw \
        -p baseline_duration_sec:=30.0 \
        -p stress_duration_sec:=30.0 \
        -p show_window:=true
"""

import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

# ---------------------------------------------------------------------------
# MediaPipe imports — graceful error if not installed
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp

    _MP_FACE_DET = mp.solutions.face_detection
    _MP_FACE_MESH = mp.solutions.face_mesh
    _MP_HANDS = mp.solutions.hands
    _MP_DRAW = mp.solutions.drawing_utils
    _MP_DRAW_STYLES = mp.solutions.drawing_styles
    _MEDIAPIPE_OK = True
except ImportError:
    _MEDIAPIPE_OK = False


# ---------------------------------------------------------------------------
# Expression heuristics from face-mesh landmarks
# Landmark indices follow the canonical MediaPipe 468-point map.
# ---------------------------------------------------------------------------
_EXPR_MOUTH_LEFT = 61
_EXPR_MOUTH_RIGHT = 291
_EXPR_MOUTH_TOP = 13
_EXPR_MOUTH_BOTTOM = 14
_EXPR_LEFT_EYE_TOP = 159
_EXPR_LEFT_EYE_BOTTOM = 145
_EXPR_RIGHT_EYE_TOP = 386
_EXPR_RIGHT_EYE_BOTTOM = 374
_EXPR_LEFT_BROW_INNER = 107
_EXPR_LEFT_EYE_INNER = 33
_EXPR_RIGHT_BROW_INNER = 336
_EXPR_RIGHT_EYE_INNER = 263
_EXPR_NOSE_TIP = 4


def _lm(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def classify_expression(landmarks, w, h):
    """Return a rough expression label from 468-point face mesh."""
    try:
        mouth_w = np.linalg.norm(_lm(landmarks, _EXPR_MOUTH_LEFT, w, h) -
                                  _lm(landmarks, _EXPR_MOUTH_RIGHT, w, h))
        mouth_h = np.linalg.norm(_lm(landmarks, _EXPR_MOUTH_TOP, w, h) -
                                  _lm(landmarks, _EXPR_MOUTH_BOTTOM, w, h))
        left_eye_h = np.linalg.norm(_lm(landmarks, _EXPR_LEFT_EYE_TOP, w, h) -
                                     _lm(landmarks, _EXPR_LEFT_EYE_BOTTOM, w, h))
        right_eye_h = np.linalg.norm(_lm(landmarks, _EXPR_RIGHT_EYE_TOP, w, h) -
                                      _lm(landmarks, _EXPR_RIGHT_EYE_BOTTOM, w, h))
        avg_eye_h = (left_eye_h + right_eye_h) / 2.0

        left_brow_y = _lm(landmarks, _EXPR_LEFT_BROW_INNER, w, h)[1]
        left_eye_y = _lm(landmarks, _EXPR_LEFT_EYE_INNER, w, h)[1]
        right_brow_y = _lm(landmarks, _EXPR_RIGHT_BROW_INNER, w, h)[1]
        right_eye_y = _lm(landmarks, _EXPR_RIGHT_EYE_INNER, w, h)[1]
        brow_raise = ((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) / 2.0

        mouth_open_ratio = mouth_h / (mouth_w + 1e-6)
        eye_open_ratio = avg_eye_h / (mouth_w + 1e-6)

        if mouth_open_ratio > 0.25:
            return "Surprised / Open mouth"
        if mouth_w / (w + 1e-6) > 0.18 and mouth_h / (mouth_w + 1e-6) < 0.15:
            return "Happy / Smiling"
        if brow_raise < h * 0.03 and eye_open_ratio < 0.04:
            return "Angry / Frowning"
        if eye_open_ratio < 0.025:
            return "Eyes closed"
        if brow_raise > h * 0.06:
            return "Surprised / Raised brows"
        return "Neutral"
    except Exception:
        return "Unknown"


# ---------------------------------------------------------------------------
# Rolling FPS counter
# ---------------------------------------------------------------------------
class RollingFPS:
    def __init__(self, window_size=60):
        self.timestamps = deque(maxlen=window_size)

    def tick(self):
        self.timestamps.append(time.perf_counter())

    def get(self):
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        return 0.0 if dt <= 0 else (len(self.timestamps) - 1) / dt


# ---------------------------------------------------------------------------
# Colours for drawing
# ---------------------------------------------------------------------------
_COL_GREEN = (0, 255, 100)
_COL_CYAN = (0, 255, 255)
_COL_ORANGE = (0, 165, 255)
_COL_WHITE = (255, 255, 255)
_COL_RED = (0, 0, 255)
_COL_YELLOW = (0, 220, 255)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
class DetectionBenchmarkNode(Node):

    def __init__(self):
        super().__init__("detection_benchmark_node")

        if not _MEDIAPIPE_OK:
            self.get_logger().fatal(
                "MediaPipe is not installed!  Run:\n"
                "    pip install mediapipe --break-system-packages\n"
                "then re-launch this node."
            )
            raise RuntimeError("mediapipe not installed")

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter("color_topic", "/d435i/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/d435i/camera/depth/image_rect_raw")
        self.declare_parameter("baseline_duration_sec", 30.0)
        self.declare_parameter("stress_duration_sec", 30.0)
        self.declare_parameter("sample_hz", 5.0)
        self.declare_parameter("render_hz", 30.0)
        self.declare_parameter(
            "output_dir",
            str(Path.home() / "ros2_ws" / "src" / "realsense_benchmark" / "results"),
        )
        self.declare_parameter("show_window", True)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.baseline_duration_sec = float(self.get_parameter("baseline_duration_sec").value)
        self.stress_duration_sec = float(self.get_parameter("stress_duration_sec").value)
        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.render_hz = float(self.get_parameter("render_hz").value)
        self.output_dir = self.get_parameter("output_dir").value
        self.show_window = bool(self.get_parameter("show_window").value)

        os.makedirs(self.output_dir, exist_ok=True)

        # ── MediaPipe pipelines (created once, kept alive) ─────────────────
        # Running all three simultaneously is the deliberate stress load.
        self.mp_face_det = _MP_FACE_DET.FaceDetection(
            model_selection=1,          # 1 = full-range model (heavier)
            min_detection_confidence=0.5,
        )
        self.mp_face_mesh = _MP_FACE_MESH.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,            # track up to 4 faces
            refine_landmarks=True,      # iris landmarks = extra cost
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_hands = _MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=4,            # track up to 4 hands
            model_complexity=1,         # 1 = heavy model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── state ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None

        self.color_fps = RollingFPS()
        self.depth_fps = RollingFPS()
        self.display_fps = RollingFPS()
        self.inference_fps = RollingFPS()

        self.first_color_logged = False
        self.first_depth_logged = False
        self.first_render_logged = False

        self.start_time = time.perf_counter()
        self.done = False

        # latest inference results (written by render_loop, read by draw)
        self._last_face_detections = None   # list of BoundingBox-like results
        self._last_mesh_results = None      # mediapipe FaceMesh result
        self._last_hand_results = None      # mediapipe Hands result
        self._last_expressions = []         # list of str

        # ── records ───────────────────────────────────────────────────────
        self.records_t = []
        self.records_color_fps = []
        self.records_depth_fps = []
        self.records_display_fps = []
        self.records_inference_fps = []
        self.records_phase = []

        # ── QoS ───────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.color_sub = self.create_subscription(
            Image, self.color_topic, self.color_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sensor_qos
        )

        self.render_timer = self.create_timer(1.0 / self.render_hz, self.render_loop)
        self.sample_timer = self.create_timer(1.0 / self.sample_hz, self.sample_metrics)

        # ── display probe ─────────────────────────────────────────────────
        self._display_available = self._check_display()
        if self.show_window and not self._display_available:
            self.get_logger().warn(
                "show_window=True but no display found. "
                "Live window disabled — results will still be saved."
            )

        self.get_logger().info(f"Subscribed color : {self.color_topic}")
        self.get_logger().info(f"Subscribed depth : {self.depth_topic}")
        self.get_logger().info(
            f"Baseline={self.baseline_duration_sec}s (raw stream)  |  "
            f"Stress={self.stress_duration_sec}s "
            f"(FaceDetection + FaceMesh[refine] + Hands x4)"
        )
        self.get_logger().info(f"Results → {self.output_dir}")

    # ── display probe ──────────────────────────────────────────────────────
    def _check_display(self) -> bool:
        if not self.show_window:
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

    # ── helpers ────────────────────────────────────────────────────────────
    def current_elapsed(self):
        return time.perf_counter() - self.start_time

    def current_phase(self):
        t = self.current_elapsed()
        if t < self.baseline_duration_sec:
            return "baseline"
        if t < self.baseline_duration_sec + self.stress_duration_sec:
            return "stress"
        return "done"

    # ── subscriptions ──────────────────────────────────────────────────────
    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.color_fps.tick()
            if not self.first_color_logged:
                self.get_logger().info(
                    f"First color frame: {msg.width}x{msg.height}"
                )
                self.first_color_logged = True
        except Exception as e:
            self.get_logger().error(f"Color callback: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_fps.tick()
            if not self.first_depth_logged:
                self.get_logger().info(
                    f"First depth frame: {msg.width}x{msg.height}"
                )
                self.first_depth_logged = True
        except Exception as e:
            self.get_logger().error(f"Depth callback: {e}")

    # ── inference ──────────────────────────────────────────────────────────
    def run_detectors(self, bgr_frame):
        """
        Run all three MediaPipe pipelines sequentially on one frame.
        This is intentionally synchronous and heavy — that's the benchmark load.
        Returns (face_det_result, face_mesh_result, hand_result, expressions)
        """
        h, w = bgr_frame.shape[:2]
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        face_det_result = self.mp_face_det.process(rgb)
        mesh_result = self.mp_face_mesh.process(rgb)
        hand_result = self.mp_hands.process(rgb)

        expressions = []
        if mesh_result.multi_face_landmarks:
            for face_lms in mesh_result.multi_face_landmarks:
                expr = classify_expression(face_lms.landmark, w, h)
                expressions.append(expr)

        self.inference_fps.tick()
        return face_det_result, mesh_result, hand_result, expressions

    # ── drawing ────────────────────────────────────────────────────────────
    def draw_face_detections(self, frame, result):
        """Draw bounding boxes from FaceDetection."""
        if not result or not result.detections:
            return
        h, w = frame.shape[:2]
        for det in result.detections:
            bb = det.location_data.relative_bounding_box
            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), _COL_GREEN, 2)
            score = det.score[0] if det.score else 0.0
            cv2.putText(
                frame,
                f"Face {score:.2f}",
                (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _COL_GREEN,
                2,
                cv2.LINE_AA,
            )

    def draw_face_mesh(self, frame, mesh_result, expressions):
        """Draw face mesh contours and expression label."""
        if not mesh_result or not mesh_result.multi_face_landmarks:
            return
        h, w = frame.shape[:2]
        for idx, face_lms in enumerate(mesh_result.multi_face_landmarks):
            # Draw tessellation (the full mesh) — expensive to draw, adds to GPU load
            _MP_DRAW.draw_landmarks(
                image=frame,
                landmark_list=face_lms,
                connections=_MP_FACE_MESH.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=_MP_DRAW_STYLES.get_default_face_mesh_tesselation_style(),
            )
            # Draw contours on top
            _MP_DRAW.draw_landmarks(
                image=frame,
                landmark_list=face_lms,
                connections=_MP_FACE_MESH.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=_MP_DRAW_STYLES.get_default_face_mesh_contours_style(),
            )
            # Expression label anchored to nose tip
            nose = face_lms.landmark[_EXPR_NOSE_TIP]
            nx, ny = int(nose.x * w), int(nose.y * h)
            expr = expressions[idx] if idx < len(expressions) else "?"
            cv2.putText(
                frame,
                expr,
                (nx - 60, ny + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _COL_YELLOW,
                2,
                cv2.LINE_AA,
            )

    def draw_hands(self, frame, hand_result):
        """Draw hand skeleton and handedness label."""
        if not hand_result or not hand_result.multi_hand_landmarks:
            return
        h, w = frame.shape[:2]
        for idx, hand_lms in enumerate(hand_result.multi_hand_landmarks):
            _MP_DRAW.draw_landmarks(
                frame,
                hand_lms,
                _MP_HANDS.HAND_CONNECTIONS,
                _MP_DRAW_STYLES.get_default_hand_landmarks_style(),
                _MP_DRAW_STYLES.get_default_hand_connections_style(),
            )
            # Handedness label above wrist
            wrist = hand_lms.landmark[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            if hand_result.multi_handedness and idx < len(hand_result.multi_handedness):
                label = hand_result.multi_handedness[idx].classification[0].label
                score = hand_result.multi_handedness[idx].classification[0].score
            else:
                label, score = "Hand", 0.0
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (wx - 30, wy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _COL_ORANGE,
                2,
                cv2.LINE_AA,
            )

    def draw_hud(self, frame, phase):
        """Draw semi-transparent HUD panel at bottom of frame."""
        h, w = frame.shape[:2]
        panel_h = 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        n_faces = 0
        n_hands = 0
        if self._last_face_detections and self._last_face_detections.detections:
            n_faces = len(self._last_face_detections.detections)
        if self._last_hand_results and self._last_hand_results.multi_hand_landmarks:
            n_hands = len(self._last_hand_results.multi_hand_landmarks)

        phase_col = _COL_GREEN if phase == "baseline" else _COL_RED
        lines = [
            (f"Phase: {phase.upper()}", phase_col),
            (
                f"Color {self.color_fps.get():.1f} fps  |  "
                f"Depth {self.depth_fps.get():.1f} fps  |  "
                f"Display {self.display_fps.get():.1f} fps  |  "
                f"Inference {self.inference_fps.get():.1f} fps",
                _COL_WHITE,
            ),
            (
                f"Faces detected: {n_faces}   Hands detected: {n_hands}   "
                f"Elapsed: {self.current_elapsed():.1f}s",
                _COL_CYAN,
            ),
        ]

        y = h - panel_h + 26
        for text, col in lines:
            cv2.putText(
                frame, text, (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA,
            )
            y += 30

    # ── render loop ────────────────────────────────────────────────────────
    def render_loop(self):
        if self.done:
            return

        phase = self.current_phase()
        if phase == "done":
            self.finish_and_shutdown()
            return

        if self.latest_color is None:
            return

        frame = self.latest_color.copy()

        # ── STRESS: run all three detectors ───────────────────────────────
        if phase == "stress":
            (
                self._last_face_detections,
                self._last_mesh_results,
                self._last_hand_results,
                self._last_expressions,
            ) = self.run_detectors(frame)

            self.draw_face_detections(frame, self._last_face_detections)
            self.draw_face_mesh(frame, self._last_mesh_results, self._last_expressions)
            self.draw_hands(frame, self._last_hand_results)

        # ── BASELINE: raw stream, no inference ────────────────────────────
        # (intentionally empty — just display the frame as-is)

        self.draw_hud(frame, phase)

        if not self.first_render_logged:
            self.get_logger().info(
                f"First render: {frame.shape[1]}x{frame.shape[0]}"
            )
            self.first_render_logged = True

        if self.show_window and self._display_available:
            try:
                cv2.imshow("Detection Benchmark", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.get_logger().info("User pressed Q — stopping early.")
                    self.finish_and_shutdown()
                    return
            except Exception as e:
                self.get_logger().error(f"Display error: {e}")
                self._display_available = False

        self.display_fps.tick()

    # ── sample loop ────────────────────────────────────────────────────────
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
        self.records_inference_fps.append(self.inference_fps.get())
        self.records_phase.append(phase)

        inf_str = (
            f"inference_fps={self.inference_fps.get():.2f}"
            if phase == "stress"
            else "no inference (baseline)"
        )
        self.get_logger().info(
            f"phase={phase} t={local_t:.1f}s "
            f"color={self.color_fps.get():.2f} "
            f"depth={self.depth_fps.get():.2f} "
            f"display={self.display_fps.get():.2f} "
            f"{inf_str}"
        )

    # ── finish ─────────────────────────────────────────────────────────────
    def finish_and_shutdown(self):
        if self.done:
            return
        self.done = True

        if self.show_window and self._display_available:
            cv2.destroyAllWindows()

        # Release MediaPipe resources
        self.mp_face_det.close()
        self.mp_face_mesh.close()
        self.mp_hands.close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.output_dir, f"detection_benchmark_{ts}.png")
        raw_path = os.path.join(self.output_dir, f"detection_benchmark_{ts}.npz")

        t = np.array(self.records_t)
        color_fps = np.array(self.records_color_fps)
        depth_fps = np.array(self.records_depth_fps)
        display_fps = np.array(self.records_display_fps)
        inf_fps = np.array(self.records_inference_fps)
        phase = np.array(self.records_phase, dtype=object)

        np.savez(
            raw_path,
            t=t,
            color_fps=color_fps,
            depth_fps=depth_fps,
            display_fps=display_fps,
            inference_fps=inf_fps,
            phase=phase,
        )

        base_mask = phase == "baseline"
        stress_mask = phase == "stress"

        t_b, t_s = t[base_mask], t[stress_mask]
        c_b, c_s = color_fps[base_mask], color_fps[stress_mask]
        d_b, d_s = depth_fps[base_mask], depth_fps[stress_mask]
        disp_b, disp_s = display_fps[base_mask], display_fps[stress_mask]
        inf_s = inf_fps[stress_mask]

        def avg(x):
            return float(np.mean(x)) if len(x) > 0 else 0.0

        fig, axes = plt.subplots(4, 1, figsize=(13, 14), dpi=150, sharex=False)
        fig.suptitle(
            "RealSense Detection Benchmark\n"
            "Baseline: raw stream   |   Stress: FaceDetection + FaceMesh + Hands",
            fontsize=14,
            weight="bold",
            y=0.99,
        )

        # Color FPS
        axes[0].plot(t_b, c_b, color="#4CAF50", lw=2,
                     label=f"Baseline avg={avg(c_b):.1f}")
        axes[0].plot(t_s, c_s, color="#F44336", lw=2,
                     label=f"Stress avg={avg(c_s):.1f}")
        axes[0].set_ylabel("Color FPS")
        axes[0].set_title("Color Stream FPS", weight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Depth FPS
        axes[1].plot(t_b, d_b, color="#4CAF50", lw=2,
                     label=f"Baseline avg={avg(d_b):.1f}")
        axes[1].plot(t_s, d_s, color="#F44336", lw=2,
                     label=f"Stress avg={avg(d_s):.1f}")
        axes[1].set_ylabel("Depth FPS")
        axes[1].set_title("Depth Stream FPS", weight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Display FPS
        axes[2].plot(t_b, disp_b, color="#4CAF50", lw=2,
                     label=f"Baseline avg={avg(disp_b):.1f}")
        axes[2].plot(t_s, disp_s, color="#F44336", lw=2,
                     label=f"Stress avg={avg(disp_s):.1f}")
        axes[2].set_ylabel("Display FPS")
        axes[2].set_title("Display (render loop) FPS", weight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        # Inference FPS (stress only)
        axes[3].plot(t_s, inf_s, color="#FF9800", lw=2,
                     label=f"Inference avg={avg(inf_s):.1f}")
        axes[3].set_ylabel("Inference FPS")
        axes[3].set_xlabel("Time within phase (s)")
        axes[3].set_title(
            "Inference Throughput (stress only)\n"
            "FaceDetection + FaceMesh[refine=True, max_faces=4] + Hands[max=4]",
            weight="bold",
        )
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        self.get_logger().info(f"Saved figure   : {fig_path}")
        self.get_logger().info(f"Saved raw data : {raw_path}")
        self.get_logger().info("Benchmark complete.")

        self.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    try:
        node = DetectionBenchmarkNode()
    except RuntimeError:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            if hasattr(node, "_display_available") and node._display_available:
                cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()
