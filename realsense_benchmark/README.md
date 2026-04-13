# `realsense_benchmark`

`realsense_benchmark` is a ROS 2 Python package for measuring how Intel RealSense RGB-D streams behave under different visualization and processing loads.

The package is useful when you want to answer questions such as:

- Is the camera publishing color and depth at the expected rate?
- How much display-side processing reduces effective frame rate?
- How much additional perception or inference load the system can tolerate?
- What changes when aligned depth is enabled instead of raw depth?

This package does not publish benchmark results as ROS topics. Its outputs are:

- terminal logs
- OpenCV debug windows
- saved `.png` figures
- saved `.npz` raw benchmark data files

## Package Contents

Console executables:

- `live_benchmark_node`
- `phase_benchmark_node`
- `detection_benchmark_node`

Launch files:

- `launch/benchmark_view.launch.py`
- `launch/stress_view.launch.py`

## What Each Node Does

### `live_benchmark_node`

Purpose:
- Subscribe to a color stream and a depth stream
- Render color and depth side by side
- Highlight a configurable depth band
- Report live stream and display rates

Inputs:
- color image topic as `sensor_msgs/msg/Image`
- depth image topic as `sensor_msgs/msg/Image`

Outputs:
- OpenCV window titled `RealSense Benchmark`
- periodic log messages with `color_fps`, `depth_fps`, `display_fps`, and `proc_ms`

This node is best for quick sanity checks and visual inspection.

### `phase_benchmark_node`

Purpose:
- Run a baseline phase and a stress phase in one session
- Apply additional synthetic image-processing load in the stress phase
- Save plots and raw measurements for later comparison

Inputs:
- color image topic as `sensor_msgs/msg/Image`
- depth image topic as `sensor_msgs/msg/Image`

Outputs:
- OpenCV window titled `Phase Benchmark`
- periodic log messages with phase-local FPS metrics
- `phase_compare_<timestamp>.png`
- `phase_compare_<timestamp>.npz`

This node is best when you want reproducible before/after measurements without adding an external inference dependency.

### `detection_benchmark_node`

Purpose:
- Run a baseline phase and an inference-heavy stress phase
- Measure the effect of MediaPipe face and hand pipelines on stream throughput and rendering

Stress-phase load:
- Face Detection
- Face Mesh
- Hand tracking

Inputs:
- color image topic as `sensor_msgs/msg/Image`
- depth image topic as `sensor_msgs/msg/Image`

Outputs:
- OpenCV window titled `Detection Benchmark`
- periodic log messages with stream FPS, display FPS, and inference FPS
- `detection_benchmark_<timestamp>.png`
- `detection_benchmark_<timestamp>.npz`

Important note:
- The detection benchmark uses the color stream for inference.
- The depth stream is still subscribed and measured, but depth is not used for the MediaPipe detections themselves.

## Topics and Message Types

All nodes subscribe to `sensor_msgs/msg/Image`.

Typical D435i topics used in this repository:

- color: `/d435i/camera/color/image_raw`
- raw depth: `/d435i/camera/depth/image_rect_raw`
- aligned depth: `/d435i/camera/aligned_depth_to_color/image_raw`

Default topic behavior by node:

- `live_benchmark_node`
  - default color topic: `/d435i/camera/color/image_raw`
  - default depth topic: `/d435i/camera/depth/image_rect_raw`
- `phase_benchmark_node`
  - default color topic: `/d435i/camera/color/image_raw`
  - default depth topic: `/d435i/camera/depth/image_rect_raw`
- `detection_benchmark_node`
  - default color topic: `/d435i/camera/color/image_raw`
  - default depth topic: `/d435i/camera/depth/image_rect_raw`

## Launch Files

### `benchmark_view.launch.py`

Starts:

- `realsense2_camera_node` in namespace `/d435i`
- `live_benchmark_node`

Behavior:

- uses raw depth topic `/d435i/camera/depth/image_rect_raw`
- sets `cpu_loops=0`
- intended for a lightweight live benchmark session

Launch argument:

- `serial_no`
  - default source: environment variable `D435I_SERIAL`
  - expected format: RealSense serial string including the leading underscore

Example:

```bash
ros2 launch realsense_benchmark benchmark_view.launch.py
```

### `stress_view.launch.py`

Starts:

- `realsense2_camera_node` in namespace `/d435i`
- `live_benchmark_node`

Behavior:

- enables aligned depth
- uses `/d435i/camera/aligned_depth_to_color/image_raw`
- sets `cpu_loops=8`
- intended for a heavier live rendering session

Launch argument:

- `serial_no`
  - default source: environment variable `D435I_SERIAL`

Example:

```bash
ros2 launch realsense_benchmark stress_view.launch.py
```

Important note:
- These launch files are D435i-focused convenience wrappers.
- If you want to benchmark a different camera namespace, run the nodes directly and override the topics.

## Node Arguments

All parameters below are standard ROS parameters and can be passed with `--ros-args -p ...`.

### `live_benchmark_node` parameters

- `color_topic` (`string`, default `/d435i/camera/color/image_raw`)
  - color image topic to subscribe to
- `depth_topic` (`string`, default `/d435i/camera/depth/image_rect_raw`)
  - depth image topic to subscribe to
- `cpu_loops` (`int`, default `0`)
  - number of extra OpenCV filter passes applied to the depth mask
- `band_min_m` (`float`, default `0.4`)
  - minimum depth in meters for the highlighted band
- `band_max_m` (`float`, default `1.5`)
  - maximum depth in meters for the highlighted band
- `report_period_sec` (`float`, default `1.0`)
  - interval between terminal metric reports
- `show_window` (`bool`, default `true`)
  - whether to show the OpenCV display window

Example:

```bash
ros2 run realsense_benchmark live_benchmark_node --ros-args \
  -p color_topic:=/d435i/camera/color/image_raw \
  -p depth_topic:=/d435i/camera/depth/image_rect_raw \
  -p cpu_loops:=4 \
  -p band_min_m:=0.4 \
  -p band_max_m:=1.2 \
  -p report_period_sec:=1.0 \
  -p show_window:=true
```

### `phase_benchmark_node` parameters

- `color_topic` (`string`, default `/d435i/camera/color/image_raw`)
- `depth_topic` (`string`, default `/d435i/camera/depth/image_rect_raw`)
- `baseline_duration_sec` (`float`, default `30.0`)
  - duration of the baseline phase
- `stress_duration_sec` (`float`, default `30.0`)
  - duration of the stress phase
- `stress_cpu_loops` (`int`, default `8`)
  - extra synthetic processing load applied during the stress phase
- `band_min_m` (`float`, default `0.4`)
- `band_max_m` (`float`, default `1.5`)
- `sample_hz` (`float`, default `5.0`)
  - metric sampling rate for saved benchmark data
- `render_hz` (`float`, default `30.0`)
  - render-loop timer frequency
- `output_dir` (`string`, default `~/ros2_ws/src/realsense_benchmark/results`)
  - directory for `.png` and `.npz` outputs
- `show_window` (`bool`, default `true`)

Example:

```bash
ros2 run realsense_benchmark phase_benchmark_node --ros-args \
  -p color_topic:=/d435i/camera/color/image_raw \
  -p depth_topic:=/d435i/camera/depth/image_rect_raw \
  -p baseline_duration_sec:=30.0 \
  -p stress_duration_sec:=30.0 \
  -p stress_cpu_loops:=8 \
  -p sample_hz:=5.0 \
  -p render_hz:=30.0 \
  -p output_dir:=/tmp/realsense_benchmark_results \
  -p show_window:=true
```

Saved `.npz` keys:

- `t`
- `color_fps`
- `depth_fps`
- `display_fps`
- `phase`

### `detection_benchmark_node` parameters

- `color_topic` (`string`, default `/d435i/camera/color/image_raw`)
- `depth_topic` (`string`, default `/d435i/camera/depth/image_rect_raw`)
- `baseline_duration_sec` (`float`, default `30.0`)
- `stress_duration_sec` (`float`, default `30.0`)
- `sample_hz` (`float`, default `5.0`)
- `render_hz` (`float`, default `30.0`)
- `output_dir` (`string`, default `~/ros2_ws/src/realsense_benchmark/results`)
- `show_window` (`bool`, default `true`)

Example:

```bash
python3 -m pip install -U mediapipe --break-system-packages

ros2 run realsense_benchmark detection_benchmark_node --ros-args \
  -p color_topic:=/d435i/camera/color/image_raw \
  -p depth_topic:=/d435i/camera/depth/image_rect_raw \
  -p baseline_duration_sec:=30.0 \
  -p stress_duration_sec:=30.0 \
  -p sample_hz:=5.0 \
  -p render_hz:=30.0 \
  -p output_dir:=/tmp/realsense_benchmark_results \
  -p show_window:=true
```

Saved `.npz` keys:

- `t`
- `color_fps`
- `depth_fps`
- `display_fps`
- `inference_fps`
- `phase`

## Dependencies

Required runtime dependencies:

- ROS 2 Humble
- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- `numpy`
- `opencv-python` or system OpenCV with Python bindings
- `matplotlib` for result plots in the phase and detection benchmarks

Optional dependency:

- `mediapipe` for `detection_benchmark_node`

If MediaPipe is not installed, `detection_benchmark_node` exits with a fatal error instead of running partially.

## Typical Workflows

### 1. Quick stream sanity check

Use:
- `live_benchmark_node`

Best when:
- you want to confirm that the camera is streaming correctly
- you want a quick side-by-side view of color and depth
- you want to see whether aligned depth changes display responsiveness

### 2. Reproducible baseline vs stress comparison

Use:
- `phase_benchmark_node`

Best when:
- you want saved benchmark plots
- you want to measure the effect of controlled synthetic load
- you want repeatable runs across machines or camera settings

### 3. Estimate impact of inference-heavy color processing

Use:
- `detection_benchmark_node`

Best when:
- you want to see how much headroom is left once vision pipelines are running
- you want a stress case closer to real perception workloads than simple image filtering

## QoS and Display Notes

- `phase_benchmark_node` and `detection_benchmark_node` use an explicit RELIABLE image QoS profile.
- `live_benchmark_node` uses BEST_EFFORT image QoS for a lighter-weight live stream view.
- If `show_window=true` but no display is available, the nodes disable the OpenCV window and continue running.

## Current Limitations

- The package is focused on benchmark visualization and measurement, not closed-loop robotics behavior.
- Launch files currently assume a D435i namespace and RealSense driver configuration.
- `detection_benchmark_node` measures the impact of color inference pipelines, not RGB-D object detection.
- Result files are written locally; there is no ROS message or service interface for benchmark reports.

## Recommended Output Handling

The package writes generated artifacts into a `results/` directory. These files are intended to be local benchmark outputs and should not normally be committed to git.

## Source Layout

- `realsense_benchmark/live_benchmark_node.py`
- `realsense_benchmark/phase_benchmark_node.py`
- `realsense_benchmark/detection_benchmark_node.py`
- `realsense_benchmark/common.py`
- `realsense_benchmark/depth_tools.py`
- `realsense_benchmark/launch_utils.py`

## License

Apache-2.0
