# `floor_object_detection`

`floor_object_detection` is a ROS 2 package for detecting and tracking objects that protrude above the floor using RGB-D data.

The package is designed for Intel RealSense cameras and is intended to be:

- easier to understand than the previous `floor_box_perception` package
- more stable frame to frame
- less sensitive to floor color and lighting
- more useful to downstream ROS nodes through explicit detection topics

## What The Package Does

The node subscribes to:

- a color image
- a depth image
- camera intrinsics

It then:

- estimates the floor geometry from depth when camera intrinsics are available
- segments foreground objects based on height above the floor
- falls back to a depth-only local background model if plane fitting is unavailable
- selects the most plausible floor object candidate
- tracks that candidate across frames to reduce twitching
- publishes debug, mask, and structured detection outputs

## Published Topics

All outputs are relative to the node name `floor_object_detector`.

- `~/debug_image` as `sensor_msgs/msg/Image`
- `~/foreground_mask` as `sensor_msgs/msg/Image`
- `~/floor_mask` as `sensor_msgs/msg/Image`
- `~/detections_2d` as `vision_msgs/msg/Detection2DArray`
- `~/box_center` as `geometry_msgs/msg/PointStamped`
- `~/box_pose` as `geometry_msgs/msg/PoseStamped`
- `~/marker_array` as `visualization_msgs/msg/MarkerArray`
- `~/distance` as `std_msgs/msg/Float32`

With the default node name, these resolve to:

- `/floor_object_detector/debug_image`
- `/floor_object_detector/foreground_mask`
- `/floor_object_detector/floor_mask`
- `/floor_object_detector/detections_2d`
- `/floor_object_detector/box_center`
- `/floor_object_detector/box_pose`
- `/floor_object_detector/marker_array`
- `/floor_object_detector/distance`

## Default Inputs

- `color_topic`: `/d435i/camera/color/image_raw`
- `depth_topic`: `/d435i/camera/aligned_depth_to_color/image_raw`
- `camera_info_topic`: `/d435i/camera/color/camera_info`

These defaults fit a D435i with aligned depth enabled. For D405, override the topics.

## Why This Is More Robust

- Detection is depth-first instead of relying on RGB edges.
- White or low-texture floors are no longer a primary failure mode.
- A temporal tracker smooths the box and 3D center instead of only smoothing a rectangle.
- The tracker keeps detections alive through short dropouts, which reduces twitching.
- The node publishes structured ROS outputs for downstream consumers.

## Parameters

Topic and runtime parameters:

- `color_topic` (`/d435i/camera/color/image_raw`)
- `depth_topic` (`/d435i/camera/aligned_depth_to_color/image_raw`)
- `camera_info_topic` (`/d435i/camera/color/camera_info`)
- `sync_slop_sec` (`0.08`)
- `show_debug_window` (`true`)
- `window_name` (`floor_object_detector`)
- `log_detection_period_sec` (`1.0`)
- `marker_lifetime_sec` (`0.25`)

Depth and geometry parameters:

- `min_depth_m` (`0.20`)
- `max_depth_m` (`2.00`)
- `depth_scale` (`0.001`)
- `plane_ransac_iterations` (`80`)
- `plane_inlier_threshold_m` (`0.02`)
- `min_floor_points` (`180`)
- `min_plane_y_component` (`0.35`)
- `plane_sample_stride` (`4`)
- `min_height_above_floor_m` (`0.03`)
- `max_height_above_floor_m` (`0.60`)

Fallback depth segmentation parameters:

- `fallback_foreground_margin_m` (`0.05`)
- `local_background_sigma` (`31.0`)

Candidate filtering parameters:

- `search_top_ignore_ratio` (`0.10`)
- `min_contour_area` (`1800`)
- `min_bbox_size_px` (`28`)
- `max_bbox_aspect_ratio` (`4.0`)
- `min_extent` (`0.20`)
- `min_solidity` (`0.45`)
- `open_kernel_size` (`5`)
- `close_kernel_size` (`7`)
- `point_depth_window` (`5`)

Tracking parameters:

- `min_confirmed_frames` (`2`)
- `max_missed_frames` (`4`)
- `bbox_smoothing_alpha` (`0.35`)
- `center_smoothing_alpha` (`0.25`)
- `max_center_jump_px` (`140.0`)
- `max_depth_jump_m` (`0.30`)

## Run Examples

D435i:

```bash
ros2 run floor_object_detection floor_object_detector --ros-args \
  -p color_topic:=/d435i/camera/color/image_raw \
  -p depth_topic:=/d435i/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/d435i/camera/color/camera_info \
  -p show_debug_window:=true
```

D405:

```bash
ros2 run floor_object_detection floor_object_detector --ros-args \
  -p color_topic:=/d405/camera/color/image_raw \
  -p depth_topic:=/d405/camera/depth/image_rect_raw \
  -p camera_info_topic:=/d405/camera/color/camera_info \
  -p show_debug_window:=true
```

Launch file:

```bash
ros2 launch floor_object_detection detector.launch.py
```

Launch arguments:

- `color_topic`
- `depth_topic`
- `camera_info_topic`
- `show_debug_window`

## Practical Notes

- The best results come from aligned depth when color and depth resolutions differ.
- `box_pose` currently represents the object center with identity orientation.
- If camera intrinsics are unavailable, the node falls back to a depth-only local background model.
- RViz can consume `~/marker_array`, while downstream logic can consume `~/detections_2d` and `~/box_center`.

## License

Apache-2.0
