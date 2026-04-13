from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_stream_benchmark_launch_description(
    *,
    aligned_depth: bool,
    cpu_loops: int,
) -> LaunchDescription:
    """Generate the standard D435i + live benchmark launch description."""

    serial_no = LaunchConfiguration("serial_no")
    depth_topic = (
        "/d435i/camera/aligned_depth_to_color/image_raw"
        if aligned_depth
        else "/d435i/camera/depth/image_rect_raw"
    )

    camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="d435i",
        name="camera",
        output="screen",
        parameters=[{
            "camera_name": "d435i",
            "serial_no": serial_no,
            "enable_color": True,
            "enable_depth": True,
            "enable_infra1": False,
            "enable_infra2": False,
            "enable_gyro": False,
            "enable_accel": False,
            "pointcloud.enable": aligned_depth,
            "align_depth.enable": aligned_depth,
            "rgb_camera.color_profile": "640x480x60",
            "depth_module.depth_profile": "640x480x60",
        }],
    )

    benchmark_node = Node(
        package="realsense_benchmark",
        executable="live_benchmark_node",
        name="live_benchmark_node",
        output="screen",
        parameters=[{
            "color_topic": "/d435i/camera/color/image_raw",
            "depth_topic": depth_topic,
            "cpu_loops": cpu_loops,
            "band_min_m": 0.4,
            "band_max_m": 1.5,
            "report_period_sec": 1.0,
            "show_window": True,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "serial_no",
            default_value=EnvironmentVariable("D435I_SERIAL", default_value=""),
            description="RealSense D435i serial number, including the leading underscore.",
        ),
        camera_node,
        benchmark_node,
    ])
