from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    serial_no = LaunchConfiguration("serial_no")

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
            "pointcloud.enable": False,
            "align_depth.enable": False,
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
            "depth_topic": "/d435i/camera/depth/image_rect_raw",
            "cpu_loops": 0,
            "band_min_m": 0.4,
            "band_max_m": 1.5,
            "report_period_sec": 1.0,
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
