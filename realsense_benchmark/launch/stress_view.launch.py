from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="d435i",
        name="camera",
        output="screen",
        parameters=[{
            "camera_name": "d435i",
            "serial_no": "_419622072439",
            "enable_color": True,
            "enable_depth": True,
            "enable_infra1": False,
            "enable_infra2": False,
            "enable_gyro": False,
            "enable_accel": False,
            "pointcloud.enable": True,
            "align_depth.enable": True,
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
            "depth_topic": "/d435i/camera/aligned_depth_to_color/image_raw",
            "cpu_loops": 8,
            "band_min_m": 0.4,
            "band_max_m": 1.5,
            "report_period_sec": 1.0,
        }],
    )

    return LaunchDescription([
        camera_node,
        benchmark_node,
    ])
