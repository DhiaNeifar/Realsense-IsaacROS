from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    show_debug_window = LaunchConfiguration("show_debug_window")

    return LaunchDescription([
        DeclareLaunchArgument("color_topic", default_value="/d435i/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/d435i/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/d435i/camera/color/camera_info"),
        DeclareLaunchArgument("show_debug_window", default_value="true"),
        Node(
            package="floor_object_detection",
            executable="floor_object_detector",
            name="floor_object_detector",
            output="screen",
            parameters=[{
                "color_topic": color_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
                "show_debug_window": ParameterValue(show_debug_window, value_type=bool),
            }],
        )
    ])
