from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='floor_box_perception',
            executable='box_floor_detector',
            name='box_floor_detector',
            output='screen',
            parameters=[{
                'color_topic': '/d435i/camera/color/image_raw',
                'depth_topic': '/d435i/camera/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/d435i/camera/color/camera_info',
                'min_depth_m': 0.20,
                'max_depth_m': 2.00,
                'min_contour_area': 2500,
                'depth_scale': 0.001,
                'foreground_margin_m': 0.05,
                'show_debug_window': True,
                'window_name': 'floor_box_detector',
                'smoothing_alpha': 0.30,
                'point_depth_window': 5,
                'log_points_period_sec': 0.50,
            }]
        )
    ])
