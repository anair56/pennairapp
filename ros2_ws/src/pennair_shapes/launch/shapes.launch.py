"""ros2 launch pennair_shapes shapes.launch.py video:=/path/to/video.mp4 [use_3d:=true] [loop:=true]"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    video = LaunchConfiguration('video')
    use_3d = LaunchConfiguration('use_3d')
    loop = LaunchConfiguration('loop')

    return LaunchDescription([
        DeclareLaunchArgument('video', description='path to the input video'),
        DeclareLaunchArgument('use_3d', default_value='true', description='estimate X/Y/Z from the circle'),
        DeclareLaunchArgument('loop', default_value='true', description='restart the video when it ends'),

        Node(package='pennair_shapes', executable='video_publisher', name='video_publisher',
             parameters=[{'video_path': video, 'loop': loop}], output='screen'),
        Node(package='pennair_shapes', executable='detector_node', name='shape_detector',
             parameters=[{'use_3d': use_3d}], output='screen'),
    ])
