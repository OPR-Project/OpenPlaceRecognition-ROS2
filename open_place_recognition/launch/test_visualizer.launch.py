from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    database_path = os.path.join(os.path.expanduser("~"), "Datasets/00_2023-10-25-night")

    return LaunchDescription([
        Node(
            package='open_place_recognition',
            executable='test_visualizer_node.py',
            name='place_recognition_visualizer',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    "database_dir": database_path,
                }
            ]
        )
    ])
