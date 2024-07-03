from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='open_place_recognition',
            executable='visualizer',
            name='place_recognition_visualizer',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    # "database_dir": "/home/docker_opr_ros2/Datasets/itlpcampus_nature_exps/databases/indoor_floor_5",
                    "database_dir": "/home/docker_opr_ros2/Datasets/itlpcampus_nature_exps/databases/outdoor_2023-04-11-day",
                }
            ]
        )
    ])
