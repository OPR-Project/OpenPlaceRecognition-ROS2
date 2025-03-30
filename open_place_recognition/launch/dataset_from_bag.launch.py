#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    rosbag_file = os.path.expanduser('~/.ros/opr_dataset') # Adjust this path as needed.

    return LaunchDescription([
        Node(
            package='open_place_recognition',
            executable='dataset_from_rosbag_node.py',
            name='dataset_from_rosbag_node',
            output='screen',
            parameters=[
                {"front_camera_topic":  "/front_cam/camera_depth/image_raw"},
                {"back_camera_topic":   "/back_cam//camera_depth/image_raw"},
                {"lidar_topic":         "/lidar/points2_raw"},
                {"pose_topic":          "/my_pose_topic"},
                {"output_path":         "~/.ros/opr_dataset"},
                {"track_name":          "my_experiment_01"},
                # Optionally pass the rosbag file path as a parameter if your node needs it:
                {"rosbag_file": rosbag_file}
            ]
        ),
        # ExecuteProcess to run ros2 bag play using the specified rosbag directory
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', rosbag_file],
            output='screen'
        )
    ])

if __name__ == '__main__':
    generate_launch_description()
