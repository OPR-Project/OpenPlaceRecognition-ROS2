#!/usr/bin/env python3
import os
import sys
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import OpaqueFunction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    dataset_path = LaunchConfiguration('dataset_path')
    map_name = LaunchConfiguration('map_name')
    output_path =   LaunchConfiguration('output_path')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'dataset_path',
            default_value='~/.ros/opr_dataset',
            description='Path to the dataset directory'
        ),
        DeclareLaunchArgument(
            'map_name',
            default_value='my_map',
            description='Map name for dataset training'
        ),
        DeclareLaunchArgument(
            'output_path',
            default_value='~/.ros/opr_dataset',
            description='Map name for dataset creation'
        ),
        Node(
            package='orca_opr',
            executable='dataset_train_node.py',
            name='dataset_train_node',
            output='screen',
            parameters=[{
                'dataset_path': dataset_path,
                'map_name': map_name,
                'output_path': output_path,
            }],
        )
    ])
