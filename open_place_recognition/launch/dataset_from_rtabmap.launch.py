#!/usr/bin/env python3
import os
import sys
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import OpaqueFunction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    map_name =      LaunchConfiguration('map_name')
    input_path =  LaunchConfiguration('input_path')
    output_path =   LaunchConfiguration('output_path')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'input_path',
            default_value='~/Sync/map',
            description='Path to the dataset directory'
        ),
        DeclareLaunchArgument(
            'map_name',
            default_value='my_map',
            description='Map name for dataset creation'
        ),
        DeclareLaunchArgument(
            'output_path',
            default_value='~/.ros/opr_dataset',
            description='Map name for dataset creation'
        ),
        Node(
            package='orca_opr',
            executable='dataset_from_rtabmap_node.py',
            name='dataset_from_rtabmap',
            output='screen',
            parameters=[{
                'input_path': input_path,
                'map_name': map_name,
                'output_path': output_path,
            }],
        )
    ])
