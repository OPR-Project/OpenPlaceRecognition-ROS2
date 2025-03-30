#!/usr/bin/env python3
import os
import sys
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import OpaqueFunction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Even though the model node only uses 'map_name', we declare it here.
    map_name =      LaunchConfiguration('map_name')
    weight_path =   LaunchConfiguration('weight_path')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name',
            default_value='my_map',
            description='Map name for model operation'
        ),
        DeclareLaunchArgument(
            'weight_path',
            default_value='~/Sync/3d_map',
            description='Map name for model operation'
        ),
        Node(
            package='orca_opr',
            executable='test_opr_odom_node.py',
            name='opr_odom_node',
            output='screen',
            parameters=[{
                'map_name': map_name,
                'weight_path': weight_path,
            }],
        )
    ])
