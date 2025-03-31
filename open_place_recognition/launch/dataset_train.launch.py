#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    dataset_path_arg = DeclareLaunchArgument(
        'dataset_path',
        default_value=f'{os.path.expanduser("~")}/Datasets/itlp_campus_outdoor/01_2023-02-21',
        description='Root directory of the database track (3D data).'
    )
    output_path_arg = DeclareLaunchArgument(
        'output_path',
        default_value=f'{os.path.expanduser("~")}/Datasets/itlp_campus_outdoor/01_2023-02-21',
        description='Where to save the index.faiss and any outputs.'
    )
    batch_size_arg = DeclareLaunchArgument(
        'batch_size',
        default_value='64',
        description='Batch size for the DataLoader.'
    )
    num_workers_arg = DeclareLaunchArgument(
        'num_workers',
        default_value='4',
        description='Number of CPU workers for DataLoader.'
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to run the model on (cuda or cpu).'
    )
    model_config_path_arg = DeclareLaunchArgument(
        'model_config_path',
        default_value=f'{os.path.expanduser("~")}/OpenPlaceRecognition/configs/model/place_recognition/minkloc3d.yaml',
        description='Path to MinkLoc3D Hydra config.'
    )
    weights_path_arg = DeclareLaunchArgument(
        'weights_path',
        default_value=f'{os.path.expanduser("~")}/OpenPlaceRecognition/weights/place_recognition/minkloc3d_nclt.pth',
        description='Path to the MinkLoc3D pre-trained weights.'
    )

    return LaunchDescription([
        dataset_path_arg,
        output_path_arg,
        batch_size_arg,
        num_workers_arg,
        device_arg,
        model_config_path_arg,
        weights_path_arg,

        Node(
            package='open_place_recognition',
            executable='dataset_train_node.py',
            name='dataset_train_node',
            output='screen',
            parameters=[{
                'dataset_path': LaunchConfiguration('dataset_path'),
                'output_path':  LaunchConfiguration('output_path'),
                'dataset_path': LaunchConfiguration('dataset_path'),
                'batch_size':   LaunchConfiguration('batch_size'),
                'num_workers':  LaunchConfiguration('num_workers'),
                'device':       LaunchConfiguration('device'),
                'model_config_path': LaunchConfiguration('model_config_path'),
                'weights_path': LaunchConfiguration('weights_path'),
            }],
        )
    ])
