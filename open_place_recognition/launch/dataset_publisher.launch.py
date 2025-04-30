#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments with default values
    dataset_dir_arg = DeclareLaunchArgument(
        'dataset_dir',
        default_value=os.path.join(os.path.expanduser("~"), "Datasets", "06_2023-08-18-night"),
        description='Path to the dataset directory (database path)'
    )
    enable_front_camera_arg = DeclareLaunchArgument(
        'enable_front_camera',
        default_value='true',
        description='Enable front camera'
    )
    enable_back_camera_arg = DeclareLaunchArgument(
        'enable_back_camera',
        default_value='true',
        description='Enable back camera'
    )
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar',
        default_value='true',
        description='Enable lidar'
    )
    enable_global_ref_arg = DeclareLaunchArgument(
        'enable_global_ref',
        default_value='false',
        description='Enable global reference'
    )
    global_ref_topic_arg = DeclareLaunchArgument(
        'global_ref_topic',
        default_value='/global_ref',
        description='Global reference topic'
    )
    reserve_arg = DeclareLaunchArgument(
        'reserve',
        default_value='false',
        description='Reserve parameter'
    )

    # Topic names
    front_cam_topic_arg = DeclareLaunchArgument(
        'front_cam_topic',
        default_value='/zed_node/left/image_rect_color/compressed',
        description='Front camera topic'
    )
    front_cam_mask_topic_arg = DeclareLaunchArgument(
        'front_cam_mask_topic',
        default_value='/zed_node/left/semantic_segmentation',
        description='Front camera mask topic'
    )
    front_cam_info_topic_arg = DeclareLaunchArgument(
        'front_cam_info_topic',
        default_value='/zed_node/left/image_rect_color/camera_info',
        description='Front camera info topic'
    )
    back_cam_topic_arg = DeclareLaunchArgument(
        'back_cam_topic',
        default_value='/realsense_back/color/image_raw/compressed',
        description='Back camera topic'
    )
    back_cam_mask_topic_arg = DeclareLaunchArgument(
        'back_cam_mask_topic',
        default_value='/realsense_back/semantic_segmentation',
        description='Back camera mask topic'
    )
    back_cam_info_topic_arg = DeclareLaunchArgument(
        'back_cam_info_topic',
        default_value='/realsense_back/color/image_raw/camera_info',
        description='Back camera info topic'
    )
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/velodyne_points',
        description='Lidar topic'
    )
    # TF frames
    tf_parent_frame_arg = DeclareLaunchArgument(
        'tf_parent_frame',
        default_value='base_link',
        description='TF parent frame'
    )
    front_cam_frame_arg = DeclareLaunchArgument(
        'front_cam_frame',
        default_value='zed_left',
        description='Front camera TF frame'
    )
    back_cam_frame_arg = DeclareLaunchArgument(
        'back_cam_frame',
        default_value='realsense_back',
        description='Back camera TF frame'
    )
    lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='velodyne',
        description='Lidar TF frame'
    )

    # New QoS arguments (separate for front camera, back camera, lidar, global ref)
    qos_front_cam_arg = DeclareLaunchArgument(
        'qos_front_cam',
        default_value='2',
        description='QoS for front camera (0=SystemDefault,1=BestEffort,2=Reliable)'
    )
    qos_back_cam_arg = DeclareLaunchArgument(
        'qos_back_cam',
        default_value='2',
        description='QoS for back camera (0=SystemDefault,1=BestEffort,2=Reliable)'
    )
    qos_lidar_arg = DeclareLaunchArgument(
        'qos_lidar',
        default_value='2',
        description='QoS for lidar (0=SystemDefault,1=BestEffort,2=Reliable)'
    )
    qos_global_ref_arg = DeclareLaunchArgument(
        'qos_global_ref',
        default_value='2',
        description='QoS for global ref subscription (0=SystemDefault,1=BestEffort,2=Reliable)'
    )

    # Use LaunchConfiguration substitutions to pass these values as parameters
    params = {
        "dataset_dir": LaunchConfiguration('dataset_dir'),
        "enable_front_camera": LaunchConfiguration('enable_front_camera'),
        "enable_back_camera": LaunchConfiguration('enable_back_camera'),
        "enable_lidar": LaunchConfiguration('enable_lidar'),
        "enable_global_ref": LaunchConfiguration('enable_global_ref'),
        "global_ref_topic": LaunchConfiguration('global_ref_topic'),
        "reserve": LaunchConfiguration('reserve'),
        # Topic names
        "front_cam_topic": LaunchConfiguration('front_cam_topic'),
        "front_cam_mask_topic": LaunchConfiguration('front_cam_mask_topic'),
        "front_cam_info_topic": LaunchConfiguration('front_cam_info_topic'),
        "back_cam_topic": LaunchConfiguration('back_cam_topic'),
        "back_cam_mask_topic": LaunchConfiguration('back_cam_mask_topic'),
        "back_cam_info_topic": LaunchConfiguration('back_cam_info_topic'),
        "lidar_topic": LaunchConfiguration('lidar_topic'),
        # TF frames
        "tf_parent_frame": LaunchConfiguration('tf_parent_frame'),
        "front_cam_frame": LaunchConfiguration('front_cam_frame'),
        "back_cam_frame": LaunchConfiguration('back_cam_frame'),
        "lidar_frame": LaunchConfiguration('lidar_frame'),
        # QoS
        "qos_front_cam": LaunchConfiguration('qos_front_cam'),
        "qos_back_cam": LaunchConfiguration('qos_back_cam'),
        "qos_lidar": LaunchConfiguration('qos_lidar'),
        "qos_global_ref": LaunchConfiguration('qos_global_ref'),
    }

    return LaunchDescription([
        # Declare all arguments
        dataset_dir_arg,
        enable_front_camera_arg,
        enable_back_camera_arg,
        enable_lidar_arg,
        enable_global_ref_arg,
        global_ref_topic_arg,
        reserve_arg,
        front_cam_topic_arg,
        front_cam_mask_topic_arg,
        front_cam_info_topic_arg,
        back_cam_topic_arg,
        back_cam_mask_topic_arg,
        back_cam_info_topic_arg,
        lidar_topic_arg,
        tf_parent_frame_arg,
        front_cam_frame_arg,
        back_cam_frame_arg,
        lidar_frame_arg,
        qos_front_cam_arg,
        qos_back_cam_arg,
        qos_lidar_arg,
        qos_global_ref_arg,
        # Launch the node with the parameters
        Node(
            package='open_place_recognition',
            executable='dataset_publisher_node.py',
            name='opr_dataset_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[params]
        )
    ])
