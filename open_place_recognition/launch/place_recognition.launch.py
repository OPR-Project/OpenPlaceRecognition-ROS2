from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the share directory of the package
    config_dir = os.path.join(
        get_package_share_directory('open_place_recognition'),
        'configs/pipelines/place_recognition'
    )

    # Declare launch arguments for configurable parameters
    launch_arguments = [
        DeclareLaunchArgument(
            'image_front_topic',
            default_value='/zed_node/left/image_rect_color/compressed',
            description='Front camera image topic.'
        ),
        DeclareLaunchArgument(
            'image_back_topic',
            default_value='/realsense_back/color/image_raw/compressed',
            description='Back camera image topic.'
        ),
        DeclareLaunchArgument(
            'mask_front_topic',
            default_value='/zed_node/left/semantic_segmentation',
            description='Front semantic segmentation mask topic.'
        ),
        DeclareLaunchArgument(
            'mask_back_topic',
            default_value='/realsense_back/semantic_segmentation',
            description='Back semantic segmentation mask topic.'
        ),
        DeclareLaunchArgument(
            'lidar_topic',
            default_value='/velodyne_points',
            description='Lidar pointcloud topic.'
        ),
        DeclareLaunchArgument(
            'dataset_dir',
            default_value=os.path.join(os.path.expanduser("~"), "Datasets/itlp_campus_outdoor/01_2023-02-21"),
            description='Path to the dataset directory (database path)'
        ),
        DeclareLaunchArgument(
            'pipeline_cfg',
            default_value=os.path.join(config_dir, 'multimodal_pr.yaml'),
            description='Path to the pipeline configuration file.'
        ),
        DeclareLaunchArgument(
            'image_resize',
            default_value='[320, 192]',
            description='Image resize dimensions.'
        ),
        # New arguments for sensor enable/disable and global reference system
        DeclareLaunchArgument(
            'enable_front_camera',
            default_value='true',
            description='Enable front camera.'
        ),
        DeclareLaunchArgument(
            'enable_back_camera',
            default_value='true',
            description='Enable back camera.'
        ),
        DeclareLaunchArgument(
            'enable_lidar',
            default_value='true',
            description='Enable lidar sensor.'
        ),
        DeclareLaunchArgument(
            'enable_global_ref',
            default_value='true',
            description='Enable global reference system'
        ),
        DeclareLaunchArgument(
            'global_ref_topic',
            default_value='/global_ref',
            description='Global reference system topic (e.g. GPS/Barometer, WGS84).'
        ),
        DeclareLaunchArgument(
            'reserve',
            default_value='false',
            description='Reserve variable for future use.'
        )
    ]

    # Use LaunchConfiguration substitutions for all parameters
    node_parameters = {
        "image_front_topic":    LaunchConfiguration("image_front_topic"),
        "image_back_topic":     LaunchConfiguration("image_back_topic"),
        "mask_front_topic":     LaunchConfiguration("mask_front_topic"),
        "mask_back_topic":      LaunchConfiguration("mask_back_topic"),
        "lidar_topic":          LaunchConfiguration("lidar_topic"),
        "dataset_dir":          LaunchConfiguration("dataset_dir"),
        "pipeline_cfg":         LaunchConfiguration("pipeline_cfg"),
        "image_resize":         LaunchConfiguration("image_resize"),
        "enable_front_camera":  LaunchConfiguration("enable_front_camera"),
        "enable_back_camera":   LaunchConfiguration("enable_back_camera"),
        "enable_lidar":         LaunchConfiguration("enable_lidar"),
        "enable_global_ref":    LaunchConfiguration("enable_global_ref"),
        "global_ref_topic":     LaunchConfiguration("global_ref_topic"),
        "reserve":              LaunchConfiguration("reserve"),
    }

    # Create the Node action with parameters from LaunchConfiguration
    node = Node(
        package='open_place_recognition',
        executable='place_recognition_node.py',
        name='multimodal_multicamera_lidar_place_recognition',
        output='screen',
        emulate_tty=True,
        parameters=[node_parameters]
    )

    return LaunchDescription(launch_arguments + [node])

if __name__ == '__main__':
    generate_launch_description()
