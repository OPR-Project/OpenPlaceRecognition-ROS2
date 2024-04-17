from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='open_place_recognition',
            executable='place_recognition',
            name='multimodal_multicamera_lidar_place_recognition',
            output='screen',
            emulate_tty=True,
            parameters=[
                {"image_front_topic": "/zed_node/left/image_rect_color/compressed",
                 "image_back_topic": "/realsense_back/color/image_raw/compressed",
                 "lidar_topic": "/velodyne_points",
                 "database_dir": "/home/docker_opr_ros2/Datasets/sample_place_recognition_db_2023-02-10",
                 "model_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multi-image_lidar_late-fusion.yaml",
                 "model_weights_path": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multi-image_lidar_late-fusion_nclt.pth",
                 "device": "cuda",
                 "image_resize": [320, 192]}
            ]
        )
    ])
