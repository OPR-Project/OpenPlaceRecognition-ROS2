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
                 "mask_front_topic": "/zed_node/left/semantic_segmentation",
                 "mask_back_topic": "/realsense_back/semantic_segmentation",
                 "lidar_topic": "/velodyne_points",
                #  "database_dir": "/home/docker_opr_ros2/Datasets/itlpcampus_nature_exps/databases/indoor_floor_5",
                 "database_dir": "/home/docker_opr_ros2/Datasets/itlpcampus_nature_exps/databases/outdoor_2023-04-11-day",

                #  "model_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multi-image_lidar_late-fusion.yaml",
                #  "model_weights_path": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multi-image_lidar_late-fusion_nclt.pth",

                #  "model_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multi-image_multi-semantic_lidar_late-fusion.yaml",
                #  "model_weights_path": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multi-image_multi-semantic_lidar_late-fusion_nclt.pth",

                 "model_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multimodal_with_soc_outdoor.yaml",
                 "model_weights_path": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/multimodal_with_soc_outdoor_nclt.pth",

                 "device": "cuda",
                 "image_resize": [320, 192]}
            ]
        )
    ])
