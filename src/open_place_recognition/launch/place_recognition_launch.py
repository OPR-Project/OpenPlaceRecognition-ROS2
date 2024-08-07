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
                 "pipeline_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/configs/pipelines/place_recognition/multimodal_pr.yaml",
                #  "pipeline_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/configs/pipelines/place_recognition/multimodal_semantic_pr.yaml",
                #  "pipeline_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/configs/pipelines/place_recognition/multimodal_with_soc_outdoor_pr.yaml",
                #  "pipeline_cfg": "/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/configs/pipelines/place_recognition/multimodal_semantic_with_soc_outdoor_pr.yaml",
                 "image_resize": [320, 192]}
            ]
        )
    ])
