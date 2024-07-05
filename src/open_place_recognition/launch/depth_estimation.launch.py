from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='open_place_recognition',
            executable='depth_estimation',
            name='depth_estimation_with_lidar',
            output='screen',
            emulate_tty=True,
            parameters=[
                {"image_front_topic": "/zed_node/left/image_rect_color/compressed",
                 "camera_info_front_topic": "/zed_node/left/camera_info",
                 "lidar_topic": "/velodyne_points",
                 "publish_point_cloud_form_depth": True,
                 "model_weights_path": "/home/docker_opr_ros2/ros2_ws/dependencies/OpenPlaceRecognition/weights/depth_estimation/res50.pth",
                 "device": "cuda",
                 "image_resize": [640, 480]}
            ]
        )
    ])
