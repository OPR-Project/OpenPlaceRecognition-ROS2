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
                 "model_weights_path": "/home/docker_opr_ros2/ros2_ws/dependencies/OpenPlaceRecognition/weights/depth_estimation/depth_anything_v2_metric_hypersim_vits.pth",
                 "model_type": "DepthAnything",
                 "align_type": "regression",
                 "mode": "indoor",
                 "device": "cuda",
                 "image_resize": [640, 480]}
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='publish_tf_lidar_to_camera',
            arguments="0.061 0.049 -0.131 -0.498, 0.498, -0.495, 0.510 velodyne zed_left_camera_optical_frame".split(" ")
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='publish_tf_base_to_lidar',
            arguments="-0.300 0.014 0.883 -0.016 0.009 -0.015 base_link velodyne".split(" ")
        )
    ])
