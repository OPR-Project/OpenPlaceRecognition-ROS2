# launch/convert.launch.py
import os
import launch
import launch_ros.actions

home_dir = os.path.expanduser("~")

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='your_package_name',
            executable='bag_converter_node',
            name='bag_converter_node',
            parameters=[
                {"input_dir":       os.path.join(home_dir, 'ros2_bags')},
                {"trajectory_file": os.path.join(home_dir, 'ros2_bags/trajectory.db3')},
                {"out_dir":         os.path.expanduser('~/.ros/opr_dataset')},
                {"distance_threshold": 5.0},
                {"max_diff": 60000000},
                {"front_cam_topic": "/front/depth_camera/image_raw"},
                {"back_cam_topic":  "/back/depth_camera/image_raw"},
                {"lidar_topic":     "/lidar/points2_raw"},
                {"trajectory_topic": "/global_trajectory_0"},
            ],
        )
    ])
