cmake_minimum_required(VERSION 3.5)
project(open_place_recognition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Find ament and any required dependencies (e.g. rclpy if used)
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Install launch files (matches glob pattern in setup.py)
install(
  DIRECTORY 
  configs
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/dataset_from_rosbag_node.py
  src/dataset_from_rtabmap_node.py
  src/dataset_publisher_node.py
  src/dataset_train_node.py
  src/localization_node.py
  src/place_recognition_node.py
#   src/test_depth_estimation_node.py
#   src/test_opr_odom_node.py
#   src/test_visualizer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
