[place_recognition_node.py-1] 2025-03-31 14:33:42.986 | WARNING  | opr.models.place_recognition.pointmamba:<module>:16 - The 'pointmamba' package is not installed. Please install it manually if neccessary.
[place_recognition_node.py-1] [INFO] [1743431623.286569389] [multimodal_multicamera_lidar_place_recognition]: Initialized PlaceRecognitionNode node.
[place_recognition_node.py-1] [INFO] [1743431696.078611049] [multimodal_multicamera_lidar_place_recognition]: Received synchronized messages.
[place_recognition_node.py-1] Traceback (most recent call last):
[place_recognition_node.py-1]   File "/home/docker_opr_ros2/ros2_ws/install/open_place_recognition/lib/open_place_recognition/place_recognition_node.py", line 377, in <module>
[place_recognition_node.py-1]     main()
[place_recognition_node.py-1]   File "/home/docker_opr_ros2/ros2_ws/install/open_place_recognition/lib/open_place_recognition/place_recognition_node.py", line 371, in main
[place_recognition_node.py-1]     rclpy.spin(pr_node)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py", line 226, in spin
[place_recognition_node.py-1]     executor.spin_once()
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 739, in spin_once
[place_recognition_node.py-1]     self._spin_once_impl(timeout_sec)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 736, in _spin_once_impl
[place_recognition_node.py-1]     raise handler.exception()
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/task.py", line 239, in __call__
[place_recognition_node.py-1]     self._handler.send(None)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 437, in handler
[place_recognition_node.py-1]     await call_coroutine(entity, arg)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 362, in _execute_subscription
[place_recognition_node.py-1]     await await_or_execute(sub.callback, msg)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 107, in await_or_execute
[place_recognition_node.py-1]     return callback(*args)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/message_filters/__init__.py", line 83, in callback
[place_recognition_node.py-1]     self.signalMessage(msg)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/message_filters/__init__.py", line 64, in signalMessage
[place_recognition_node.py-1]     cb(*(msg + args))
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/message_filters/__init__.py", line 313, in add
[place_recognition_node.py-1]     self.signalMessage(*msgs)
[place_recognition_node.py-1]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/message_filters/__init__.py", line 64, in signalMessage
[place_recognition_node.py-1]     cb(*(msg + args))
[place_recognition_node.py-1]   File "/home/docker_opr_ros2/ros2_ws/install/open_place_recognition/lib/open_place_recognition/place_recognition_node.py", line 354, in listener_callback
[place_recognition_node.py-1]     output = self.pipeline.infer(input_data)
[place_recognition_node.py-1]   File "/home/docker_opr_ros2/OpenPlaceRecognition/src/opr/pipelines/place_recognition/base.py", line 124, in infer
[place_recognition_node.py-1]     _, pred_i = self.database_index.search(descriptor, 1)
[place_recognition_node.py-1]   File "/usr/local/lib/python3.10/dist-packages/faiss-1.7.4-py3.10.egg/faiss/class_wrappers.py", line 329, in replacement_search
[place_recognition_node.py-1]     assert d == self.d
[place_recognition_node.py-1] AssertionError
[ERROR] [place_recognition_node.py-1]: process has died [pid 2597, exit code 1, cmd '/home/docker_opr_ros2/ros2_ws/install/open_place_recognition/lib/open_place_recognition/place_recognition_node.py --ros-args -r __node:=multimodal_multicamera_lidar_place_recognition --params-file /tmp/launch_params_wdk5sjdl'].