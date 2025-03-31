#!/usr/bin/env python3

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
import cv2

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, CameraInfo, NavSatFix
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import tf2_ros
from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory

class DatabasePublisherNode(Node):
    def __init__(self):
        super().__init__('database_publisher')
        
        # Declare parameters
        self.declare_parameter('dataset_dir',           '', ParameterDescriptor(description="Path to the dataset directory."))
        self.declare_parameter('enable_front_camera',   True, ParameterDescriptor(description="Enable front camera publishing."))
        self.declare_parameter('enable_back_camera',    True, ParameterDescriptor(description="Enable back camera publishing."))
        self.declare_parameter('enable_lidar',          True, ParameterDescriptor(description="Enable lidar publishing."))
        self.declare_parameter('enable_global_ref',     True, ParameterDescriptor(description="Enable global reference subscription."))
        self.declare_parameter('global_ref_topic',      '/global_ref', ParameterDescriptor(description="Global reference topic."))
        self.declare_parameter('reserve',               False, ParameterDescriptor(description="Reserved for future use."))
        
        # Declare topic parameters
        self.declare_parameter('front_cam_topic',       '/zed_node/left/image_rect_color/compressed')
        self.declare_parameter('front_cam_mask_topic',  '/zed_node/left/semantic_segmentation')
        self.declare_parameter('front_cam_info_topic',  '/zed_node/left/image_rect_color/camera_info')
        self.declare_parameter('back_cam_topic',        '/realsense_back/color/image_raw/compressed')
        self.declare_parameter('back_cam_mask_topic',   '/realsense_back/semantic_segmentation')
        self.declare_parameter('back_cam_info_topic',   '/realsense_back/color/image_raw/camera_info')
        self.declare_parameter('lidar_topic',           '/velodyne_points')
        
        # Declare TF frame parameters
        self.declare_parameter('tf_parent_frame', 'base_link')
        self.declare_parameter('front_cam_frame', 'zed_left')
        self.declare_parameter('back_cam_frame', 'realsense_back')
        self.declare_parameter('lidar_frame', 'velodyne')
        
        # Retrieve parameters
        self.dataset_dir = self.get_parameter('dataset_dir').get_parameter_value().string_value
        if not self.dataset_dir:
            self.get_logger().error("Dataset directory not provided!")
            sys.exit(1)
        
        self.enable_front_camera    = self.get_parameter('enable_front_camera').get_parameter_value().bool_value
        self.enable_back_camera     = self.get_parameter('enable_back_camera').get_parameter_value().bool_value
        self.enable_lidar           = self.get_parameter('enable_lidar').get_parameter_value().bool_value
        self.enable_global_ref      = self.get_parameter('enable_global_ref').get_parameter_value().bool_value
        self.global_ref_topic       = self.get_parameter('global_ref_topic').get_parameter_value().string_value
        self.reserve                = self.get_parameter('reserve').get_parameter_value().bool_value
        
        # Topics and frames
        self.front_cam_topic        = self.get_parameter('front_cam_topic').get_parameter_value().string_value
        self.front_cam_mask_topic   = self.get_parameter('front_cam_mask_topic').get_parameter_value().string_value
        self.front_cam_info_topic   = self.get_parameter('front_cam_info_topic').get_parameter_value().string_value
        self.back_cam_topic         = self.get_parameter('back_cam_topic').get_parameter_value().string_value
        self.back_cam_mask_topic    = self.get_parameter('back_cam_mask_topic').get_parameter_value().string_value
        self.back_cam_info_topic    = self.get_parameter('back_cam_info_topic').get_parameter_value().string_value
        self.lidar_topic            = self.get_parameter('lidar_topic').get_parameter_value().string_value
        
        self.tf_parent_frame        = self.get_parameter('tf_parent_frame').get_parameter_value().string_value
        self.front_cam_frame        = self.get_parameter('front_cam_frame').get_parameter_value().string_value
        self.back_cam_frame         = self.get_parameter('back_cam_frame').get_parameter_value().string_value
        self.lidar_frame            = self.get_parameter('lidar_frame').get_parameter_value().string_value
        
        # Create publishers using the parameterized topics
        if self.enable_front_camera:
            self.pub_front_cam      = self.create_publisher(CompressedImage, self.front_cam_topic, 1)
            self.pub_front_cam_mask = self.create_publisher(Image, self.front_cam_mask_topic, 1)
            self.pub_front_cam_info = self.create_publisher(CameraInfo, self.front_cam_info_topic, 1)
        else:
            self.pub_front_cam      = self.pub_front_cam_mask = self.pub_front_cam_info = None
        
        if self.enable_back_camera:
            self.pub_back_cam       = self.create_publisher(CompressedImage, self.back_cam_topic, 1)
            self.pub_back_cam_mask  = self.create_publisher(Image, self.back_cam_mask_topic, 1)
            self.pub_back_cam_info  = self.create_publisher(CameraInfo, self.back_cam_info_topic, 1)
        else:
            self.pub_back_cam       = self.pub_back_cam_mask = self.pub_back_cam_info = None
        
        if self.enable_lidar:
            self.pub_lidar = self.create_publisher(PointCloud2, self.lidar_topic, 1)
        else:
            self.pub_lidar = None
        
        # TF Broadcaster for publishing transforms
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Global reference subscription (if enabled)
        if self.enable_global_ref:
            self.global_ref_sub = self.create_subscription(NavSatFix, self.global_ref_topic, self.global_ref_callback, 10)
        else:
            self.global_ref_sub = None
        self.global_ref = None
        
        # Load CSV (track.csv)
        csv_path = os.path.join(self.dataset_dir, 'track.csv')
        dtypes = {'floor': int, 'timestamp': np.int64, 'front_cam_ts': np.int64,
                  'back_cam_ts': np.int64, 'lidar_ts': np.int64}
        self.track_df = pd.read_csv(csv_path, dtype=dtypes)
        self.get_logger().info(f"Loaded track.csv with {len(self.track_df)} rows.")
        
        self.cv_bridge = CvBridge()
        
        # Load sensor configuration from husky.yaml
        config_path = os.path.join(get_package_share_directory('open_place_recognition'),
                                   'configs', 'sensors', 'husky.yaml')
        try:
            with open(config_path, 'r') as f:
                self.sensor_config = yaml.safe_load(f)
            self.get_logger().info("Loaded sensor configuration from husky.yaml")
        except Exception as e:
            self.get_logger().error(f"Failed to load sensor configuration: {e}")
            self.sensor_config = None
        
        self.get_logger().info("DatabasePublisherNode initialized.")
    
    def global_ref_callback(self, msg):
        self.global_ref = msg
        
    def publish_sensor_config(self):
        if not self.sensor_config:
            return
        
        now = self.get_clock().now().to_msg()
        
        # Front camera configuration publishing
        if self.enable_front_camera and self.pub_front_cam_info is not None:
            try:
                front = self.sensor_config['front_cam']['left']
                tfs = TransformStamped()
                tfs.header.stamp = now
                tfs.header.frame_id = self.tf_parent_frame
                tfs.child_frame_id = self.front_cam_frame
                t = front['baselink2cam']['t']
                q = front['baselink2cam']['q']
                tfs.transform.translation.x = t[0]
                tfs.transform.translation.y = t[1]
                tfs.transform.translation.z = t[2]
                tfs.transform.rotation.x = q[1]
                tfs.transform.rotation.y = q[2]
                tfs.transform.rotation.z = q[3]
                tfs.transform.rotation.w = q[0]
                self.tf_broadcaster.sendTransform(tfs)
                
                cam_info = CameraInfo()
                cam_info.header.stamp = now
                cam_info.header.frame_id = self.front_cam_frame
                resolution = front['resolution']
                cam_info.width = resolution[0]
                cam_info.height = resolution[1]
                cam_info.distortion_model = "plumb_bob"
                cam_info.d = [0.0]*5
                P = front['rect']['P']
                cam_info.p = [elem for row in P for elem in row]
                cam_info.k = [P[0][0], P[0][1], P[0][2],
                              P[1][0], P[1][1], P[1][2],
                              P[2][0], P[2][1], P[2][2]]
                cam_info.r = [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0]
                self.pub_front_cam_info.publish(cam_info)
            except Exception as e:
                self.get_logger().error(f"Error publishing front_cam config: {e}")
        
        # Back camera configuration publishing
        if self.enable_back_camera and self.pub_back_cam_info is not None:
            try:
                back = self.sensor_config['back_cam']['left']
                tbs = TransformStamped()
                tbs.header.stamp = now
                tbs.header.frame_id = self.tf_parent_frame
                tbs.child_frame_id = self.back_cam_frame
                t = back['baselink2cam']['t']
                q = back['baselink2cam']['q']
                tbs.transform.translation.x = t[0]
                tbs.transform.translation.y = t[1]
                tbs.transform.translation.z = t[2]
                tbs.transform.rotation.x = q[1]
                tbs.transform.rotation.y = q[2]
                tbs.transform.rotation.z = q[3]
                tbs.transform.rotation.w = q[0]
                self.tf_broadcaster.sendTransform(tbs)
                
                cam_info = CameraInfo()
                cam_info.header.stamp = now
                cam_info.header.frame_id = self.back_cam_frame
                resolution = back['resolution']
                cam_info.width = resolution[0]
                cam_info.height = resolution[1]
                cam_info.distortion_model = "plumb_bob"
                cam_info.d = [0.0]*5
                P = back['rect']['P']
                cam_info.p = [elem for row in P for elem in row]
                cam_info.k = [P[0][0], P[0][1], P[0][2],
                              P[1][0], P[1][1], P[1][2],
                              P[2][0], P[2][1], P[2][2]]
                cam_info.r = [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0]
                self.pub_back_cam_info.publish(cam_info)
            except Exception as e:
                self.get_logger().error(f"Error publishing back_cam config: {e}")
        
        # Lidar configuration publishing
        if self.enable_lidar:
            try:
                lidar = self.sensor_config['lidar']
                tls = TransformStamped()
                tls.header.stamp = now
                tls.header.frame_id = self.tf_parent_frame
                tls.child_frame_id = self.lidar_frame
                t = lidar['baselink2lidar']['t']
                q = lidar['baselink2lidar']['q']
                tls.transform.translation.x = t[0]
                tls.transform.translation.y = t[1]
                tls.transform.translation.z = t[2]
                tls.transform.rotation.x = q[1]
                tls.transform.rotation.y = q[2]
                tls.transform.rotation.z = q[3]
                tls.transform.rotation.w = q[0]
                self.tf_broadcaster.sendTransform(tls)
            except Exception as e:
                self.get_logger().error(f"Error publishing lidar config: {e}")
    
    def publish_one_row(self, i):
        self.publish_sensor_config()
        floor_num = int(self.track_df['floor'][i])
        floor_folder = f"floor_{floor_num}"
        back_cam_ts = int(self.track_df['back_cam_ts'][i])
        front_cam_ts = int(self.track_df['front_cam_ts'][i])
        lidar_ts = int(self.track_df['lidar_ts'][i])
        self.get_logger().info(f"Publishing data for {floor_folder}")
        
        if self.enable_back_camera:
            back_cam_path = os.path.join(self.dataset_dir, floor_folder, 'back_cam', f"{back_cam_ts}.png")
            msg = self.read_image_as_compressed(back_cam_path, frame_id=self.back_cam_frame)
            if msg and self.pub_back_cam:
                self.pub_back_cam.publish(msg)
            mask_path = os.path.join(self.dataset_dir, floor_folder, 'masks', 'back_cam', f"{back_cam_ts}.png")
            mask_msg = self.read_image_as_uncompressed(mask_path, frame_id=self.back_cam_frame)
            if mask_msg and self.pub_back_cam_mask:
                self.pub_back_cam_mask.publish(mask_msg)
        
        if self.enable_front_camera:
            front_cam_path = os.path.join(self.dataset_dir, floor_folder, 'front_cam', f"{front_cam_ts}.png")
            msg = self.read_image_as_compressed(front_cam_path, frame_id=self.front_cam_frame)
            if msg and self.pub_front_cam:
                self.pub_front_cam.publish(msg)
            mask_path = os.path.join(self.dataset_dir, floor_folder, 'masks', 'front_cam', f"{front_cam_ts}.png")
            mask_msg = self.read_image_as_uncompressed(mask_path, frame_id=self.front_cam_frame)
            if mask_msg and self.pub_front_cam_mask:
                self.pub_front_cam_mask.publish(mask_msg)
        
        if self.enable_lidar:
            lidar_path = os.path.join(self.dataset_dir, floor_folder, 'lidar', f"{lidar_ts}.bin")
            lidar_msg = self.read_lidar_as_pointcloud2(lidar_path, frame_id=self.lidar_frame)
            if lidar_msg and self.pub_lidar:
                self.pub_lidar.publish(lidar_msg)
    
    def publish_loop(self):
        if len(self.track_df) < 1:
            self.get_logger().warn("track.csv is empty. Nothing to publish.")
            return
        
        timestamps = self.track_df['timestamp']
        for i in range(len(self.track_df)):
            self.publish_one_row(i)
            if i < len(self.track_df) - 1:
                dt_ns = timestamps[i + 1] - timestamps[i]
                dt_s = dt_ns / 1e9
                if dt_s < 0 or dt_s > 10:
                    self.get_logger().warn(f"Skipping sleep due to invalid dt at index {i}.")
                    continue
                time.sleep(dt_s)
        self.get_logger().info("Finished publishing all rows from track.csv.")
    
    def read_image_as_compressed(self, image_path, frame_id='camera'):
        if not os.path.exists(image_path):
            self.get_logger().error(f"Image not found: {image_path}")
            return None
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = frame_id
            msg.format = "png"
            msg.data = list(image_data)
            return msg
        except Exception as e:
            self.get_logger().error(f"Failed to read compressed image: {e}")
            return None
    
    def read_image_as_uncompressed(self, image_path, frame_id='camera'):

        if not os.path.exists(image_path):
            self.get_logger().error(f"Mask image not found: {image_path}")
            return None

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().error(f"Failed to read image {image_path}")
            return None

        if len(img.shape) == 2:
            encoding = "mono8"
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                encoding = "mono8"
            elif img.shape[2] == 3:
                encoding = "bgr8"
            elif img.shape[2] == 4:
                encoding = "bgra8"
            else:
                encoding = "bgr8"
        else:
            encoding = "bgr8"
        try:
            msg = self.cv_bridge.cv2_to_imgmsg(img, encoding=encoding)
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion error: {e}")
            return None
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        return msg
    
    def read_lidar_as_pointcloud2(self, lidar_path, frame_id='velodyne'):
        if not os.path.exists(lidar_path):
            self.get_logger().error(f"Lidar file not found: {lidar_path}")
            return None
        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            self.get_logger().error(f"Error reading lidar file: {e}")
            return None
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        from sensor_msgs.msg import PointField
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg = point_cloud2.create_cloud(header, fields, points)
        return pc2_msg

def main(args=None):
    rclpy.init(args=args)
    node = DatabasePublisherNode()
    try:
        node.publish_loop()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
