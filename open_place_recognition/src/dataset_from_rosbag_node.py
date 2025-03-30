#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time

# Messages we’ll be subscribing to:
from sensor_msgs.msg import Image, LaserScan  # or PointCloud2
from geometry_msgs.msg import PoseStamped

# For writing images:
import os
import cv2
import numpy as np

import math
import struct

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (qx, qy, qz, qw).
    Assuming roll = x-rotation, pitch = y-rotation, yaw = z-rotation.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)

class DatasetCreateNode(Node):
    """
    This node subscribes to front camera, back camera, lidar, and pose topics
    and builds a dataset in a specified output directory.

    The user can configure the following ROS parameters:
      - front_camera_topic   (default: /camera_front/image_raw)
      - back_camera_topic    (default: /camera_back/image_raw)
      - lidar_topic          (default: /scan)
      - pose_topic           (default: /robot_pose)
      - output_path          (default: ~/.ros/opr_dataset)
      - track_name           (default: 00_my_map)

    For each message received, the node:
      - Decodes images and writes them to disk.
      - Buffers topic timestamps and relevant data in memory.
      - On shutdown, writes out a CSV describing all data.

    NOTE: This example is not time-synchronizing the topics.
          Each message will appear in the CSV under its own stamp entry.
    """
    def __init__(self):
        super().__init__('dataset_create_node')

        # Declare parameters
        self.declare_parameter('front_camera_topic', '/camera_front/image_raw')
        self.declare_parameter('back_camera_topic', '/camera_back/image_raw')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('pose_topic', '/robot_pose')
        self.declare_parameter('output_path', '~/.ros/opr_dataset')
        self.declare_parameter('track_name', '00_my_map')

        # Retrieve parameter values
        self.front_cam_topic = self.get_parameter('front_camera_topic').value
        self.back_cam_topic = self.get_parameter('back_camera_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.output_path = os.path.expanduser(self.get_parameter('output_path').value)
        self.track_name = self.get_parameter('track_name').value

        # Prepare output directories
        self.images_dir = os.path.join(self.output_path, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_path, "tracker.csv")

        # Data storage: dictionary keyed by stamp (float)
        # We store a subdict with e.g.:
        #   { 'front_cam_ts': stamp_if_front_camera,
        #     'back_cam_ts':  stamp_if_back_camera,
        #     'lidar_ts':     stamp_if_lidar,
        #     'pose_stamp':   stamp_if_pose,
        #     'x': ...,
        #     'y': ...,
        #     'z': ...,
        #     'qx': ...,
        #     'qy': ...,
        #     'qz': ...,
        #     'qw': ... }
        # 
        # In practice, you may want more sophisticated time alignment or
        # separate dictionaries for each topic, etc.
        self.data = {}

        # Subscriptions
        self.sub_front_cam = self.create_subscription(
            Image,
            self.front_cam_topic,
            self.front_camera_callback,
            10
        )
        self.sub_back_cam = self.create_subscription(
            Image,
            self.back_cam_topic,
            self.back_camera_callback,
            10
        )
        self.sub_lidar = self.create_subscription(
            LaserScan,
            self.lidar_topic,
            self.lidar_callback,
            10
        )
        self.sub_pose = self.create_subscription(
            PoseStamped,
            self.pose_callback,
            10
        )

        self.get_logger().info("DatasetCreateNode started. Listening to topics:")
        self.get_logger().info(f"  front_camera_topic: {self.front_cam_topic}")
        self.get_logger().info(f"  back_camera_topic:  {self.back_cam_topic}")
        self.get_logger().info(f"  lidar_topic:        {self.lidar_topic}")
        self.get_logger().info(f"  pose_topic:         {self.pose_topic}")
        self.get_logger().info(f"Writing results to:   {self.output_path}")

    def _to_float_stamp(self, stamp_msg):
        """
        Convert a builtin_interfaces.msg.Time or ROS2 stamp to a float of seconds.
        """
        return float(stamp_msg.sec) + 1e-9 * float(stamp_msg.nanosec)

    def _ensure_entry(self, t: float):
        """
        Ensure our data dictionary has an entry for this time.
        Returns the subdict for that time.
        """
        if t not in self.data:
            self.data[t] = {
                "front_cam_ts": 0,
                "back_cam_ts": 0,
                "lidar_ts": 0,
                # We'll store the pose separately:
                "pose_ts": 0,
                "tx": 0.0,
                "ty": 0.0,
                "tz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0
            }
        return self.data[t]

    def front_camera_callback(self, msg: Image):
        """
        Front camera image callback.
        Write out the image to file, store timestamp in our dictionary.
        """
        stamp_float = self._to_float_stamp(msg.header.stamp)
        entry = self._ensure_entry(stamp_float)

        # Mark that we have a front camera image at this time
        entry["front_cam_ts"] = stamp_float

        # Decode the image if it's something like raw or compressed:
        # We'll assume it's an 8UC3 (BGR) or 8UC1 for demonstration.
        # The encoding can vary (e.g. 'bgr8', 'rgb8', 'mono8', etc.).
        # We demonstrate a simple approach with OpenCV.
        try:
            # Convert ROS Image (raw bytes) to a NumPy array
            dtype = np.uint8
            channels = 3
            if msg.encoding == 'mono8':
                channels = 1
            # Construct a 2D array from the raw data
            img_np = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)

            # If needed to convert from RGB->BGR or vice versa
            # depending on the encoding. Here we assume it's already BGR.
            # If it was 'rgb8', you might do: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Write to disk
            out_name = f"front_{stamp_float:.6f}.png"
            out_path = os.path.join(self.images_dir, out_name)
            cv2.imwrite(out_path, img_np)
        except Exception as e:
            self.get_logger().warn(f"Front camera image decode failed at stamp {stamp_float}: {e}")

    def back_camera_callback(self, msg: Image):
        """
        Back camera image callback.
        """
        stamp_float = self._to_float_stamp(msg.header.stamp)
        entry = self._ensure_entry(stamp_float)
        entry["back_cam_ts"] = stamp_float

        try:
            dtype = np.uint8
            channels = 3
            if msg.encoding == 'mono8':
                channels = 1
            img_np = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)

            out_name = f"back_{stamp_float:.6f}.png"
            out_path = os.path.join(self.images_dir, out_name)
            cv2.imwrite(out_path, img_np)
        except Exception as e:
            self.get_logger().warn(f"Back camera image decode failed at stamp {stamp_float}: {e}")

    def lidar_callback(self, msg: LaserScan):
        """
        Example LIDAR callback. We only store the timestamp here.
        If needed, you could also dump the data (ranges, intensities) to disk.
        """
        stamp_float = self._to_float_stamp(msg.header.stamp)
        entry = self._ensure_entry(stamp_float)
        entry["lidar_ts"] = stamp_float
        # If you want to save the entire scan, you might do so here.
        # E.g., out_name = f"lidar_{stamp_float:.6f}.txt" or .csv
        # For demonstration, we only store the timestamp in memory.

    def pose_callback(self, msg: PoseStamped):
        """
        Pose callback. We'll store the pose in our dictionary.
        """
        stamp_float = self._to_float_stamp(msg.header.stamp)
        entry = self._ensure_entry(stamp_float)
        entry["pose_ts"] = stamp_float

        # Extract position
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z

        # Extract orientation
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        entry["tx"] = px
        entry["ty"] = py
        entry["tz"] = pz
        entry["qx"] = qx
        entry["qy"] = qy
        entry["qz"] = qz
        entry["qw"] = qw

    def write_csv(self):
        """
        Write out a CSV of all the data we've collected.
        """
        self.get_logger().info(f"Writing CSV to: {self.csv_path}")

        with open(self.csv_path, 'w') as f:
            f.write("track,floor,timestamp,front_cam_ts,back_cam_ts,lidar_ts,tx,ty,tz,qx,qy,qz,qw\n")

            # We do not have a 'floor' concept in this example, so let's store '0' or something static.
            floor_id = 0

            # Sort all entries by the earliest stamp
            sorted_stamps = sorted(self.data.keys())
            for t in sorted_stamps:
                row = self.data[t]
                # We'll store 't' as the primary timestamp
                # front_cam_ts, back_cam_ts, etc. are in row
                front_cam_ts = row["front_cam_ts"]
                back_cam_ts  = row["back_cam_ts"]
                lidar_ts     = row["lidar_ts"]
                tx = row["tx"]
                ty = row["ty"]
                tz = row["tz"]
                qx = row["qx"]
                qy = row["qy"]
                qz = row["qz"]
                qw = row["qw"]

                csv_line = (
                    f"{self.track_name},{floor_id},{t:.6f},"
                    f"{front_cam_ts:.6f},{back_cam_ts:.6f},{lidar_ts:.6f},"
                    f"{tx:.6f},{ty:.6f},{tz:.6f},"
                    f"{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}\n"
                )
                f.write(csv_line)

    def destroy_node(self):
        """
        Override destroy_node to write out the CSV before shutting down.
        """
        self.write_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCreateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
