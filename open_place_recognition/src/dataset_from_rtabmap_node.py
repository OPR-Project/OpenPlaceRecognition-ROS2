#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import sys
from ament_index_python.packages import get_package_share_directory
import time
import sqlite3
import struct
import math

# If needed for image decoding:
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


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
    def __init__(self):
        super().__init__('dataset_create_node')
        self.declare_parameter('map_name', 'my_map')
        self.declare_parameter('input_path', '~/Sync/map')
        self.declare_parameter('output_path', '~/.ros/opr_dataset')

        map_name = self.get_parameter('map_name').value
        output_path = self.get_parameter('output_path').value
        dataset_dir = os.path.join(os.path.expanduser(self.get_parameter('input_path').value), map_name)
        share_directory = self.get_parameter('use_share_directory').value
        if not os.path.exists(dataset_dir):
            self.get_logger().error(f"Databse directory {dataset_dir} not found.")
            sys.exit(1)

        # Actually create the dataset
        if not self.create_dataset(dataset_dir, map_name, output_path):
            self.get_logger().error(f"Error creating dataset for map {map_name}")
            sys.exit(1)

    def create_dataset(self, dataset_dir: str, map_name: str, output_path: str):
        """
        Main routine to:
        1. Connect to the RTAB-Map database
        2. Extract node poses, camera timestamps, LIDAR timestamps, etc.
        3. Write them to a CSV
        4. Extract images from the DB
        """
        self.get_logger().info(f"[CREATE] Creating dataset for map '{map_name}' at '{dataset_dir}' ...")

        # The RTAB-Map database file
        rtabmap_db = os.path.join(dataset_dir, f"{map_name}.db")

        if not os.path.exists(rtabmap_db):
            self.get_logger().error(f"RTAB-Map DB '{rtabmap_db}' does not exist!")
            return False

        # Prepare output directories
        dataset_path = os.path.join(os.path.expanduser(output_path), map_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        csv_path = os.path.join(dataset_path, "tracker.csv")
        images_dir = os.path.join(dataset_path, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # --------------------
        # Extract data from DB
        # --------------------
        try:
            # Connect to the .db
            conn = sqlite3.connect(rtabmap_db)
            c = conn.cursor()

            # 1) Gather Node info (id, mapId, stamp, pose, etc.)
            #
            #   Typically the Nodes table might be called "Node" or "Nodes". 
            #   The columns often are something like:
            #   - id (PRIMARY KEY)
            #   - mapId
            #   - stamp
            #   - pose (BLOB)
            #   - ground_truth_pose (BLOB)
            #   - ...
            #
            #   Adjust to match your DB schema. 
            c.execute("SELECT id, mapId, stamp, pose FROM Node ORDER BY stamp ASC")
            node_rows = c.fetchall()

            # We will store aggregated info in a dict keyed by node_id
            # so we can combine cameras, lidar, pose, etc.
            node_dict = {}
            for (node_id, map_id, stamp, pose_blob) in node_rows:
                # Convert stamp from float to int (or keep as float, depending on your usage)
                # Some DBs store stamps in seconds, others in nanoseconds, etc.
                # For demonstration, let's keep them as is, or cast to int if you prefer:
                timestamp = int(stamp)

                # Decode the pose (you may need '6f', '7f', '6d', '7d', etc.):
                # Example: x, y, z, roll, pitch, yaw = struct.unpack('6f', pose_blob)
                # or x, y, z, qx, qy, qz, qw = struct.unpack('7f', pose_blob)
                # Adapt to your exact RTAB-Map version!

                x = y = z = 0.0
                qx = qy = qz = qw = 0.0

                # Example assume 6 floats [x, y, z, roll, pitch, yaw]
                try:
                    x, y, z, roll, pitch, yaw = struct.unpack('6f', pose_blob)
                    qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
                except:
                    # Fallback if your DB actually stores x,y,z,qx,qy,qz,qw
                    x, y, z, qx, qy, qz, qw = struct.unpack('7f', pose_blob)

                node_dict[node_id] = {
                    "floor": map_id,
                    "timestamp": timestamp,  # This will be the 'primary' or 'node' timestamp
                    "front_cam_ts": None,
                    "back_cam_ts": None,
                    "lidar_ts": None,
                    "tx": x,
                    "ty": y,
                    "tz": z,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "qw": qw
                }

            # 2) Gather camera images (front, back) by node
            #
            #   Typically in the "Images" table:
            #   - image_id
            #   - node_id
            #   - data (BLOB)
            #   - stamp
            #   - camera_id
            #   - ...
            #
            #   Adjust to match your DB schema (and camera numbering).
            #   We'll guess camera_id=0 => front, camera_id=1 => back.
            try:
                c.execute("SELECT image_id, node_id, stamp, camera_id, data FROM Images")
                image_rows = c.fetchall()
                for (img_id, node_id, img_stamp, cam_id, img_blob) in image_rows:
                    # Convert image_stamp to int if needed
                    img_ts = int(img_stamp)

                    if node_id not in node_dict:
                        # This might be a node not in the Node table, skip
                        continue

                    # Save the camera stamp in our dictionary
                    if cam_id == 0:
                        node_dict[node_id]["front_cam_ts"] = img_ts
                    elif cam_id == 1:
                        node_dict[node_id]["back_cam_ts"] = img_ts

                    # Optionally decode and save the image
                    if OPENCV_AVAILABLE:
                        np_arr = np.frombuffer(img_blob, np.uint8)
                        img_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img_decoded is not None:
                            out_name = f"node_{node_id}_cam_{cam_id}_{img_ts}.jpg"
                            out_path = os.path.join(images_dir, out_name)
                            cv2.imwrite(out_path, img_decoded)
                    else:
                        # If you can't decode, but the data is e.g. raw JPEG,
                        # you could just write the BLOB directly to file:
                        # with open(os.path.join(images_dir, f"node_{node_id}_cam_{cam_id}.jpg"), 'wb') as img_file:
                        #     img_file.write(img_blob)
                        pass
            except sqlite3.OperationalError:
                # Possibly no Images table
                self.get_logger().warn("No Images table found or query failed. Skipping camera extraction.")

            # 3) Gather LIDAR scans by node (if stored), stamp, etc.
            #    Typical table might be "LaserScans" with columns:
            #      - id
            #      - node_id
            #      - stamp
            #      - scan (BLOB)
            #      ...
            try:
                c.execute("SELECT node_id, stamp FROM LaserScans")
                lidar_rows = c.fetchall()
                for (node_id, scan_stamp) in lidar_rows:
                    if node_id not in node_dict:
                        continue
                    node_dict[node_id]["lidar_ts"] = int(scan_stamp)
            except sqlite3.OperationalError:
                # Possibly no LaserScans table
                self.get_logger().warn("No LaserScans table found or query failed. Skipping lidar extraction.")

            conn.close()

        except Exception as e:
            self.get_logger().error(f"Error reading RTAB-Map DB: {e}")
            return False

        # --------------------
        # Write out the CSV
        # --------------------
        self.get_logger().info(f"Writing CSV file: {csv_path}")
        with open(csv_path, 'w') as f:
            # Header
            f.write("track,floor,timestamp,front_cam_ts,back_cam_ts,lidar_ts,tx,ty,tz,qx,qy,qz,qw\n")

            # For your sample, track could be "00_<map_name>", or directly map_name, etc.
            # Just adapt to your usage:
            track_name = f"00_{map_name}"

            # We'll iterate in ascending node timestamp order:
            sorted_nodes = sorted(node_dict.values(), key=lambda x: x["timestamp"])
            for row in sorted_nodes:
                floor = row["floor"]
                timestamp = row["timestamp"]
                front_cam_ts = row["front_cam_ts"] if row["front_cam_ts"] else 0
                back_cam_ts = row["back_cam_ts"] if row["back_cam_ts"] else 0
                lidar_ts = row["lidar_ts"] if row["lidar_ts"] else 0
                tx, ty, tz = row["tx"], row["ty"], row["tz"]
                qx, qy, qz, qw = row["qx"], row["qy"], row["qz"], row["qw"]

                csv_line = (
                    f"{track_name},"
                    f"{floor},"
                    f"{timestamp},"
                    f"{front_cam_ts},"
                    f"{back_cam_ts},"
                    f"{lidar_ts},"
                    f"{tx},"
                    f"{ty},"
                    f"{tz},"
                    f"{qx},"
                    f"{qy},"
                    f"{qz},"
                    f"{qw}\n"
                )
                f.write(csv_line)

        self.get_logger().info("[CREATE] Dataset creation completed.")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCreateNode()
    # Spin briefly to process log messages
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
