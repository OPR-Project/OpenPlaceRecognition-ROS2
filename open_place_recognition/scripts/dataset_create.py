#!/usr/bin/env python3
"""
Script to extract data from an RTAB-Map database (rtabmap.db) for AI training.
It extracts:
  - Node data from the Node table:
      • Pose (as a 3x4 float matrix) converted to a quaternion (for CSV) 
        and saved as a raw text file in the "pose" folder.
      • Velocity (6 floats) saved in the "velocity" folder.
      • GPS (6 doubles) saved in the "gps" folder.
  - Sensor data from the Data table:
      • RGB images (from the "image" column) saved in "rgb".
      • Depth images (from the "depth" column) saved in "depth".
      • Calibration data (from the "calibration" column) saved in "calib".
      • Laser scan data (from the "scan" column) saved in "scan".
      • Scan info (from the "scan_info" column) saved in "scan_info".
  - A CSV file with node pose (as quaternion) and sensor timestamps.
  
Usage:
    python extract_rtabmap_all.py /path/to/rtabmap.db [optional: output_dir]

Requires:
  - Python 3.x
  - sqlite3 (built-in)
  - (Optional) OpenCV (cv2) and numpy to decode and save images.
"""

import os
import sys
import struct
import math
import sqlite3

# Optional: install with `pip install opencv-python numpy`
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix R (list of 3 lists of 3 floats)
    into a quaternion (qx, qy, qz, qw).
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2  # S = 4*qw
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2  # S = 4*qx
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2  # S = 4*qy
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2  # S = 4*qz
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return (qx, qy, qz, qw)

class RtabMapDatasetExtractor:
    """
    Extracts dataset from an RTAB-Map database for AI training.
    It extracts node poses, velocity, gps (from the Node table) and sensor data
    (RGB, depth, calibration, scan, scan_info from the Data table). Each type is saved
    in its own folder.
    """
    def __init__(self, db_path="~/Sync/3d_map/rtabmap.db", output_dir="~/.ros/opr_dataset"):
        self.db_path = os.path.expanduser(db_path)
        self.output_dir = os.path.expanduser(output_dir)
        # Use the base name of the DB as the map name.
        self.map_name = os.path.splitext(os.path.basename(self.db_path))[0]

    def run(self):
        if not os.path.isfile(self.db_path):
            print(f"[ERROR] Database not found: {self.db_path}")
            sys.exit(1)

        # Create output directories.
        map_out = os.path.join(self.output_dir, self.map_name)
        folders = {
            "rgb": os.path.join(map_out, "rgb"),
            "depth": os.path.join(map_out, "depth"),
            "calib": os.path.join(map_out, "calib"),
            "scan": os.path.join(map_out, "scan"),
            "scan_info": os.path.join(map_out, "scan_info"),
            "pose": os.path.join(map_out, "pose"),
            "velocity": os.path.join(map_out, "velocity"),
            "gps": os.path.join(map_out, "gps")
        }
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        csv_file = os.path.join(map_out, "tracker.csv")

        print(f"[INFO] Reading RTAB-Map DB: {self.db_path}")
        node_data = self._extract_node_data()
        if not node_data:
            print("[WARN] No node data found in the DB!")
        
        # Extract sensor data from Data table.
        self._attach_sensor_data(node_data, folders)
        # Write additional node data to separate folders.
        self._write_node_extras(node_data, folders)
        # Write CSV file.
        print(f"[INFO] Writing CSV data to: {csv_file}")
        self._write_csv(csv_file, node_data)
        print(f"[DONE] Dataset extraction complete. See '{map_out}'.")

    def _extract_node_data(self):
        """
        Query the Node table to extract:
          - id, map_id, stamp, pose, velocity, gps.
        The pose is stored as a 3x4 float matrix (12 floats). The rotation (first 9)
        is converted to a quaternion. Also store the raw 12-float pose.
        The velocity is 6 floats and gps is 6 doubles.
        """
        node_dict = {}
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            query = """
                SELECT id, map_id, stamp, pose, velocity, gps
                FROM Node
                ORDER BY stamp ASC
            """
            c.execute(query)
            rows = c.fetchall()
            for row in rows:
                node_id, map_id, stamp, pose_blob, velocity_blob, gps_blob = row
                timestamp = int(stamp)
                try:
                    # Unpack pose as 12 floats.
                    raw_pose = struct.unpack('12f', pose_blob)
                    # Build rotation matrix from values [0,1,2], [4,5,6], [8,9,10]
                    R = [
                        [raw_pose[0], raw_pose[1], raw_pose[2]],
                        [raw_pose[4], raw_pose[5], raw_pose[6]],
                        [raw_pose[8], raw_pose[9], raw_pose[10]]
                    ]
                    # Translation: indices 3,7,11.
                    tx, ty, tz = raw_pose[3], raw_pose[7], raw_pose[11]
                    qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                except Exception as e:
                    print(f"[ERROR] Failed to unpack pose for node {node_id}: {e}")
                    continue

                # Extract velocity if available.
                velocity_val = None
                if velocity_blob:
                    try:
                        velocity_val = struct.unpack('6f', velocity_blob)
                    except Exception as e:
                        print(f"[WARN] Failed to unpack velocity for node {node_id}: {e}")

                # Extract gps if available.
                gps_val = None
                if gps_blob:
                    try:
                        gps_val = struct.unpack('6d', gps_blob)
                    except Exception as e:
                        print(f"[WARN] Failed to unpack GPS for node {node_id}: {e}")

                node_dict[node_id] = {
                    "floor": map_id,
                    "timestamp": timestamp,
                    "tx": tx, "ty": ty, "tz": tz,
                    "qx": qx, "qy": qy, "qz": qz, "qw": qw,
                    "raw_pose": raw_pose,
                    "velocity": velocity_val,
                    "gps": gps_val,
                    # Placeholders for sensor timestamps.
                    "rgb_ts": 0,
                    "depth_ts": 0,
                    "calib_ts": 0,
                    "scan_ts": 0,
                    "scan_info_ts": 0
                }
            conn.close()
        except sqlite3.Error as e:
            print(f"[ERROR] SQLite error while reading Node table: {e}")
        return node_dict

    def _attach_sensor_data(self, node_dict, folders):
        """
        Query the Data table to extract sensor data.
        Uses columns:
          - image: compressed RGB image
          - depth: compressed depth image
          - calibration: calibration data
          - scan: laser scan data
          - scan_info: scan information data
        Data.id is assumed to match Node.id.
        """
        if not node_dict:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT * FROM Data")
            rows = c.fetchall()
            cols = [desc[0] for desc in c.description]
            # Get indices.
            try:
                data_id_idx = cols.index("id")
                stamp_idx = cols.index("time_enter") if "time_enter" in cols else None
            except ValueError:
                print("[WARN] Required columns missing in Data table.")
                conn.close()
                return

            image_idx = cols.index("image") if "image" in cols else None
            depth_idx = cols.index("depth") if "depth" in cols else None
            calib_idx = cols.index("calibration") if "calibration" in cols else None
            scan_idx  = cols.index("scan") if "scan" in cols else None
            scan_info_idx = cols.index("scan_info") if "scan_info" in cols else None

            for row in rows:
                data_node_id = row[data_id_idx]  # assume Data.id == Node.id
                if data_node_id not in node_dict:
                    continue
                # Use Data.time_enter if available.
                if stamp_idx is not None and row[stamp_idx]:
                    try:
                        data_ts = float(row[stamp_idx])
                    except Exception:
                        data_ts = node_dict[data_node_id]["timestamp"]
                else:
                    data_ts = node_dict[data_node_id]["timestamp"]

                # Extract RGB image.
                if image_idx is not None:
                    img_blob = row[image_idx]
                    if img_blob and OPENCV_AVAILABLE:
                        np_arr = np.frombuffer(img_blob, dtype=np.uint8)
                        rgb_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if rgb_img is not None:
                            fname = f"node_{data_node_id}_{int(data_ts)}_rgb.jpg"
                            cv2.imwrite(os.path.join(folders["rgb"], fname), rgb_img)
                            if node_dict[data_node_id]["rgb_ts"] == 0:
                                node_dict[data_node_id]["rgb_ts"] = int(data_ts)

                # Extract depth image.
                if depth_idx is not None:
                    depth_blob = row[depth_idx]
                    if depth_blob and OPENCV_AVAILABLE:
                        np_arr = np.frombuffer(depth_blob, dtype=np.uint8)
                        depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                        if depth_img is not None:
                            fname = f"node_{data_node_id}_{int(data_ts)}_depth.png"
                            cv2.imwrite(os.path.join(folders["depth"], fname), depth_img)
                            if node_dict[data_node_id]["depth_ts"] == 0:
                                node_dict[data_node_id]["depth_ts"] = int(data_ts)

                # Extract calibration data.
                if calib_idx is not None:
                    calib_blob = row[calib_idx]
                    if calib_blob:
                        try:
                            calib_text = calib_blob.decode('utf-8')
                        except Exception:
                            calib_text = str(calib_blob)
                        fname = f"node_{data_node_id}_{int(data_ts)}_calib.txt"
                        with open(os.path.join(folders["calib"], fname), 'w') as f:
                            f.write(calib_text)
                        if node_dict[data_node_id]["calib_ts"] == 0:
                            node_dict[data_node_id]["calib_ts"] = int(data_ts)

                # Extract scan data.
                if scan_idx is not None:
                    scan_blob = row[scan_idx]
                    if scan_blob:
                        fname = f"node_{data_node_id}_{int(data_ts)}_scan.bin"
                        with open(os.path.join(folders["scan"], fname), 'wb') as f:
                            f.write(scan_blob)
                        if node_dict[data_node_id]["scan_ts"] == 0:
                            node_dict[data_node_id]["scan_ts"] = int(data_ts)

                # Extract scan_info data.
                if scan_info_idx is not None:
                    si_blob = row[scan_info_idx]
                    if si_blob:
                        # Attempt to decode as text; if not, write as binary.
                        try:
                            si_text = si_blob.decode('utf-8')
                            fname = f"node_{data_node_id}_{int(data_ts)}_scan_info.txt"
                            with open(os.path.join(folders["scan_info"], fname), 'w') as f:
                                f.write(si_text)
                        except Exception:
                            fname = f"node_{data_node_id}_{int(data_ts)}_scan_info.bin"
                            with open(os.path.join(folders["scan_info"], fname), 'wb') as f:
                                f.write(si_blob)
                        if node_dict[data_node_id]["scan_info_ts"] == 0:
                            node_dict[data_node_id]["scan_info_ts"] = int(data_ts)
            conn.close()
        except sqlite3.OperationalError as e:
            print(f"[WARN] Data table query failed: {e}. Skipping sensor data extraction.")
        except sqlite3.Error as e:
            print(f"[ERROR] SQLite error while reading Data table: {e}")

    def _write_node_extras(self, node_dict, folders):
        """
        Write additional node data (raw pose, velocity, gps) to separate folders.
          - In folder "pose": write the 12 float values (raw pose) as a text file.
          - In folder "velocity": write 6 float values if available.
          - In folder "gps": write 6 double values if available.
        """
        # Write raw pose.
        for node_id, data in node_dict.items():
            if "raw_pose" in data:
                fname = f"node_{node_id}_pose.txt"
                with open(os.path.join(folders["pose"], fname), 'w') as f:
                    f.write(" ".join(f"{v:.6f}" for v in data["raw_pose"]))
            # Write velocity.
            if data.get("velocity"):
                fname = f"node_{node_id}_velocity.txt"
                with open(os.path.join(folders["velocity"], fname), 'w') as f:
                    f.write(" ".join(f"{v:.6f}" for v in data["velocity"]))
            # Write GPS.
            if data.get("gps"):
                fname = f"node_{node_id}_gps.txt"
                with open(os.path.join(folders["gps"], fname), 'w') as f:
                    f.write(" ".join(f"{v:.6f}" for v in data["gps"]))

    def _write_csv(self, csv_file_path, node_dict):
        """
        Write a CSV file with columns:
          track, floor, timestamp, rgb_ts, depth_ts, calib_ts, scan_ts, scan_info_ts,
          tx, ty, tz, qx, qy, qz, qw
        """
        if not node_dict:
            print("[WARN] Node dictionary empty, no data to write to CSV.")
            return
        sorted_nodes = sorted(node_dict.values(), key=lambda x: x["timestamp"])
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        with open(csv_file_path, 'w') as f:
            header = ("track,floor,timestamp,rgb_ts,depth_ts,calib_ts,scan_ts,scan_info_ts,"
                      "tx,ty,tz,qx,qy,qz,qw\n")
            f.write(header)
            track = f"00_{self.map_name}"
            for node in sorted_nodes:
                row = (
                    f"{track},"
                    f"{node['floor']},"
                    f"{node['timestamp']},"
                    f"{node['rgb_ts']},"
                    f"{node['depth_ts']},"
                    f"{node['calib_ts']},"
                    f"{node['scan_ts']},"
                    f"{node['scan_info_ts']},"
                    f"{node['tx']},"
                    f"{node['ty']},"
                    f"{node['tz']},"
                    f"{node['qx']},"
                    f"{node['qy']},"
                    f"{node['qz']},"
                    f"{node['qw']}\n"
                )
                f.write(row)

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_rtabmap_all.py [db_path] [optional: output_dir]")
        sys.exit(1)
    db_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "~/.ros/opr_dataset"
    extractor = RtabMapDatasetExtractor(db_path=db_path, output_dir=output_dir)
    extractor.run()

if __name__ == '__main__':
    main()
