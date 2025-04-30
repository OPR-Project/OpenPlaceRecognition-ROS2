#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message

import rosbag2_py

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs.point_cloud2 import read_points

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from pandas import DataFrame
from typing import Dict, List, Tuple, Union
from tqdm import tqdm

matplotlib.use("Agg")

###############################################################################
# Helper functions (preprocessing)
###############################################################################
def closest_values_indices(in_array: np.ndarray, from_array: np.ndarray) -> np.ndarray:
    """For each element in the first array find the closest value from the second array."""
    closest_idxs = np.zeros(len(in_array), dtype=np.int64)
    for i, a_val in enumerate(in_array):
        abs_diffs = np.abs(from_array - a_val)
        closest_idxs[i] = np.argmin(abs_diffs)
    return closest_idxs


def filter_timestamps(
    pose_ts: np.ndarray,
    front_ts: np.ndarray,
    back_ts: np.ndarray,
    lidar_ts: np.ndarray,
    max_diff: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter timestamps. For each pose_ts, find the closest front_ts, back_ts, lidar_ts
    that differ by less than `max_diff`. Return the filtered indices.
    """
    filtered_pose_idxs = []
    filtered_front_idxs = []
    filtered_back_idxs = []
    filtered_lidar_idxs = []

    for i, ts in enumerate(pose_ts):
        front_idx = closest_values_indices(np.array([ts]), front_ts)[0]
        back_idx = closest_values_indices(np.array([ts]), back_ts)[0]
        lidar_idx = closest_values_indices(np.array([ts]), lidar_ts)[0]

        if (
            abs(ts - front_ts[front_idx]) <= max_diff
            and abs(ts - back_ts[back_idx]) <= max_diff
            and abs(ts - lidar_ts[lidar_idx]) <= max_diff
        ):
            filtered_pose_idxs.append(i)
            filtered_front_idxs.append(front_idx)
            filtered_back_idxs.append(back_idx)
            filtered_lidar_idxs.append(lidar_idx)

    return (
        np.array(filtered_pose_idxs),
        np.array(filtered_front_idxs),
        np.array(filtered_back_idxs),
        np.array(filtered_lidar_idxs),
    )


def filter_by_distance_indices(utm_points: np.ndarray, distance: float = 5.0) -> np.ndarray:
    """
    Filter points so that each point is ~ `distance` meters away from the previous one.
    """
    filtered_points = np.array([0], dtype=int)  # start with index 0
    for i in range(1, utm_points.shape[0]):
        # distance from current to last filtered
        right_dist = np.linalg.norm(utm_points[i] - utm_points[filtered_points[-1]])
        if right_dist >= distance:
            left_dist = np.linalg.norm(utm_points[i - 1] - utm_points[filtered_points[-1]])
            if abs(right_dist - distance) < abs(left_dist - distance):
                filtered_points = np.append(filtered_points, i)
            else:
                filtered_points = np.append(filtered_points, i - 1)
    return filtered_points


def plot_track_map(utms: np.ndarray) -> np.ndarray:
    """
    Plot a 2D track (X-Y) and return it as a BGR image (for saving via OpenCV).
    """
    x, y = utms[:, 0], utms[:, 1]
    x_min, x_max = np.min(x) - 2, np.max(x) + 2
    y_min, y_max = np.min(y) - 2, np.max(y) + 2
    fig, ax = plt.subplots(dpi=200)
    ax.scatter(x, y, s=0.5)
    ax.set_xlabel("x")
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("y")
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    fig.canvas.draw()

    # Convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # convert from RGB to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


###############################################################################
# Helper functions (reading + unpacking)
###############################################################################
def merge_dicts(dict1: Dict[str, List[int]], dict2: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Merge two dicts of lists, appending values of matching keys."""
    for k in dict2:
        dict1[k] += dict2[k]
    return dict1


def read_ros2_bag_messages(
    bag_file_path: Path,
    wanted_topics: List[str],
    max_count: int = -1,
):
    """
    Reads messages from a ROS 2 bag using rosbag2_py.
    Yields tuples: (topic_name, message, time_ns).
    """
    # Create storage and converter options to open the bag
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_file_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Map topic name -> type so we can deserialize messages properly
    type_map = {}
    for topic_metadata in reader.get_all_topics_and_types():
        if topic_metadata.name in wanted_topics:
            type_map[topic_metadata.name] = topic_metadata.type

    count = 0
    while reader.has_next():
        (topic, raw_data, t) = reader.read_next()
        if topic not in wanted_topics:
            continue
        # If the topic is known, we can figure out how to deserialize
        if type_map[topic] == "sensor_msgs/msg/CompressedImage":
            msg = deserialize_message(raw_data, CompressedImage)
        elif type_map[topic] == "sensor_msgs/msg/PointCloud2":
            msg = deserialize_message(raw_data, PointCloud2)
        else:
            # For other types: geometry_msgs/msg/TransformStamped, etc.
            # This is a minimal example, add more if needed
            msg = deserialize_message(raw_data, PointCloud2)  # or appropriate type
        count += 1
        yield (topic, msg, t)
        if max_count > 0 and count >= max_count:
            break


def list_images_and_points_ros2(
    bag_file_path: Union[str, Path],
    front_cam_topic: str,
    back_cam_topic: str,
    lidar_topic: str,
) -> Dict[str, List[int]]:
    """
    Return timestamps for images and LiDAR from a ROS 2 bag file.
    """
    bag_file_path = Path(bag_file_path)
    out_dict = {"front_cam": [], "back_cam": [], "lidar": []}
    wanted_topics = [front_cam_topic, back_cam_topic, lidar_topic]

    for topic, msg, t_ns in tqdm(
        read_ros2_bag_messages(bag_file_path, wanted_topics),
        desc=f"Listing {bag_file_path.name}",
        leave=False,
    ):
        if topic == front_cam_topic:
            out_dict["front_cam"].append(t_ns)
        elif topic == back_cam_topic:
            out_dict["back_cam"].append(t_ns)
        elif topic == lidar_topic:
            out_dict["lidar"].append(t_ns)

    return out_dict


def export_from_bag_ros2(
    bag_file_path: Union[str, Path],
    output_dir: Union[str, Path],
    timestamps_dict: Dict[str, np.ndarray],
    front_cam_topic: str,
    back_cam_topic: str,
    lidar_topic: str,
):
    """
    Extract images (PNG) and LiDAR points (BIN) from a ROS 2 bag file.
    """
    bag_file_path = Path(bag_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    front_cam_dir = output_dir / "front_cam"
    front_cam_dir.mkdir(exist_ok=True)
    back_cam_dir = output_dir / "back_cam"
    back_cam_dir.mkdir(exist_ok=True)
    lidar_dir = output_dir / "lidar"
    lidar_dir.mkdir(exist_ok=True)

    bridge = CvBridge()

    # Turn them into sets for quick membership tests
    front_cam_set = set(timestamps_dict["front_cam"].tolist())
    back_cam_set = set(timestamps_dict["back_cam"].tolist())
    lidar_set = set(timestamps_dict["lidar"].tolist())

    wanted_topics = [front_cam_topic, back_cam_topic, lidar_topic]

    for topic, msg, t_ns in tqdm(
        read_ros2_bag_messages(bag_file_path, wanted_topics),
        desc=f"Exporting {bag_file_path.name}",
        leave=False,
    ):
        if topic == front_cam_topic:
            if t_ns in front_cam_set:
                # msg is sensor_msgs/CompressedImage
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite(str(front_cam_dir / f"{t_ns}.png"), cv_image)
        elif topic == back_cam_topic:
            if t_ns in back_cam_set:
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite(str(back_cam_dir / f"{t_ns}.png"), cv_image)
        elif topic == lidar_topic:
            if t_ns in lidar_set:
                # msg is sensor_msgs/PointCloud2
                points_arr = np.array(list(read_points(msg)), dtype=np.float32)
                # scale intensity from [0..255] to [0..1], if needed
                points_arr[:, 3] /= 255.0
                points_file_path = lidar_dir / f"{t_ns}.bin"
                points_arr[:, :4].tofile(points_file_path)


def read_trajectory_bag_ros2(filepath: Path, trajectory_topic: str) -> DataFrame:
    """
    Reads a trajectory from a ROS 2 bag file, if it’s stored as e.g. geometry_msgs/TransformStamped.
    """
    # We assume the topic is geometry_msgs/msg/TransformStamped
    # If your transform is a different type, adjust accordingly.
    from geometry_msgs.msg import TransformStamped

    data = {
        "timestamp": [], "tx": [], "ty": [], "tz": [],
        "qx": [], "qy": [], "qz": [], "qw": []
    }
    wanted_topics = [trajectory_topic]

    for topic, msg, t_ns in read_ros2_bag_messages(filepath, wanted_topics):
        # deserialize as TransformStamped
        # (adjust if your bag actually uses a different type)
        trans_msg = deserialize_message(msg.serialize(), TransformStamped)  
        data["timestamp"].append(t_ns)
        data["tx"].append(trans_msg.transform.translation.x)
        data["ty"].append(trans_msg.transform.translation.y)
        data["tz"].append(trans_msg.transform.translation.z)
        data["qx"].append(trans_msg.transform.rotation.x)
        data["qy"].append(trans_msg.transform.rotation.y)
        data["qz"].append(trans_msg.transform.rotation.z)
        data["qw"].append(trans_msg.transform.rotation.w)

    return DataFrame(data=data)


def read_trajectory_tum(filepath: Path) -> DataFrame:
    """
    Reads a trajectory from a TUM file: <timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
    """
    data = {"timestamp": [], "tx": [], "ty": [], "tz": [], "qx": [], "qy": [], "qz": [], "qw": []}
    with open(filepath, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 8:
                continue
            ts_float = float(vals[0])
            timestamp = int(ts_float * 1e9)  # float secs -> nanosecs

            tx, ty, tz = [float(x) for x in vals[1:4]]
            qx, qy, qz, qw = [float(x) for x in vals[4:8]]
            data["timestamp"].append(timestamp)
            data["tx"].append(tx)
            data["ty"].append(ty)
            data["tz"].append(tz)
            data["qx"].append(qx)
            data["qy"].append(qy)
            data["qz"].append(qz)
            data["qw"].append(qw)
    return DataFrame(data=data)


###############################################################################
# Main Node
###############################################################################
class BagConverterNode(Node):
    def __init__(self):
        super().__init__("bag_converter_node")

        # Declare ROS 2 parameters so we can override them in a launch file.
        self.declare_parameter("input_dir", "/path/to/input_dir")
        self.declare_parameter("trajectory_file", "/path/to/trajectory.db3")  # e.g. a ROS 2 bag
        self.declare_parameter("out_dir", "/path/to/output_dir")
        self.declare_parameter("distance_threshold", 5.0)
        self.declare_parameter("max_diff", 60000000)  # 60 ms in nanoseconds

        # Topics to read from the bags
        self.declare_parameter("front_cam_topic", "/zed_node/left/image_rect_color/compressed")
        self.declare_parameter("back_cam_topic", "/realsense_back/color/image_raw/compressed")
        self.declare_parameter("lidar_topic", "/velodyne_points")
        # If your trajectory is published as geometry_msgs/TransformStamped on this topic:
        self.declare_parameter("trajectory_topic", "/global_trajectory_0")

        # Fetch parameter values
        input_dir = Path(self.get_parameter("input_dir").value)
        trajectory_file = Path(self.get_parameter("trajectory_file").value)
        out_dir = Path(self.get_parameter("out_dir").value)
        distance_threshold = float(self.get_parameter("distance_threshold").value)
        max_diff = int(self.get_parameter("max_diff").value)

        front_cam_topic = self.get_parameter("front_cam_topic").value
        back_cam_topic = self.get_parameter("back_cam_topic").value
        lidar_topic = self.get_parameter("lidar_topic").value
        trajectory_topic = self.get_parameter("trajectory_topic").value

        # Run the core logic
        self.run_conversion(
            input_dir,
            trajectory_file,
            out_dir,
            distance_threshold,
            max_diff,
            front_cam_topic,
            back_cam_topic,
            lidar_topic,
            trajectory_topic,
        )

    def run_conversion(
        self,
        input_dir: Path,
        trajectory_file: Path,
        out_dir: Path,
        dist_threshold: float,
        max_diff_ns: int,
        front_cam_topic: str,
        back_cam_topic: str,
        lidar_topic: str,
        trajectory_topic: str,
    ):
        # Find all ROS 2 bag files in the input directory. By default, ros2 bag creates a folder
        # (e.g. "my_bag") containing a "metadata.yaml" and .db3 files. 
        # You might want to gather all `.db3` or all subfolders. Adjust logic as needed.
        bag_files_list = sorted(
            f for f in input_dir.iterdir() 
            if f.is_file() and f.suffix in [".db3", ".mcap"]
        )

        if not bag_files_list:
            self.get_logger().error(f"No ROS 2 bag files found in {input_dir}")
            return

        # Merge timestamps found in the bag files
        timestamps_dict = {"front_cam": [], "back_cam": [], "lidar": []}
        for bag_file_path in tqdm(bag_files_list, desc="Reading timestamps", position=0):
            sub_dict = list_images_and_points_ros2(
                bag_file_path,
                front_cam_topic,
                back_cam_topic,
                lidar_topic,
            )
            timestamps_dict = merge_dicts(timestamps_dict, sub_dict)

        # Convert all to NumPy
        timestamps_dict = {k: np.array(v) for k, v in timestamps_dict.items()}

        # Read the trajectory from either a TUM file or a ROS 2 bag
        if trajectory_file.suffix in [".db3", ".mcap"]:
            # assume ROS 2 bag file with geometry_msgs/TransformStamped
            poses_df = read_trajectory_bag_ros2(trajectory_file, trajectory_topic)
        elif trajectory_file.suffix in [".tum", ".txt"]:
            poses_df = read_trajectory_tum(trajectory_file)
        else:
            self.get_logger().error(f"Unsupported trajectory file: {trajectory_file}")
            return

        # Filter timestamps
        filtered_indices = filter_timestamps(
            pose_ts=poses_df["timestamp"].to_numpy(),
            front_ts=timestamps_dict["front_cam"],
            back_ts=timestamps_dict["back_cam"],
            lidar_ts=timestamps_dict["lidar"],
            max_diff=max_diff_ns,
        )

        # Apply the filter
        poses_df = poses_df.iloc[filtered_indices[0]]
        timestamps_dict["front_cam"] = timestamps_dict["front_cam"][filtered_indices[1]]
        timestamps_dict["back_cam"] = timestamps_dict["back_cam"][filtered_indices[2]]
        timestamps_dict["lidar"] = timestamps_dict["lidar"][filtered_indices[3]]

        # Filter by distance
        distance_filtered_indices = filter_by_distance_indices(
            poses_df[["tx", "ty", "tz"]].to_numpy(), distance=dist_threshold
        )
        poses_df = poses_df.iloc[distance_filtered_indices]
        timestamps_dict["front_cam"] = timestamps_dict["front_cam"][distance_filtered_indices]
        timestamps_dict["back_cam"] = timestamps_dict["back_cam"][distance_filtered_indices]
        timestamps_dict["lidar"] = timestamps_dict["lidar"][distance_filtered_indices]

        # Plot the final track map
        track_map_img = plot_track_map(poses_df[["tx", "ty"]].to_numpy())

        # Create output directory
        if out_dir.exists():
            self.get_logger().warn(f"Output directory already exists: {out_dir}")
        else:
            self.get_logger().info(f"Creating output directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save the track map
        cv2.imwrite(str(out_dir / "track_map.png"), track_map_img)

        # Save final CSV
        out_df = poses_df.copy()
        out_df["front_cam_ts"] = timestamps_dict["front_cam"]
        out_df["back_cam_ts"] = timestamps_dict["back_cam"]
        out_df["lidar_ts"] = timestamps_dict["lidar"]
        out_df = out_df[
            ["timestamp", "front_cam_ts", "back_cam_ts", "lidar_ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
        ]
        out_df.to_csv(out_dir / "track.csv", index=False)

        # Extract images + LiDAR from each bag
        for bag_file_path in tqdm(bag_files_list, desc="Exporting data", position=0):
            export_from_bag_ros2(
                bag_file_path,
                out_dir,
                timestamps_dict,
                front_cam_topic,
                back_cam_topic,
                lidar_topic,
            )

        self.get_logger().info("Dataset conversion is complete.")


def main(args=None):
    rclpy.init(args=args)
    node = BagConverterNode()
    # For one-shot execution, just destroy after finishing
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
