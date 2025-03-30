#!/usr/bin/env python3
"""
Place Recognition Node

This node subscribes to front/back camera images, semantic masks, and lidar pointclouds.
It synchronizes these messages, prepares the input for the place recognition pipeline,
and publishes the resulting pose and database match index.
"""

import os
import numpy as np
import torch
import rclpy
from cv_bridge import CvBridge
from hydra.utils import instantiate
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omegaconf import OmegaConf
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, NavSatFix
from sensor_msgs_py.point_cloud2 import read_points
from std_msgs.msg import Int32
from torch import Tensor
from ament_index_python.packages import get_package_share_directory

# Import custom message and processing functions
from opr_interfaces.msg import DatabaseMatchIndex
from opr.datasets.augmentations import DefaultImageTransform, DefaultSemanticTransform
from opr.datasets.projection import Projector
from opr.datasets.soc_utils import (
    get_points_labels_by_mask,
    instance_masks_to_objects,
    pack_objects,
    semantic_mask_to_instances,
)
from opr.pipelines.place_recognition import PlaceRecognitionPipeline

class PlaceRecognitionNode(Node):
    """ROS2 Node for Place Recognition using camera images, semantic masks, and lidar data."""
    
    def __init__(self):
        super().__init__("place_recognition")
        # Declare required parameters directly in __init__
        self.declare_parameter("image_front_topic", "", ParameterDescriptor(description="Front camera image topic."))
        self.declare_parameter("image_back_topic", "", ParameterDescriptor(description="Back camera image topic."))
        self.declare_parameter("mask_front_topic", "", ParameterDescriptor(description="Front semantic segmentation mask topic."))
        self.declare_parameter("mask_back_topic", "", ParameterDescriptor(description="Back semantic segmentation mask topic."))
        self.declare_parameter("lidar_topic", "", ParameterDescriptor(description="Lidar pointcloud topic."))
        self.declare_parameter("pipeline_cfg", "", ParameterDescriptor(description="Path to the pipeline configuration file."))
        self.declare_parameter("image_resize", [], ParameterDescriptor(description="Image resize dimensions."))
        self.declare_parameter("enable_front_camera", True, ParameterDescriptor(description="Enable front camera."))
        self.declare_parameter("enable_back_camera", True, ParameterDescriptor(description="Enable back camera."))
        self.declare_parameter("enable_lidar", True, ParameterDescriptor(description="Enable lidar sensor."))
        self.declare_parameter("global_ref_topic", "", ParameterDescriptor(description="Global reference system topic (e.g. GPS/Barometer, WGS84)."))
        self.declare_parameter("enable_global_ref", True, ParameterDescriptor(description="Enable global reference subscription."))
        self.declare_parameter("reserve", False, ParameterDescriptor(description="Reserve variable for future use."))

        # Retrieve topics and configuration from parameters.
        image_front_topic = self.get_parameter("image_front_topic").get_parameter_value().string_value
        image_back_topic = self.get_parameter("image_back_topic").get_parameter_value().string_value
        mask_front_topic = self.get_parameter("mask_front_topic").get_parameter_value().string_value
        mask_back_topic = self.get_parameter("mask_back_topic").get_parameter_value().string_value
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        pipeline_cfg = self.get_parameter("pipeline_cfg").get_parameter_value().string_value
        image_resize = self.get_parameter("image_resize").get_parameter_value().integer_array_value

        # Retrieve sensor enable/disable parameters and global reference topic.
        self.enable_front_camera = self.get_parameter("enable_front_camera").get_parameter_value().bool_value
        self.enable_back_camera = self.get_parameter("enable_back_camera").get_parameter_value().bool_value
        self.enable_lidar = self.get_parameter("enable_lidar").get_parameter_value().bool_value
        self.enable_global_ref = self.get_parameter("enable_global_ref").get_parameter_value().bool_value
        self.global_ref_topic = self.get_parameter("global_ref_topic").get_parameter_value().string_value
        self.reserve = self.get_parameter("reserve").get_parameter_value().bool_value

        # Initialize CvBridge for image conversions.
        self.cv_bridge = CvBridge()

        # Create subscribers and map sensor names to synchronizer indices.
        subscribers = []
        mapping = {}
        if self.enable_front_camera:
            self.image_front_sub = Subscriber(self, CompressedImage, image_front_topic)
            subscribers.append(self.image_front_sub)
            mapping["front_image"] = len(subscribers) - 1

            self.mask_front_sub = Subscriber(self, Image, mask_front_topic)
            subscribers.append(self.mask_front_sub)
            mapping["front_mask"] = len(subscribers) - 1
        else:
            self.image_front_sub = None
            self.mask_front_sub = None

        if self.enable_back_camera:
            self.image_back_sub = Subscriber(self, CompressedImage, image_back_topic)
            subscribers.append(self.image_back_sub)
            mapping["back_image"] = len(subscribers) - 1

            self.mask_back_sub = Subscriber(self, Image, mask_back_topic)
            subscribers.append(self.mask_back_sub)
            mapping["back_mask"] = len(subscribers) - 1
        else:
            self.image_back_sub = None
            self.mask_back_sub = None

        if self.enable_lidar:
            self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)
            subscribers.append(self.lidar_sub)
            mapping["lidar"] = len(subscribers) - 1
        else:
            self.lidar_sub = None

        # Save the mapping for use in the callback.
        self.subscriber_mapping = mapping

        # Create the synchronizer if any subscribers exist.
        if subscribers:
            self.ts = ApproximateTimeSynchronizer(subscribers, queue_size=1, slop=1.05)
            self.ts.registerCallback(self.listener_callback)
        else:
            self.get_logger().error("No sensor subscribers created; check sensor enable parameters.")

        # Create publishers for pose and database match index.
        self.pose_pub = self.create_publisher(PoseStamped, "/place_recognition/pose", 10)
        self.idx_pub = self.create_publisher(DatabaseMatchIndex, "/place_recognition/db_idx", 10)

        # Subscribe to global reference system messages if enabled.
        if self.enable_global_ref:
            self.global_ref_sub = self.create_subscription(NavSatFix, self.global_ref_topic, self.global_ref_callback, 10)
        else:
            self.global_ref_sub = None
        self.global_ref = None

        # Instantiate the place recognition pipeline from configuration.
        pipeline_config = OmegaConf.load(pipeline_cfg)
        self.pr_pipe = instantiate(pipeline_config)

        # Check for Scene Object Context (SOC) module support.
        if self.pr_pipe.model.soc_module is not None:
            self.load_soc = True
            self.get_logger().info("self.load_soc is set to True.")
            sensors_cfg = OmegaConf.load(os.path.join(get_package_share_directory("open_place_recognition"), "configs/sensors/husky.yaml"))
            anno_cfg = OmegaConf.load(os.path.join(get_package_share_directory("open_place_recognition"), "configs/anno/oneformer.yaml"))
            self.front_cam_proj = Projector(sensors_cfg.front_cam, sensors_cfg.lidar)
            self.back_cam_proj = Projector(sensors_cfg.back_cam, sensors_cfg.lidar)
            self.max_distance_soc = 50.0
            self.top_k_soc = self.pr_pipe.model.soc_module.num_objects
            self.special_classes = anno_cfg.special_classes
            self.soc_coords_type = "euclidean"
        else:
            self.load_soc = False

        # Set up image and mask transformations.
        self.image_transform = DefaultImageTransform(train=False, resize=image_resize)
        self.mask_transform = DefaultSemanticTransform(train=False, resize=image_resize)

        self.get_logger().info(f"Initialized {self.__class__.__name__} node.")

    def global_ref_callback(self, msg: NavSatFix) -> None:
        """Callback to update the global reference system message."""
        self.global_ref = msg
        self.get_logger().debug(f"Received global reference message: {msg}")

    def _prepare_input(
        self,
        images: np.ndarray | list[np.ndarray] = None,
        masks: np.ndarray | list[np.ndarray] = None,
        pointcloud: np.ndarray = None,
    ) -> dict[str, Tensor]:
        """
        Prepare and transform input data for the pipeline.

        Args:
            images: Single image or list of images (in numpy format).
            masks: Single mask or list of masks.
            pointcloud: Lidar pointcloud data as a numpy array.

        Returns:
            Dictionary with transformed images, masks, and pointcloud tensors.
        """
        input_data: dict[str, Tensor] = {}

        if images is not None:
            if isinstance(images, list):
                for i, image in enumerate(images):
                    if image is not None:
                        input_data[f"image_{i}"] = self.image_transform(image)
                    else:
                        self.get_logger().info(f"Image {i} is disabled or not available.")
            elif isinstance(images, np.ndarray):
                input_data["image_0"] = self.image_transform(images)
            else:
                self.get_logger().warning(f"Invalid type for images in '_prepare_input': {type(images)}")

        if masks is not None:
            if isinstance(masks, list):
                for i, mask in enumerate(masks):
                    if mask is not None:
                        input_data[f"mask_{i}"] = self.mask_transform(mask)
                    else:
                        self.get_logger().info(f"Mask {i} is disabled or not available.")
            elif isinstance(masks, np.ndarray):
                input_data["mask_0"] = self.mask_transform(masks)
            else:
                self.get_logger().warning(f"Invalid type for masks in '_prepare_input': {type(masks)}")

        if pointcloud is not None:
            pointcloud_tensor = torch.tensor(pointcloud).contiguous()
            input_data["pointcloud_lidar_coords"] = pointcloud_tensor[:, :3]
            if pointcloud_tensor.shape[1] > 3:
                input_data["pointcloud_lidar_feats"] = pointcloud_tensor[:, 3].unsqueeze(1)
            else:
                input_data["pointcloud_lidar_feats"] = torch.ones_like(pointcloud_tensor[:, :1])

        if self.load_soc and masks is not None and isinstance(masks, list) and len(masks) >= 2:
            input_data["soc"] = self._get_soc(mask_front=masks[0], mask_back=masks[1], lidar_scan=pointcloud)

        return input_data

    def _get_soc(self, mask_front: np.ndarray, mask_back: np.ndarray, lidar_scan: np.ndarray) -> Tensor:
        """
        Compute the Scene Object Context (SOC) tensor based on the provided masks and lidar scan.

        Args:
            mask_front: Semantic mask from the front camera.
            mask_back: Semantic mask from the back camera.
            lidar_scan: Lidar pointcloud data.

        Returns:
            A PyTorch tensor representing packed objects.
        """
        coords_front, _, in_image_front = self.front_cam_proj(lidar_scan)
        coords_back, _, in_image_back = self.back_cam_proj(lidar_scan)
        point_labels = np.zeros(len(lidar_scan), dtype=np.uint8)
        point_labels[in_image_front] = get_points_labels_by_mask(coords_front, mask_front)
        point_labels[in_image_back] = get_points_labels_by_mask(coords_back, mask_back)
        instances_front = semantic_mask_to_instances(mask_front, area_threshold=10, labels_whitelist=self.special_classes)
        instances_back = semantic_mask_to_instances(mask_back, area_threshold=10, labels_whitelist=self.special_classes)
        objects_front = instance_masks_to_objects(instances_front, coords_front,
                                                  point_labels[in_image_front], lidar_scan[in_image_front])
        objects_back = instance_masks_to_objects(instances_back, coords_back,
                                                 point_labels[in_image_back], lidar_scan[in_image_back])
        objects = {**objects_front, **objects_back}
        packed_objects = pack_objects(objects, self.top_k_soc, self.max_distance_soc, self.special_classes)

        if self.soc_coords_type == "cylindrical_3d":
            packed_objects = np.concatenate((
                np.linalg.norm(packed_objects, axis=-1, keepdims=True),
                np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                packed_objects[..., 2:],
            ), axis=-1)
        elif self.soc_coords_type == "cylindrical_2d":
            packed_objects = np.concatenate((
                np.linalg.norm(packed_objects[..., :2], axis=-1, keepdims=True),
                np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                packed_objects[..., 2:],
            ), axis=-1)
        elif self.soc_coords_type == "euclidean":
            pass
        elif self.soc_coords_type == "spherical":
            packed_objects = np.concatenate((
                np.linalg.norm(packed_objects, axis=-1, keepdims=True),
                np.arccos(packed_objects[..., 2] / np.linalg.norm(packed_objects, axis=-1, keepdims=True)),
                np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
            ), axis=-1)
        else:
            raise ValueError(f"Unknown soc_coords_type: {self.soc_coords_type!r}")
        objects_tensor = torch.from_numpy(packed_objects).float()
        return objects_tensor

    def _create_pose_msg(self, pose: np.ndarray, timestamp: Time) -> PoseStamped:
        """
        Create a PoseStamped message from a pose array.

        Args:
            pose: A 7-element array [x, y, z, qx, qy, qz, qw].
            timestamp: The message timestamp.

        Returns:
            A populated PoseStamped message.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]
        pose_msg.pose.orientation.x = pose[3]
        pose_msg.pose.orientation.y = pose[4]
        pose_msg.pose.orientation.z = pose[5]
        pose_msg.pose.orientation.w = pose[6]
        return pose_msg

    def _create_idx_msg(self, idx: int, timestamp: Time) -> DatabaseMatchIndex:
        """
        Create a DatabaseMatchIndex message.

        Args:
            idx: The index value.
            timestamp: The message timestamp.

        Returns:
            A populated DatabaseMatchIndex message with the provided index.
        """
        idx_msg = DatabaseMatchIndex()
        idx_msg.header.stamp = timestamp
        idx_msg.index = idx
        return idx_msg

    def listener_callback(self, *msgs) -> None:
        """
        Callback for synchronized sensor messages.

        Converts incoming messages to OpenCV and numpy formats, prepares input data,
        runs inference through the place recognition pipeline, and publishes the results.
        """
        self.get_logger().info("Received synchronized messages.")
        t_start = self.get_clock().now()

        mapping = self.subscriber_mapping
        front_image_msg = msgs[mapping["front_image"]] if "front_image" in mapping else None
        front_mask_msg = msgs[mapping["front_mask"]] if "front_mask" in mapping else None
        back_image_msg = msgs[mapping["back_image"]] if "back_image" in mapping else None
        back_mask_msg = msgs[mapping["back_mask"]] if "back_mask" in mapping else None
        lidar_msg = msgs[mapping["lidar"]] if "lidar" in mapping else None

        front_image = self.cv_bridge.compressed_imgmsg_to_cv2(front_image_msg) if front_image_msg is not None else None
        back_image = self.cv_bridge.compressed_imgmsg_to_cv2(back_image_msg) if back_image_msg is not None else None
        front_mask = self.cv_bridge.imgmsg_to_cv2(front_mask_msg) if front_mask_msg is not None else None
        back_mask = self.cv_bridge.imgmsg_to_cv2(back_mask_msg) if back_mask_msg is not None else None

        if lidar_msg is not None:
            points = read_points(lidar_msg, field_names=("x", "y", "z"))
            pointcloud = np.array([points["x"], points["y"], points["z"]]).T
        else:
            pointcloud = None

        input_data = self._prepare_input(
            images=[front_image, back_image],
            masks=[front_mask, back_mask],
            pointcloud=pointcloud
        )

        output = self.pr_pipe.infer(input_data)
        t_taken = self.get_clock().now() - t_start
        self.get_logger().info(f"Place recognition inference took: {t_taken.nanoseconds / 1e6} ms.")

        timestamp = lidar_msg.header.stamp if lidar_msg is not None else self.get_clock().now().to_msg()
        pose_msg = self._create_pose_msg(output["pose"], timestamp)
        idx_msg = self._create_idx_msg(int(output["idx"]), timestamp)
        self.pose_pub.publish(pose_msg)
        self.get_logger().info(f"Published pose message: {pose_msg.pose}")
        self.idx_pub.publish(idx_msg)
        self.get_logger().info(f"Published database index message: {idx_msg.index}")


def main(args=None):
    """Main entry point for the Place Recognition node."""
    rclpy.init(args=args)
    pr_node = PlaceRecognitionNode()
    rclpy.spin(pr_node)
    pr_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
