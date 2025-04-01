#!/usr/bin/env python3
"""
Localization Node for image and lidar-based localization.
This node subscribes to front/back camera images, semantic masks, and lidar pointclouds.
It then synchronizes the messages, preprocesses the data, and runs a localization pipeline.
"""

import os
import cv2
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
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from std_msgs.msg import Int32
from torch import Tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ament_index_python.packages import get_package_share_directory

# Import QoS classes for separate QoS handling
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

# Importing required modules from our package
from opr.pipelines.localization import ArucoLocalizationPipeline
from opr_interfaces.msg import DatabaseMatchIndex
from opr.datasets.augmentations import DefaultImageTransform, DefaultSemanticTransform
from opr.datasets.projection import Projector
from opr.datasets.soc_utils import (
    get_points_labels_by_mask,
    instance_masks_to_objects,
    pack_objects,
    semantic_mask_to_instances,
)


class ImageTransformWithoutNormalize:
    """Custom image transformation without normalization.
    
    Converts an image in cv2 format to a PyTorch tensor (channel-first) using albumentations.
    """
    def __call__(self, img: np.ndarray) -> Tensor:
        augmented = A.Compose([ToTensorV2()])(image=img)
        return augmented["image"]


class LocalizationNode(Node):
    """ROS2 node for localization based on synchronized image and lidar data."""
    
    def __init__(self):
        super().__init__("localization")
        
        # Declare required parameters directly in __init__
        self.declare_parameter("qos_front_camera", 2, ParameterDescriptor(description="QoS for front camera"))
        self.declare_parameter("qos_back_camera", 2,  ParameterDescriptor(description="QoS for back camera"))
        self.declare_parameter("qos_lidar", 2,        ParameterDescriptor(description="QoS for lidar"))
        self.declare_parameter("qos_global_ref", 2,   ParameterDescriptor(description="QoS for global reference subscription"))
        # Declare other parameters
        self.declare_parameter("image_front_topic", "", ParameterDescriptor(description="Front camera image topic."))
        self.declare_parameter("image_back_topic",  "", ParameterDescriptor(description="Back camera image topic."))
        self.declare_parameter("mask_front_topic",  "", ParameterDescriptor(description="Front semantic segmentation mask topic."))
        self.declare_parameter("mask_back_topic",   "", ParameterDescriptor(description="Back semantic segmentation mask topic."))
        self.declare_parameter("lidar_topic",       "", ParameterDescriptor(description="Lidar pointcloud topic."))
        self.declare_parameter("dataset_dir",       "", ParameterDescriptor(description="dataset directory."))
        self.declare_parameter("pipeline_cfg",      "", ParameterDescriptor(description="Path to the pipeline configuration file."))
        self.declare_parameter("exclude_dynamic_classes", False, ParameterDescriptor(description="Exclude dynamic objects from the input data."))
        self.declare_parameter("image_resize", rclpy.Parameter.Type.INTEGER_ARRAY, ParameterDescriptor(description="Image resize dimensions."))
        
        # New parameters for enabling/disabling sensors and global reference.
        self.declare_parameter("enable_front_camera", True, ParameterDescriptor(description="Enable front camera."))
        self.declare_parameter("enable_back_camera",  True, ParameterDescriptor(description="Enable back camera."))
        self.declare_parameter("enable_lidar",        True, ParameterDescriptor(description="Enable lidar sensor."))
        self.declare_parameter("enable_global_ref",   True, ParameterDescriptor(description="Enable global reference system subscription."))
        self.declare_parameter("global_ref_topic",  "", ParameterDescriptor(description="Global reference topic (e.g. GPS/Barometer, WGS84)."))
        
        # Retrieve QoS parameter values
        self.qos_front_camera_value = self.get_parameter("qos_front_camera").get_parameter_value().integer_value
        self.qos_back_camera_value  = self.get_parameter("qos_back_camera").get_parameter_value().integer_value
        self.qos_lidar_value        = self.get_parameter("qos_lidar").get_parameter_value().integer_value
        self.qos_global_ref_value   = self.get_parameter("qos_global_ref").get_parameter_value().integer_value
        
        # Retrieve other parameters
        self.image_front_topic      = self.get_parameter("image_front_topic").get_parameter_value().string_value
        self.image_back_topic       = self.get_parameter("image_back_topic").get_parameter_value().string_value
        self.mask_front_topic       = self.get_parameter("mask_front_topic").get_parameter_value().string_value
        self.mask_back_topic        = self.get_parameter("mask_back_topic").get_parameter_value().string_value
        self.lidar_topic            = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.dataset_dir            = self.get_parameter("dataset_dir").get_parameter_value().string_value
        self.pipeline_cfg           = self.get_parameter("pipeline_cfg").get_parameter_value().string_value
        self.exclude_dynamic_classes = self.get_parameter("exclude_dynamic_classes").get_parameter_value().bool_value
        self.image_resize           = self.get_parameter("image_resize").get_parameter_value().integer_array_value

        self.enable_front_camera    = self.get_parameter("enable_front_camera").get_parameter_value().bool_value
        self.enable_back_camera     = self.get_parameter("enable_back_camera").get_parameter_value().bool_value
        self.enable_lidar           = self.get_parameter("enable_lidar").get_parameter_value().bool_value
        self.enable_global_ref      = self.get_parameter("enable_global_ref").get_parameter_value().bool_value
        self.global_ref_topic       = self.get_parameter("global_ref_topic").get_parameter_value().string_value

        # Initialize cv_bridge for converting ROS image messages to OpenCV format.
        self.cv_bridge = CvBridge()

        # Create QoS profiles
        self.qos_front_cam_profile  = self._create_qos_profile(self.qos_front_camera_value)
        self.qos_back_cam_profile   = self._create_qos_profile(self.qos_back_camera_value)
        self.qos_lidar_profile      = self._create_qos_profile(self.qos_lidar_value)
        self.qos_global_ref_profile = self._create_qos_profile(self.qos_global_ref_value, depth=10)

        # Build a list of subscribers conditionally based on enabled sensors,
        # each with its own QoS profile.
        subscribers = []
        mapping = {}  # Will map sensor name to its index in the synchronizer arguments.

        if self.enable_front_camera:
            # Subscriber for front camera image
            self.image_front_sub = Subscriber(
                self,
                CompressedImage,
                self.image_front_topic,
                qos_profile=self.qos_front_cam_profile
            )
            subscribers.append(self.image_front_sub)
            mapping['front_image'] = len(subscribers) - 1

            # Subscriber for front camera mask
            self.mask_front_sub = Subscriber(
                self,
                Image,
                self.mask_front_topic,
                qos_profile=self.qos_front_cam_profile
            )
            subscribers.append(self.mask_front_sub)
            mapping['front_mask'] = len(subscribers) - 1
        else:
            self.image_front_sub = None
            self.mask_front_sub  = None

        if self.enable_back_camera:
            # Subscriber for back camera image
            self.image_back_sub = Subscriber(
                self,
                CompressedImage,
                self.image_back_topic,
                qos_profile=self.qos_back_cam_profile
            )
            subscribers.append(self.image_back_sub)
            mapping['back_image'] = len(subscribers) - 1

            # Subscriber for back camera mask
            self.mask_back_sub = Subscriber(
                self,
                Image,
                self.mask_back_topic,
                qos_profile=self.qos_back_cam_profile
            )
            subscribers.append(self.mask_back_sub)
            mapping['back_mask'] = len(subscribers) - 1
        else:
            self.image_back_sub = None
            self.mask_back_sub  = None

        if self.enable_lidar:
            # Subscriber for LiDAR
            self.lidar_sub = Subscriber(
                self,
                PointCloud2,
                self.lidar_topic,
                qos_profile=self.qos_lidar_profile
            )
            subscribers.append(self.lidar_sub)
            mapping['lidar'] = len(subscribers) - 1
        else:
            self.lidar_sub = None

        # Save mapping for use in the callback.
        self.subscriber_mapping = mapping

        # Create synchronizer only if at least one sensor is enabled.
        if subscribers:
            self.ts = ApproximateTimeSynchronizer(
                subscribers,
                queue_size=1,
                slop=0.05,
            )
            self.ts.registerCallback(self.listener_callback)
        else:
            self.get_logger().error("No sensors enabled; cannot create synchronizer.")

        # Create publishers for pose and database match index (use default QoS or 10).
        self.db_match_pose_pub  = self.create_publisher(PoseStamped, "/place_recognition/pose", 10)
        self.idx_pub            = self.create_publisher(DatabaseMatchIndex, "/place_recognition/db_idx", 10)
        self.estimated_pose_pub = self.create_publisher(PoseStamped, "/localization/pose", 10)

        # If enabled, create a subscriber for the global reference system with its QoS.
        if self.enable_global_ref:
            from sensor_msgs.msg import NavSatFix
            self.global_ref_sub = self.create_subscription(
                NavSatFix,
                self.global_ref_topic,
                self.global_ref_callback,
                self.qos_global_ref_profile
            )
        else:
            self.global_ref_sub = None
        self.global_ref = None

        # Instantiate the localization pipeline from configuration.
        if not os.path.exists(self.pipeline_cfg):
            exit(1)
        cfg = OmegaConf.load(self.pipeline_cfg)

        if not os.path.exists(self.dataset_dir):
            exit(1)

        # Check out the open_place_recognition/configs/pipelines/localization_pipeline.yaml
        print("This node is not fully developed")
        exit(1)
        model_weights_path = os.path.join(os.path.expanduser("~"), "OpenPlaceRecognition", cfg.model_weights_path)
        if not os.path.exists(self.dataset_dir):
            exit(1)
        self.get_logger().error(f"dataset_dir does not exist: {self.dataset_dir}")
        self.get_logger().error(f"model_weights_path does not exist: {model_weights_path}")
        cfg.database_dir = self.dataset_dir
        cfg.model_weights_path = model_weights_path
        # ----------------------------------

        self.pipeline = instantiate(cfg)

        # Check if the pipeline uses a SOC (scene object context) module.
        if self.pipeline.pr_pipe.model.soc_module is not None:
            self.load_soc = True
            self.get_logger().info("self.load_soc is set to True.")
            sensors_cfg = OmegaConf.load(os.path.join(get_package_share_directory("open_place_recognition"), "configs/sensors/husky.yaml"))
            anno_cfg = OmegaConf.load(os.path.join(get_package_share_directory("open_place_recognition"), "configs/anno/oneformer.yaml"))
            self.front_cam_proj = Projector(sensors_cfg.front_cam, sensors_cfg.lidar)
            self.back_cam_proj = Projector(sensors_cfg.back_cam, sensors_cfg.lidar)
            self.max_distance_soc = 50.0
            self.top_k_soc = self.pipeline.pr_pipe.model.soc_module.num_objects
            self.special_classes = anno_cfg.special_classes
            self.soc_coords_type = "euclidean"
        else:
            self.load_soc = False

        # Set up image and mask transformations based on the pipeline type.
        if isinstance(self.pipeline, ArucoLocalizationPipeline):
            self.image_transform = ImageTransformWithoutNormalize()
            self.mask_transform = DefaultSemanticTransform(train=False, resize=None)
        else:
            self.image_transform = DefaultImageTransform(train=False, resize=self.image_resize)
            self.mask_transform = DefaultSemanticTransform(train=False, resize=self.image_resize)

        # Dynamic class index to be excluded.
        self._ade20k_dynamic_idx = [12]

        # Transformation matrices for lidar to camera conversions.
        self.lidar2front = np.array([
            [ 0.01509615, -0.99976457, -0.01558544,  0.04632156],
            [ 0.00871086,  0.01571812, -0.99983852, -0.13278588],
            [ 0.9998481,   0.01495794,  0.0089461,  -0.06092749],
            [ 0.,          0.,          0.,          1.        ]
        ])
        self.lidar2back = np.array([
            [-1.50409674e-02,  9.99886421e-01,  9.55906151e-04,  1.82703304e-02],
            [-1.30440106e-02,  7.59716299e-04, -9.99914635e-01, -1.41787545e-01],
            [-9.99801792e-01, -1.50521522e-02,  1.30311022e-02, -6.72336358e-02],
            [ 0.,              0.,              0.,              1.        ]
        ])
        self.front_matrix = np.array([
            [683.6199340820312, 0.0, 615.1160278320312],
            [0.0, 683.6199340820312, 345.32354736328125],
            [0.0, 0.0, 1.0]
        ])
        self.front_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.back_matrix = np.array([
            [910.4178466796875, 0.0, 648.44140625],
            [0.0, 910.4166870117188, 354.0118408203125],
            [0.0, 0.0, 1.0]
        ])
        self.back_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.get_logger().info(f"Initialized {self.__class__.__name__} node.")

    def _create_qos_profile(self, qos_value: int, depth=1) -> QoSProfile:
        """
        Create a QoSProfile based on integer (0=SystemDefault,1=BestEffort,2=Reliable).
        """
        qos_profile = QoSProfile(depth=depth)
        if qos_value == 0:
            qos_profile.reliability = QoSReliabilityPolicy.SYSTEM_DEFAULT
        elif qos_value == 1:
            qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        else:
            qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        return qos_profile

    def global_ref_callback(self, msg) -> None:
        """Callback to update the global reference message."""
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
            images: A single image or list of images (cv2 format).
            masks: A single mask or list of masks.
            pointcloud: Lidar pointcloud data.
        
        Returns:
            A dictionary with transformed images, masks, and pointcloud tensors.
        """
        input_data: dict[str, Tensor] = {}

        # Process image inputs using the configured image transform.
        if images is not None and isinstance(images, list):
            for i, image in enumerate(images):
                if image is None:
                    self.get_logger().info(f"Image {i} is disabled or not available.")
                    continue
                img_processed = image.copy()
                if self.exclude_dynamic_classes and masks is not None and i < len(masks) and masks[i] is not None:
                    for dyn_idx in self._ade20k_dynamic_idx:
                        img_processed = np.where(masks[i][:, :, np.newaxis] == dyn_idx, 0, img_processed)
                input_data[f"image_{i}"] = self.image_transform(img_processed)
        elif isinstance(images, np.ndarray):
            img_processed = images.copy()
            if self.exclude_dynamic_classes and masks is not None:
                for dyn_idx in self._ade20k_dynamic_idx:
                    img_processed = np.where(masks == dyn_idx, 0, img_processed)
            input_data["image_0"] = self.image_transform(img_processed)
        else:
            self.get_logger().warning(f"Invalid type for images in '_prepare_input': {type(images)}")

        # Process mask inputs using the configured mask transform.
        if masks is not None and isinstance(masks, list):
            for i, mask in enumerate(masks):
                if mask is None:
                    self.get_logger().info(f"Mask {i} is disabled or not available.")
                    continue
                input_data[f"mask_{i}"] = self.mask_transform(mask)
        elif isinstance(masks, np.ndarray):
            input_data["mask_0"] = self.mask_transform(masks)
        else:
            self.get_logger().warning(f"Invalid type for masks in '_prepare_input': {type(masks)}")

        # Process pointcloud data.
        if pointcloud is not None:
            if self.exclude_dynamic_classes and masks is not None:
                if isinstance(masks, list):
                    for mask in masks:
                        pointcloud = self._remove_dynamic_points(pointcloud, mask,
                                                                  self.lidar2back, self.back_matrix, self.back_dist)
                elif isinstance(masks, np.ndarray):
                    pointcloud = self._remove_dynamic_points(pointcloud, masks,
                                                              self.lidar2back, self.back_matrix, self.back_dist)
            pc_tensor = torch.tensor(pointcloud).contiguous()
            pc_tensor = pc_tensor[~torch.any(pc_tensor.isnan(), dim=1)]
            input_data["pointcloud_lidar_coords"] = pc_tensor[:, :3]
            if pc_tensor.shape[1] > 3:
                input_data["pointcloud_lidar_feats"] = pc_tensor[:, 3].unsqueeze(1)
            else:
                input_data["pointcloud_lidar_feats"] = torch.ones_like(pc_tensor[:, :1])

        # Process SOC (scene object context) if enabled and if masks are available.
        if self.load_soc and masks is not None and isinstance(masks, list) and len(masks) >= 2:
            input_data["soc"] = self._get_soc(mask_front=masks[0], mask_back=masks[1], lidar_scan=pointcloud)

        return input_data

    def _get_soc(self, mask_front: np.ndarray, mask_back: np.ndarray, lidar_scan: np.ndarray) -> Tensor:
        """
        Compute the scene object context (SOC) tensor based on the provided masks and lidar scan.
        
        Args:
            mask_front: Front camera semantic mask.
            mask_back: Back camera semantic mask.
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

    def _remove_dynamic_points(self, pointcloud: np.ndarray, semantic_map: np.ndarray,
                                 lidar2sensor: np.ndarray, sensor_intrinsics: np.ndarray,
                                 sensor_dist: np.ndarray) -> np.ndarray:
        """
        Remove points corresponding to dynamic objects from the pointcloud.
        
        Args:
            pointcloud: The original pointcloud data.
            semantic_map: The semantic segmentation mask.
            lidar2sensor: Transformation matrix from lidar to sensor frame.
            sensor_intrinsics: Camera intrinsic matrix.
            sensor_dist: Camera distortion coefficients.
        
        Returns:
            The filtered pointcloud with dynamic points removed.
        """
        pc_values = np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))], axis=1).T
        camera_values = lidar2sensor @ pc_values
        camera_values = camera_values.T[:, :3]
        points_2d, _ = cv2.projectPoints(
            camera_values,
            np.zeros((3, 1), np.float32),
            np.zeros((3, 1), np.float32),
            sensor_intrinsics,
            sensor_dist
        )
        points_2d = points_2d[:, 0, :]
        classes_in_map = set(np.unique(semantic_map))
        dynamic_classes = set(self._ade20k_dynamic_idx)
        if classes_in_map.intersection(dynamic_classes):
            valid = (~np.isnan(points_2d[:, 0])) & (~np.isnan(points_2d[:, 1]))
            in_bounds_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < 1280)
            in_bounds_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < 720)
            look_forward = (camera_values[:, 2] > 0)
            overall_mask = valid & in_bounds_x & in_bounds_y & look_forward
            indices = np.where(overall_mask)[0]
            mask_for_points = np.full((points_2d.shape[0], 3), True)
            dynamic_idx_array = np.array(self._ade20k_dynamic_idx)
            semantic_values = semantic_map[
                np.floor(points_2d[indices, 1]).astype(int),
                np.floor(points_2d[indices, 0]).astype(int)
            ]
            matching_indices = np.where(np.isin(semantic_values, dynamic_idx_array))
            mask_for_points[indices[matching_indices[0]]] = np.array([False, False, False])
            return pointcloud[mask_for_points].reshape((-1, 3))
        else:
            return pointcloud

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
            A populated DatabaseMatchIndex message.
        """
        idx_msg = DatabaseMatchIndex()
        idx_msg.header.stamp = timestamp
        idx_msg.index = idx
        return idx_msg

    def listener_callback(self, *msgs) -> None:
        """
        Callback function invoked when synchronized messages are received.
        
        Converts messages to OpenCV images and numpy arrays, prepares input data,
        runs inference through the pipeline, and publishes the results.
        """
        self.get_logger().info("Received synchronized messages.")
        t_start = self.get_clock().now()

        # Extract messages from the synchronizer based on our mapping.
        mapping = self.subscriber_mapping
        front_image_msg = msgs[mapping['front_image']] if 'front_image' in mapping else None
        front_mask_msg  = msgs[mapping['front_mask']]  if 'front_mask'  in mapping else None
        back_image_msg  = msgs[mapping['back_image']]  if 'back_image'  in mapping else None
        back_mask_msg   = msgs[mapping['back_mask']]   if 'back_mask'   in mapping else None
        lidar_msg       = msgs[mapping['lidar']]       if 'lidar'       in mapping else None

        # Convert image and mask messages to OpenCV format if available.
        front_image = self.cv_bridge.compressed_imgmsg_to_cv2(front_image_msg) if front_image_msg is not None else None
        back_image  = self.cv_bridge.compressed_imgmsg_to_cv2(back_image_msg) if back_image_msg is not None else None
        front_mask  = self.cv_bridge.imgmsg_to_cv2(front_mask_msg) if front_mask_msg is not None else None
        back_mask   = self.cv_bridge.imgmsg_to_cv2(back_mask_msg) if back_mask_msg is not None else None

        # Convert lidar message to a numpy pointcloud if available.
        if lidar_msg is not None:
            points = read_points(lidar_msg, field_names=("x", "y", "z"))
            pointcloud = np.array([points["x"], points["y"], points["z"]]).T
        else:
            pointcloud = None

        # Prepare input data for the pipeline.
        input_data = self._prepare_input(
            images=[front_image, back_image],
            masks=[front_mask, back_mask],
            pointcloud=pointcloud
        )

        # Run inference.
        output = self.pipeline.infer(input_data)
        t_taken = self.get_clock().now() - t_start
        self.get_logger().debug(f"Localization inference took: {t_taken.nanoseconds / 1e6} ms.")
        self.get_logger().info(f"output['db_match_pose'] = {output['db_match_pose']}")

        # Create and publish messages.
        db_match_pose = output["db_match_pose"].tolist()
        estimated_pose = output["estimated_pose"]
        db_idx = int(output["db_idx"])

        db_match_pose_msg = self._create_pose_msg(
            db_match_pose,
            lidar_msg.header.stamp if lidar_msg is not None else self.get_clock().now().to_msg()
        )
        estimated_pose_msg = self._create_pose_msg(
            estimated_pose,
            lidar_msg.header.stamp if lidar_msg is not None else self.get_clock().now().to_msg()
        )
        idx_msg = self._create_idx_msg(
            db_idx,
            lidar_msg.header.stamp if lidar_msg is not None else self.get_clock().now().to_msg()
        )

        self.db_match_pose_pub.publish(db_match_pose_msg)
        self.get_logger().info(f"Published db match pose message: {db_match_pose_msg.pose}")
        self.estimated_pose_pub.publish(estimated_pose_msg)
        self.get_logger().info(f"Published estimated pose message: {estimated_pose_msg.pose}")
        self.idx_pub.publish(idx_msg)
        self.get_logger().info(f"Published database index message: {idx_msg.index}")

def main(args=None):
    """Main function to initialize and spin the localization node."""
    rclpy.init(args=args)
    loc_node = LocalizationNode()
    rclpy.spin(loc_node)
    loc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
