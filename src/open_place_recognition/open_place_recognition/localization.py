import numpy as np
import rclpy
import torch
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

from opr_interfaces.msg import DatabaseMatchIndex
from opr.datasets.augmentations import DefaultImageTransform, DefaultSemanticTransform
from opr.datasets.projection import Projector
from opr.datasets.soc_utils import (
    get_points_labels_by_mask,
    instance_masks_to_objects,
    pack_objects,
    semantic_mask_to_instances,
)


TOPIC = "/zed_node/left/image_rect_color/compressed"
DATABASE_DIR = "/path/to/database"
MODEL_CFG = "/path/to/model/config.yaml"
MODEL_WEIGHTS_PATH = "/path/to/model/weights.pth"
DEVICE = "cuda"
IMAGE_RESIZE = (320, 192)


class LocalizationNode(Node):

    def __init__(self):
        super().__init__("localization")
        self._declare_parameters()

        image_front_topic = self.get_parameter("image_front_topic").get_parameter_value().string_value
        image_back_topic = self.get_parameter("image_back_topic").get_parameter_value().string_value
        mask_front_topic = self.get_parameter("mask_front_topic").get_parameter_value().string_value
        mask_back_topic = self.get_parameter("mask_back_topic").get_parameter_value().string_value
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        pipeline_cfg = self.get_parameter("pipeline_cfg").get_parameter_value().string_value
        image_resize = self.get_parameter("image_resize").get_parameter_value().integer_array_value

        self.cv_bridge = CvBridge()

        self.image_front_sub = Subscriber(self, CompressedImage, image_front_topic)
        self.image_back_sub = Subscriber(self, CompressedImage, image_back_topic)
        self.mask_front_sub = Subscriber(self, Image, mask_front_topic)
        self.mask_back_sub = Subscriber(self, Image, mask_back_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_front_sub, self.image_back_sub, self.mask_front_sub, self.mask_back_sub, self.lidar_sub],
            queue_size=1,
            slop=0.05,
        )
        self.ts.registerCallback(self.listener_callback)

        self.db_match_pose_pub = self.create_publisher(PoseStamped, "/place_recognition/pose", 10)
        self.idx_pub = self.create_publisher(DatabaseMatchIndex, "/place_recognition/db_idx", 10)
        self.estimated_pose_pub = self.create_publisher(PoseStamped, "/localization/pose", 10)

        self.pipeline = instantiate(OmegaConf.load(pipeline_cfg))

        if self.pipeline.pr_pipe.model.soc_module is not None:
            self.load_soc = True
            self.get_logger().info(f"self.load_soc is set to True.")
            sensors_cfg = OmegaConf.load("/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/configs/sensors/husky.yaml")
            anno_cfg = OmegaConf.load("/home/docker_opr_ros2/ros2_ws/src/open_place_recognition/configs/anno/oneformer.yaml")
            self.front_cam_proj = Projector(sensors_cfg.front_cam, sensors_cfg.lidar)
            self.back_cam_proj = Projector(sensors_cfg.back_cam, sensors_cfg.lidar)
            self.max_distance_soc = 50.0
            self.top_k_soc = self.pipeline.pr_pipe.model.soc_module.num_objects
            self.special_classes = anno_cfg.special_classes
            self.soc_coords_type = "euclidean"
        else:
            self.load_soc = False

        self.image_transform = DefaultImageTransform(train=False, resize=image_resize)
        self.mask_transform = DefaultSemanticTransform(train=False, resize=image_resize)

        self.get_logger().info(f"Initialized {self.__class__.__name__} node.")

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "image_front_topic",
            rclpy.Parameter.Type.STRING, #"/zed_node/left/image_rect_color/compressed",
            ParameterDescriptor(description="Front camera image topic.")
        )
        self.declare_parameter(
            "image_back_topic",
            rclpy.Parameter.Type.STRING, #"/realsense_back/color/image_raw/compressed",
            ParameterDescriptor(description="Back camera image topic.")
        )
        self.declare_parameter(
            "mask_front_topic",
            rclpy.Parameter.Type.STRING, #"/zed_node/left/semantic_segmentation",
            ParameterDescriptor(description="Front semantic segmentation mask topic.")
        )
        self.declare_parameter(
            "mask_back_topic",
            rclpy.Parameter.Type.STRING, #"/realsense_back/semantic_segmentation",
            ParameterDescriptor(description="Back semantic segmentation mask topic.")
        )
        self.declare_parameter(
            "lidar_topic",
            rclpy.Parameter.Type.STRING, #"/velodyne_points",
            ParameterDescriptor(description="Lidar pointcloud topic.")
        )
        self.declare_parameter(
            "pipeline_cfg",
            rclpy.Parameter.Type.STRING, #"",
            ParameterDescriptor(description="Path to the pipeline configuration file.")
        )
        self.declare_parameter(
            "image_resize",
            rclpy.Parameter.Type.INTEGER_ARRAY, #(320, 192),
            ParameterDescriptor(description="Image resize dimensions.")
        )

    def _prepare_input(
        self,
        images: np.ndarray | list[np.ndarray] = None,
        masks: np.ndarray | list[np.ndarray] = None,
        pointcloud: np.ndarray = None,
    ) -> dict[str, Tensor]:
        input_data = {}
        if images is not None:
            if isinstance(images, list):
                for i, image in enumerate(images):
                    input_data[f"image_{i}"] = self.image_transform(image)
            elif isinstance(images, np.ndarray):
                input_data["image_0"] = self.image_transform(images)
            else:
                self.get_logger().warning(f"Invalid type for images in '_prepare_input': {type(images)}")
        if masks is not None:
            if isinstance(masks, list):
                for i, mask in enumerate(masks):
                    input_data[f"mask_{i}"] = self.mask_transform(mask)
            elif isinstance(masks, np.ndarray):
                input_data["mask_0"] = self.mask_transform(masks)
            else:
                self.get_logger().warning(f"Invalid type for masks in '_prepare_input': {type(masks)}")
        if pointcloud is not None:
            pointcloud = torch.tensor(pointcloud).contiguous()
            input_data["pointcloud_lidar_coords"] = pointcloud[:, :3]
            if pointcloud.shape[1] > 3:
                input_data["pointcloud_lidar_feats"] = pointcloud[:, 3].unsqueeze(1)
            else:
                input_data["pointcloud_lidar_feats"] = torch.ones_like(pointcloud[:, :1])
        if self.load_soc:
            input_data["soc"] = self._get_soc(mask_front=masks[0], mask_back=masks[1], lidar_scan=pointcloud)
        return input_data

    def _get_soc(self, mask_front: np.ndarray, mask_back: np.ndarray, lidar_scan: np.ndarray) -> Tensor:
        coords_front, _, in_image_front = self.front_cam_proj(lidar_scan)
        coords_back, _, in_image_back = self.back_cam_proj(lidar_scan)

        point_labels = np.zeros(len(lidar_scan), dtype=np.uint8)
        point_labels[in_image_front] = get_points_labels_by_mask(coords_front, mask_front)
        point_labels[in_image_back] = get_points_labels_by_mask(coords_back, mask_back)

        instances_front = semantic_mask_to_instances(
            mask_front,
            area_threshold=10,
            labels_whitelist=self.special_classes,
        )
        instances_back = semantic_mask_to_instances(
            mask_back,
            area_threshold=10,
            labels_whitelist=self.special_classes,
        )

        objects_front = instance_masks_to_objects(
            instances_front,
            coords_front,
            point_labels[in_image_front],
            lidar_scan[in_image_front],
        )
        objects_back = instance_masks_to_objects(
            instances_back,
            coords_back,
            point_labels[in_image_back],
            lidar_scan[in_image_back],
        )

        objects = {**objects_front, **objects_back}
        packed_objects = pack_objects(objects, self.top_k_soc, self.max_distance_soc, self.special_classes)

        if self.soc_coords_type == "cylindrical_3d":
            packed_objects = np.concatenate(
                (
                    np.linalg.norm(packed_objects, axis=-1, keepdims=True),
                    np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                    packed_objects[..., 2:],
                ),
                axis=-1,
            )
        elif self.soc_coords_type == "cylindrical_2d":
            packed_objects = np.concatenate(
                (
                    np.linalg.norm(packed_objects[..., :2], axis=-1, keepdims=True),
                    np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                    packed_objects[..., 2:],
                ),
                axis=-1,
            )
        elif self.soc_coords_type == "euclidean":
            pass
        elif self.soc_coords_type == "spherical":
            packed_objects = np.concatenate(
                (
                    np.linalg.norm(packed_objects, axis=-1, keepdims=True),
                    np.arccos(
                        packed_objects[..., 2] / np.linalg.norm(packed_objects, axis=-1, keepdims=True)
                    ),
                    np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Unknown soc_coords_type: {self.soc_coords_type!r}")

        objects_tensor = torch.from_numpy(packed_objects).float()

        return objects_tensor

    def _create_pose_msg(self, pose: np.ndarray, timestamp: Time) -> PoseStamped:
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
        idx_msg = DatabaseMatchIndex()
        idx_msg.header.stamp = timestamp
        idx_msg.index = idx
        return idx_msg

    def listener_callback(
            self,
            front_image_msg: CompressedImage,
            back_image_msg: CompressedImage,
            front_mask_msg: Image,
            back_mask_msg: Image,
            lidar_msg: PointCloud2,
        ) -> None:
        self.get_logger().info("Received synchronized messages.")
        t_start = self.get_clock().now()
        lidar_timestamp = lidar_msg.header.stamp
        front_image = self.cv_bridge.compressed_imgmsg_to_cv2(front_image_msg)
        back_image = self.cv_bridge.compressed_imgmsg_to_cv2(back_image_msg)
        front_mask = self.cv_bridge.imgmsg_to_cv2(front_mask_msg)
        back_mask = self.cv_bridge.imgmsg_to_cv2(back_mask_msg)
        pointcloud = read_points(lidar_msg, field_names=("x", "y", "z"))
        pointcloud = np.array([pointcloud["x"], pointcloud["y"], pointcloud["z"]]).T
        input_data = self._prepare_input(
            images=[front_image, back_image], masks=[front_mask, back_mask], pointcloud=pointcloud
        )
        output = self.pipeline.infer(input_data)
        t_taken = self.get_clock().now() - t_start
        self.get_logger().debug(f"Localization inference took: {t_taken.nanoseconds / 1e6} ms.")
        self.get_logger().info(f"output['db_match_pose'] = {output['db_match_pose']}")
        output['db_match_pose'] = output['db_match_pose'].tolist()
        db_match_pose_msg = self._create_pose_msg(output["db_match_pose"], lidar_timestamp)
        estimated_pose_msg = self._create_pose_msg(output["estimated_pose"], lidar_timestamp)
        idx_msg = self._create_idx_msg(int(output["db_idx"]), lidar_timestamp)
        self.db_match_pose_pub.publish(db_match_pose_msg)
        self.get_logger().info(f"Published db match pose message: {db_match_pose_msg.pose}")
        self.estimated_pose_pub.publish(estimated_pose_msg)
        self.get_logger().info(f"Published estimated pose message: {estimated_pose_msg.pose}")
        self.idx_pub.publish(idx_msg)
        self.get_logger().info(f"Published database index message: {idx_msg.index}")


def main(args=None):
    rclpy.init(args=args)

    loc_node = LocalizationNode()

    rclpy.spin(loc_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    loc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
