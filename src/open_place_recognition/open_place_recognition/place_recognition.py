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
from opr.pipelines.place_recognition import PlaceRecognitionPipeline


TOPIC = "/zed_node/left/image_rect_color/compressed"
DATABASE_DIR = "/path/to/database"
MODEL_CFG = "/path/to/model/config.yaml"
MODEL_WEIGHTS_PATH = "/path/to/model/weights.pth"
DEVICE = "cuda"
IMAGE_RESIZE = (320, 192)


class PlaceRecognitionNode(Node):

    def __init__(self):
        super().__init__("place_recognition")
        self._declare_parameters()

        image_front_topic = self.get_parameter("image_front_topic").get_parameter_value().string_value
        image_back_topic = self.get_parameter("image_back_topic").get_parameter_value().string_value
        mask_front_topic = self.get_parameter("mask_front_topic").get_parameter_value().string_value
        mask_back_topic = self.get_parameter("mask_back_topic").get_parameter_value().string_value
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        database_dir = self.get_parameter("database_dir").get_parameter_value().string_value
        model_cfg = self.get_parameter("model_cfg").get_parameter_value().string_value
        model_weights_path = self.get_parameter("model_weights_path").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
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

        self.pose_pub = self.create_publisher(PoseStamped, "/place_recognition/pose", 10)
        self.idx_pub = self.create_publisher(DatabaseMatchIndex, "/place_recognition/db_idx", 10)

        model_config = OmegaConf.load(model_cfg)
        model = instantiate(model_config)

        self.pr_pipe = PlaceRecognitionPipeline(
            database_dir=database_dir,
            model=model,
            model_weights_path=model_weights_path,
            device=device,
            pointcloud_quantization_size=0.5,
        )

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
            "database_dir",
            rclpy.Parameter.Type.STRING, #"",
            ParameterDescriptor(description="Path to the database directory with faiss index.")
        )
        self.declare_parameter(
            "model_cfg",
            rclpy.Parameter.Type.STRING, #"",
            ParameterDescriptor(description="Path to the model configuration file.")
        )
        self.declare_parameter(
            "model_weights_path",
            rclpy.Parameter.Type.STRING, #"",
            ParameterDescriptor(description="Path to the model weights.")
        )
        self.declare_parameter(
            "device",
            rclpy.Parameter.Type.STRING, #"cuda",
            ParameterDescriptor(description="Device to use for inference.")
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

        return input_data

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
        output = self.pr_pipe.infer(input_data)
        t_taken = self.get_clock().now() - t_start
        self.get_logger().debug(f"Place recognition inference took: {t_taken.nanoseconds / 1e6} ms.")
        pose_msg = self._create_pose_msg(output["pose"], lidar_timestamp)
        idx_msg = self._create_idx_msg(int(output["idx"]), lidar_timestamp)
        self.pose_pub.publish(pose_msg)
        self.get_logger().info(f"Published pose message: {pose_msg.pose}")
        self.idx_pub.publish(idx_msg)
        self.get_logger().info(f"Published database index message: {idx_msg.index}")


def main(args=None):
    rclpy.init(args=args)

    pr_node = PlaceRecognitionNode()

    rclpy.spin(pr_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pr_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
