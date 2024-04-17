import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from hydra.utils import instantiate
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omegaconf import OmegaConf
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from torch import Tensor

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
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        database_dir = self.get_parameter("database_dir").get_parameter_value().string_value
        model_cfg = self.get_parameter("model_cfg").get_parameter_value().string_value
        model_weights_path = self.get_parameter("model_weights_path").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        image_resize = self.get_parameter("image_resize").get_parameter_value().integer_array_value

        self.cv_bridge = CvBridge()

        self.image_front_sub = Subscriber(self, CompressedImage, image_front_topic)
        self.image_back_sub = Subscriber(self, CompressedImage, image_back_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_front_sub, self.image_back_sub, self.lidar_sub], queue_size=1, slop=0.05,
        )
        self.ts.registerCallback(self.listener_callback)

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

    def listener_callback(
            self, front_image_msg: CompressedImage, back_image_msg: CompressedImage, lidar_msg: PointCloud2
        ) -> None:
        front_image = self.cv_bridge.compressed_imgmsg_to_cv2(front_image_msg)
        back_image = self.cv_bridge.compressed_imgmsg_to_cv2(back_image_msg)
        pointcloud = read_points(lidar_msg, field_names=("x", "y", "z"))
        pointcloud = np.array([pointcloud["x"], pointcloud["y"], pointcloud["z"]]).T
        input_data = self._prepare_input(images=[front_image, back_image], pointcloud=pointcloud)
        output = self.pr_pipe.infer(input_data)
        self.get_logger().info(f"Place recognition output: {output.keys()}")


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
