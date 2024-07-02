import os
from typing import Tuple

import cv2
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
import pandas as pd

from opr_interfaces.msg import DatabaseMatchIndex
from opr.datasets.augmentations import DefaultImageTransform, DefaultSemanticTransform
from opr.pipelines.place_recognition import PlaceRecognitionPipeline


class VisualizerNode(Node):

    def __init__(self):
        super().__init__("visualizer")
        self._declare_parameters()

        self.database_dir = self.get_parameter("database_dir").get_parameter_value().string_value

        self.cv_bridge = CvBridge()

        self.pose_sub = Subscriber(self, PoseStamped, "/place_recognition/pose")
        self.idx_sub = Subscriber(self, DatabaseMatchIndex, "/place_recognition/db_idx")

        self.ts = ApproximateTimeSynchronizer(
            [self.pose_sub, self.idx_sub],
            queue_size=1,
            slop=0.01,
        )
        self.ts.registerCallback(self.listener_callback)

        self.db_front_cam_pub = self.create_publisher(Image, "/place_recognition/db_front_cam", 10)
        self.db_back_cam_pub = self.create_publisher(Image, "/place_recognition/db_back_cam", 10)

        self.database_df = pd.read_csv(os.path.join(self.database_dir, "track.csv"))

        self.get_logger().info(f"Initialized {self.__class__.__name__} node.")

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "database_dir",
            rclpy.Parameter.Type.STRING, #"",
            ParameterDescriptor(description="Path to the database directory with faiss index.")
        )

    def _get_images(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_front = cv2.imread(
            os.path.join(self.database_dir, f"front_cam/{int(self.database_df['front_cam_ts'].iloc[idx])}.png")
        )
        img_back = cv2.imread(
            os.path.join(self.database_dir, f"back_cam/{int(self.database_df['back_cam_ts'].iloc[idx])}.png")
        )
        return img_front, img_back

    def listener_callback(
            self,
            pose_msg: PoseStamped,
            idx_msg: DatabaseMatchIndex,
        ) -> None:
            idx = idx_msg.index
            img_front, img_back = self._get_images(idx)
            img_front_msg = self.cv_bridge.cv2_to_imgmsg(img_front)
            img_front_msg.header.stamp = pose_msg.header.stamp
            img_back_msg = self.cv_bridge.cv2_to_imgmsg(img_back)
            img_back_msg.header.stamp = pose_msg.header.stamp
            self.db_front_cam_pub.publish(img_front_msg)
            self.db_back_cam_pub.publish(img_back_msg)


def main(args=None):
    rclpy.init(args=args)

    vis_node = VisualizerNode()

    rclpy.spin(vis_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vis_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
