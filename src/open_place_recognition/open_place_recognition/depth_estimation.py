from rclpy.node import Node
import numpy as np
import rclpy
import torch
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from hydra.utils import instantiate
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omegaconf import OmegaConf
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, CameraInfo
from sensor_msgs_py.point_cloud2 import read_points, create_cloud_xyz32
from tf2_ros import Buffer
from rcl_interfaces.msg import ParameterDescriptor

import sys
sys.path.append('/home/docker_opr_ros2/ros2_ws/dependencies/OpenPlaceRecognition/third_party/AdelaiDepth/LeReS/Minist_Test')
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
import argparse

from opr.pipelines.depth_estimation import DepthEstimationPipeline

def parse_args(a):
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args(a)
    return args

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__("depth_estimation")
        self._declare_parameters()

        image_front_topic = self.get_parameter("image_front_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_front_topic").get_parameter_value().string_value
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        model_weights_path = self.get_parameter("model_weights_path").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        image_resize = self.get_parameter("image_resize").get_parameter_value().integer_array_value
        self.publish_point_cloud_from_depth = True#self.get_parameter("publish_point_cloud_from_depth").get_parameter_value().bool_value

        self.cv_bridge = CvBridge()

        self.image_front_sub = Subscriber(self, CompressedImage, image_front_topic)
        self.camera_info_sub = Subscriber(self, CameraInfo, camera_info_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)
        self.tf_buffer = Buffer()

        self.ts = ApproximateTimeSynchronizer(
            [self.image_front_sub, self.lidar_sub, self.camera_info_sub],
            queue_size=1,
            slop=0.05,
        )
        self.ts.registerCallback(self.listener_callback)

        self.depth_pub = self.create_publisher(Image, "/depth_estimation/depth", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/depth_estimation/point_cloud_from_depth", 10)
        self.image_raw_pub = self.create_publisher(Image, "zed_node/left/image_rect_color", 10)

        arguments = "--load_ckpt {} \
                    --backbone resnet50".format(model_weights_path).split()
        args = parse_args(arguments)
        rel_depth_model = RelDepthModel(backbone='resnet50').cuda()
        load_ckpt(args, rel_depth_model, None, None)
        self.pipeline = DepthEstimationPipeline(rel_depth_model)


    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "image_front_topic",
            rclpy.Parameter.Type.STRING, #"/zed_node/left/image_rect_color/compressed",
            ParameterDescriptor(description="Front camera image topic.")
        )
        self.declare_parameter(
            "camera_info_front_topic",
            rclpy.Parameter.Type.STRING, #"/zed_node/left/image_rect_color/compressed",
            ParameterDescriptor(description="Front camera info topic.")
        )
        self.declare_parameter(
            "lidar_topic",
            rclpy.Parameter.Type.STRING, #"/velodyne_points",
            ParameterDescriptor(description="Lidar pointcloud topic.")
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
        #self.declare_parameter(
        #    "publish_point_cloud_from_depth", #True,
        #    rclpy.Parameter.Type.BOOL,
        #    ParameterDescriptor(description="Publish poinr cloud from depth or not.")
        #)

    def get_point_cloud_from_depth(self, depth, camera_matrix):
        f = camera_matrix['f']
        cx = camera_matrix['cx']
        cy = camera_matrix['cy']
        h, w = depth.shape
        i = np.tile(np.arange(h), w).reshape((w, h)).T
        j = np.tile(np.arange(w), h).reshape((h, w))
        z = depth.ravel()
        x = (j.ravel() - cx) / f * z
        y = (i.ravel() - cy) / f * z
        pcd = np.zeros((x.shape[0], 3))
        pcd[:, 0] = x
        pcd[:, 1] = y
        pcd[:, 2] = z
        return pcd

    def listener_callback(
            self,
            front_image_msg: CompressedImage,
            lidar_msg: PointCloud2,
            camera_info_msg: CameraInfo
        ) -> None:
        print('Start getting transform')
        try:
            transform_msg = self.tf_buffer.lookup_transform(lidar_msg.header.frame_id, front_image_msg.header.frame_id, 0)
        except:
            print('No transform from {} to {}'.format(lidar_msg.header.frame_id, front_image_msg.header.frame_id))
            translation = np.array([[0.061], [0.049], [-0.131]])
            rotation = [-0.498, 0.498, -0.495, 0.510]
        R = Rotation.from_quat(rotation).as_matrix()
        #R = np.linalg.inv(R)
        tf_matrix = np.concatenate([R, translation], axis=1)
        tf_matrix = np.concatenate([tf_matrix, np.array([[0, 0, 0, 1]])], axis=0)
        print('Transform:', tf_matrix)
        self.pipeline.set_lidar_to_camera_transform(tf_matrix)
        camera_matrix = {'f': camera_info_msg.k[0], 
                         'cx': camera_info_msg.k[2], 
                         'cy': camera_info_msg.k[5]}
        self.pipeline.set_camera_matrix(camera_matrix)

        image = self.cv_bridge.compressed_imgmsg_to_cv2(front_image_msg)
        pointcloud = read_points(lidar_msg, field_names=("x", "y", "z"))
        pointcloud = np.array([pointcloud["x"], pointcloud["y"], pointcloud["z"]]).T
        depth = self.pipeline.get_depth_with_lidar(image, pointcloud)
        print(depth.min(), depth.max())

        image_raw = self.cv_bridge.cv2_to_imgmsg(image)
        image_raw.header = front_image_msg.header
        self.image_raw_pub.publish(image_raw)

        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth)
        #depth_msg.is_bigendian = True
        depth_msg.header.stamp = front_image_msg.header.stamp
        depth_msg.header.frame_id = front_image_msg.header.frame_id
        self.depth_pub.publish(depth_msg)

        if self.publish_point_cloud_from_depth:
            point_cloud_from_depth = self.get_point_cloud_from_depth(depth, camera_matrix)
            point_cloud_msg = create_cloud_xyz32(front_image_msg.header, point_cloud_from_depth)
            self.cloud_pub.publish(point_cloud_msg)


def main(args=None):
    rclpy.init(args=args)

    de_node = DepthEstimationNode()

    rclpy.spin(de_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    de_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()