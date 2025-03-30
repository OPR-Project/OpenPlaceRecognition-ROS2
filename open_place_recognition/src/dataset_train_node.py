#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import sys
from ament_index_python.packages import get_package_share_directory
import time

class DatasetTrainNode(Node):
    def __init__(self):
        super().__init__('dataset_train_node')
        # Get parameters
        self.declare_parameter('dataset_path', 'datasets')
        self.declare_parameter('map_name', 'default_map')
        self.declare_parameter('output_path', '~/.ros/opr_dataset')
        
        dataset_path = self.get_parameter('dataset_path').value
        map_name = self.get_parameter('map_name').value
        output_path = os.path.expanduser(self.get_parameter('output_path').value)

        # Get package share directory and build paths for datasets
        package_dir = get_package_share_directory('orca_opr')
        dataset_dir = os.path.join(package_dir, dataset_path)
        
        # Check that required folders exist
        if not os.path.exists(dataset_dir):
            self.get_logger().error(f"Datasets directory {dataset_dir} does not exist. Please create it and try again.")
            sys.exit(1)
        if not os.path.exists(output_path):
            self.get_logger().error(f"Output directory {output_path} does not exist. Please create it and try again.")
            sys.exit(1)

        if not self.run_torch_training(dataset_dir, output_path, map_name):
            self.get_logger().error(f"Error during training {map_name}")
            sys.exit(1)

    def run_torch_training(self, dataset_path: str, output_path: str, map_name: str):
        """
        Loads dataset from dataset_path and performs training for map_name.
        """
        print(f"[TRAINER] Starting training on dataset: {dataset_path} for map: {map_name}")
        time.sleep(3)  # Simulate training time

        # Simulate training by creating a dummy .pt file in the output_path folder
        dummy_model_file = os.path.join(output_path, f"{map_name}.pt")
        with open(dummy_model_file, 'w') as f:
            f.write("dummy model weights")

        self.get_logger().info(f"Dummy model file created at: {dummy_model_file}")
        print(f"[TRAINER] Finished training for map: {map_name} using dataset: {dataset_path}")
        return True

def main(args=None):
    rclpy.init(args=args)
    node = DatasetTrainNode()
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
