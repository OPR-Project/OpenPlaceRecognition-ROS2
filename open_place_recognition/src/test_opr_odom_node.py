#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import sys
from ament_index_python.packages import get_package_share_directory

class ModelOPRNode(Node):
    def __init__(self):
        super().__init__('model_opr_node')
        # Get parameter (only map_name is needed)
        self.declare_parameter('map_name', 'default_map')
        map_name = self.get_parameter('map_name').value

        # Get package share directory and build the weights path
        package_dir = get_package_share_directory('orca_opr')
        weights_path = os.path.join(package_dir, 'weights')
        
        if not os.path.exists(weights_path):
            self.get_logger().error(f"Weights directory {weights_path} does not exist. Please create it and try again.")
            sys.exit(1)
        
        dummy_model_file = os.path.join(weights_path, f"{map_name}.pt")
        if not os.path.exists(dummy_model_file):
            self.get_logger().error(f"Model file {dummy_model_file} does not exist. Please run the training node first.")
            sys.exit(1)
        else:
            self.get_logger().info(f"Model file {dummy_model_file} found. Loading model...")
            # Simulate model loading (in future, actual torch code will be used)
            self.get_logger().info("Model loaded successfully.")

def main(args=None):
    rclpy.init(args=args)
    node = ModelOPRNode()
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
