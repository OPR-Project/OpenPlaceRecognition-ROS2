#!/usr/bin/env python3
import os
import sys
import time

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf

from opr.datasets.itlp import ITLPCampus

class DatasetTrainNode(Node):
    def __init__(self):
        super().__init__('dataset_train_node')
        #
        # ---------------------------
        # Declare ROS 2 Parameters
        # ---------------------------
        #
        # General dataset paths
        self.declare_parameter('dataset_path', '')
        self.declare_parameter('output_path', '~/.ros/opr_dataset')

        # MinkLoc3D / descriptor extraction parameters
        self.declare_parameter('batch_size', 64)
        self.declare_parameter('num_workers', 4)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('model_config_path', 'configs/model/place_recognition/minkloc3d.yaml')
        self.declare_parameter('weights_path', 'weights/place_recognition/minkloc3d_nclt.pth')

        # Additional dataset options
        self.declare_parameter('mink_quantization_size', 0.5)
        self.declare_parameter('load_semantics', False)
        self.declare_parameter('load_text_descriptions', False)
        self.declare_parameter('load_text_labels', False)
        self.declare_parameter('load_aruco_labels', False)
        self.declare_parameter('indoor', False)

        #
        # ---------------------------
        # Fetch Parameter Values
        # ---------------------------
        #
        dataset_path = os.path.expanduser(self.get_parameter('dataset_path').value)
        output_path = os.path.expanduser(self.get_parameter('output_path').value)

        # MinkLoc3D / descriptor extraction parameters
        dataset_path = os.path.expanduser(self.get_parameter('dataset_path').value)
        batch_size = int(self.get_parameter('batch_size').value)
        num_workers = int(self.get_parameter('num_workers').value)
        device = self.get_parameter('device').value
        model_config_path = self.get_parameter('model_config_path').value
        weights_path = self.get_parameter('weights_path').value

        # Additional dataset options
        mink_quantization_size = float(self.get_parameter('mink_quantization_size').value)
        load_semantics = bool(self.get_parameter('load_semantics').value)
        load_text_descriptions = bool(self.get_parameter('load_text_descriptions').value)
        load_text_labels = bool(self.get_parameter('load_text_labels').value)
        load_aruco_labels = bool(self.get_parameter('load_aruco_labels').value)
        indoor = bool(self.get_parameter('indoor').value)

        #
        # ---------------------------
        # Check and prepare directories
        # ---------------------------
        #
        if not os.path.exists(dataset_path):
            self.get_logger().error(
                f"Datasets directory {dataset_path} does not exist. Please create it and try again."
            )
            sys.exit(1)

        if not os.path.exists(output_path):
            self.get_logger().error(
                f"Output directory {output_path} does not exist. Please create it and try again."
            )
            sys.exit(1)

        #
        # ---------------------------
        # Run the descriptor-extraction (or "train") function
        # ---------------------------
        #
        status_ok = self.run_descriptor_extraction(
            dataset_path=dataset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            model_config_path=model_config_path,
            weights_path=weights_path,
            output_path=output_path,
            mink_quantization_size=mink_quantization_size,
            load_semantics=load_semantics,
            load_text_descriptions=load_text_descriptions,
            load_text_labels=load_text_labels,
            load_aruco_labels=load_aruco_labels,
            indoor=indoor
        )

        if not status_ok:
            self.get_logger().error("Error during descriptor extraction.")
            sys.exit(1)
        else:
            self.get_logger().info("Descriptor extraction completed successfully.")

    def run_descriptor_extraction(
        self,
        dataset_path: str,
        batch_size: int,
        num_workers: int,
        device: str,
        model_config_path: str,
        weights_path: str,
        output_path: str,
        mink_quantization_size: float,
        load_semantics: bool,
        load_text_descriptions: bool,
        load_text_labels: bool,
        load_aruco_labels: bool,
        indoor: bool
    ):
        """
        Load dataset, run the MinkLoc3D model to extract descriptors, and build a FAISS index.
        """
        try:
            #
            # 1. Prepare the dataset
            #
            self.get_logger().info(f"Loading dataset from: {dataset_path}")
            db_dataset = ITLPCampus(
                dataset_root=dataset_path,
                sensors=["lidar"],  # or parameterize if you want
                mink_quantization_size=mink_quantization_size,
                load_semantics=load_semantics,
                load_text_descriptions=load_text_descriptions,
                load_text_labels=load_text_labels,
                load_aruco_labels=load_aruco_labels,
                indoor=indoor,
            )

            self.get_logger().info(f"Dataset size: {len(db_dataset)}")

            db_dataloader = DataLoader(
                db_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=db_dataset.collate_fn,
            )

            #
            # 2. Load the MinkLoc3D model
            #
            self.get_logger().info(f"Loading MinkLoc3D config from: {model_config_path}")
            model_config = OmegaConf.load(model_config_path)
            model = instantiate(model_config)

            self.get_logger().info(f"Loading MinkLoc3D weights from: {weights_path}")
            model.load_state_dict(torch.load(weights_path))
            model = model.to(device)
            model.eval()

            #
            # 3. Extract descriptors
            #
            descriptors_list = []
            self.get_logger().info("Extracting descriptors...")
            with torch.no_grad():
                for batch in tqdm(db_dataloader, desc="Descriptor Extraction", leave=True):
                    # Move input tensors to the correct device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    # Forward pass
                    output = model(batch)
                    final_descriptor = output["final_descriptor"]  # MinkLoc3D's key
                    descriptors_list.append(final_descriptor.detach().cpu().numpy())

            descriptors = np.concatenate(descriptors_list, axis=0)
            self.get_logger().info(f"Descriptors shape: {descriptors.shape}")

            #
            # 4. Build a FAISS index
            #
            index_dim = descriptors.shape[1]
            index = faiss.IndexFlatL2(index_dim)
            index.add(descriptors)
            self.get_logger().info(f"FAISS index trained: {index.is_trained}, total vectors: {index.ntotal}")

            #
            # 5. Write the index to disk
            #
            faiss_index_path = os.path.join(output_path, "index.faiss")
            faiss.write_index(index, faiss_index_path)
            self.get_logger().info(f"FAISS index saved to: {faiss_index_path}")

            #
            # 6. List directory contents (optional logging)
            #
            dirs = [d for d in os.listdir(dataset_path)
                    if os.path.isdir(os.path.join(dataset_path, d))]
            files = [f for f in os.listdir(dataset_path)
                     if os.path.isfile(os.path.join(dataset_path, f))]

            self.get_logger().info("Directory structure in dataset root:")
            for item in sorted(dirs):
                self.get_logger().info(f"  {item}/")
            for item in sorted(files):
                self.get_logger().info(f"  {item}")

            return True

        except Exception as e:
            self.get_logger().error(f"Exception in run_descriptor_extraction: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = DatasetTrainNode()
    # Spin once and exit (this node does a one-shot operation)
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
