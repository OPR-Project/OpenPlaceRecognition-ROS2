# OpenPlaceRecognition-ROS2

The OpenPlaceRecognition ROS2 wrapper is intended for:

- 🔌 **Easy integration** of multimodal localization into robots and autonomous systems based on ROS2.
- 🚘 **Application in existing robotic platforms and autopilots** running ROS2.
- ⚙️ **Rapid deployment** of neural network-based global localization methods into real-world ROS2 projects.

# Requirements

#### Hardware

- **CPU**: 6 or more physical cores
- **RAM**: at least 8 GB
- **GPU**: NVIDIA RTX 2060 or higher (to ensure adequate performance)
- **Video memory**: at least 4 GB
- **Storage**: SSD recommended for faster loading of data and models

#### Software

- **Operating System**:
  - Any OS with support for Docker and CUDA >= 11.1.
    *Ubuntu 20.04 or later is recommended.*

- **Dependencies** (if not using Docker):
  - Python >= 3.10
  - CUDA Toolkit >= 11.1
  - cuDNN >= 7.5
  - [OpenPlaceRecognition](https://github.com/OPR-Project/OpenPlaceRecognition) 
  - [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
  - Docker

# Installation

Firstly retreave all dependencies for this project

```bash
# Step 1: Clone the main repository with submodules
cd ~
git clone --recursive https://github.com/OPR-Project/OpenPlaceRecognition

# Step 2: Clone the ITLP-Campus-Outdoor dataset
git clone https://huggingface.co/datasets/OPR-Project/ITLP-Campus-Outdoor ~/Datasets/ITLP-Campus-Outdoor

# Step 3: Create required directories for pretrained weights
mkdir -p ~/OpenPlaceRecognition/weights/place_recognition
mkdir -p ~/OpenPlaceRecognition/weights/registration

# Step 4: Download pretrained weights
wget https://huggingface.co/OPR-Project/PlaceRecognition-NCLT/resolve/main/minkloc3d_nclt.pth \
     -O ~/OpenPlaceRecognition/weights/place_recognition/minkloc3d_nclt.pth

wget https://huggingface.co/OPR-Project/Registration-KITTI/resolve/main/geotransformer_kitti.pth \
     -O ~/OpenPlaceRecognition/weights/registration/geotransformer_kitti.pth

# Step 5: Export environment variables
export OPR_PATH=$HOME/OpenPlaceRecognition
export DATASETS_DIR=$HOME/Datasets
export DISPLAY=:0
```

### Quick Start with Docker

It is highly recommended to use the provided Dockerfile to build the complete environment. The Docker scripts are located in the `docker/` directory. For now, we only support x86_64 architecture. The aarch64 version will be released soon.

```bash
# 0. Clone the repository
git clone https://github.com/OPR-Project/OpenPlaceRecognition-ROS2.git
cd OpenPlaceRecognition-ROS2

# 1. Build the Docker image
bash docker/build_x86_64.sh

# 2. Start the container with the data directory mounted
bash docker/start_x86_64.sh

# 3. Enter the container
bash docker/into.sh
```

Inside the container, install the additional Python dependencies:

```bash
pip install -e ~/OpenPlaceRecognition
pip install -e ~/OpenPlaceRecognition/third_party/GeoTransformer
pip install -e ~/OpenPlaceRecognition/third_party/HRegNet
pip install -e ~/OpenPlaceRecognition/third_party/PointMamba
```

### Build the ROS2 Workspace

```bash
cd ~/ros2_ws/
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```


## Launch Files and Usage

The project provides a variety of launch files to support testing, visualization, and deployment. Each launch file is configurable via command-line arguments to adapt to different sensor setups and dataset paths.

- **Dataset Conversion from Bag Files:**  
  Convert ROS2 bag files into a standardized OPR dataset format for further processing.
  ```bash
  ros2 launch open_place_recognition dataset_from_bag.launch.py
  ```

- **Dataset Conversion from RTAB-Map:**  
  Convert RTAB-Map datasets to the compatible format required by the OPR pipeline.
  ```bash
  ros2 launch open_place_recognition dataset_from_rtabmap.launch.py
  ```

- **Dataset Publisher:**  
  Publish sensor streams (cameras, LiDAR, etc.) for dataset creation or real-time monitoring.
  ```bash
  ros2 launch open_place_recognition dataset_publisher.launch.py
  ```

- **Dataset Indexing:**  
  Index your dataset features to prepare for efficient retrieval and subsequent training.
  ```bash
  ros2 launch open_place_recognition dataset_indexing.launch.py
  ```

- **Place Recognition:**  
  This node performs a simple database search using the current sensor inputs (e.g. images, LiDAR) to identify the closest matching location in the pre-built database.
  ```bash
  ros2 launch open_place_recognition place_recognition.launch.py
  ```

- **Localization:**  
  This pipeline extends place recognition by adding position matching algorithms, improving accuracy and robustness for real-world deployment.
  ```bash
  ros2 launch open_place_recognition localization.launch.py
  ```


## Additional Tools

For further analysis and visualization, you can run Jupyter Lab from within the container:

```bash
export PATH=$PATH:$HOME/.local/bin
cd ~/OpenPlaceRecognition/notebooks
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```


## License

This project is licensed under the [Apache 2.0 License](./LICENSE).