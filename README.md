# OpenPlaceRecognition-ROS2

The ROS2 wrapper is intended for:

- 🔌 **Easy integration** of multimodal localization into robots and autonomous systems based on ROS2.
- 🚘 **Application in existing robotic platforms and autopilots** running ROS2.
- ⚙️ **Rapid deployment** of neural network-based global localization methods into real-world ROS2 projects.

# Installation

## Prerequisites

- [OpenPlaceRecognition](https://github.com/OPR-Project/OpenPlaceRecognition) recommended environment:
    - Ubuntu 22.04
    - Python 3.10
    - CUDA 12.1.1
    - cuDNN 8
    - PyTorch 2.1.2
    - torhvision 0.16.2
    - MinkowskiEngine
    - faiss
- ROS2 Humble

It is highly recommended to use the provided Dockerfile to build the environment.
The scripts to build, run and enter the container are provided in the [docker/](./docker) directory.
You can use the following commands:

```bash
# run from the repo root directory

# 1. build the image
bash docker/build.sh

# 2. start the container and mount the data directory
bash docker/start.sh [DATA_DIR]

# 3. enter the container
bash docker/into.sh
```

To use [OpenPlaceRecognition](https://github.com/OPR-Project/OpenPlaceRecognition) library in the container, you need to install it first.
Run the following command inside the container:

```bash
pip install -e ~/ros2_ws/dependencies/OpenPlaceRecognition
```

# Running nodes

## Build the workspace

Inside `ros2_ws/` directory, run the following command:

```bash
colcon build --packages-select open_place_recognition opr_interfaces
```

## Run the nodes

Open the new terminal and run:

```bash
source ros2_ws/install/setup.bash
```

Run the place recognition node using launch file:

```bash
ros2 launch open_place_recognition place_recognition_launch.py
```

Run the visualizer node using launch file:

```bash
ros2 launch open_place_recognition visualizer_launch.py
```

# License

[Apache 2.0 license](./LICENSE)
