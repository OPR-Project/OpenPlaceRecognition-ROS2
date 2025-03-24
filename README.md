# OpenPlaceRecognition-ROS2

The ROS2 wrapper is intended for:

- 🔌 **Easy integration** of multimodal localization into robots and autonomous systems based on ROS2.
- 🚘 **Application in existing robotic platforms and autopilots** running ROS2.
- ⚙️ **Rapid deployment** of neural network-based global localization methods into real-world ROS2 projects.

> Check out out [OpenPlaceRecognition library](https://github.com/OPR-Project/OpenPlaceRecognition).
> It is a library for place recognition and localization, which includes a collection of datasets, including ITLP-Campus.
> The library provides a unified API for loading datasets, training and evaluating models, and performing place recognition tasks.
> The library is designed to be easy to use and extensible, allowing researchers and developers to quickly experiment with different models and datasets.

## Installation

### Requirements

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

- **Dependencies**:
  - Python >= 3.10
  - CUDA Toolkit >= 11.1
  - cuDNN >= 7.5
  - [OpenPlaceRecognition](https://github.com/OPR-Project/OpenPlaceRecognition) 
  - ROS2 Humble

### Quick start

It is highly recommended to use the provided Dockerfile to build the environment.
The scripts to build, run and enter the container are provided in the [docker/](./docker) directory.
You can use the following commands:

```bash
# 0. clone the repository and init submodules
git clone https://github.com/OPR-Project/OpenPlaceRecognition-ROS2.git
cd OpenPlaceRecognition-ROS2
git submodule update --init --recursive

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
