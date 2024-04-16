# OpenPlaceRecognition-ROS2

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
