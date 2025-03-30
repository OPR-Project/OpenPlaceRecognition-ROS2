# OpenPlaceRecognition-ROS2
Note: This project is the revisions for the Open PLace Recognition project for ROS2 Humble

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

create a folder to store:

cd ~/Downloads
mkdir OpenPlaceRecognition
cd OpenPlaceRecognition
git clone Datasets
git clone OpenPlaceRecognition


To use [OpenPlaceRecognition](https://github.com/OPR-Project/OpenPlaceRecognition) library in the container, you need to install it first.

```bash
# 1. build the image
export OPR_PATH=$HOME/Downloads/OpenPlaceRecognition/OpenPlaceRecognition
export DATASETS_DIR=$HOME/Downloads/OpenPlaceRecognition/Datasets
export DISPLAY=:0
bash docker/build_x86_64.sh

# 2. start the container and mount the data directory
bash docker/start_x86_64.sh

# 3. enter the container
bash docker/into.sh
```

Run the following command inside the container:
```bash
pip install -e ~/OpenPlaceRecognition
pip install -e ~/OpenPlaceRecognition/third_party/GeoTransformer
pip install -e ~/OpenPlaceRecognition/third_party/HRegNet
pip install -e ~/OpenPlaceRecognition/third_party/PointMamba
```

# Running nodes

## Build the workspace

Inside `ros2_ws/` directory, run the following command:

```bash
cd ~/ros2_ws/
colcon build --symlink-install
```

## Run the nodes

Open the new terminal and run:

```bash
source ~/ros2_ws/install/setup.bash
```

Run the place recognition node using launch file:

```bash
ros2 launch open_place_recognition database_publisher.launch.py
```

```bash
ros2 launch open_place_recognition place_recognition.launch.py
```

Run the visualizer node using launch file:

```bash
ros2 launch open_place_recognition visualizer.launch.py
```


## Package Overview

The package includes three main nodes:

1. **Dataset Creation Node (`dataset_create_node.py`)**  
   - Creates a dataset for a given map.
   - Accepts parameters:
     - `dataset_path`: Directory path to store or access the dataset.
     - `map_name`: Name of the map for which the dataset is generated.

2. **Dataset Training Node (`dataset_train_node.py`)**  
   - Simulates training on the created dataset using Torch.
   - Accepts parameters:
     - `dataset_path`: Directory path where the dataset is stored.
     - `map_name`: Name of the map used for training.

3. **Model Operation Node (`model_opr_node.py`)**  
   - Simulates reading a trained model and performing operations with it.
   - Accepts parameters:
     - `trained_model_path`: Directory path of the trained model.
     - `map_name`: Name of the map used during training.


## Package Structure

```
open_place_recognition/
├── launch/
│   ├── dataset_create.launch.py
│   ├── dataset_train.launch.py
│   └── opr_odometry.launch.py
├── src/
│   ├── dataset_create_node.py
│   ├── dataset_train_node.py
│   └── opr_odom_node.py
├── package.xml
└── README.md
```


## Building the Package

1. **Clone the repository into your ROS2 workspace's `src` directory:**

    ```bash
    cd ~/ros2_ws/src
    git clone <repository_url> open_place_recognition
    ```

2. **Build the workspace using `colcon`:**

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select open_place_recognition
    ```

3. **Source your workspace:**

    ```bash
    source install/setup.bash
    ```

## Running the Nodes

After building the package, you can launch the nodes using the provided launch files.

### 1. Create Dataset

```bash
ros2 launch open_place_recognition dataset_create.launch.py
```

This command starts the Dataset Creation Node using the parameters provided in the launch file (e.g., dataset path and map name).

### 2. Train Dataset

```bash
ros2 launch open_place_recognition dataset_train.launch.py
```

This command starts the Dataset Training Node, which simulates training on the dataset created in the previous step.

### 3. Operate on Trained Model

```bash
ros2 launch open_place_recognition opr.launch.py
```

This command starts the Model Operation Node, which simulates reading the trained model and performing operations on it.

export PATH=$PATH:$HOME/.local/bin
docker_opr_ros2@rover2:~/OpenPlaceRecognition/notebooks$ jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

## License

[Apache 2.0 license](./LICENSE)