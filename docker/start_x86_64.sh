#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

# get_real_path(){
#     if [ "${1:0:1}" == "/" ]; then
#         echo "$1"
#     else
#         realpath -m "$PWD"/"$1"
#     fi
# }

ARCH=`uname -m`
if [ $ARCH == "x86_64" ]; then
    if command -v nvidia-smi &> /dev/null; then
        DEVICE=cuda
        ARGS="--ipc host --gpus all"
    else
        echo "${orange}CPU-only${reset_color} build is not supported yet"
        exit 1
    fi
else
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi


if [ -z "$DATASETS_DIR" ]; then
    echo "Error: DATASETS_DIR environment variable is not set. Please set it to your Datasets directory."
    exit 1
fi

if [ ! -d $DATASETS_DIR ]; then
    echo "Error: DATASETS_DIR=$DATASETS_DIR is not an existing directory."
    exit 1
fi

if [ -z "$OPR_PATH" ]; then
    echo "Error: OPR_PATH environment variable is not set. Please set it to your OpenPlaceRecognition directory."
    exit 1
fi

PROJECT_ROOT_DIR=$(cd ./"`dirname $0`"/.. || exit; pwd)

echo "Running on ${orange}${ARCH}${reset_color} with ${orange}${DEVICE}${reset_color}"

xhost +
    docker run -it -d --rm \
        $ARGS \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name ${USER}_opr_ros2 \
        --net host \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $PROJECT_ROOT_DIR:/home/docker_opr_ros2/ros2_ws/src:rw \
        -v $DATASETS_DIR:/home/docker_opr_ros2/Datasets:rw \
        -v $OPR_PATH:/home/docker_opr_ros2/OpenPlaceRecognition:rw \
        open-place-recognition-ros2:devel
xhost -

docker exec --user root \
    ${USER}_opr_ros2 bash -c "/etc/init.d/ssh start"
