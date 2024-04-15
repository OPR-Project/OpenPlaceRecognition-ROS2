#!/bin/bash

docker exec --user docker_opr_ros2 -it ${USER}_opr_ros2 \
    /bin/bash -c "cd /home/docker_opr_ros2; echo ${USER}_opr_ros2 container; echo ; /bin/bash"
