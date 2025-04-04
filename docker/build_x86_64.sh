#!/bin/bash

orange=$(tput setaf 3)
reset_color=$(tput sgr0)

ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    echo "Detected ${orange}CUDA${reset_color} hardware"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile.x86_64"
    DEVICE=cuda
else
    echo "${orange}CPU-only${reset_color} build is not supported yet"
    exit 1
fi

echo "Building for ${orange}${ARCH}${reset_color} with ${orange}${DEVICE}${reset_color}"

# OPR_PATH must be provided as an absolute path.
if [ -z "$OPR_PATH" ]; then
    echo "Error: Please set the OPR_PATH environment variable to your external OpenPlaceRecognition directory."
    exit 1
fi

# Check if OPR_PATH exists.
if [ ! -d "$OPR_PATH" ]; then
    echo "Error: OPR_PATH ($OPR_PATH) does not exist."
    exit 1
fi
echo "Using OpenPlaceRecognition path: ${orange}${OPR_PATH}${reset_color}"

# Create a temporary build context directory (outside of your repository)
TMP_BUILD_CONTEXT=$(mktemp -d)
echo "Using temporary build context: ${orange}${TMP_BUILD_CONTEXT}${reset_color}"

# Copy the Dockerfile into the temporary build context.
cp "$DOCKERFILE_PATH" "$TMP_BUILD_CONTEXT/"

# Copy the external OPR directory into the temporary build context under a dedicated folder.
# This prevents polluting your repository.
mkdir -p "$TMP_BUILD_CONTEXT/OpenPlaceRecognition_external"
cp -r "$OPR_PATH" "$TMP_BUILD_CONTEXT/OpenPlaceRecognition_external"

# The build argument OPR_PATH will be set to point to the external folder inside the temporary build context.
# For example, if OPR_PATH=/home/rover2/Downloads/OpenPlaceRecognition/OpenPlaceRecognition,
# then $(basename "$OPR_PATH") will be "OpenPlaceRecognition", and the full relative path becomes:
# "OpenPlaceRecognition_external/OpenPlaceRecognition"
BUILD_OPR_PATH="OpenPlaceRecognition_external/$(basename "$OPR_PATH")"

# Allow external specification of the base image.
if [ -z "$BASE_IMAGE" ]; then
    BASE_IMAGE=alexmelekhin/open-place_recognition:base
fi
echo "Using base image: ${orange}${BASE_IMAGE}${reset_color}"

docker build "$TMP_BUILD_CONTEXT" \
    --build-arg MAX_JOBS=4 \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    --build-arg OPR_PATH="${BUILD_OPR_PATH}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -f "$TMP_BUILD_CONTEXT/$(basename "$DOCKERFILE_PATH")" \
    -t open-place-recognition-ros2:devel \
    --network=host

# Clean up the temporary build context.
rm -rf "$TMP_BUILD_CONTEXT"
