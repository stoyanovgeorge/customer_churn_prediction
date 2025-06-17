#!/bin/bash

# Check if version argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.0"
    exit 1
fi

VERSION=$1
IMAGE_NAME="rf-api"
REGISTRY="127.0.0.1:5000"

# Build the Docker image
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}":"${VERSION}" .

# Tag the image for the local registry
echo "Tagging image for local registry..."
docker tag "${IMAGE_NAME}":"${VERSION}" "${REGISTRY}"/"${IMAGE_NAME}":"${VERSION}"

# Push the image to the local registry
echo "Pushing image to local registry..."
docker push "${REGISTRY}"/"${IMAGE_NAME}":"${VERSION}"

echo "Done! Image ${IMAGE_NAME}:${VERSION} has been built and pushed to ${REGISTRY}"