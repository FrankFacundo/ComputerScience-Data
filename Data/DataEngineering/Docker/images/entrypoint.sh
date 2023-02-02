#!/bin/bash
set -e

# Command example: ./entrypoint simple
# To list images: docker images
# To execute: docker run <image_id>
image=$1

echo $image

if [[ -d $image ]]; then
    docker build $image --tag $image:1.0
else
    echo "Image not found. Choose one in this directory."
fi
