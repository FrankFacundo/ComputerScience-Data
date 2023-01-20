#!/bin/bash
set -e

image=$1

echo $image

if [[ -d $image ]]; then
    docker build $image --tag $image
else
    echo "Image not found. Choose one in this directory."
fi
