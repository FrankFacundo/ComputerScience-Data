#!/bin/bash

set -x

docker build -t files .
docker run --rm files ls -alh /files
docker tag files frankfacundo/files
# docker login -u frankfacundo
docker push frankfacundo/files
