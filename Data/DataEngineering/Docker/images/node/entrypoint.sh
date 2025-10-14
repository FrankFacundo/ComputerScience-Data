#!/bin/bash

set -x

docker build -t node .
docker tag node frankfacundo/node
# docker login -u frankfacundo
docker push frankfacundo/node
