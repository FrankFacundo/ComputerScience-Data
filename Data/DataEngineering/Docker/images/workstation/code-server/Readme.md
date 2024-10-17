# Commands

IMAGE_VERSION=0.96.3

docker build -t code-server-${IMAGE_VERSION} .

docker run -it --mount type=bind,source=/home/frank/Code/ComputerScience-Data,target=/workspace -p 40000:40000 code-server-${IMAGE_VERSION}

docker tag code-server-${IMAGE_VERSION} frankfacundo/workstation:code-server-${IMAGE_VERSION}

docker push frankfacundo/workstation:code-server-${IMAGE_VERSION}
