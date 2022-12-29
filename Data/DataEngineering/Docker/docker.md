Docker is a tool (container software) that helps to create **"images"**. An **"image"** (or **Docker image**) is a portable package that contains the application and its dependencies. An "image" can be instantiated multiple numbers of times to create **"containers"**.

Containers
----------

OS level virtualization allows us to run **multiple isolated user-space instances** in parallel. A **"container"** (or **Docker container**) is the isolated user-space instance that has the application code, the required dependencies, and the necessary runtime environment to run the application. Containers can dash on heterogeneous platforms.

### Benefit of Containers

-   Docker images make it easier for developers to create, deploy, and run applications on different hardware and platforms, quickly and easily. Docker has become an essential tool in CI/CD pipeline as it allows software developers a consistent and automated way to build, package, and test applications.
-   Containers share a single kernel and share application libraries.
-   Containers cause a lower system overhead as compared to Virtual Machines.

Refer to the [Docker documentation](https://docs.docker.com/) for more information.

### Commands (console)
https://docs.docker.com/engine/reference/commandline/docker/

 The following are basic commands used with Docker:

-   `docker build .` will run the Dockerfile to create an image. A Dockerfile is a text file that contains commands as a step-by-step recipe on how to build up your image. In our case, we would not use a Dockerfile because we will use a pre-created `jenkinsci/blueocean` image to instantiate a container. For more details about Dockerfile, refer the [Build and run your image](https://docs.docker.com/get-started/part2/) page.
-   `docker images` will print all the available images
-   `docker run {IMAGE_ID}` will run a container with the image
-   `docker exec -it sh` to attach to a container
-   `docker exec -it {IMAGE_ID} bash`
-   `docker cp <src-path> <container>:<dest-path> `

```
The difference between “docker run” and “docker exec” is that “docker exec” executes a command on a running container. On the other hand, “docker run” creates a temporary container, executes the command in it and stops the container when it is done.
```

-   `docker ps` will print all the running containers
	-   For docker-compose with id : `docker-compose ps --services | awk '{ print "echo -n \\""$1": \\" ; docker-compose ps -q "$1 }' | bash`
-   `docker kill {CONTAINER_ID}` will terminate the container


### Commands (Dockerfile)

- CMD : The `CMD` instruction has three forms:
	-   `CMD ["executable","param1","param2"]` (_exec_ form, this is the preferred form)
	-   `CMD ["param1","param2"]` (as _default parameters to ENTRYPOINT_)
	-   `CMD command param1 param2` (_shell_ form)

### Key Terms - Docker

This is additional learning so that you stay aware of key terminologies of Docker.

- Base Image : A set of common dependencies built into a Docker image that acts as a starting point to build an application’s Docker images to reduce build times

- Container : Grouped software dependencies and packages that make it easier and more reliable to deploy software

- Container Registry : A centralized place to store container images

- Docker-compose : A tool used to run multiple Docker containers at once; often used to specify dependent relationships between containers

- Dockerfile : A file containing instructions on how to translate an application into an image that can be run in containers

- Ephemeral : Software property where an application is expected to be short-lived

- Image - A snapshot of dependencies and code used by Docker containers to run an application

### Docker or Kubernetes ?
In short, Docker and Kubernetes do not pursue the same objective: Docker allows you to develop, deploy and therefore iterate faster on your product, while Kubernetes is the solution for the safe "runner" in production.
A equivalent of Kubernetes in Docker is Docker Swarm. Otherwise usually Kubernetes is used together with Docker.

### Docker Desktop (Windows and Mac)
https://docs.docker.com/desktop/
Docker Desktop is an easy-to-install application for your Mac or Windows environment that enables you to build and share containerized applications and microservices. Docker Desktop includes [Docker Engine](https://docs.docker.com/engine/), Docker CLI client, [Docker Compose](https://docs.docker.com/compose/), [Notary](https://docs.docker.com/notary/getting_started/), [Kubernetes](https://github.com/kubernetes/kubernetes/), and [Credential Helper](https://github.com/docker/docker-credential-helpers/).

### Ubuntu
To get started with Docker Engine on Ubuntu.
https://docs.docker.com/engine/install/ubuntu/

- Post-installation
	- https://docs.docker.com/engine/install/linux-postinstall/
	- https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04-es
	- [Udacity tutorial](file:///media/frank/FrankFacundo/Courses/Udacity/Updates/JavaWeb/Java%20Web%20Developer%20Nanodegree%20%20v2.0.0/Part%2005-Module%2001-Lesson%2005_CICD/03.%20Docker.html)

