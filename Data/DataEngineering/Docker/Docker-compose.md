A tool used to run multiple Docker containers at once; often used to specify dependent relationships between containers.


In `docker-compose.yml` we use this commands : 
Ref. https://docs.docker.com/compose/compose-file/compose-file-v3

- Image: Specify the image to start the container from. Can either be a repository/tag or a partial image ID.
	```
	image: redis
	```
	```
	image: ubuntu:18.04
	```
	```
	image: tutum/influxdb
	```
	```
	image: example-registry.com:4000/postgresql
	```
	```
	image: a4bc65fd
	```