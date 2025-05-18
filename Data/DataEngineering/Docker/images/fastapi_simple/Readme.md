# Commands

docker build -t simple_fastapi .

docker run -p 8000:8000 simple_fastapi

curl -X GET "http://localhost:8000/"

docker tag simple_fastapi frankfacundo/simple_fastapi:latest

docker push frankfacundo/simple_fastapi:latest
