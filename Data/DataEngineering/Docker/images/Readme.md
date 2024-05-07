# Docker

## Example command

```bash
./entrypoint.sh simple
```

# Ollama

docker run -d -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run llama3
docker exec -it ollama ollama run starcoder2:3b
docker exec -it ollama ollama run starcoder2:7b
docker exec -it ollama ollama run deepseek-coder:6.7b-base
// models are in "/root/.ollama" or "~/.ollama"

// docker ps -> https://docs.docker.com/reference/cli/docker/container/commit/
docker commit 7e8253978351 ollama/ollama:models
docker tag ollama/ollama:models frankfacundo/ollama:models
docker push frankfacundo/ollama:models

// check docker ollama with "curl ollama:11434" or "curl <ip>:11434"
