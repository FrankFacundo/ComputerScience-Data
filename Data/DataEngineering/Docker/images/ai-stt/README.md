Check: https://github.com/ahmetoner/whisper-asr-webservice

poetry run gunicorn --bind 0.0.0.0:4242 --workers 1 --timeout 0 app.main:app -k uvicorn.workers.UvicornWorker

docker build -t ai-stt .

docker run -p 4242:4242 ai-stt
docker run -p 4242:4242 -e ASR_MODEL=base -v //c/tmp/whisper:/root/.cache/whisper ai-stt
