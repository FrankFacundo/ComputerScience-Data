sudo docker build -t traductor .
sudo docker run -it --mount type=bind,source="$(pwd)",target=/app -p 1611:1611 traductor uvicorn app.main:app --host 0.0.0.0 --port 1611 --reload
