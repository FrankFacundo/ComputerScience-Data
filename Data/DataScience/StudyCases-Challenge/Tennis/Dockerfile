FROM ubuntu:20.04

ENV LANG C.UTF-8

WORKDIR /app
ADD app /app/app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt install -y build-essential software-properties-common && \
  apt-get update -y && apt install -y \
  python3-dev \
  python-numpy \
  python3-pip \
  wget

RUN python3 -m pip install \
  aiohttp \
  fastapi \
  matplotlib \
  pillow \
  pandas \ 
  scipy \
  uvicorn \
  scikit-learn \
  lightgbm

RUN cd /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4004"]

EXPOSE 4004