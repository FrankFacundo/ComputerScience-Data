FROM python:3.9.18-bullseye

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . /app
WORKDIR /app

ENTRYPOINT ["entrypoint.sh"]
