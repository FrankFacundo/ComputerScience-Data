#!/bin/bash

# TO FIX docker compose command
if [[ "$1" == 'run' ]]; then

    git clone https://github.com/apache/superset
    pushd superset
    TAG=3.1.1
    docker compose --network=host -f docker-compose-image-tag.yml up
    popd

fi

if [[ "$1" == 'db' ]]; then

    pushd db
    docker build -t ai-postgres .
    # docker run --network=host --name ai-postgres -p 5433:5432 ai-postgres
    docker run --name ai-postgres -p 5433:5432 ai-postgres
    popd

fi

if [[ "$1" == 'stop_db' ]]; then

    pushd db
    docker stop ai-postgres
    docker rm ai-postgres
    popd

fi