#!/bin/bash
# This entrypoint use minikube and hyperviseur kvm2
set -e

kubectl="minikube kubectl --"

function start_minikube_cluster() {
    minikube start --driver=kvm2
    minikube status
    # alias kubectl="minikube kubectl --"
    
    # This command allows to use docker local images
    eval $(minikube docker-env)
}

function stop_minikube_cluster(){
    minikube stop
}

function deploy(){
    $kubectl apply -f simple-app.yaml
    # kubectl get pods
}

function stop_deploy(){
    $kubectl delete -f simple-app.yaml
}

if [[ "$1" == 'start' ]]; then
    start_minikube_cluster

elif [[ "$1" == 'deploy' ]]; then
    deploy

elif [[ "$1" == 'stop_deploy' ]]; then
    stop_deploy

elif [[ "$1" == 'stop' ]]; then
    stop_minikube_cluster
fi
