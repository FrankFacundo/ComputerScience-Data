#!/bin/bash
# This entrypoint use minikube and hyperviseur kvm2
set -e

function start_minikube_cluster() {
    minikube start --driver=kvm2
    minikube status
    alias kubectl="minikube kubectl --"

}

function stop_minikube_cluster(){
    minikube stop
}

if [[ "$1" == 'start' ]]; then
    start_minikube_cluster


elif [[ "$1" == 'stop' ]]; then
    stop_minikube_cluster
fi

# minikube image build "../Docker/images/simple" --tag 'simple'
