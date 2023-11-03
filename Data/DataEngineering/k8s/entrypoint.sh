#!/bin/bash
# This entrypoint use minikube and hyperviseur kvm2
set -e

image=$2
echo $image

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
    $kubectl apply -f $image.yaml
    
    # grab the name of your active pod
    PODNAME=$(kubectl get pods --output=template \
        --template="{{with index .items 0}}{{.metadata.name}}{{end}}")
    echo $PODNAME
    
    # # # open a port-forward session to the pod
    # kubectl port-forward $PODNAME 4242:4242
    minikube service 192.168.39.78:30148
    # kubectl get pods
}

function stop_deploy(){
    $kubectl delete -f $image.yaml
    # minikube image rm $image
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
