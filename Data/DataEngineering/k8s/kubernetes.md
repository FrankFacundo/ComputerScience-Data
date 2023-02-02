# Kubernetes

## Concepts

- https://medium.com/google-cloud/kubernetes-101-pods-nodes-containers-and-clusters-c1509e409e16
- https://medium.com/google-cloud/kubernetes-110-your-first-deployment-bf123c1d3f8
- https://medium.com/google-cloud/kubernetes-120-networking-basics-3b903f13093a
- https://medium.com/faun/understanding-nodes-pods-containers-and-clusters-778dbd56ade8#:~:text=Each%20pod%20has%20a%20unique,Pod%2C%20they%20expose%20a%20port.&text=A%20Node%20is%20a%20worker,contains%20services%20to%20run%20pods.

## Kubernetes in local

Options: https://www.padok.fr/en/blog/minikube-kubeadm-kind-k3s
Minikube install: https://kubernetes.io/fr/docs/tasks/tools/install-minikube/

## First deploy

https://medium.com/google-cloud/kubernetes-110-your-first-deployment-bf123c1d3f8

## If working with Google Cloud, to connect a database with cluster

https://cloud.google.com/sql/docs/postgres/connect-kubernetes-engine?hl=fr#proxy-with-workload-identity

## List images

```shell
minikube image ls --format table
```

## Pods

```shell
kubectl get pods
```

## Describe

```shell
kubectl describe pod simple-deployment-85db6cd64c-rjhrj
```

## Logs
```shell
kubectl logs pod simple-deployment-85db6cd64c-rjhrj
```

## Yamel Structure

A YAML file in Kubernetes typically consists of the following structure:

```yaml
apiVersion: <api version used by the object>
kind: <type of Kubernetes object, e.g. Deployment, Service, Pod>
metadata:
  name: <name of the object>
  labels: <key-value pairs used for labeling and organization>
spec:
  <object-specific configuration details>
```

For example, here's a YAML file for a Deployment in Kubernetes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.17.1
        ports:
        - containerPort: 80apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.17.1
        ports:
        - containerPort: 80
```

This YAML file creates a Deployment with three replicas of the nginx container, using the nginx:1.17.1 image and exposing port 80.