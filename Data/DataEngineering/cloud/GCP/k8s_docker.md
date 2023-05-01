# Kubernetes and Docker commands

## List Docker images

```bash
gcloud container images list --project {PROJECT_ID}
```

## List tags of a Docker image

```bash
gcloud container images list-tags gcr.io/trainingdev1/ai-stt --limit=999999 --sort-by=TIMESTAMP
```

## Get gcloud auth

```bash
gcloud auth activate-service-account --key-file ai.json
```

## Configure Docker

```bash
gcloud auth configure-docker --quiet
```

## Obtain cluster credentials

```bash
gcloud container clusters get-credentials autopilot-cluster-1 --region europe-west9 --project ${DEVELOPMENT_PROJECT}
```

## Check clusters

```bash
kubectl cluster-info
```

## Apply Kubernetes file to cluster

```bash
kubectl apply -f ai-stt-deployment.yaml --cluster gke_trainingdev1_europe-west9_autopilot-cluster-1
```
