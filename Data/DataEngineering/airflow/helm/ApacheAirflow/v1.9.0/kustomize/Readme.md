# Airflow Kustomize

The yaml files on this kustomize repertory were been generated with `helm template masai helm --output-dir ./kustomize` in path `Apache Airflow/v1.9.0`

## Run Airflow

```bash
minikube start --vm-driver kvm2 --disk-size 20GB
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/v2.7.4/manifests/install.yaml
kubectl get all -n argocd
kubectl port-forward svc/argocd-server -n argocd 8090:443
# Open ArgoCD with: localhost:8090/
# User: admin
# To get password: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo
kubectl port-forward svc/masai-webserver -n airflow 8077:8080
# Open Airflow with: localhost:8077/
# User: admin
# Password: admin
```
