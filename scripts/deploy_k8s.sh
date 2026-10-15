#!/usr/bin/env bash
# Build the API image and deploy API + Prometheus + Grafana to a local Kubernetes cluster.
#   ./scripts/deploy_k8s.sh minikube | docker-desktop
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-minikube}"
IMAGE="${IMAGE:-heart-disease-api:1.0.0}"

[[ -f models/heart_disease_model.joblib ]] || { echo "Train first: python -m src.models.train"; exit 1; }

if [[ "$TARGET" == "minikube" ]]; then
  minikube status >/dev/null
  minikube addons enable ingress
  minikube addons enable metrics-server
  eval "$(minikube -p minikube docker-env)"
else
  kubectl config use-context docker-desktop
fi

echo ">> Building $IMAGE"
docker build -t "$IMAGE" .

echo ">> Applying manifests"
kubectl apply -k k8s/
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl create configmap grafana-dashboards -n heart-disease \
  --from-file=monitoring/grafana/dashboards/ --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring/grafana.yaml

kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=180s
kubectl get pods,svc,ingress,hpa -n heart-disease -o wide

if [[ "$TARGET" == "minikube" ]]; then
  echo "API URL: $(minikube service heart-disease-api -n heart-disease --url)"
  echo "Ingress: echo \"$(minikube ip) heart-disease.local\" | sudo tee -a /etc/hosts"
else
  echo "API: http://localhost/docs  Prometheus: http://localhost:30090  Grafana: http://localhost:30030"
fi
