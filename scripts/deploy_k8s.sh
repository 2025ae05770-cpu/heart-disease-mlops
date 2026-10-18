#!/usr/bin/env bash
# Build the API image and deploy API + Prometheus + Grafana to a local Kubernetes cluster.
#   ./scripts/deploy_k8s.sh minikube | docker-desktop
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-minikube}"
IMAGE="${IMAGE:-heart-disease-api:1.0.0}"

case "$TARGET" in
  minikube|docker-desktop) ;;
  *) echo "Usage: $0 [minikube|docker-desktop]" >&2; exit 1 ;;
esac

command -v docker >/dev/null || { echo "Docker is required and must be on PATH." >&2; exit 1; }
[[ -f models/heart_disease_model.joblib ]] || { echo "Train first: python -m src.models.train"; exit 1; }

if [[ "$TARGET" == "minikube" ]]; then
  command -v minikube >/dev/null || { echo "Minikube is required and must be on PATH." >&2; exit 1; }
  echo ">> Ensuring the minikube cluster is running (Docker driver, 2 CPUs, 4096 MB)"
  minikube start -p minikube --driver=docker --cpus=2 --memory=4096
  minikube -p minikube addons enable ingress
  minikube -p minikube addons enable metrics-server
  if command -v kubectl >/dev/null; then
    kube() { kubectl --context=minikube "$@"; }
  else
    echo ">> kubectl not found; using Minikube's bundled kubectl"
    kube() { minikube -p minikube kubectl -- "$@"; }
  fi
  docker_env=$(minikube -p minikube docker-env --shell bash)
  eval "$docker_env"
else
  command -v kubectl >/dev/null || { echo "kubectl is required and must be on PATH." >&2; exit 1; }
  kube() { kubectl --context=docker-desktop "$@"; }
fi

echo ">> Building $IMAGE"
docker build -t "$IMAGE" .

echo ">> Applying manifests"
kube apply -k k8s/
kube apply -f k8s/monitoring/prometheus.yaml
kube create configmap grafana-dashboards -n heart-disease \
  --from-file=monitoring/grafana/dashboards/ --dry-run=client -o yaml | kube apply -f -
kube apply -f k8s/monitoring/grafana.yaml

for deployment in heart-disease-api prometheus grafana; do
  kube rollout status "deployment/$deployment" -n heart-disease --timeout=180s
done
kube get pods,svc,ingress,hpa -n heart-disease -o wide

if [[ "$TARGET" == "minikube" ]]; then
  echo ">> Deployment ready. Run each port-forward in a separate terminal:"
  echo "minikube -p minikube kubectl -- -n heart-disease port-forward service/heart-disease-api 8000:80"
  echo "minikube -p minikube kubectl -- -n heart-disease port-forward service/prometheus 9090:9090"
  echo "minikube -p minikube kubectl -- -n heart-disease port-forward service/grafana 3000:3000"
  echo "For a remote machine, forward these localhost ports to your laptop over SSH."
else
  echo "API: http://localhost/docs  Prometheus: http://localhost:30090  Grafana: http://localhost:30030"
fi
