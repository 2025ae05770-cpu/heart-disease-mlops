<#
.SYNOPSIS
  Build the API image and deploy API + Prometheus + Grafana to a local Kubernetes cluster.
.EXAMPLE
  ./scripts/deploy_k8s.ps1 -Target docker-desktop
  ./scripts/deploy_k8s.ps1 -Target minikube
#>
param(
    [ValidateSet("docker-desktop", "minikube")]
    [string]$Target = "docker-desktop",
    [string]$Image = "heart-disease-api:1.0.0"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path "models/heart_disease_model.joblib")) {
    throw "models/heart_disease_model.joblib not found - run 'python -m src.models.train' first."
}

if ($Target -eq "minikube") {
    minikube status | Out-Null
    minikube addons enable ingress
    minikube addons enable metrics-server
    # Build directly inside minikube's Docker daemon so no registry is needed.
    & minikube -p minikube docker-env --shell powershell | Invoke-Expression
} else {
    kubectl config use-context docker-desktop
}

Write-Host ">> Building $Image" -ForegroundColor Cyan
docker build -t $Image .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host ">> Applying manifests" -ForegroundColor Cyan
kubectl apply -k k8s/
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl create configmap grafana-dashboards -n heart-disease `
    --from-file=monitoring/grafana/dashboards/ --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring/grafana.yaml

Write-Host ">> Waiting for rollout" -ForegroundColor Cyan
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=180s
kubectl get pods,svc,ingress,hpa -n heart-disease -o wide

if ($Target -eq "minikube") {
    Write-Host "`nRun 'minikube tunnel' in another terminal for the LoadBalancer IP, or:"
    Write-Host "  minikube service heart-disease-api -n heart-disease --url"
    Write-Host "Ingress: add '$(minikube ip)  heart-disease.local' to C:\Windows\System32\drivers\etc\hosts"
} else {
    Write-Host "`nAPI:        http://localhost/docs   (LoadBalancer on port 80)"
    Write-Host "Smoke test: python scripts/smoke_test.py --url http://localhost"
    Write-Host "Prometheus: http://localhost:30090   Grafana: http://localhost:30030 (admin/admin)"
}
