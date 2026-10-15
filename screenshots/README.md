# Screenshots

Images in this folder are embedded in `reports/Heart_Disease_MLOps_Report.docx`. After adding the missing ones, rebuild the report with `node scripts/build_report_docx.js`.

| File | Status | How to capture |
|---|---|---|
| `01_mlflow_experiment_runs.png` | captured | MLflow UI → experiment `heart-disease-risk` |
| `02_mlflow_metric_charts.png` | captured | Runs → chart view |
| `03_mlflow_champion_run.png` | captured | Champion run → Overview (nested child runs) |
| `04_mlflow_artifacts.png` | captured | Champion run → Artifacts → plots |
| `05_mlflow_model_registry.png` | captured | Models → `HeartDiseaseRiskChampion` |
| `06_api_swagger_overview.png` | captured | `http://127.0.0.1:8000/docs` |
| `07_api_swagger_predict_response.png` | captured | `/docs` → POST `/predict` → Try it out → Execute |
| `08_api_prometheus_metrics.png` | captured | `GET /metrics` after `scripts/generate_traffic.py` |
| `09_lint_and_unit_tests.png` | captured | `flake8 ...` and `pytest -v --cov=src` |
| `10_api_smoke_test.png` | captured | `python scripts/smoke_test.py` |
| `11_github_actions_pipeline.png` | **to add** | GitHub → Actions → latest run (all 4 jobs green + artifacts list) |
| `12_docker_build_run.png` | **to add** | Terminal: `docker build -t heart-disease-api:1.0.0 .` and `docker run -p 8000:8000 ...` with a `curl` to `/predict` |
| `13_k8s_resources.png` | **to add** | `kubectl get pods,svc,ingress,hpa -n heart-disease` |
| `14_k8s_endpoint_test.png` | **to add** | `python scripts/smoke_test.py --url http://localhost` (Docker Desktop LoadBalancer) or via the Ingress host |
| `15_prometheus_targets.png` | optional | Prometheus → Status → Targets (pods UP) |
| `16_grafana_dashboard.png` | **to add** | Grafana → MLOps → Heart Disease API (after `generate_traffic.py`) |
