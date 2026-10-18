# Screenshots

Evidence for the report (`reports/Heart_Disease_MLOPS_Report.docx`, figures referenced from `reports/REPORT.md`).

Generated figures live alongside the operational screenshots: `architecture.png`,
`EDA/` (six exploratory plots), and `models/` (model comparisons and diagnostics).
Regenerate them with `python scripts/make_architecture_diagram.py`,
`python -m src.eda.run_eda`, and `python -m src.models.train`, respectively.
Summary tables and the submitted Word report remain in `reports/`.

| Area | Files |
|---|---|
| MLflow experiment tracking | `mlflow_commandline_run_deploy.png`, `mlflow_experiment_runs.png`, `mlflow_metric_charts_1.png`, `mlflow_comparison_cv_metrics.png`, `mlflow_comparison_roc.png`, `mlflow_model_comparison_1.png`, `mlflow_model_comparison_2.png`, `mlflow_model_champion.png`, `mlflow_model_registry.png`, `modeldb.png` |
| Lint and unit tests | `lint_and_unit_tests.png` |
| CI/CD (GitHub Actions) | `Github_Actions.png` |
| API (FastAPI / Swagger) | `OpenAPI_Swagger.png`, `OpenAPI_Swagger_POST_API.png`, `api_smoke_test.png` |
| Docker | `Docker_build.png`, `Docker Run.png`, `Docker_run_with_curl_api_check.png`, `Docker_build_compose.png`, `Docker_containers_running.png` |
| Kubernetes (Minikube) | `kucectl_deployment.png`, `kubectl_container_pods.png`, `kube_container_swagger_api.png`, `kubectl_swagger_post_api.png` |
| Monitoring | `Prometheous_metrics.png`, `Prometheous_metrics_2.png`, `Grafana_login.png`, `Grafana_metrics.png`, `Grafana_kubectl_container_metrics.png` |
