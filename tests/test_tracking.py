import mlflow
import pytest
from mlflow import MlflowClient

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.model_pipelines import get_model_specs
from src.models.train_models import train_models
from src.tracking.mlflow_tracking import artifact_location_is_foreign, log_model_run, setup_mlflow
from src.tracking.register_model import register_champion


@pytest.mark.parametrize("location,os_name,expected", [
    ("file:///D:/proj/mlruns", "posix", True),        # Windows store opened from WSL
    ("file:///mnt/d/proj/mlruns", "posix", False),
    ("file:///D:/proj/mlruns", "nt", False),
    ("file:///home/u/proj/mlruns", "nt", True),       # Linux store opened from Windows
    ("s3://bucket/mlruns", "posix", False),
    (None, "posix", False),
])
def test_artifact_location_is_foreign(location, os_name, expected):
    assert artifact_location_is_foreign(location, os_name) is expected


@pytest.fixture
def isolated_mlflow(tmp_path):
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    setup_mlflow(tracking_uri=uri, experiment_name="pytest-exp", artifact_root=tmp_path / "artifacts")
    yield uri
    mlflow.set_tracking_uri(previous)


def test_log_model_run_records_params_metrics_artifacts_and_children(isolated_mlflow, clean_df, tmp_path):
    X, y = clean_df[FEATURE_COLUMNS], clean_df[TARGET_COLUMN]
    specs = {"logistic_regression": get_model_specs(quick=True)["logistic_regression"]}
    search = train_models(specs, X, y, cv_folds=3, n_jobs=1)["logistic_regression"]
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"\x89PNG fake")

    run_id, model_uri = log_model_run("logistic_regression", search, {"roc_auc": 0.9}, [plot], X, y)

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.data.params["C"] == "0.01"
    assert "cv_roc_auc_mean" in run.data.metrics and run.data.metrics["test_roc_auc"] == 0.9
    artifact_dirs = {a.path for a in client.list_artifacts(run_id)}
    assert {"plots", "cv_results"} <= artifact_dirs

    children = client.search_runs([run.info.experiment_id],
                                  filter_string=f"tags.mlflow.parentRunId = '{run_id}'")
    assert len(children) == 1

    version = register_champion(model_uri, name="pytest-model")
    assert client.get_model_version_by_alias("pytest-model", "champion").version == version

    loaded = mlflow.sklearn.load_model(model_uri)
    assert loaded.predict_proba(X.head(2)).shape == (2, 2)
