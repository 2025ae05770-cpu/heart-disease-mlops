"""MLflow experiment tracking: one parent run per model family + nested runs per grid candidate."""

import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.config import EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI, PROJECT_ROOT
from src.models.train_models import CV_SCORING, cv_summary

# Bundled with the logged model so the custom ClinicalFeatureEngineer is importable on load.
SRC_DIR = PROJECT_ROOT / "src"

logger = logging.getLogger(__name__)


def artifact_location_is_foreign(location, os_name=None):
    """True if a local file: artifact location was written by a different OS (Windows <-> Linux/WSL)."""
    os_name = os_name or os.name
    if not location or not location.startswith("file:"):
        return False
    path = unquote(urlparse(location).path)
    has_drive = re.match(r"^/?[A-Za-z]:", path) is not None
    return has_drive if os_name != "nt" else not has_drive


def setup_mlflow(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=EXPERIMENT_NAME,
                 artifact_root=MLFLOW_ARTIFACT_ROOT):
    mlflow.set_tracking_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        artifact_location = None
        if tracking_uri.startswith("sqlite"):
            artifact_root = Path(artifact_root)
            artifact_root.mkdir(parents=True, exist_ok=True)
            artifact_location = artifact_root.as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    elif artifact_location_is_foreign(existing.artifact_location):
        raise RuntimeError(
            f"MLflow store '{tracking_uri}' was created on another OS (artifact location "
            f"'{existing.artifact_location}'). Use a separate store per OS, e.g.\n"
            "  export MLFLOW_TRACKING_URI=sqlite:///$PWD/mlflow-linux.db MLFLOW_ARTIFACT_ROOT=$PWD/mlruns-linux\n"
            "(scripts/run_pipeline.sh does this automatically when run from a Windows drive in WSL)."
        )
    experiment = mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking at %s, experiment '%s'", tracking_uri, experiment_name)
    return experiment


def build_signature(X_sample, model):
    X_float = X_sample.astype("float64")
    return infer_signature(X_float, model.predict_proba(X_float)[:, 1]), X_float.head(3)


def _clean_params(params):
    return {key.replace("model__", ""): ("None" if value is None else value) for key, value in params.items()}


def log_model_run(name, search, test_metrics, plot_paths, X_train, y_train, extra_params=None):
    """Log one tuned model family. Returns (run_id, model_uri)."""
    model = search.best_estimator_
    signature, input_example = build_signature(X_train, model)
    cv_results = pd.DataFrame(search.cv_results_)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "model_family": name,
            "estimator": type(model.named_steps["model"]).__name__,
            "stage": "candidate",
        })
        mlflow.log_params(_clean_params(search.best_params_))
        mlflow.log_params({
            "n_train_rows": len(X_train),
            "n_features_raw": X_train.shape[1],
            "cv_folds": search.n_splits_,
            "n_grid_candidates": len(cv_results),
            "refit_metric": search.refit,
            **(extra_params or {}),
        })

        try:
            train_frame = X_train.astype("float64").assign(target=y_train.values)
            mlflow.log_input(mlflow.data.from_pandas(train_frame, name="heart-cleveland-train",
                                                     targets="target"), context="training")
        except Exception as exc:  # dataset logging is best-effort
            logger.warning("Could not log dataset input: %s", exc)

        mlflow.log_metrics(cv_summary(search))
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        for path in plot_paths:
            if path:
                mlflow.log_artifact(str(path), artifact_path="plots")

        with tempfile.TemporaryDirectory() as tmp:
            cv_path = Path(tmp) / f"{name}_cv_results.csv"
            cv_results.drop(columns=["params"]).to_csv(cv_path, index=False)
            mlflow.log_artifact(str(cv_path), artifact_path="cv_results")

        model_info = mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            code_paths=[str(SRC_DIR)],
        )

        for i, params in enumerate(cv_results["params"]):
            with mlflow.start_run(run_name=f"{name}-cand-{i:02d}", nested=True):
                mlflow.set_tag("stage", "grid_candidate")
                mlflow.log_params(_clean_params(params))
                mlflow.log_metrics({
                    f"cv_{metric}_mean": float(cv_results.loc[i, f"mean_test_{metric}"])  # type: ignore
                    for metric in CV_SCORING
                })
                mlflow.log_metric("rank_by_refit_metric", int(cv_results.loc[i, f"rank_test_{search.refit}"]))

    logger.info("Logged %s to MLflow run %s", name, run.info.run_id)
    return run.info.run_id, model_info.model_uri


def log_comparison_run(comparison, champion_name, artifact_paths):
    with mlflow.start_run(run_name="model-comparison") as run:
        mlflow.set_tags({"stage": "comparison", "champion": champion_name})
        for name, row in comparison.iterrows():
            mlflow.log_metric(f"{name}_cv_roc_auc", float(row["cv_roc_auc_mean"]))
            mlflow.log_metric(f"{name}_test_roc_auc", float(row["test_roc_auc"]))
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="comparison")
    return run.info.run_id
