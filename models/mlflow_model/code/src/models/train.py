"""End-to-end training pipeline.

clean data -> split -> tune 3 model families with CV -> hold-out evaluation -> plots
-> MLflow tracking -> champion selection -> registry -> save deployable artifacts.

Usage:
    python -m src.models.train                 # full run with MLflow tracking + registry
    python -m src.models.train --quick         # tiny grids (CI smoke run)
    python -m src.models.train --no-mlflow     # skip tracking
"""

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from src.config import (
    CV_FOLDS,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METADATA_PATH,
    MLFLOW_MODEL_DIR,
    MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    SELECTION_METRIC,
    TEST_SIZE,
)
from src.data.clean_data import load_clean_data
from src.features.prepare_data import split_train_test
from src.models.evaluate_models import (
    evaluate_model,
    plot_confusion_matrix,
    plot_cv_metric_bars,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_pr,
)
from src.models.model_pipelines import get_model_specs
from src.models.persistence import save_mlflow_model, save_model
from src.models.train_models import cv_summary, train_models
from src.tracking.select_champion import select_champion

logger = logging.getLogger(__name__)


def evaluate_all(searches, X_test, y_test, figures_dir):
    results = {}
    for name, search in searches.items():
        model = search.best_estimator_
        test_metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        plots = [
            plot_confusion_matrix(y_test, y_pred, f"{name} - confusion matrix",
                                  figures_dir / f"{name}_confusion_matrix.png"),
            plot_roc_pr(y_test, y_proba, name, figures_dir / f"{name}_roc_pr.png"),
            plot_feature_importance(model, f"{name} - top features",
                                    figures_dir / f"{name}_feature_importance.png"),
        ]
        results[name] = {
            "search": search,
            "best_params": search.best_params_,
            "cv": cv_summary(search),
            "test": test_metrics,
            "y_proba": y_proba,
            "plots": [p for p in plots if p],
        }
        logger.info("%s hold-out metrics: %s", name, json.dumps(test_metrics))
    return results


def build_comparison(results):
    rows = {}
    for name, result in results.items():
        row = dict(result["cv"])
        row.update({f"test_{k}": v for k, v in result["test"].items()})
        row["best_params"] = json.dumps({k.replace("model__", ""): v for k, v in result["best_params"].items()})
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("cv_roc_auc_mean", ascending=False)


def run_training(quick=False, track=True, register=True, data=None, model_path=MODEL_PATH,
                 metadata_path=METADATA_PATH, mlflow_model_dir=MLFLOW_MODEL_DIR,
                 figures_dir=FIGURES_DIR / "models", reports_dir=REPORTS_DIR, n_jobs=-1):
    figures_dir, reports_dir = Path(figures_dir), Path(reports_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_clean_data() if data is None else data
    X_train, X_test, y_train, y_test = split_train_test(data)
    logger.info("Train rows=%d, test rows=%d, positive rate train=%.3f",
                len(X_train), len(X_test), y_train.mean())

    searches = train_models(get_model_specs(quick=quick), X_train, y_train, n_jobs=n_jobs)
    results = evaluate_all(searches, X_test, y_test, figures_dir)

    comparison = build_comparison(results)
    comparison_csv = reports_dir / "model_comparison.csv"
    comparison.to_csv(comparison_csv)
    comparison_plots = [
        plot_model_comparison(results, y_test, figures_dir / "comparison_roc.png"),
        plot_cv_metric_bars(comparison, figures_dir / "comparison_cv_metrics.png"),
    ]

    flat = {name: {**r["cv"], **{f"test_{k}": v for k, v in r["test"].items()}} for name, r in results.items()}
    champion = select_champion(flat)
    champion_name = champion["model_name"]
    champion_model = results[champion_name]["search"].best_estimator_
    logger.info("Champion: %s (CV ROC-AUC %.4f)", champion_name, flat[champion_name]["cv_roc_auc_mean"])

    run_ids, model_version = {}, None
    if track:
        from src.tracking.mlflow_tracking import log_comparison_run, log_model_run, setup_mlflow

        setup_mlflow()
        model_uris = {}
        for name, result in results.items():
            run_ids[name], model_uris[name] = log_model_run(
                name, result["search"], result["test"], result["plots"], X_train, y_train,
                extra_params={"random_state": RANDOM_STATE, "test_size": TEST_SIZE, "quick_mode": quick},
            )
        log_comparison_run(comparison, champion_name, [comparison_csv, *comparison_plots])
        if register:
            from src.tracking.register_model import register_champion

            model_version = register_champion(model_uris[champion_name], tags={
                "model_family": champion_name,
                "cv_roc_auc": round(flat[champion_name]["cv_roc_auc_mean"], 4),
            })

    metadata = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_family": champion_name,
        "model_version": str(model_version) if model_version else datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": f"cv_{SELECTION_METRIC}_mean",
        "decision_threshold": 0.5,
        "best_params": {k.replace("model__", ""): v for k, v in results[champion_name]["best_params"].items()},
        "cv_metrics": results[champion_name]["cv"],
        "test_metrics": results[champion_name]["test"],
        "feature_columns": FEATURE_COLUMNS,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "cv_folds": CV_FOLDS,
        "mlflow_run_id": run_ids.get(champion_name),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    save_model(champion_model, metadata, model_path, metadata_path)

    from src.tracking.mlflow_tracking import build_signature

    signature, input_example = build_signature(X_train, champion_model)
    save_mlflow_model(champion_model, signature, input_example, mlflow_model_dir)

    (reports_dir / "model_comparison.md").write_text(_comparison_markdown(comparison, champion_name))
    return {"champion": champion_name, "comparison": comparison, "metadata": metadata, "run_ids": run_ids}


def _comparison_markdown(comparison, champion_name):
    cols = ["cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean", "cv_f1_mean", "cv_roc_auc_mean",
            "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    header = "| model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for name, row in comparison.iterrows():
        label = f"**{name}** (champion)" if name == champion_name else name
        body += f"| {label} | " + " | ".join(f"{row[c]:.3f}" for c in cols) + " |\n"
    return header + body


def main():
    parser = argparse.ArgumentParser(description="Train heart-disease models")
    parser.add_argument("--quick", action="store_true", help="use 1-point grids (fast CI smoke run)")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    parser.add_argument("--no-register", action="store_true", help="track but do not register the champion")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "training.log")],
    )
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    # cloudpickle is chosen deliberately (custom transformer); silence the repeated advisory.
    logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

    outcome = run_training(quick=args.quick, track=not args.no_mlflow, register=not args.no_register)
    print("\n=== Model comparison ===")
    print(outcome["comparison"].drop(columns=["best_params"]).round(4).T.to_string())
    print(f"\nChampion: {outcome['champion']}  ->  {MODEL_PATH}")


if __name__ == "__main__":
    main()
