"""Hold-out evaluation metrics and diagnostic plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return compute_metrics(y_test, y_pred, y_proba), y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["No disease", "Disease"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    return _save(fig, path)


def plot_roc_pr(y_true, y_proba, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0], name=title)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set_title("ROC curve (hold-out)")
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[1], name=title)
    axes[1].set_title("Precision-Recall curve (hold-out)")
    return _save(fig, path)


def feature_importance(model):
    """Return a Series of importances (|coef| for linear models) indexed by feature name."""
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocess.named_steps["columns"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


def plot_feature_importance(model, title, path, top_n=15):
    importance = feature_importance(model)
    if importance is None:
        return None
    top = importance.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#457b9d")
    ax.set_title(title)
    ax.set_xlabel("Importance" if hasattr(model.named_steps["model"], "feature_importances_") else "|coefficient|")
    return _save(fig, path)


def plot_model_comparison(results, y_true, path):
    """Overlay ROC curves of all candidates on the hold-out set."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, result in results.items():
        RocCurveDisplay.from_predictions(y_true, result["y_proba"], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("Model comparison - ROC on hold-out set")
    return _save(fig, path)


def plot_cv_metric_bars(comparison, path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    means = comparison[[f"cv_{m}_mean" for m in metrics]].copy()
    stds = comparison[[f"cv_{m}_std" for m in metrics]].copy()
    means.columns = stds.columns = metrics
    fig, ax = plt.subplots(figsize=(9, 4.5))
    means.T.plot(kind="bar", yerr=stds.T, capsize=3, ax=ax, rot=0)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"{len(comparison)} models - {len(metrics)} CV metrics (mean +/- std)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
