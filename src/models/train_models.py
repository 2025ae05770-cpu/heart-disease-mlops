"""Hyper-parameter tuning with stratified k-fold cross-validation."""

import logging

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import CV_FOLDS, RANDOM_STATE, SELECTION_METRIC

logger = logging.getLogger(__name__)

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def tune_model(name, pipeline, param_grid, X_train, y_train, cv_folds=CV_FOLDS, n_jobs=-1):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=CV_SCORING,
        refit=SELECTION_METRIC,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    logger.info("Tuning %s over %d candidates x %d folds", name, _grid_size(param_grid), cv_folds)
    search.fit(X_train, y_train)
    logger.info("%s best CV %s = %.4f with %s", name, SELECTION_METRIC,
                search.best_score_, search.best_params_)
    return search


def cv_summary(search):
    """Mean/std of every CV metric for the best candidate."""
    index = search.best_index_
    results = search.cv_results_
    summary = {}
    for metric in CV_SCORING:
        summary[f"cv_{metric}_mean"] = float(results[f"mean_test_{metric}"][index])
        summary[f"cv_{metric}_std"] = float(results[f"std_test_{metric}"][index])
    return summary


def train_models(specs, X_train, y_train, cv_folds=CV_FOLDS, n_jobs=-1):
    return {
        name: tune_model(name, pipeline, grid, X_train, y_train, cv_folds=cv_folds, n_jobs=n_jobs)
        for name, (pipeline, grid) in specs.items()
    }


def _grid_size(param_grid):
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size
