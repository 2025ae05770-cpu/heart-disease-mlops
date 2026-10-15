"""Candidate models: each is a full Pipeline (preprocessing + classifier) with a tuning grid."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.features.prepare_data import build_preprocessor


def build_pipeline(estimator):
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def get_model_specs(quick=False):
    """Return {name: (pipeline, param_grid)}. `quick=True` shrinks grids for smoke tests."""
    specs = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 0.5, 1.0, 5.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 4],
                "model__max_features": ["sqrt"],
            },
        ),
        "xgboost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=2, verbosity=0),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
        ),
    }

    result = {}
    for name, (estimator, grid) in specs.items():
        if quick:
            grid = {key: values[:1] for key, values in grid.items()}
        result[name] = (build_pipeline(estimator), grid)
    return result
