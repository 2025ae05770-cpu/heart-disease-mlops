import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURE_COLUMNS, RAW_COLUMNS, TARGET_COLUMN
from src.models.model_pipelines import build_pipeline
from src.models.persistence import save_model

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


def make_synthetic(n=200, seed=7):
    """Synthetic patients in valid clinical ranges with a learnable signal."""
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    data = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": np.where(risk > 0.5, 4, rng.integers(1, 4, n)),
        "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": (190 - 70 * risk + rng.normal(0, 8, n)).round(),
        "exang": (risk + rng.normal(0, 0.2, n) > 0.6).astype(int),
        "oldpeak": (risk * 4).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })
    data[TARGET_COLUMN] = (risk > 0.5).astype(int)
    data.loc[[0, 5], "ca"] = np.nan
    data.loc[[3], "thal"] = np.nan
    return data


@pytest.fixture
def clean_df():
    return make_synthetic()


@pytest.fixture
def raw_df(clean_df):
    raw = clean_df.copy()
    raw["num"] = np.where(raw[TARGET_COLUMN] == 1, np.random.default_rng(1).integers(1, 5, len(raw)), 0)
    raw = raw.drop(columns=[TARGET_COLUMN]).astype(object)
    raw.loc[0, "ca"] = "?"
    raw.loc[3, "thal"] = "?"
    return raw[RAW_COLUMNS]


@pytest.fixture
def trained_pipeline(clean_df):
    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(clean_df[FEATURE_COLUMNS], clean_df[TARGET_COLUMN])
    return model


@pytest.fixture
def model_files(tmp_path, trained_pipeline):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    save_model(trained_pipeline, {"model_name": "test-model", "model_family": "logistic_regression",
                                  "model_version": "test-1"}, model_path, metadata_path)
    assert json.loads(metadata_path.read_text())["model_version"] == "test-1"
    return model_path, metadata_path


@pytest.fixture
def sample_patient():
    return dict(SAMPLE_PATIENT)
