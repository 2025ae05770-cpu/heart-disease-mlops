import numpy as np
import pytest

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN
from src.features.feature_engineering import ClinicalFeatureEngineer
from src.features.prepare_data import build_preprocessor, split_features_target, split_train_test


def test_feature_engineer_adds_hr_pct_of_max(clean_df):
    X = clean_df[FEATURE_COLUMNS]
    out = ClinicalFeatureEngineer().fit_transform(X)
    assert "hr_pct_of_max" in out.columns
    expected = X["thalach"] / (220 - X["age"])
    np.testing.assert_allclose(out["hr_pct_of_max"], expected)
    assert "hr_pct_of_max" not in X.columns  # input not mutated


def test_split_is_stratified_and_sized(clean_df):
    X_train, X_test, y_train, y_test = split_train_test(clean_df, test_size=0.25)
    assert len(X_test) == pytest.approx(len(clean_df) * 0.25, abs=1)
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert TARGET_COLUMN not in X_train.columns


def test_preprocessor_outputs_no_missing_values(clean_df):
    X, _ = split_features_target(clean_df)
    assert X.isna().any().any()
    transformed = build_preprocessor().fit_transform(X)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_preprocessor_scales_numeric_and_encodes_categoricals(clean_df):
    X, _ = split_features_target(clean_df)
    preprocessor = build_preprocessor().fit(X)
    names = list(preprocessor.named_steps["columns"].get_feature_names_out())
    assert "age" in names and "hr_pct_of_max" in names
    assert "sex_1.0" in names and "sex_0.0" not in names  # drop="if_binary" -> single column
    assert {"cp_1.0", "cp_4.0", "thal_7.0"} <= set(names)
    assert not set(CATEGORICAL_FEATURES) & set(names)  # raw categoricals replaced by one-hot columns
    transformed = preprocessor.transform(X)
    age_col = transformed[:, names.index("age")]
    assert abs(age_col.mean()) < 1e-6 and abs(age_col.std() - 1) < 1e-6


def test_preprocessor_handles_unseen_category(clean_df):
    X, _ = split_features_target(clean_df)
    preprocessor = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["restecg"] = 9
    assert preprocessor.transform(unseen).shape[0] == 1
