"""Feature/target split, train/test split and the reusable preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.features.feature_engineering import ClinicalFeatureEngineer


def split_features_target(data):
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].astype(int).copy()


def split_train_test(data, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X, y = split_features_target(data)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor():
    """Raw patient features -> engineered -> imputed -> scaled / one-hot encoded matrix."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("engineer", ClinicalFeatureEngineer()),
            ("columns", column_transformer),
        ]
    )
