"""Schema / range validation for raw and cleaned heart-disease data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Plausible clinical ranges (inclusive). Values outside are flagged.
VALUE_RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (40, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}
ALLOWED_THAL = {3, 6, 7}


class DataValidationError(ValueError):
    pass


def validate_data(data: pd.DataFrame, required_columns, target_column=None, target_values=None):
    """Raise on structural problems; return a report of soft issues (out-of-range, missing)."""
    if data.empty:
        raise DataValidationError("Dataset is empty")

    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if target_column is not None:
        if data[target_column].isna().any():
            raise DataValidationError(f"Target column '{target_column}' has missing values")
        if target_values is not None:
            unexpected = set(data[target_column].unique()) - set(target_values)
            if unexpected:
                raise DataValidationError(f"Unexpected target values: {sorted(unexpected)}")

    out_of_range = {}
    for column, (low, high) in VALUE_RANGES.items():
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        bad = int(((values < low) | (values > high)).sum())
        if bad:
            out_of_range[column] = bad

    if "thal" in data.columns:
        bad_thal = int((~data["thal"].dropna().isin(ALLOWED_THAL)).sum())
        if bad_thal:
            out_of_range["thal"] = bad_thal

    report = {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "missing_values": {k: int(v) for k, v in data.isna().sum().items() if v},
        "duplicate_rows": int(data.duplicated().sum()),
        "out_of_range": out_of_range,
    }
    logger.info("Validation report: %s", report)
    return report
