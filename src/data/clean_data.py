"""Clean the raw UCI data and write the processed CSV.

Cleaning decisions:
* '?' placeholders -> NaN, all columns coerced to numeric.
* Exact duplicate rows dropped.
* Diagnosis `num` (0-4) collapsed to binary `target` (0 = absent, 1 = present).
* Remaining NaNs (only `ca` and `thal`) are intentionally *kept*: they are imputed
  inside the sklearn pipeline so that imputation statistics are learned on the
  training folds only (no leakage) and the same logic runs at inference time.

Usage:
    python -m src.data.clean_data
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLEAN_DATA_PATH, FEATURE_COLUMNS, RAW_COLUMNS, TARGET_COLUMN
from src.data.load_data import load_raw_data
from src.data.validate_data import validate_data

logger = logging.getLogger(__name__)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy().replace("?", np.nan)
    for column in RAW_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    before = len(data)
    data = data.drop_duplicates().dropna(subset=["num"])
    logger.info("Dropped %d duplicate/unlabelled rows", before - len(data))

    data[TARGET_COLUMN] = (data["num"] > 0).astype(int)
    data = data.drop(columns=["num"])
    return data[FEATURE_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)


def load_clean_data(path=CLEAN_DATA_PATH, create_if_missing=True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if not create_if_missing:
            raise FileNotFoundError(path)
        run_cleaning(output_path=path)
    return pd.read_csv(path)


def run_cleaning(output_path=CLEAN_DATA_PATH):
    raw = load_raw_data()
    validate_data(raw, RAW_COLUMNS, target_column="num", target_values=range(5))

    clean = clean_data(raw)
    report = validate_data(clean, FEATURE_COLUMNS + [TARGET_COLUMN],
                           target_column=TARGET_COLUMN, target_values=[0, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)

    report["class_balance"] = {str(k): int(v) for k, v in clean[TARGET_COLUMN].value_counts().items()}
    report_path = output_path.with_name("data_quality_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Clean dataset written to %s", output_path)
    return clean, report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clean, report = run_cleaning()
    print(f"Clean dataset shape: {clean.shape}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
