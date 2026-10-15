"""Model inference helpers shared by the API and batch scoring.

Batch usage:
    python -m src.inference.predict --input data/processed/heart_disease_clean.csv \
        --output data/processed/batch_predictions.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLEAN_DATA_PATH, FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH, PROJECT_ROOT
from src.models.persistence import load_metadata, load_model

logger = logging.getLogger(__name__)

LABELS = {0: "no_heart_disease", 1: "heart_disease"}


def risk_band(probability):
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "moderate"
    return "high"


def load_model_bundle(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    return load_model(model_path), load_metadata(metadata_path)


def to_frame(records):
    frame = pd.DataFrame(records)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature(s): {missing}")
    return frame[FEATURE_COLUMNS].astype("float64")


def predict_records(model, records, threshold=0.5):
    """records: list of dicts or DataFrame -> list of prediction dicts."""
    frame = records if isinstance(records, pd.DataFrame) else to_frame(records)
    frame = frame[FEATURE_COLUMNS].astype("float64")
    probabilities = model.predict_proba(frame)[:, 1]
    outputs = []
    for p in probabilities:
        p = float(p)
        prediction = int(p >= threshold)
        outputs.append({
            "prediction": prediction,
            "label": LABELS[prediction],
            "confidence": round(max(p, 1 - p), 4),
            "probability_heart_disease": round(p, 4),
            "risk_level": risk_band(p),
        })
    return outputs


def run_batch_inference(input_path=CLEAN_DATA_PATH, output_path=None, model_path=MODEL_PATH):
    input_path = Path(input_path)
    output_path = Path(output_path or PROJECT_ROOT / "data" / "processed" / "batch_predictions.csv")
    data = pd.read_csv(input_path)
    model = load_model(model_path)
    predictions = pd.DataFrame(predict_records(model, data[FEATURE_COLUMNS]), index=data.index)
    result = pd.concat([data, predictions], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Scored %d rows -> %s", len(result), output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch scoring with the packaged model")
    parser.add_argument("--input", default=str(CLEAN_DATA_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_batch_inference(args.input, args.output, args.model)
    print(result[["prediction", "probability_heart_disease", "risk_level"]].head(10).to_string())
    if "target" in result.columns:
        print(f"Agreement with labels: {np.mean(result['prediction'] == result['target']):.3f}")


if __name__ == "__main__":
    main()
