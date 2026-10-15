"""Save / load the final model bundle (joblib pipeline + JSON metadata + MLflow format)."""

import json
import logging
import shutil
from pathlib import Path

import joblib

from src.config import METADATA_PATH, MLFLOW_MODEL_DIR, MODEL_PATH

logger = logging.getLogger(__name__)


def save_model(model, metadata, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved model to %s and metadata to %s", model_path, metadata_path)
    return model_path


def save_mlflow_model(model, signature, input_example, output_dir=MLFLOW_MODEL_DIR):
    """Also persist in MLflow's portable format (servable via `mlflow models serve`)."""
    import mlflow.sklearn

    from src.config import PROJECT_ROOT

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    mlflow.sklearn.save_model(
        model, path=str(output_dir), signature=signature, input_example=input_example,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        code_paths=[str(PROJECT_ROOT / "src")],
    )
    return output_dir


def load_model(model_path=MODEL_PATH):
    return joblib.load(Path(model_path))


def load_metadata(metadata_path=METADATA_PATH):
    path = Path(metadata_path)
    return json.loads(path.read_text()) if path.exists() else {}
