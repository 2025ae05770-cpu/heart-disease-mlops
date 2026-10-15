"""Central project configuration (paths, column groups, experiment settings)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data -------------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"

RAW_COLUMNS = [
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type 1-4
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl
    "restecg",   # resting ECG result 0-2
    "thalach",   # max heart rate achieved
    "exang",     # exercise induced angina
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment 1-3
    "ca",        # number of major vessels coloured by fluoroscopy 0-3
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # diagnosis 0 (absent) .. 4
]
TARGET_COLUMN = "target"
FEATURE_COLUMNS = RAW_COLUMNS[:-1]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ENGINEERED_FEATURES = ["hr_pct_of_max"]

# --- Modelling --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SELECTION_METRIC = "roc_auc"

# --- Artifacts --------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "heart_disease_model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", MODELS_DIR / "model_metadata.json"))
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- MLflow -----------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_ROOT = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", PROJECT_ROOT / "mlruns")).resolve()
EXPERIMENT_NAME = "heart-disease-risk"
REGISTERED_MODEL_NAME = "HeartDiseaseRiskChampion"
