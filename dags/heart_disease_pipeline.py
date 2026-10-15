"""Airflow orchestration of the heart-disease pipeline (mirrors the Webinar 1 DAG pattern).

download_data -> clean_validate -> eda -> train_track_register -> batch_inference

The DAG only orchestrates; all ML logic lives in the `src` package.
Run:  airflow standalone   then trigger `heart_disease_mlops_pipeline`.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG

try:  # Airflow 3.x
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def download_data():
    from src.data.load_data import download_dataset

    return str(download_dataset())


def clean_validate():
    from src.data.clean_data import run_cleaning

    _, report = run_cleaning()
    if report["out_of_range"]:
        raise ValueError(f"Out-of-range values found: {report['out_of_range']}")
    return report


def eda():
    from src.eda.run_eda import run_eda

    _, summary = run_eda()
    return summary["class_balance"]


def train_track_register():
    from src.models.train import run_training

    outcome = run_training()
    return {"champion": outcome["champion"], "model_version": outcome["metadata"]["model_version"]}


def batch_inference():
    from src.inference.predict import run_batch_inference

    return len(run_batch_inference())


with DAG(
    dag_id="heart_disease_mlops_pipeline",
    description="UCI heart disease: data -> EDA -> train/track -> register -> batch scoring",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["mlops", "heart-disease"],
) as dag:
    t_download = PythonOperator(task_id="download_data", python_callable=download_data)
    t_clean = PythonOperator(task_id="clean_validate", python_callable=clean_validate)
    t_eda = PythonOperator(task_id="eda", python_callable=eda)
    t_train = PythonOperator(task_id="train_track_register", python_callable=train_track_register)
    t_batch = PythonOperator(task_id="batch_inference", python_callable=batch_inference)

    t_download >> t_clean >> t_eda >> t_train >> t_batch
