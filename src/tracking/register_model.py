"""Register the champion in the MLflow Model Registry and tag it with the `champion` alias."""

import logging

import mlflow
from mlflow import MlflowClient

from src.config import REGISTERED_MODEL_NAME

logger = logging.getLogger(__name__)


def register_champion(model_uri, name=REGISTERED_MODEL_NAME, alias="champion", tags=None):
    version = mlflow.register_model(model_uri, name)
    client = MlflowClient()
    client.set_registered_model_alias(name, alias, version.version)
    for key, value in (tags or {}).items():
        client.set_model_version_tag(name, version.version, key, str(value))
    logger.info("Registered %s version %s with alias '%s'", name, version.version, alias)
    return version.version
