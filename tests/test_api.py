import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client(model_files, monkeypatch, tmp_path):
    model_path, metadata_path = model_files
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "api.log"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("LOG_FILE", "")
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "test-1"}


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    response = client.post("/predict", json=sample_patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["risk_level"] in ("low", "moderate", "high")
    assert body["model_version"] == "test-1"
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    sample_patient.pop("ca")
    sample_patient.pop("thal")
    assert client.post("/predict", json=sample_patient).status_code == 200


@pytest.mark.parametrize("field,value", [("age", 200), ("sex", 2), ("cp", 0), ("thal", 5), ("chol", -1)])
def test_predict_rejects_invalid_values(client, sample_patient, field, value):
    sample_patient[field] = value
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_rejects_missing_and_unknown_fields(client, sample_patient):
    assert client.post("/predict", json={k: v for k, v in sample_patient.items() if k != "age"}).status_code == 422
    assert client.post("/predict", json={**sample_patient, "foo": 1}).status_code == 422


def test_batch_predict(client, sample_patient):
    response = client.post("/predict/batch", json={"records": [sample_patient, sample_patient]})
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_metrics_endpoint_exposes_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_model_predictions_total" in text
    assert "heart_model_loaded 1.0" in text


def test_requests_are_logged_to_file(client, sample_patient, tmp_path):
    client.post("/predict", json=sample_patient)
    log_text = (tmp_path / "api.log").read_text()
    assert '"message": "request"' in log_text and '"message": "prediction"' in log_text


def test_model_info(client):
    assert client.get("/model-info").json()["model_family"] == "logistic_regression"


def test_service_degrades_without_model(client_without_model, sample_patient):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=sample_patient).status_code == 503


def test_unwritable_log_file_falls_back_to_stdout(tmp_path):
    from src.monitoring.logging_config import configure_logging

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    logger = configure_logging(log_file=str(blocker / "api.log"))
    assert len(logger.handlers) == 1  # stream handler only, no crash
