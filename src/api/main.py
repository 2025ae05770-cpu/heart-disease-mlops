"""FastAPI model-serving application.

Run locally:
    uvicorn src.api.main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import BatchRequest, BatchResponse, HealthResponse, PatientRecord, PredictionResponse
from src.config import METADATA_PATH, MODEL_PATH
from src.inference.predict import load_model_bundle, predict_records
from src.monitoring import metrics
from src.monitoring.logging_config import configure_logging

logger = logging.getLogger("heart_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    metadata_path = Path(os.getenv("METADATA_PATH", METADATA_PATH))
    app.state.model, app.state.metadata = None, {}
    try:
        app.state.model, app.state.metadata = load_model_bundle(model_path, metadata_path)
        metrics.MODEL_LOADED.set(1)
        metrics.MODEL_INFO.info({
            "name": str(app.state.metadata.get("model_name", "unknown")),
            "family": str(app.state.metadata.get("model_family", "unknown")),
            "version": str(app.state.metadata.get("model_version", "unknown")),
        })
        logger.info("model_loaded", extra={
            "model_path": str(model_path),
            "model_version": app.state.metadata.get("model_version"),
        })
    except Exception:
        metrics.MODEL_LOADED.set(0)
        logger.exception("model_load_failed", extra={"model_path": str(model_path)})
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description="Predicts the risk of heart disease from 13 clinical attributes (UCI Cleveland).",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    start = time.perf_counter()
    metrics.IN_PROGRESS.inc()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        metrics.IN_PROGRESS.dec()
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = route.path if route is not None else "unmatched"
        if endpoint != "/metrics":
            metrics.REQUEST_COUNT.labels(request.method, endpoint, str(status_code)).inc()
            metrics.REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
            logger.info("request", extra={
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status_code": status_code, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None,
            })
    response.headers["X-Request-ID"] = request_id
    return response


def _model_version():
    return str(app.state.metadata.get("model_version", "unknown"))


def _require_model():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return app.state.model


def _score(records, request_id):
    model = _require_model()
    try:
        outputs = predict_records(model, [r.model_dump() for r in records])
    except Exception as exc:
        metrics.PREDICTION_ERRORS.inc()
        logger.exception("prediction_failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    responses = []
    for record, output in zip(records, outputs):
        metrics.PREDICTIONS.labels(output["label"]).inc()
        metrics.PREDICTION_PROBABILITY.observe(output["probability_heart_disease"])
        logger.info("prediction", extra={"request_id": request_id, "input": record.model_dump(), **output})
        responses.append(PredictionResponse(**output, model_version=_model_version(), request_id=request_id))
    return responses


@app.get("/", tags=["info"])
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "health": "/health",
            "predict": "/predict", "metrics": "/metrics"}


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    loaded = getattr(app.state, "model", None) is not None
    body = HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                          model_version=_model_version() if loaded else None)
    return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())


@app.get("/model-info", tags=["info"])
def model_info():
    _require_model()
    keys = ["model_name", "model_family", "model_version", "trained_at", "selection_metric",
            "best_params", "cv_metrics", "test_metrics", "feature_columns", "decision_threshold"]
    return {k: app.state.metadata.get(k) for k in keys}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(record: PatientRecord, request: Request):
    return _score([record], request.state.request_id)[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request):
    predictions = _score(batch.records, request.state.request_id)
    return BatchResponse(predictions=predictions, count=len(predictions))


@app.get("/metrics", tags=["monitoring"])
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
