# syntax=docker/dockerfile:1
# Heart Disease Risk API - FastAPI + scikit-learn pipeline served by uvicorn.
# Build:  docker build -t heart-disease-api:latest .
# Run:    docker run --rm -p 8000:8000 heart-disease-api:latest

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/heart_disease_model.joblib \
    METADATA_PATH=/app/models/model_metadata.json \
    LOG_FILE=/app/logs/api.log \
    LOG_LEVEL=INFO

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --uid 10001 --home /app app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Application code (needed to unpickle the custom ClinicalFeatureEngineer) + trained artifacts.
COPY src/ ./src/
COPY models/heart_disease_model.joblib models/model_metadata.json ./models/

RUN mkdir -p /app/logs && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
