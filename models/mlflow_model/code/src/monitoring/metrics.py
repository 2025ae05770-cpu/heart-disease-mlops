"""Prometheus metrics exposed by the API at /metrics."""

from prometheus_client import Counter, Gauge, Histogram, Info

REQUEST_COUNT = Counter(
    "heart_api_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "heart_api_request_latency_seconds", "HTTP request latency in seconds", ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
IN_PROGRESS = Gauge("heart_api_requests_in_progress", "Requests currently being processed")
PREDICTIONS = Counter("heart_model_predictions_total", "Predictions served by label", ["label"])
PREDICTION_PROBABILITY = Histogram(
    "heart_model_probability", "Predicted probability of heart disease",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
PREDICTION_ERRORS = Counter("heart_model_prediction_errors_total", "Failed prediction attempts")
MODEL_INFO = Info("heart_model", "Metadata of the loaded model")
MODEL_LOADED = Gauge("heart_model_loaded", "1 if the model is loaded and ready")
