"""All Prometheus metrics for the FastAPI service."""
from prometheus_client import Counter, Gauge, Histogram, Summary

# Counters
REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions by class",
    ["predicted_class", "instance_id"],
)

ERRORS_TOTAL = Counter(
    "errors_total",
    "Total errors by type",
    ["error_type"],
)

FEEDBACK_TOTAL = Counter(
    "feedback_total",
    "User feedback events",
    ["was_correct"],
)

# Gauges
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of in-flight requests",
)

DRIFT_SCORE = Gauge(
    "drift_score",
    "Current data drift score vs baseline",
)

MODEL_VERSION = Gauge(
    "model_version_active",
    "Active model version label",
    ["version", "stage"],
)

# Histograms
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference time per request",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0),
)

TEXT_LENGTH = Histogram(
    "input_text_length_chars",
    "Input text length in characters",
    buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000),
)

# Summary (rolling feedback-based accuracy)
ROLLING_ACCURACY = Summary(
    "rolling_accuracy",
    "Rolling accuracy computed from user feedback",
)
