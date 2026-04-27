"""
FastAPI backend.
Endpoints:
    POST /predict      - classify text
    POST /feedback     - log feedback (used for retraining)
    GET  /health       - liveness
    GET  /ready        - readiness
    GET  /metrics      - Prometheus
    GET  /model_info   - model metadata for dashboard
    GET  /live_stats   - aggregate live stats for dashboard
"""
import json
import logging
import os
import socket
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.drift import DriftDetector
from src.api.metrics import (
    ACTIVE_REQUESTS,
    DRIFT_SCORE,
    ERRORS_TOTAL,
    FEEDBACK_TOTAL,
    INFERENCE_LATENCY,
    MODEL_VERSION,
    PREDICTIONS_TOTAL,
    REQUESTS_TOTAL,
    ROLLING_ACCURACY,
    TEXT_LENGTH,
)
from src.api.model_client import ModelClient
from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.data.clean import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

CONTAINER_ID = socket.gethostname()
BASELINE = os.getenv("BASELINE_STATS_PATH", "/app/data/baseline_stats.json")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
FEEDBACK_LOG = Path("/app/logs/feedback.log")
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)

model_client = ModelClient()
drift_detector = DriftDetector(BASELINE, window_size=200)

# In-memory live stats
LIVE_STATS = {
    "total_predictions": 0,
    "total_latency_s": 0.0,
    "feedback_count": 0,
}
stats_lock = Lock()

# Rolling feedback window
feedback_window = deque(maxlen=100)
feedback_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Starting on container {CONTAINER_ID}")
    try:
        model_client.load()
        MODEL_VERSION.labels(
            version=model_client.version, stage=model_client.stage
        ).set(1)
    except Exception as e:
        log.error(f"Model load failed at startup: {e}")
    yield


app = FastAPI(title="Mental Health Text Classifier", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    try:
        response = await call_next(request)
        REQUESTS_TOTAL.labels(
            endpoint=request.url.path, method=request.method, status=response.status_code
        ).inc()
        return response
    except Exception as e:
        ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        REQUESTS_TOTAL.labels(
            endpoint=request.url.path, method=request.method, status=500
        ).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_client.bundle is not None,
        container_id=CONTAINER_ID,
    )


@app.get("/ready", response_model=HealthResponse)
def ready():
    if model_client.bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ready", model_loaded=True, container_id=CONTAINER_ID)


@app.get("/metrics")
def metrics():
    DRIFT_SCORE.set(drift_detector.compute())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model_info")
def model_info():
    """Model metadata for the dashboard."""
    if model_client.bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    bundle = model_client.bundle
    labels = bundle.get("labels", [])
    model_type = bundle.get("model_type", "unknown")

    # Load metrics: local file first, MLflow run as fallback
    metrics_data = {}
    metrics_file = Path(f"/app/metrics/{model_type}_metrics.json")
    if metrics_file.exists():
        try:
            metrics_data = json.loads(metrics_file.read_text())
        except Exception:
            pass
    if not metrics_data:
        try:
            from mlflow import MlflowClient as _MFC
            _client = _MFC(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            _alias = os.getenv("MODEL_ALIAS", "champion")
            _v = _client.get_model_version_by_alias(
                os.getenv("MODEL_NAME", "mental_health_classifier"), _alias
            )
            _run = _client.get_run(_v.run_id)
            m = _run.data.metrics
            metrics_data = {
                "accuracy":     m.get("accuracy", 0),
                "macro_f1":     m.get("macro_f1", 0),
                "weighted_f1":  m.get("weighted_f1", 0),
                "val_accuracy": m.get("val_accuracy", 0),
                "val_macro_f1": m.get("val_macro_f1", 0),
                "run_id":       _v.run_id,
            }
        except Exception:
            pass

    # Class distribution from baseline stats
    class_dist = {}
    baseline_file = Path(BASELINE)
    if baseline_file.exists():
        try:
            stats = json.loads(baseline_file.read_text())
            class_dist = stats.get("class_distribution", {})
        except Exception:
            pass

    return {
        "version": model_client.version,
        "stage": model_client.stage,
        "model_type": model_type,
        "labels": labels,
        "metrics": metrics_data,
        "class_distribution": class_dist,
    }


@app.get("/live_stats")
def live_stats():
    """Aggregate stats for dashboard."""
    with stats_lock:
        total = LIVE_STATS["total_predictions"]
        total_lat = LIVE_STATS["total_latency_s"]
        fb = LIVE_STATS["feedback_count"]

    avg_latency_ms = (total_lat / total * 1000) if total > 0 else 0
    return {
        "total_predictions": total,
        "avg_latency_ms": avg_latency_ms,
        "drift_score": drift_detector.compute(),
        "feedback_count": fb,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model_client.bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = clean_text(req.text)
    TEXT_LENGTH.observe(len(req.text))
    drift_detector.add_text(text)

    t0 = time.time()
    try:
        result = model_client.predict(text)
    except Exception as e:
        ERRORS_TOTAL.labels(error_type="prediction_error").inc()
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.time() - t0

    INFERENCE_LATENCY.observe(elapsed)
    PREDICTIONS_TOTAL.labels(
        predicted_class=result["predicted_class"], instance_id=CONTAINER_ID
    ).inc()

    with stats_lock:
        LIVE_STATS["total_predictions"] += 1
        LIVE_STATS["total_latency_s"] += elapsed

    current_drift = drift_detector.compute()
    DRIFT_SCORE.set(current_drift)

    return PredictResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=result["model_version"],
        model_stage=result["model_stage"],
        container_id=CONTAINER_ID,
        drift_score=current_drift,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    FEEDBACK_TOTAL.labels(was_correct=str(req.was_correct).lower()).inc()

    with feedback_lock:
        feedback_window.append(1 if req.was_correct else 0)
        if feedback_window:
            ROLLING_ACCURACY.observe(sum(feedback_window) / len(feedback_window))

    with stats_lock:
        LIVE_STATS["feedback_count"] += 1

    # Append to feedback log — consumed by Airflow retraining DAG
    entry = {
        "ts": time.time(),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": req.text,
        "predicted_label": req.predicted_label,
        "correct_label": req.correct_label,
        "was_correct": req.was_correct,
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return FeedbackResponse(status="logged")