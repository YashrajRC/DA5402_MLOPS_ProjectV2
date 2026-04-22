"""
FastAPI backend.
Endpoints:
    POST /predict    - classify text
    POST /feedback   - log feedback
    GET  /health     - liveness
    GET  /ready      - readiness (model loaded?)
    GET  /metrics    - Prometheus
"""
import json
import logging
import os
import socket
import time
from collections import deque
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

app = FastAPI(title="Mental Health Text Classifier", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

model_client = ModelClient()
drift_detector = DriftDetector(BASELINE, window_size=200)

# Rolling feedback for accuracy summary
feedback_window = deque(maxlen=100)
feedback_lock = Lock()


@app.on_event("startup")
def on_startup():
    log.info(f"Starting on container {CONTAINER_ID}")
    try:
        model_client.load()
        MODEL_VERSION.labels(
            version=model_client.version, stage=model_client.stage
        ).set(1)
    except Exception as e:
        log.error(f"Model load failed at startup: {e}")


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
    # Refresh drift gauge on each scrape
    DRIFT_SCORE.set(drift_detector.compute())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

    entry = {
        "ts": time.time(),
        "text": req.text,
        "predicted_label": req.predicted_label,
        "correct_label": req.correct_label,
        "was_correct": req.was_correct,
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return FeedbackResponse(status="logged")
