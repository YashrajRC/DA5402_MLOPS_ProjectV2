# Low-Level Design — API Specification

## Base URL

`http://localhost:8000` (via Nginx → FastAPI replicas)

## Endpoints

### 1. `POST /predict`

Classifies input text into one of 7 mental health categories.

**Request**
```json
{
  "text": "I have been feeling anxious and can't sleep"
}
```

Schema:
| Field | Type | Constraints |
|---|---|---|
| text | string | 1 ≤ len ≤ 10,000 |

**Response 200**
```json
{
  "predicted_class": "Anxiety",
  "confidence": 0.87,
  "probabilities": {
    "Anxiety": 0.87,
    "Depression": 0.05,
    "Normal": 0.03,
    "Stress": 0.02,
    "Bipolar": 0.01,
    "PTSD": 0.01,
    "Personality disorder": 0.01
  },
  "model_version": "3",
  "model_stage": "Production",
  "container_id": "7f3b2a19e4c5",
  "drift_score": 0.12
}
```

**Error responses**
| Code | Reason |
|---|---|
| 422 | Validation failure (missing text, length out of range) |
| 500 | Internal prediction error |
| 503 | Model not loaded |

---

### 2. `POST /feedback`

Records whether the last prediction was correct.

**Request**
```json
{
  "text": "original text",
  "predicted_label": "Anxiety",
  "correct_label": "Stress",
  "was_correct": false
}
```

**Response 200**
```json
{"status": "logged"}
```

Logged to `/app/logs/feedback.log` as JSONL.

---

### 3. `GET /health`

Liveness check — used by Docker healthchecks.

**Response 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "container_id": "7f3b2a19e4c5"
}
```

---

### 4. `GET /ready`

Readiness — 503 if model not yet loaded.

**Response 200**
```json
{"status": "ready", "model_loaded": true, "container_id": "..."}
```

---

### 5. `GET /metrics`

Prometheus exposition format.

Sample output:
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/predict",method="POST",status="200"} 42
...
```

## Internal Module Contracts

### `src/api/model_client.py::ModelClient`
- `.load()` → loads bundle from MLflow registry (fallback: `models/*.joblib`)
- `.predict(text: str) -> dict` returns predicted_class, confidence, probabilities, model_version, model_stage

### `src/api/drift.py::DriftDetector`
- `.add_text(text)` appends to rolling window (thread-safe)
- `.compute() -> float` returns JS divergence vs baseline in [0, 1]

### `src/training/features.py`
- `build_tfidf(max_features, ngram_range) -> TfidfVectorizer`
- `HandcraftedFeatures().fit_transform(texts) -> np.ndarray (n, 12)`
- `combine_features(tfidf_sparse, handcrafted_dense) -> scipy.sparse.csr_matrix`

## Airflow DAG Tasks

| Task | Type | Pool | Depends on |
|---|---|---|---|
| wait_for_csv | FileSensor | data_prep_pool | — |
| find_latest_csv | PythonOperator | data_prep_pool | wait_for_csv |
| validate_csv | PythonOperator | data_prep_pool | find_latest_csv |
| clean_and_stats | PythonOperator | data_prep_pool | validate_csv |
| detect_drift | PythonOperator | data_prep_pool | clean_and_stats |
| archive | PythonOperator | data_prep_pool | detect_drift |
| build_stats_email | PythonOperator | — | archive |
| email_stats | EmailOperator | — | build_stats_email |
| email_broken_csv | EmailOperator (ONE_FAILED) | — | validate_csv |
| email_dry_pipeline | EmailOperator (ALL_FAILED) | — | wait_for_csv |

Retries: 3 with exponential backoff (30s → 60s → 120s, capped at 5 min).
