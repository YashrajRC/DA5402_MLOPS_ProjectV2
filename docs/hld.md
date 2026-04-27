
# High-Level Design (HLD)
## Mental Health Text Classifier — MLOps System

---

## 1. Business Problem

Mental health platforms receive thousands of user posts daily across social media, journals, and peer-support channels. Manual triage is impossible at scale, and mis-routing a user in crisis to the wrong resource has real human cost.

This system provides automated, real-time classification of free-form text into seven mental health categories, allowing downstream applications to route users to appropriate resources. It is explicitly positioned as a **pattern-detection tool, not a clinical diagnostic instrument**.

---

## 2. Goals and Success Criteria

| Category | Metric | Target | Achieved |
|---|---|---|---|
| ML Quality | Macro F1-score | ≥ 0.75 | **0.780** (XGBoost) |
| ML Quality | Per-class F1 (minority classes) | ≥ 0.60 | Min: **0.641** (Stress) |
| Serving | p95 inference latency | < 200ms | **~9.5ms** |
| Serving | API availability | ≥ 99% | 3-replica failover |
| Operations | Drift detection lag | < 1 hour | **Real-time** (rolling window) |
| Operations | Batch drift detection | Per batch | **Per Airflow run** |
| Dev | Test coverage | 20 tests | **20/20 passing** |

---

## 3. Classification Target

**Dataset:** Kaggle "Sentiment Analysis for Mental Health" — 52,675 labelled samples  
**Source:** Reddit posts across mental health subreddits  
**Split:** 80% train (42,140) / 20% test (10,535), stratified by label

| Class | Train samples | % of data |
|---|---|---|
| Normal | 13,070 | 31.0% |
| Depression | 12,323 | 29.2% |
| Suicidal | 8,521 | 20.2% |
| Anxiety | 3,073 | 7.3% |
| Bipolar | 2,222 | 5.3% |
| Stress | 2,069 | 4.9% |
| Personality Disorder | 862 | 2.0% |

---

## 4. System Architecture Layers

```
┌────────────────────────────────────────────────────────────────┐
│  USER LAYER                                                    │
│  Streamlit :8501  ──────────────────────────────────────────  │
│       │                                                        │
│  REST (HTTP)                                                   │
│       ↓                                                        │
├────────────────────────────────────────────────────────────────┤
│  SERVING LAYER                                                 │
│  Nginx :8000 (load balancer)                                   │
│     ├──▶ FastAPI replica 1 :8000                               │
│     ├──▶ FastAPI replica 2 :8000                               │
│     └──▶ FastAPI replica 3 :8000                               │
│                    │                                           │
│            /metrics endpoint                                   │
├────────────────────────────────────────────────────────────────┤
│  MODEL LAYER                                                   │
│  MLflow :5000  ←── champion alias ─── promote_model.py        │
│     ├── Logistic Regression (v7)                               │
│     ├── XGBoost (v8, champion)                                 │
│     └── LinearSVC (v9)                                         │
├────────────────────────────────────────────────────────────────┤
│  TRAINING LAYER (DVC pipeline)                                 │
│  prepare → train_logreg / train_linearsvc / train_xgboost     │
│         → promote (picks best macro-F1)                        │
├────────────────────────────────────────────────────────────────┤
│  INGESTION & RETRAINING LAYER (Airflow)                        │
│  data_prep_pipeline: FileSensor → validate → clean →           │
│    drift_detect → archive → notify                             │
│  retrain_pipeline: feedback threshold → split → retrain →      │
│    promote → notify                                            │
├────────────────────────────────────────────────────────────────┤
│  OBSERVABILITY LAYER                                           │
│  Prometheus :9090 ──▶ Grafana :3001                            │
│  Node Exporter :9100                                           │
│  AlertManager :9093                                            │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Responsibilities

| Component | Technology | Responsibility |
|---|---|---|
| Streamlit | Python 3.11 / Streamlit 1.39 | 3-page UI: Analyze, Dashboard, User Manual |
| Nginx | nginx:alpine | Round-robin load balancer across 3 FastAPI replicas |
| FastAPI | FastAPI 0.115 / Uvicorn | REST API: predict, feedback, health, metrics, model_info |
| MLflow | MLflow 2.16 / SQLite | Experiment tracking, artifact store, model registry |
| DVC | DVC 3.55 | Reproducible ML pipeline with stage-level caching |
| Airflow | Apache Airflow 2.9.2 | Two DAGs: batch ingestion + drift detection (`data_prep_pipeline`); feedback-triggered retraining + promotion (`retrain_pipeline`) |
| Prometheus | prom/prometheus | Pull-based metrics collection (15s interval) |
| Grafana | grafana 11.4.0 | Dashboards, alert visualization |
| AlertManager | prom/alertmanager | Route Prometheus alerts to email/Slack |
| Node Exporter | prom/node-exporter | Host CPU/memory/disk metrics |
| Postgres | postgres:15 | Airflow metadata backend |

---

## 6. Key Design Decisions

### 6.1 Three-Model Strategy with Automatic Promotion
Three model families train on every pipeline run. The `promote` DVC stage runs `promote_model.py` which queries all registered MLflow versions, compares macro-F1, and sets the `champion` alias on the winner. No manual step required.

**Why three models:** Different families excel at different data patterns. Logistic Regression provides a probabilistic linear baseline; LinearSVC finds maximum-margin boundaries; XGBoost captures non-linear feature interactions. Keeping all three lets the system always serve the empirically best option.

### 6.2 MLflow Aliases over Lifecycle Stages
MLflow 2.9+ deprecated `Production`/`Staging` stages. This system uses named aliases: `champion` (current best model) and `challenger` (previous champion, archived for rollback).

**Why:** Future-proof API; semantic clarity; rollback is trivial (`set_registered_model_alias(name, "champion", challenger_version)`).

### 6.3 Stateless Replicas with Shared Registry
Each FastAPI replica loads the champion model bundle independently from MLflow at startup. No shared in-memory state between replicas. Nginx uses DNS service discovery (`dns_sd_configs`) to find all FastAPI container IPs.

**Why:** True horizontal scalability. A replica crash does not affect the others. Adding replicas requires only changing `deploy.replicas` in docker-compose.

### 6.4 Validation Split in Training
Training data is split 85% inner-train / 15% validation before feature fitting. Vectorizers are fit only on inner-train. Validation metrics (`val_accuracy`, `val_macro_f1`, `val_weighted_f1`) are logged to MLflow separately from test metrics.

**Why:** Prevents data leakage from validation into the feature representation. Validation metrics give a realistic estimate of generalisation before seeing the held-out test set.

### 6.5 DVC per-Model Stage Isolation
Each model (`train_logreg`, `train_linearsvc`, `train_xgboost`) is its own DVC stage. If XGBoost fails, Logistic Regression and LinearSVC results are preserved and the promote stage can still run.

**Why:** A single `train_all` stage (previous design) marked everything failed if any model failed, forcing full re-training on the next `dvc repro`.

### 6.6 Airflow Reschedule-Mode FileSensor
The FileSensor runs with `mode="reschedule"` and `soft_fail=True`. It checks every 60 seconds, releases its worker slot between checks, and soft-fails after 12 hours (triggering `notify_dry_pipeline` instead of a hard failure).

**Why:** Poke mode holds a worker slot indefinitely. With pool size 3 and a 12-hour timeout, poke mode would starve the pool. Reschedule mode allows the pool to serve other tasks between sensor checks.

### 6.8 Feedback-Driven Retraining Loop

The `/feedback` endpoint writes user corrections as JSON lines to `data/feedback.log` (shared bind-mount between FastAPI and Airflow). The `retrain_pipeline` DAG polls this file every 2 hours. When `FEEDBACK_THRESHOLD` entries accumulate it:

1. Splits feedback 80/20 (stratified by `correct_label`) and appends each slice to the existing `train.csv`/`test.csv` — so every retraining run trains on all historical data plus feedback.
2. Retrains the current champion model type (queried from MLflow) as a new registered version.
3. Promotes the new version only if its `macro_f1` beats the incumbent — ensuring feedback noise cannot degrade the serving model.

The test set grows alongside the train set, so evaluation metrics remain representative of the current data distribution rather than freezing at initial-training time.

### 6.7 Notification Fallback
All email alerts use a `_notify()` helper (PythonOperator) instead of Airflow's built-in `EmailOperator`. `_notify()` tries SMTP first; if SMTP is not configured or fails, it writes to `/opt/airflow/logs/notifications.log` without raising.

**Why:** `EmailOperator` throws uncaught exceptions when SMTP credentials are absent, failing the entire DAG run. The fallback keeps the pipeline functional in development.

---

## 7. Feature Engineering

Features are computed identically at training time and inference time to prevent training-serving skew:

| Feature group | Implementation | Dimensionality |
|---|---|---|
| Word-level TF-IDF | 1–3 n-grams, 30,000 features, sublinear TF, English stopwords | 30,000 |
| Char-level TF-IDF | 3–4 char n-grams, 10,000 features, `analyzer=char_wb` | 10,000 |
| Handcrafted | Text length, word count, avg word length, punct density, uppercase ratio, first-person ratio, `!` count, `?` count, VADER compound/pos/neg/neu | 12 |
| **Total** | | **40,012** |

The char-level TF-IDF captures morphological patterns (affixes, misspellings) that word-level TF-IDF misses. VADER sentiment adds an explicit emotional signal that complements distributional word features.

---

## 8. Drift Detection Architecture

Two independent drift detection paths operate in parallel:

**Path 1 — API real-time (per-request):**
- FastAPI `DriftDetector` maintains a thread-safe rolling deque of the last 200 cleaned texts
- On every `/predict` call, the current window's top-1000 word frequencies are compared to `baseline_stats.json` via Jensen-Shannon divergence
- Result exposed as `drift_score` Prometheus gauge
- Grafana fires alert if `drift_score > 0.3` for 1 minute

**Path 2 — Airflow batch (per-ingestion):**
- Each batch CSV processed by the `detect_drift` task computes the same JSD against the baseline
- Score logged in task output and included in the stats email
- Threshold 0.3 triggers a ⚠️ warning in the email subject

**Baseline:** Computed from a stratified 500-text sample of the training set (proportional to class frequency) to ensure the baseline and runtime window are calibrated to the same scale.

---

## 9. Observability Strategy

### Metrics Taxonomy

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `http_requests_total` | Counter | endpoint, method, status | Request rate, error rate |
| `predictions_total` | Counter | predicted_class, instance_id | Class distribution over time |
| `errors_total` | Counter | error_type | Error classification |
| `feedback_total` | Counter | was_correct | User satisfaction |
| `active_requests` | Gauge | — | Concurrent load |
| `drift_score` | Gauge | — | Data drift monitoring |
| `model_version_active` | Gauge | version, stage | Model provenance |
| `inference_latency_seconds` | Histogram | — | Latency distribution |
| `input_text_length_chars` | Histogram | — | Input size distribution |
| `rolling_accuracy` | Summary | — | Feedback-based accuracy |

### Recording Rules (pre-computed)
- `job:request_rate:1m` — requests/second over 1-minute window
- `job:error_rate:2m` — 5xx fraction over 2-minute window (with `or vector(0)` guard)
- `job:inference_latency_p95:2m` — p95 latency bucket quantile

### Alert Rules
| Alert | Expression | For | Severity |
|---|---|---|---|
| HighErrorRate | error_rate > 5% | 2m | critical |
| HighInferenceLatency | p95 > 200ms | 2m | warning |
| DataDriftDetected | drift_score > 0.3 | 1m | warning |
| ModelServingDown | up{job="fastapi"} == 0 | 1m | critical |
| HighCPUUsage | CPU > 80% | 2m | warning |

---

## 10. Continuous Integration / Continuous Delivery

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | Push & PR to main | Unit tests (pytest tests/unit/) + Docker build validation (all 4 images) |
| `integration.yml` | Push to main + manual | Start mlflow+fastapi+nginx, wait for health, run integration tests, teardown |

CI blocks merge if any unit test fails or any Dockerfile fails to build. Integration tests that require a loaded model skip gracefully in CI (no model is trained in the runner environment).

---

## 11. Security and Secrets Management

| Secret | Storage | Rotation |
|---|---|---|
| Mailtrap SMTP credentials | `.env` (gitignored) | Manual |
| Airflow DB password | Hardcoded in compose (dev only) | Change before production |
| Grafana admin password | Compose env var | Change before production |
| MLflow | No auth (internal network only) | Add reverse proxy + auth for production |

All services communicate on an isolated Docker bridge network (`appnet`). Only Nginx (:8000), Streamlit (:8501), Airflow (:8080), MLflow (:5000), Grafana (:3001), Prometheus (:9090) are exposed to the host.

---

## 12. Scalability Path

| Bottleneck | Current (dev) | Production path |
|---|---|---|
| FastAPI replicas | 3 fixed | Kubernetes HPA on CPU/RPS |
| MLflow backend | SQLite | PostgreSQL, object store (S3/GCS) for artifacts |
| Airflow executor | LocalExecutor | CeleryExecutor + Redis broker + worker fleet |
| Drift window | 200 texts in-memory per replica | Redis-backed shared sliding window |
| Model training | Docker Compose on laptop | GPU-backed Kubernetes Jobs |
| Feature computation | Single-threaded sklearn | Distributed Dask/Spark for large corpora |
