# Project Report
## End-to-End MLOps System for Mental Health Text Classification

**Course:** DA5402 — Machine Learning Operations  
**Author:** Yashraj RC  
**Date:** April 2026  
**Repository:** github.com/YashrajRC/DA5402_MLOPS_ProjectV2

---

## Abstract

This report describes the design, implementation, and evaluation of a production-grade MLOps system for classifying mental health conditions from free-form text. The system covers the full machine learning lifecycle: automated data ingestion via Apache Airflow, reproducible training via DVC with MLflow experiment tracking, scalable model serving via three FastAPI replicas behind an Nginx load balancer, real-time observability via Prometheus and Grafana, and automated testing and build validation via GitHub Actions CI/CD. The best-performing model — an XGBoost classifier — achieves a test macro-F1 of 0.780 and p95 inference latency of 9.5ms across 7 mental health categories on the Kaggle "Sentiment Analysis for Mental Health" dataset. All 20 unit and integration tests pass. The complete system runs as 14 Docker Compose services on a single machine.

---

## 1. Introduction

### 1.1 Motivation

Mental health awareness has increased significantly in recent years, yet access to timely, appropriate resources remains a challenge at scale. Platforms that host mental health discussions — social networks, peer-support forums, journalling applications — receive large volumes of text that could be automatically triaged if reliable classification were available. A machine learning system that can distinguish between a post expressing ordinary stress and one expressing suicidal ideation could help route users to appropriate resources faster than human moderators.

This project is motivated by the engineering challenge rather than the clinical one: given an existing labelled dataset, how do you build, ship, and maintain a reliable ML system in production? The answer is MLOps — the discipline of applying DevOps principles to machine learning systems.

### 1.2 Problem Statement

Given a piece of free-form English text, classify it into one of seven mental health categories:
**Normal, Depression, Anxiety, Bipolar, Suicidal, Stress, Personality Disorder.**

The system must:
- Process new data automatically without manual intervention
- Reproduce any historical training run exactly
- Serve predictions at low latency across multiple replicas
- Detect when incoming data drifts from the training distribution
- Alert operators when system health degrades
- Support continuous testing and build validation

### 1.3 Scope

This project is a **proof-of-concept MLOps platform**. The classifier is explicitly not a clinical tool. Results must not be used to make diagnostic or treatment decisions.

---

## 2. Dataset

**Source:** Kaggle "Sentiment Analysis for Mental Health" dataset  
**Origin:** Reddit posts from mental health-focused subreddits  
**Size:** 52,675 samples  
**Format:** CSV with columns `statement` (text) and `status` (label)

### 2.1 Class Distribution

| Class | Samples | Percentage |
|---|---|---|
| Normal | 16,352 | 31.0% |
| Depression | 15,404 | 29.2% |
| Suicidal | 10,652 | 20.2% |
| Anxiety | 3,842 | 7.3% |
| Bipolar | 2,778 | 5.3% |
| Stress | 2,587 | 4.9% |
| Personality Disorder | 1,077 | 2.0% |

The dataset is moderately imbalanced. Normal and Depression together account for 60% of samples. Personality Disorder represents only 2%, making it the hardest class to learn. All models use `class_weight="balanced"` to compensate.

### 2.2 Data Preparation

The raw dataset undergoes the following preprocessing (`src/data/prepare.py`):
1. Column normalisation: `statement` → `text`, `status` → `label`
2. Text cleaning: lowercase, URL removal, @mention removal, whitespace normalisation
3. Minimum length filter: texts shorter than 3 characters dropped
4. Stratified 80/20 train/test split (random seed 42)
5. Baseline statistics computed from a 500-text stratified sample (for drift detection)

---

## 3. System Architecture

### 3.1 Overview

The system comprises 14 Docker Compose services organised into six functional layers:

| Layer | Services | Responsibility |
|---|---|---|
| Ingestion | Airflow scheduler, webserver, Postgres | Automated CSV batch processing |
| Training | Trainer container (DVC + MLflow) | Reproducible model training |
| Registry | MLflow server | Experiment tracking, model versioning |
| Serving | FastAPI × 3, Nginx | Low-latency prediction API |
| Frontend | Streamlit | End-user UI and ops dashboard |
| Observability | Prometheus, Grafana, AlertManager, Node Exporter | Metrics, dashboards, alerts |

### 3.2 Technology Stack

| Component | Technology | Version | Rationale |
|---|---|---|---|
| Orchestration | Docker Compose | 2.x | Single-command boot; isolates all services |
| Data versioning | DVC | 3.55 | Git-compatible ML pipeline reproducibility |
| Experiment tracking | MLflow | 2.16 | Unified tracking + registry in one tool |
| Data pipeline | Apache Airflow | 2.9.2 | FileSensor, pools, SMTP, retries built-in |
| API framework | FastAPI | 0.115 | Async, Pydantic validation, auto-docs |
| Load balancer | Nginx | alpine | Round-robin; DNS service discovery |
| Frontend | Streamlit | 1.39 | Data-app UI without JavaScript |
| Metrics | Prometheus | latest | Pull-based, multi-replica via DNS SD |
| Dashboards | Grafana | 11.4.0 | Recording rules, alert rules |
| CI/CD | GitHub Actions | — | Cloud runners, matrix builds |
| ML framework | scikit-learn 1.5 + XGBoost 2.1 | — | Classical ML, well-supported |

### 3.3 Serving Architecture

Three FastAPI replicas run behind Nginx in a round-robin configuration. Prometheus uses DNS service discovery (`dns_sd_configs`) to find all replica IPs automatically, enabling per-replica metric scraping without static configuration. Each replica loads the champion model independently from MLflow at startup. If MLflow is unavailable, the replica falls back to the highest-priority local `.joblib` bundle (LinearSVC > XGBoost > LogReg priority order).

This design provides:
- **Fault tolerance:** a replica crash does not disrupt service
- **Load distribution:** verified by observing ≥ 2 distinct `container_id` values in 30 consecutive requests
- **Zero-downtime model updates:** replicas can be restarted individually

---

## 4. Feature Engineering

### 4.1 Feature Pipeline

All three models share an identical 40,012-dimensional feature vector assembled by horizontal stacking of three groups:

**Word-level TF-IDF (30,000 features)**
- Vocabulary: top 30,000 terms
- N-gram range: 1–3 (unigrams, bigrams, trigrams)
- Sublinear TF scaling: reduces influence of very frequent terms
- English stop words removed; `min_df=2`, `max_df=0.95`

**Character-level TF-IDF (10,000 features)**
- Vocabulary: top 10,000 char sequences
- N-gram range: 3–4 characters
- Analyzer: `char_wb` (character sequences within word boundaries)
- Captures misspellings, affixes, and informal writing style common in Reddit data

**Handcrafted features (12 features)**
- Character count, word count, average word length
- Punctuation density (punctuation chars / total chars)
- Uppercase character ratio
- First-person pronoun ratio (`I`, `me`, `my`, `mine`, `myself`)
- Exclamation count, question mark count
- VADER sentiment scores: compound, positive, negative, neutral

### 4.2 Training/Serving Parity

Vectorizers are fitted **only on the inner-train split** (85% of train data) and serialised as part of the model bundle. At inference time, the identical fitted vectorizers are loaded from the bundle, ensuring no training-serving skew.

The 15% validation split is held out from feature fitting entirely, providing an unbiased measure of generalisation before the final test evaluation.

---

## 5. Model Training

### 5.1 Training Split Strategy

```
Full dataset (52,675)
├── Test set (10,535) ← 20%, stratified, never seen during training
└── Train set (42,140) ← 80%
    ├── Inner train (35,819) ← 85%, used for fitting
    └── Validation set (6,321) ← 15%, held out for validation metrics
```

### 5.2 Pipeline Reproducibility (DVC)

The training pipeline is defined as five DVC stages (`dvc.yaml`):

```
prepare → train_logreg ─┐
        → train_linearsvc─┤→ promote
        → train_xgboost ──┘
```

DVC tracks all stage inputs (code, data, parameters), outputs (bundles, metrics), and a `dvc.lock` file. Any stage only re-runs when its dependencies change. Stage failures are isolated — if XGBoost fails, Logistic Regression and LinearSVC results are preserved.

### 5.3 Experiment Tracking (MLflow)

Every training run logs to MLflow:
- **Parameters:** all hyperparameters and feature engineering settings
- **Metrics:** `accuracy`, `macro_f1`, `weighted_f1`, `val_accuracy`, `val_macro_f1`, `val_weighted_f1`, per-class F1 for all 7 classes
- **Artifacts:** confusion matrix PNG, serialised model bundle (`.joblib`)
- **Registry:** model registered under `mental_health_classifier`

### 5.4 Model Promotion

`scripts/promote_model.py` queries all registered model versions, compares `macro_f1`, and sets the `champion` alias on the winner. The previous champion is archived as `challenger`. The system uses MLflow aliases rather than deprecated lifecycle stages (`Production`/`Staging`).

---

## 6. Experimental Results

### 6.1 Overall Model Comparison

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7928 | 0.7672 | 0.7978 | 0.7729 |
| **XGBoost ← champion** | **0.8070** | **0.7767** | **0.8042** | **0.7803** |
| LinearSVC | 0.6921 | 0.5237 | 0.6868 | 0.5176 |

XGBoost outperforms Logistic Regression by +0.74pp macro-F1 on the test set. LinearSVC performs significantly worse than the other two models, likely because CalibratedClassifierCV (3-fold) overfits to calibration folds under the class imbalance.

Validation and test metrics for XGBoost are very close (0.777 vs 0.780), confirming no overfitting.

### 6.2 Per-Class F1 (XGBoost — Champion)

| Class | Test F1 | Training samples |
|---|---|---|
| Normal | 0.929 | 13,070 |
| Bipolar | 0.847 | 2,222 |
| Anxiety | 0.833 | 3,073 |
| Personality Disorder | 0.759 | 862 |
| Depression | 0.756 | 12,323 |
| Suicidal | 0.697 | 8,521 |
| Stress | 0.641 | 2,069 |

**Key observations:**
- Normal achieves 0.929 F1, driven by its large training set and distinctive "everyday" vocabulary
- Bipolar achieves 0.847 despite only 5.3% of training data, likely due to discriminative vocabulary (mania, mood swings, cycling)
- Stress is the weakest class (0.641); work/deadline language overlaps with Normal
- Depression and Suicidal show vocabulary overlap; the model correctly distinguishes them most of the time

### 6.3 Latency and Throughput

| Metric | Value |
|---|---|
| p95 inference latency | 9.5ms |
| Mean inference latency | 7.4ms |
| Error rate | 0.0% |
| Replicas | 3 |
| Load distribution | Round-robin, all 3 replicas active |

Latency is dominated by TF-IDF vectorisation (~5ms) rather than classifier inference. At 40,012 features, XGBoost inference over 250 trees is still substantially faster than transformer-based models.

---

## 7. Data Pipeline (Airflow)

### 7.1 DAG Overview

The `data_prep_pipeline` DAG runs every 10 minutes and watches `data/incoming/` for new CSV files. When a file appears, it executes a 6-task pipeline:

1. **wait_for_csv** — FileSensor with reschedule mode (releases worker slot between polls)
2. **find_latest_csv** — selects most recently modified file, pushes path via XCom
3. **validate_csv** — checks required columns (`text`/`label` or Kaggle aliases `statement`/`status`), row count, null fraction
4. **clean_and_stats** — applies `clean_text()`, computes batch statistics, writes cleaned batch to `data/batches/`
5. **detect_drift** — computes Jensen-Shannon divergence between batch word frequencies and training baseline; threshold 0.3
6. **archive** — moves original file to `data/incoming_archive/`

Three notification tasks fire on: successful completion (`notify_stats`), validation failure (`notify_broken_csv`), and 12-hour timeout with no data (`notify_dry_pipeline`). All use a `_notify()` helper that attempts SMTP and falls back to a local log file, never raising an exception.

### 7.2 Drift Detection

Jensen-Shannon divergence measures the distributional difference between two probability distributions. For word frequencies:

```
P = batch word frequencies (normalised over top-1000 words)
Q = baseline word frequencies (from 500-text training sample)
JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)   where M = 0.5*(P+Q)
```

JSD is symmetric, bounded in [0, 1], and zero only when P = Q. A score above 0.3 indicates the incoming batch's vocabulary has shifted meaningfully from the training distribution.

**Calibration note:** The baseline is computed from a 500-text stratified sample (matching the expected batch size) rather than all 42k training texts. Computing from the full training set produces an overly smooth baseline that makes even small batches appear highly drifted due to the sample-size difference.

---

## 8. Observability

### 8.1 Prometheus Metrics

Ten metrics are exposed at `/metrics` on each FastAPI replica:

| Metric | Type | Key labels |
|---|---|---|
| `http_requests_total` | Counter | endpoint, method, status |
| `predictions_total` | Counter | predicted_class, instance_id |
| `errors_total` | Counter | error_type |
| `feedback_total` | Counter | was_correct |
| `active_requests` | Gauge | — |
| `drift_score` | Gauge | — |
| `model_version_active` | Gauge | version, stage |
| `inference_latency_seconds` | Histogram | — |
| `input_text_length_chars` | Histogram | — |
| `rolling_accuracy` | Summary | — |

Recording rules pre-compute request rate, error rate, and p95 latency to ensure dashboard queries respond instantly.

### 8.2 Alerting

Five alert rules are configured:

| Alert | Condition | Severity |
|---|---|---|
| HighErrorRate | 5xx fraction > 5% for 2m | critical |
| HighInferenceLatency | p95 > 200ms for 2m | warning |
| DataDriftDetected | drift_score > 0.3 for 1m | warning |
| ModelServingDown | any FastAPI replica unreachable | critical |
| HighCPUUsage | CPU > 80% for 2m | warning |

---

## 9. CI/CD

### 9.1 Continuous Integration

**`ci.yml`** runs on every push and pull request to `main`:
- **Unit tests** (Job 1): Python 3.11, install dependencies, download NLTK data, `pytest tests/unit/ -v` — 12 tests
- **Docker build validation** (Job 2): Build all 4 images (fastapi/trainer, airflow, streamlit, mlflow); validate `docker compose config`

A pull request cannot be merged if any unit test fails or any Dockerfile errors.

### 9.2 Integration Testing

**`integration.yml`** runs on push to `main` and on manual dispatch:
1. Spin up `mlflow + fastapi + nginx` via Docker Compose
2. Poll `/health` until healthy (up to 150 seconds)
3. Run `pytest tests/integration/ -v` — 8 tests
4. Predict and load-distribution tests skip gracefully if no champion model is loaded in the CI environment
5. Teardown with `docker compose down -v`

---

## 10. Testing

### 10.1 Test Suite Summary

| Suite | Tests | Coverage |
|---|---|---|
| `tests/unit/test_clean.py` | 6 | `clean_text()` — all edge cases |
| `tests/unit/test_drift.py` | 3 | `DriftDetector` — empty, similar, OOD |
| `tests/unit/test_features.py` | 3 | `HandcraftedFeatures`, `build_tfidf` |
| `tests/integration/test_api.py` | 8 | Full API contract + load balancing |
| **Total** | **20** | — |

### 10.2 Results

All 20 tests pass (run: 2026-04-27, champion model XGBoost v8 loaded).

```
20 passed in 1.75s
```

### 10.3 Test Design Decisions

**Unit tests** do not mock any collaborator classes. `DriftDetector` tests create real temporary JSON baseline files. `HandcraftedFeatures` tests use a real VADER `SentimentIntensityAnalyzer`. This approach catches real integration issues between components.

**Integration tests** include a `model_ready` session fixture that checks `/ready`. Tests depending on prediction output (`test_predict_returns_valid_probs`, `test_load_distribution_across_containers`) skip if the model is not loaded, making the suite safe to run in CI where no model is trained.

---

## 11. Challenges and Resolutions

### 11.1 Airflow Log Permission Errors (WSL2)

**Problem:** `airflow-init` ran `chown -R 50000:0 /opt/airflow/logs` before `airflow db migrate`. The migrate command created `logs/scheduler/YYYY-MM-DD/` as root after the chown, so the scheduler (uid 50000) could not write log files → DAG processor crashed → no DAGs visible in UI.

**Resolution:** Moved `chown` and `chmod 777` to the end of the init command, after all Airflow CLI commands complete. This ensures all directories created during init are correctly owned before the scheduler starts.

### 11.2 DVC Stage Isolation

**Problem:** A single `train_all` DVC stage ran all three models sequentially. If XGBoost failed, DVC wrote no `dvc.lock` entry and the next `dvc repro` re-ran all three models from scratch.

**Resolution:** Split into three independent stages (`train_logreg`, `train_linearsvc`, `train_xgboost`), each with `cache: false` on outputs. Individual failures are now isolated.

### 11.3 MLflow Stage Deprecation

**Problem:** `get_latest_versions(stages=["Production"])` returned empty in MLflow 2.9+, which deprecated the lifecycle stages API. `promote_model.py` never actually promoted any model.

**Resolution:** Switched entirely to the aliases API. `promote_model.py` uses `set_registered_model_alias("champion", version)`. `ModelClient` uses `get_model_version_by_alias("champion")`.

### 11.4 Drift Calibration

**Problem:** Computing the word-frequency baseline from all 42,000 training texts produced a very smooth distribution. Comparing a 200-text rolling window against it always showed JSD ≈ 0.3–0.5, even for in-distribution text, because the smaller sample's vocabulary was less diverse.

**Resolution:** Recomputed the baseline from a stratified 500-text sample (proportional to class frequency). This calibrates the baseline to the same scale as the rolling window, producing meaningful contrast between in-distribution and OOD inputs.

### 11.5 Training-Serving Skew (Validation Split)

**Problem:** The original training code trained on all train.csv data and evaluated only on test.csv, with no validation split. Vectorizers were fit on full training data including what would become validation examples.

**Resolution:** Added a `StratifiedShuffleSplit` (15% of train) before any feature fitting. All vectorizers and feature transformers are fitted only on the inner-train split and serialised into the model bundle for inference.

### 11.6 SMTP AlertManager Crashes

**Problem:** The original email tasks used Airflow's `EmailOperator`, which throws an uncaught exception when SMTP credentials are not configured, failing the entire DAG run.

**Resolution:** Replaced all email tasks with `PythonOperator` calling a `_notify()` helper. `_notify()` attempts SMTP if `MAILTRAP_USER` is set, catches any exception, and always falls back to writing to a local notification log. The DAG run never fails due to email configuration.

---

## 12. Limitations

| Limitation | Description | Mitigation path |
|---|---|---|
| Classical ML only | No transformer-based model; likely leaves 5–10pp macro-F1 on the table | Fine-tune DistilBERT when GPU is available |
| Minority classes | Personality Disorder (2% of data) and Stress are the weakest classes | Synthetic oversampling; active learning from feedback |
| Drift calibration | JSD is sensitive to window size vs baseline size | Bootstrap-calibrated threshold; or MMD/UMAP-based drift |
| No auth on API | `/predict` and `/feedback` are unauthenticated | Add OAuth2/API key middleware for production |
| SQLite MLflow backend | Not concurrent-write safe | Migrate to PostgreSQL backend |
| Single-host Docker Compose | Cannot scale beyond one machine | Kubernetes migration path documented in HLD |
| English only | TF-IDF and VADER are English-specific | Multilingual tokenisation; mBERT for non-English |

---

## 13. Future Work

1. **Transformer fine-tuning:** Replace TF-IDF + XGBoost with a fine-tuned DistilBERT or MentalBERT model. Expected improvement: +5–8pp macro-F1.

2. **Active learning loop:** Route low-confidence predictions (confidence < 0.5) to a human reviewer queue. Retrain on corrected labels monthly.

3. **Model confidence monitoring:** Complement word-frequency JSD with a confidence-score distribution monitor. The model's own uncertainty is a more direct signal of distribution shift than vocabulary change.

4. **Kubernetes deployment:** Containerise each service as a Kubernetes Deployment. Use HPA for FastAPI (scale on RPS) and a CronJob for Airflow triggers.

5. **A/B testing framework:** Serve two models simultaneously (champion and challenger) and compare prediction distributions and feedback rates before full promotion.

6. **Crisis detection specialisation:** Build a dedicated binary classifier for Suicidal vs. non-Suicidal as a higher-precision safety layer on top of the 7-class model.

---

## 14. Conclusion

This project demonstrates a complete, functioning MLOps system built from open-source components. The system automates every stage of the ML lifecycle — data ingestion, validation, drift detection, reproducible training, experiment tracking, model promotion, scalable serving, observability, alerting, and CI/CD — with all components containerised and runnable from a single `docker compose up` command.

The champion XGBoost model achieves a test macro-F1 of 0.780 (exceeding the 0.75 target) with p95 inference latency of 9.5ms (well within the 200ms SLA). All 20 tests pass. The system gracefully handles component failures, SMTP outages, model loading failures, and minority-class imbalance.

The primary engineering contributions are:
- A **five-stage DVC pipeline** with per-model failure isolation and automatic best-model promotion
- A **three-replica FastAPI serving layer** with DNS-based Prometheus scraping and local model fallback
- A **two-path drift detection system** (API rolling window + Airflow batch) using Jensen-Shannon divergence with calibrated baseline
- A **resilient Airflow notification system** that never fails a DAG run due to SMTP misconfiguration
- A **CI/CD pipeline** that validates builds and runs tests on every push

---

## Appendix A — File Structure

```
project/
├── airflow/
│   └── dags/data_prep_pipeline.py   ← Airflow DAG
├── configs/
│   ├── prometheus.yml               ← scrape config + DNS SD
│   ├── alert_rules.yml              ← 5 alert rules
│   ├── recording_rules.yml          ← 3 recording rules
│   └── grafana/                     ← dashboard + datasource provisioning
├── data/
│   ├── raw/data.csv                 ← Kaggle dataset (DVC tracked)
│   ├── processed/                   ← train.csv, test.csv (DVC tracked)
│   ├── baseline_stats.json          ← drift baseline (DVC tracked)
│   ├── incoming/                    ← drop CSVs here
│   ├── incoming_archive/            ← processed CSVs moved here
│   └── batches/                     ← cleaned batch output
├── docker/
│   ├── fastapi/Dockerfile           ← API + trainer image
│   ├── airflow/Dockerfile           ← Airflow image
│   ├── streamlit/Dockerfile         ← Streamlit image
│   ├── mlflow/Dockerfile            ← MLflow image
│   └── nginx/nginx.conf             ← load balancer config
├── docs/                            ← this documentation
├── metrics/                         ← DVC metric JSON files
├── models/                          ← local .joblib bundles
├── scripts/
│   ├── promote_model.py             ← champion alias promotion
│   ├── simulate_drift.py            ← OOD traffic generator
│   ├── load_tester.py               ← latency benchmarking
│   └── sample_incoming.csv          ← demo batch for Airflow
├── src/
│   ├── api/                         ← FastAPI application
│   ├── data/                        ← prepare.py, clean.py
│   ├── frontend/app.py              ← Streamlit UI
│   └── training/                    ← train.py, features.py
├── tests/
│   ├── unit/                        ← 12 unit tests
│   └── integration/                 ← 8 integration tests
├── .github/workflows/               ← ci.yml, integration.yml
├── docker-compose.yml               ← 14-service stack definition
├── dvc.yaml                         ← 5-stage pipeline
├── params.yaml                      ← all hyperparameters
├── MLproject                        ← MLflow project entry point
└── pytest.ini                       ← test configuration
```

## Appendix B — Key Commands Reference

```bash
# Start the full stack
docker compose up -d

# Run all tests
.venv/bin/pytest -v

# Retrain all models
docker compose exec -T trainer mlflow run /app -e train -P model=logreg --env-manager local --experiment-name mental_health_classifier
docker compose exec -T trainer mlflow run /app -e train -P model=linearsvc --env-manager local --experiment-name mental_health_classifier
docker compose exec -T trainer mlflow run /app -e train -P model=xgboost --env-manager local --experiment-name mental_health_classifier

# Promote best model
docker compose exec -T trainer python /app/scripts/promote_model.py

# Reload serving layer
docker compose restart fastapi

# Trigger Airflow batch processing
cp scripts/sample_incoming.csv data/incoming/demo.csv

# Generate drift traffic
python scripts/simulate_drift.py

# View current drift score
curl -s http://localhost:8000/live_stats | python3 -m json.tool

# Check all service health
docker compose ps
```

## Appendix C — Service Ports Reference

| Service | Port | URL |
|---|---|---|
| Streamlit | 8501 | http://localhost:8501 |
| API (via Nginx) | 8000 | http://localhost:8000 |
| MLflow | 5000 | http://localhost:5000 |
| Airflow | 8080 | http://localhost:8080 |
| Grafana | 3001 | http://localhost:3001 |
| Prometheus | 9090 | http://localhost:9090 |
| AlertManager | 9093 | http://localhost:9093 |
| Node Exporter | 9100 | http://localhost:9100 |
