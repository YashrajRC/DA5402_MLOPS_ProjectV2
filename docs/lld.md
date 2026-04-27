# Low-Level Design (LLD)
## Mental Health Text Classifier — Component Specifications

---

## 1. REST API Specification

**Base URL:** `http://localhost:8000` (via Nginx → FastAPI replicas)  
**Content-Type:** `application/json`  
**Framework:** FastAPI 0.115 + Uvicorn 0.30 + Pydantic 2.9

---

### `POST /predict`

Classifies input text into one of 7 mental health categories.

**Request body**

| Field | Type | Constraints |
|---|---|---|
| `text` | string | 1 ≤ len ≤ 10,000 chars |

```json
{ "text": "I have been feeling anxious and cannot sleep for weeks" }
```

**Response 200**

| Field | Type | Description |
|---|---|---|
| `predicted_class` | string | One of the 7 class labels |
| `confidence` | float | Probability of the top class [0, 1] |
| `probabilities` | object | Score for each of the 7 classes (sum ≈ 1.0) |
| `model_version` | string | MLflow registry version number |
| `model_stage` | string | Alias under which the model was loaded |
| `container_id` | string | Hostname of the FastAPI replica that served this |
| `drift_score` | float | Current rolling-window JS divergence [0, 1] |

```json
{
  "predicted_class": "Anxiety",
  "confidence": 0.864,
  "probabilities": {
    "Anxiety": 0.864,
    "Depression": 0.071,
    "Normal": 0.031,
    "Stress": 0.018,
    "Suicidal": 0.008,
    "Bipolar": 0.005,
    "Personality disorder": 0.003
  },
  "model_version": "8",
  "model_stage": "champion",
  "container_id": "75ab725b7119",
  "drift_score": 0.302
}
```

**Error responses**

| Code | Cause |
|---|---|
| 422 | Validation failure — empty text, length > 10,000, missing field |
| 500 | Internal prediction error |
| 503 | Model not loaded at startup |

**Processing pipeline (per request):**
1. Pydantic validates request schema
2. `clean_text()` applied: lowercase, strip URLs, strip @mentions, collapse whitespace
3. `drift_detector.add_text(cleaned)` — appends to rolling deque (thread-safe lock)
4. `model_client.predict(cleaned)` — vectorize → classify → softmax probabilities
5. Prometheus counters/histograms updated
6. `drift_detector.compute()` — JSD computed and set as Gauge
7. Response serialised and returned

---

### `POST /feedback`

Records user correction for a previous prediction.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | The original input text |
| `predicted_label` | string | yes | What the model predicted |
| `correct_label` | string | no | The true label if known |
| `was_correct` | bool | yes | Whether the prediction was right |

**Response 200**
```json
{ "status": "logged" }
```

Feedback appended as JSON lines to `/app/data/feedback.log` (shared bind-mount, readable by Airflow at `/opt/airflow/data/feedback.log`). Prometheus `feedback_total` counter incremented. `rolling_accuracy` Summary updated. The `retrain_pipeline` Airflow DAG consumes this file when it reaches the configured threshold.

---

### `GET /health`

Liveness check — always returns 200 if the process is alive.

```json
{ "status": "ok", "model_loaded": true, "container_id": "75ab725b7119" }
```

---

### `GET /ready`

Readiness check — returns 503 if model bundle not yet loaded.

```json
{ "status": "ready", "model_loaded": true, "container_id": "75ab725b7119" }
```

---

### `GET /model_info`

Returns the champion model's metadata and metrics.

```json
{
  "version": "8",
  "stage": "champion",
  "model_type": "xgboost",
  "labels": ["Anxiety","Bipolar","Depression","Normal","Personality disorder","Stress","Suicidal"],
  "metrics": {
    "accuracy": 0.8042,
    "macro_f1": 0.7803,
    "weighted_f1": 0.8025,
    "val_accuracy": 0.8070,
    "val_macro_f1": 0.7767,
    "run_id": "8fcc2fce7f8b4a7cbe1921817541fcd1"
  },
  "class_distribution": { "Normal": 0.310, "Depression": 0.292, "..." : "..." }
}
```

Metrics priority: local `metrics/{model_type}_metrics.json` → MLflow run fallback → empty dict.

---

### `GET /live_stats`

Returns aggregate runtime statistics.

```json
{
  "total_predictions": 1452,
  "avg_latency_ms": 7.4,
  "drift_score": 0.302,
  "feedback_count": 23
}
```

---

### `GET /metrics`

Prometheus text exposition format. Scraped by Prometheus every 15 seconds.

---

## 2. Internal Module Contracts

### `src/api/model_client.py — ModelClient`

```python
class ModelClient:
    bundle: Optional[dict]   # loaded joblib bundle
    version: str             # MLflow version or "local"
    stage: str               # alias or "fallback"

    def load(self) -> None:
        # 1. Try MLflow: get_model_version_by_alias("champion")
        # 2. download_artifacts(run_id, "model_bundle")
        # 3. joblib.load the first *_bundle.joblib found
        # Fallback: scan /app/models/ for *.joblib
        #   priority: linearsvc > xgboost > logreg

    def predict(self, text: str) -> dict:
        # vectorize → tfidf_word + tfidf_char + handcrafted
        # hstack into sparse matrix
        # clf.predict_proba() or clf.predict() + one-hot
        # return predicted_class, confidence, probabilities, version, stage
```

Bundle structure (joblib dict):
| Key | Type | Description |
|---|---|---|
| `tfidf_word` | TfidfVectorizer | Fitted word-level TF-IDF |
| `tfidf_char` | TfidfVectorizer | Fitted char-level TF-IDF |
| `handcrafted` | HandcraftedFeatures | Fitted feature extractor |
| `classifier` | sklearn/xgboost estimator | Trained classifier |
| `label_encoder` | LabelEncoder | Fitted label encoder |
| `labels` | list[str] | Ordered class names |
| `model_type` | str | "logreg" / "linearsvc" / "xgboost" |

---

### `src/api/drift.py — DriftDetector`

```python
class DriftDetector:
    baseline: dict           # word_freq_top1000 from baseline_stats.json
    recent_texts: deque      # maxlen=200, thread-safe via Lock
    window_size: int = 200

    def add_text(self, text: str) -> None:
        # acquires lock, appends cleaned text

    def compute(self) -> float:
        # requires n >= 20 texts, else returns 0.0
        # extracts top-1000 words from window, normalises by top-1000 sum
        # computes Jensen-Shannon divergence vs baseline_freq
        # clamps result to [0, 1]
```

JSD formula:
```
M = 0.5 * (P + Q)
JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
```

---

### `src/data/clean.py — clean_text`

```python
def clean_text(text) -> str:
    # 1. Cast to str; return "" for None/non-string
    # 2. Lowercase
    # 3. Strip URLs (http://, https://, www.)
    # 4. Strip @mentions
    # 5. Collapse whitespace
    # Note: punctuation PRESERVED (! ? . carry signal)
```

---

### `src/training/features.py`

```python
class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    # 12 features per text:
    # [0]  len(text) — character count
    # [1]  len(tokens) — word count
    # [2]  mean word length
    # [3]  punctuation count / char count
    # [4]  uppercase char count / char count
    # [5]  first-person pronoun count / word count
    # [6]  exclamation count
    # [7]  question mark count
    # [8]  VADER compound score
    # [9]  VADER positive score
    # [10] VADER negative score
    # [11] VADER neutral score

def build_tfidf(max_features, ngram_range) -> TfidfVectorizer:
    # sublinear_tf=True, min_df=2, max_df=0.95, stop_words="english"
```

---

### `src/data/prepare.py — compute_baseline_stats`

```python
def compute_baseline_stats(df, sample_size=500) -> dict:
    # Stratified sample: proportional to class frequency
    # Computes word frequencies from sample (not full 42k dataset)
    # Returns: avg_text_length, std_text_length, class_distribution,
    #          vocab_size, word_freq_top1000, total_samples
```

---

## 3. DVC Pipeline Stages

| Stage | Input deps | Params | Outputs |
|---|---|---|---|
| `prepare` | `data/raw/data.csv`, `src/data/prepare.py`, `src/data/clean.py` | `prepare.*` | `data/processed/train.csv`, `data/processed/test.csv`, `data/baseline_stats.json` |
| `train_logreg` | `src/training/train.py`, `src/training/features.py`, `data/processed/train.csv`, `data/processed/test.csv`, `MLproject` | `features.*`, `train.logreg.*` | `models/logreg_bundle.joblib` (cache:false), `metrics/logreg_metrics.json` |
| `train_linearsvc` | same as logreg | `features.*`, `train.linearsvc.*` | `models/linearsvc_bundle.joblib`, `metrics/linearsvc_metrics.json` |
| `train_xgboost` | same as logreg | `features.*`, `train.xgboost.*` | `models/xgboost_bundle.joblib`, `metrics/xgboost_metrics.json` |
| `promote` | `scripts/promote_model.py`, all 3 metric files | — | (side-effect: sets MLflow champion alias) |

DVC commands reference:

```bash
dvc repro                          # run all changed stages
dvc repro --force train_xgboost    # force re-run one stage
dvc metrics show                   # compare metrics across runs
dvc diff                           # see what changed vs last commit
```

---

## 4. Model Training Parameters

### Shared feature parameters (`params.yaml`)

| Parameter | Value | Effect |
|---|---|---|
| `max_tfidf_features` | 30,000 | Word-level vocabulary cap |
| `ngram_range` | [1, 3] | Unigrams, bigrams, trigrams |
| `max_tfidf_char_features` | 10,000 | Char n-gram vocabulary cap |
| `char_ngram_range` | [3, 4] | 3- and 4-char sequences |
| `min_df` | 2 | Ignore terms in fewer than 2 docs |
| `max_df` | 0.95 | Ignore terms in more than 95% of docs |

### Logistic Regression

| Parameter | Value |
|---|---|
| C | 2.0 |
| solver | liblinear |
| penalty | l2 |
| max_iter | 2000 |
| class_weight | balanced |

### LinearSVC (wrapped in `CalibratedClassifierCV`)

| Parameter | Value |
|---|---|
| C | 0.5 |
| max_iter | 2000 |
| class_weight | balanced |
| cv | 3 (for calibration) |

### XGBoost

| Parameter | Value |
|---|---|
| n_estimators | 250 |
| max_depth | 4 |
| learning_rate | 0.15 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| subsample | 0.85 |
| colsample_bytree | 0.85 |
| min_child_weight | 3 |
| tree_method | hist |
| eval_metric | mlogloss |

---

## 5. Airflow DAG — Task-Level Specification

**DAG ID:** `data_prep_pipeline`  
**Schedule:** `*/10 * * * *` (every 10 minutes)  
**Max active runs:** 1  
**Catchup:** False

| Task | Operator | Pool | Trigger rule | Retry |
|---|---|---|---|---|
| `wait_for_csv` | FileSensor | data_prep_pool | — | 2 |
| `find_latest_csv` | PythonOperator | data_prep_pool | ALL_SUCCESS | 2 |
| `validate_csv` | PythonOperator | data_prep_pool | ALL_SUCCESS | 2 |
| `clean_and_stats` | PythonOperator | data_prep_pool | ALL_SUCCESS | 2 |
| `detect_drift` | PythonOperator | data_prep_pool | ALL_SUCCESS | 2 |
| `archive` | PythonOperator | data_prep_pool | ALL_SUCCESS | 2 |
| `notify_stats` | PythonOperator | — | ALL_SUCCESS | 2 |
| `notify_broken_csv` | PythonOperator | — | ONE_FAILED | 2 |
| `notify_dry_pipeline` | PythonOperator | — | ALL_SKIPPED | 2 |

**Retry policy:** exponential backoff, 30s base, 3-minute cap.

**FileSensor config:**
- `filepath`: `/opt/airflow/data/incoming/*.csv`
- `poke_interval`: 60s
- `timeout`: 43,200s (12 hours)
- `mode`: `reschedule`
- `soft_fail`: `True`

**Pool:** `data_prep_pool`, size 3 — limits concurrent task slots to prevent resource contention.

**XCom keys used:**

| Key | Set by | Read by |
|---|---|---|
| `csv_path` | `find_latest_csv` | `validate_csv`, `clean_and_stats`, `archive` |
| `row_count` | `validate_csv` | — |
| `classes` | `validate_csv` | — |
| `batch_path` | `clean_and_stats` | `detect_drift`, `notify_stats` |
| `batch_stats` | `clean_and_stats` | `notify_stats` |
| `drift_score` | `detect_drift` | `notify_stats` |
| `drift_note` | `detect_drift` | `notify_stats` |

---

### DAG 2: `retrain_pipeline`

**DAG ID:** `retrain_pipeline`  
**Schedule:** `0 */2 * * *` (every 2 hours)  
**Max active runs:** 1  
**Catchup:** False

Polls `feedback.log` for user corrections. Skips the entire run if the threshold is not met; otherwise merges feedback into the processed dataset and retrains.

| Task | Operator | Trigger rule | Description |
|---|---|---|---|
| `check_feedback_threshold` | PythonOperator | ALL_SUCCESS | Count lines in feedback.log; raise `AirflowSkipException` if < `FEEDBACK_THRESHOLD` |
| `split_and_append_feedback` | PythonOperator | ALL_SUCCESS | Parse JSON lines (use `correct_label` as ground truth); stratified 80/20 split → append to train.csv / test.csv; archive and delete feedback.log |
| `get_champion_model_type` | PythonOperator | ALL_SUCCESS | Query MLflow for champion alias → extract model_type param; default `logreg` |
| `run_training` | PythonOperator | ALL_SUCCESS | `subprocess.run` → `python -m src.training.train --model {type}`; 35-minute timeout |
| `promote_model` | PythonOperator | ALL_SUCCESS | Compare all registered versions by `macro_f1`; promote winner to `champion` alias only if it beats the current champion |
| `notify_success` | PythonOperator | ALL_SUCCESS | Log retraining summary (rows added, promotion result) to notifications.log |
| `notify_failure` | PythonOperator | ONE_FAILED | Log alert to notifications.log when any core task fails |

**XCom keys used:**

| Key | Set by | Read by |
|---|---|---|
| `n_train` | `split_and_append_feedback` | `notify_success` |
| `n_test` | `split_and_append_feedback` | `notify_success` |
| `model_type` | `get_champion_model_type` | `run_training` |
| `promoted_version` | `promote_model` | `notify_success` |
| `promoted_type` | `promote_model` | `notify_success` |
| `promoted_f1` | `promote_model` | `notify_success` |

**Feedback log format** (`/app/data/feedback.log`, JSON lines):
```json
{
  "ts": 1777313261.256,
  "datetime": "2026-04-27 18:07:41",
  "text": "original input text",
  "predicted_label": "Depression",
  "correct_label": "Anxiety",
  "was_correct": false
}
```

---

## 6. Prometheus Metrics Schema

### Recording Rules (`recording_rules.yml`)

```promql
job:inference_latency_p95:2m =
    histogram_quantile(0.95,
        sum(rate(inference_latency_seconds_bucket[2m])) by (le))

job:request_rate:1m =
    sum(rate(http_requests_total[1m]))

job:error_rate:2m =
    (sum(rate(http_requests_total{status=~"5.."}[2m])) or vector(0))
    / sum(rate(http_requests_total[2m]))
```

### DNS Service Discovery for FastAPI replicas

```yaml
dns_sd_configs:
  - names: ["fastapi"]   # resolves to all 3 replica IPs
    type: A
    port: 8000
```

Prometheus scrapes `/metrics` on each resolved IP independently. All 3 replicas are aggregated in queries.

---

## 7. Nginx Configuration

```
upstream fastapi_backend {
    server fastapi:8000;   # Docker DNS resolves to all 3 replicas
}
# Round-robin by default; no sticky sessions
# All /predict, /feedback, /health, /metrics routed upstream
```

Each request carries the originating `container_id` in the response body, allowing load-balancing verification without Nginx access logs.

---

## 8. Docker Volumes and Mounts

| Service | Host path | Container path | Purpose |
|---|---|---|---|
| trainer | `./src` | `/app/src` | Training source code |
| trainer | `./scripts` | `/app/scripts` | promote_model.py |
| trainer | `./data` | `/app/data` | Train/test CSV, baseline stats |
| trainer | `./models` | `/app/models` | Saved joblib bundles |
| trainer | `./metrics` | `/app/metrics` | DVC metric JSON files |
| fastapi | `./src` | `/app/src` | API source code |
| fastapi | `./data` | `/app/data` | baseline_stats.json for drift |
| fastapi | `./models` | `/app/models` | Local fallback bundles |
| fastapi | `./data` | `/app/data` | baseline_stats.json, feedback.log |
| airflow-* | `./airflow/dags` | `/opt/airflow/dags` | DAG files |
| airflow-* | `./airflow/logs` | `/opt/airflow/logs` | Task logs |
| airflow-* | `./data` | `/opt/airflow/data` | incoming/, batches/, baseline, feedback.log |
| airflow-* | `./params.yaml` | `/opt/airflow/params.yaml` | Hyperparameters for retraining |
| airflow-* | `./models` | `/opt/airflow/models` | Joblib bundles written by retraining |
| airflow-* | `./metrics` | `/opt/airflow/metrics` | Metric JSON/PNG files written by retraining |
| mlflow | `./mlruns` | `/mlruns` | Experiment runs and artifacts |

---

## 9. Environment Variables

| Variable | Service | Default | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | trainer, fastapi | `http://mlflow:5000` | MLflow server URL |
| `MODEL_NAME` | fastapi | `mental_health_classifier` | Registry model name |
| `MODEL_ALIAS` | fastapi | `champion` | Alias to load |
| `BASELINE_STATS_PATH` | fastapi | `/app/data/baseline_stats.json` | Drift baseline file |
| `DRIFT_THRESHOLD` | fastapi | `0.3` | Alert threshold (informational) |
| `MAILTRAP_USER` | airflow | (from .env) | SMTP username |
| `MAILTRAP_PASS` | airflow | (from .env) | SMTP password |
| `ALERT_EMAIL` | airflow | (from .env) | Recipient address |
| `PYTHONPATH` | trainer, fastapi | `/app` | Python module root |
| `NLTK_DATA` | airflow, fastapi | `/home/airflow/nltk_data` | NLTK data directory |
| `FEEDBACK_LOG_PATH` | fastapi | `/app/data/feedback.log` | Override feedback log location |
| `FEEDBACK_THRESHOLD` | airflow | `10` | Min entries before retraining triggers |
| `MLFLOW_TRACKING_URI` | airflow | `http://mlflow:5000` | MLflow server (used by retrain DAG) |

---

## 10. CI/CD Workflow Specifications

### `ci.yml` — Runs on push and PR to `main`

**Job 1: unit-tests**
- Python 3.11, install pinned deps from requirements.txt subset
- Download NLTK stopwords, punkt, punkt_tab
- `pytest tests/unit/ -v`

**Job 2: docker-build**
- Generate `.env` from `.env.example` (dummy values)
- `docker compose config --quiet` — validates YAML
- Build fastapi, airflow, streamlit, mlflow images sequentially

### `integration.yml` — Runs on push to `main` and `workflow_dispatch`

1. Generate dummy `.env`
2. `docker compose up -d mlflow fastapi nginx`
3. Poll `/health` for up to 150 seconds
4. Install pytest + requests
5. `pytest tests/integration/ -v` (predict tests skip if no model loaded)
6. `docker compose logs` on failure
7. `docker compose down -v` always

---

## 11. Test Module Specifications

### `tests/unit/test_clean.py` (6 cases)
Tests `clean_text()`: lowercasing, URL removal, @mention removal, whitespace normalisation, non-string input, punctuation preservation.

### `tests/unit/test_drift.py` (3 cases)
Tests `DriftDetector`: returns 0.0 with empty window; low score for similar vocabulary; higher score for OOD vocabulary.

### `tests/unit/test_features.py` (3 cases)
Tests `HandcraftedFeatures` and `build_tfidf`: output shape (2, 12); all values finite; TF-IDF output shape matches input count.

### `tests/integration/test_api.py` (8 cases)
Tests live API: health, readiness, Prometheus metrics, predict validity, predict validation errors (empty, missing), feedback logging, load distribution (≥ 2 container IDs across 30 requests). Predict/load tests skip gracefully if model not loaded.

**Session fixture:** `model_ready` — checks `/ready` once per session; predict and load tests skip if False.
