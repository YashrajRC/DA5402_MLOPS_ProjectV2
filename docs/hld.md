# High-Level Design — Mental Health Text Classifier

## 1. Business Problem

Mental health platforms receive thousands of user posts daily across social media, journals, and peer-support channels. Manual triage is impossible at scale, and mis-routing a user in crisis to the wrong resource has real human cost.

## 2. Desired Outcome

An automated triage system that classifies text into 7 mental health categories (Normal, Depression, Anxiety, Bipolar, PTSD, Stress, Personality Disorder) with sufficient accuracy and speed to route users to appropriate resources in real time — **explicitly positioned as a pattern-detection tool, not a diagnostic one**.

## 3. Success Criteria

| Type | Metric | Target |
|---|---|---|
| ML | Macro F1-score | ≥ 0.75 |
| ML | Per-class F1 (minority) | ≥ 0.60 |
| Business | p95 inference latency | < 200ms |
| Business | API availability | ≥ 99% |
| Business | Throughput | ≥ 50 req/s |
| Business | Drift detection lag | < 1 hour |
| Business | Error rate | < 5% |

## 4. Architecture Overview

```
┌──────────┐         ┌─────────┐      ┌──────────────┐
│ Streamlit│──HTTP──▶│  Nginx  │─────▶│ FastAPI × 3  │
└──────────┘         └─────────┘      └──────┬───────┘
                                             │ REST
                                      ┌──────▼──────┐
                                      │   MLflow    │
                                      │  (models,   │
                                      │  registry)  │
                                      └─────────────┘
                                             │ scrape /metrics
┌────────────┐   ┌──────────┐        ┌──────▼──────┐
│ AlertMgr   │◀──┤Prometheus├───────▶│  Grafana    │
│ (email)    │   │          │        │ (dashboards)│
└────────────┘   └────┬─────┘        └─────────────┘
                      │ scrape
                ┌─────▼────────┐
                │ node-exporter│
                └──────────────┘

 Airflow (separate DAG) → prepares data → feeds training + baseline stats
```

## 5. Technology Choices & Rationale

| Layer | Tool | Why |
|---|---|---|
| Frontend | Streamlit | Fast to build, no HTML/JS required, good for data apps |
| API | FastAPI | Async, auto-docs, Pydantic validation, Prometheus-friendly |
| Load balancer | Nginx | Proves load distribution across FastAPI replicas |
| Model tracking | MLflow | Registry + artifact store + REST serving in one tool |
| Data orchestration | Airflow | Required by rubric; FileSensor + Pool + SMTP built-in |
| Data versioning | DVC | Git-compatible, reproducible via `dvc repro` |
| Metrics | Prometheus | Pull-based, works with multi-replica via DNS SD |
| Dashboards | Grafana | Standard pairing with Prometheus |
| Alerting | AlertManager + Mailtrap | Email-based alerts without needing a real SMTP server |
| Packaging | Docker Compose | Single-command boot for all services |

## 6. Loose Coupling

The frontend and backend communicate **only** via REST. The frontend never imports model code, never touches MLflow directly, and never reads `models/`. This means:
- Either side can be replaced independently
- The backend can be scaled horizontally (we run 3 replicas)
- The API contract is the only integration point

## 7. Model Strategy

Three model families, same feature extraction (TF-IDF + handcrafted features):

1. **Logistic Regression** — linear probabilistic baseline; calibrated probabilities
2. **LinearSVC** — linear margin-based; wrapped in `CalibratedClassifierCV` for probabilities
3. **XGBoost** — tree ensemble; captures non-linear interactions

Handcrafted features: text length, avg word length, punctuation density, uppercase ratio, first-person pronoun ratio, `!`/`?` counts, VADER sentiment (compound, pos, neg, neu).

After training, all three are logged to MLflow. The best (by macro-F1) is promoted to the `Production` stage via `scripts/promote_model.py`.

## 8. Drift Detection

- During data prep, we compute word-frequency distribution (top 1000) on training data → `baseline_stats.json`
- FastAPI maintains a rolling window of the last 200 input texts
- Every `/predict` call computes Jensen-Shannon divergence between the current window's word-freq and the baseline
- Divergence exposed as a Prometheus gauge (`drift_score`)
- Grafana alert fires if `drift_score > 0.3` for 1 minute
- Airflow DAG also computes drift per batch and includes it in the stats email

## 9. Observability Strategy

**Counters:** `http_requests_total`, `predictions_total`, `errors_total`, `feedback_total`
**Gauges:** `active_requests`, `drift_score`, `model_version_active`
**Histograms:** `inference_latency_seconds`, `input_text_length_chars`
**Summary:** `rolling_accuracy` (from feedback)

All metrics carry custom labels (endpoint, method, status, predicted_class, instance_id) enabling multi-dimensional slicing in Grafana.

## 10. Alerting Strategy

| Alert | Condition | Severity | Justification |
|---|---|---|---|
| HighErrorRate | error rate > 5% for 2m | critical | Business SLA |
| HighInferenceLatency | p95 > 200ms for 2m | warning | Business SLA |
| DataDriftDetected | drift_score > 0.3 for 1m | warning | Indicates model degradation risk |
| ModelServingDown | up{job=fastapi} == 0 for 1m | critical | Full outage |
| HighCPUUsage | CPU > 80% for 2m | warning | Resource exhaustion risk |

## 11. Security & Secrets

- All SMTP credentials loaded from `.env` — never committed
- AlertManager configured with TLS to SMTP relay (Mailtrap)
- No real user data in the dataset (public Reddit posts)
