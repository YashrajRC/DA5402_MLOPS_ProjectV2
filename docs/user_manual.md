# User Manual
## Mental Health Text Classifier

---

> **Important Disclaimer**
> This tool is a **pattern-detection system, not a clinical diagnostic instrument**. Results must not replace evaluation by a qualified mental health professional. If you or someone you know is in crisis, please contact a licensed professional or a crisis helpline immediately.

---

## 1. Overview

The Mental Health Text Classifier is a web application that reads a piece of text and identifies which of seven mental health patterns it most closely resembles. It can process journal entries, social media posts, or any free-form written expression in English.

**Seven output categories:**

| Category | Description |
|---|---|
| Normal | No significant mental health indicator detected |
| Depression | Persistent low mood, hopelessness, loss of interest |
| Anxiety | Excessive worry, panic, fear of situations |
| Bipolar | Cycling between elevated and depressed mood states |
| Suicidal | Thoughts of self-harm or ending one's life |
| Stress | Pressure from external demands (work, relationships, deadlines) |
| Personality Disorder | Unstable identity, relationship extremes, emotional dysregulation |

---

## 2. Opening the Application

1. Make sure the system is running (`docker compose up -d` from the project directory)
2. Open your browser and go to **http://localhost:8501**
3. You will see the main interface with three tabs at the top

---

## 3. Tab 1 — Analyze (Main Feature)

### Step 1: Enter your text

Click the large text box labelled **"Enter text to analyze"** and type or paste any text. The character counter below updates as you type. Minimum 1 character, maximum 10,000 characters.

**Example texts to try:**

| To test this class | Try this text |
|---|---|
| Depression | "I haven't left my bed in days. Everything feels pointless and I have no energy to do anything." |
| Anxiety | "My heart races every morning. I can't stop worrying about everything that could go wrong." |
| Suicidal | "I don't want to be alive anymore. I've been thinking about ending it all." |
| Normal | "Had a great day today. Went for a walk, had coffee with a friend, feeling good." |
| Stress | "Too many deadlines, not enough time. My boss keeps adding more work and I can't cope." |

### Step 2: Click Analyze

Press the **Analyze** button. Results appear within 1–2 seconds.

### Step 3: Read your results

The result section shows:

- **Predicted category** — the most likely classification, shown in a coloured banner
- **Confidence score** — how certain the model is (0–100%). Higher is more reliable.
- **Probability chart** — a horizontal bar chart showing scores for all 7 classes. Hover over a bar to see the exact percentage.
- **Model info** — which model version produced this prediction (shown at the bottom)

### Step 4: Give feedback (optional)

Below the result, two buttons appear:
- **✓ Correct** — click if the prediction matched your expectation
- **✗ Wrong** — click if the prediction was wrong; a dropdown appears so you can select the correct label

Feedback is logged and used to track model accuracy over time. It does not immediately change the model.

---

## 4. Tab 2 — Dashboard

Shows live operational statistics. No interaction needed — it refreshes automatically.

### Panel: Model Information

| Field | What it means |
|---|---|
| Model type | Algorithm family (XGBoost, Logistic Regression, or LinearSVC) |
| Version | MLflow registry version number |
| Stage | Alias (champion = currently serving) |
| Test accuracy | Accuracy on held-out test set |
| Test macro-F1 | F1 averaged equally across all 7 classes |
| Val accuracy | Accuracy on validation set (held out during training) |
| Val macro-F1 | F1 on validation set |

### Panel: Live Statistics

| Metric | What it means |
|---|---|
| Total predictions | Number of /predict calls since last service restart |
| Avg latency (ms) | Average response time per request |
| Drift score | How different current input vocabulary is from training data (0 = same, 1 = completely different) |
| Feedback count | Number of feedback submissions received |

### Panel: Class Distribution (Training Data)

Pie chart showing how many training examples exist per class. Use this to understand model bias — classes with fewer examples (Personality Disorder at 2%) are harder for the model to detect reliably.

---

## 5. Tab 3 — User Manual

Displays this document in-app for quick reference.

---

## 6. For Operators — Monitoring

### Grafana Dashboard

Open **http://localhost:3001** (login: admin / admin123).

The main dashboard shows:
- **Request Rate** — requests per second over the last minute
- **p95 Latency** — 95th-percentile response time (SLA: < 200ms)
- **Error Rate** — fraction of 5xx responses (target: < 5%)
- **Drift Score** — current rolling-window drift gauge

**What the panels mean:**

| Panel value | Interpretation |
|---|---|
| Request rate = 0 | No traffic — normal in idle periods |
| p95 latency > 200ms | Model may be under load or cold; investigate |
| Error rate > 5% | API issues — check FastAPI logs immediately |
| Drift score > 0.3 | Input vocabulary has shifted from training data |

### Prometheus

Open **http://localhost:9090** and try these queries:

```promql
# Requests per second
job:request_rate:1m

# p95 inference latency
job:inference_latency_p95:2m

# Error rate
job:error_rate:2m

# Drift score
drift_score

# Prediction class distribution
sum by (predicted_class) (predictions_total)
```

---

## 7. For Operators — Airflow (Data Pipeline)

Open **http://localhost:8080** (login: admin / admin).

The `data_prep_pipeline` DAG runs every 10 minutes. It watches the `data/incoming/` folder for CSV files.

### Dropping a new data batch

Place a CSV file in `data/incoming/`. The file must have columns `text` and `label` (or Kaggle format: `statement` and `status`). The sensor picks it up within 60 seconds.

```bash
cp your_data.csv data/incoming/batch_$(date +%s).csv
```

### What happens to the file

1. **wait_for_csv** — sensor detects the file
2. **find_latest_csv** — identifies the most recently modified file
3. **validate_csv** — checks columns, row count; rejects empty or malformed files
4. **clean_and_stats** — cleans text, writes a cleaned batch to `data/batches/`
5. **detect_drift** — computes JS divergence vs training baseline; logs score and warning flag
6. **archive** — moves original file to `data/incoming_archive/`
7. **notify_stats** — sends email (if SMTP configured) or writes to `airflow/logs/notifications.log`

### Reading notification logs (when email is not configured)

```bash
cat airflow/logs/notifications.log
```

### Triggering a broken-CSV alert (for demo)

```bash
printf "wrong,columns\na,b\n" > data/incoming/broken_test.csv
```

The `validate_csv` task will fail and `notify_broken_csv` will fire.

---

## 8. For Operators — Retraining the Model

Run the full DVC training pipeline:

```bash
# Retrain all three models
docker compose exec -T trainer mlflow run /app -e train -P model=logreg --env-manager local --experiment-name mental_health_classifier
docker compose exec -T trainer mlflow run /app -e train -P model=linearsvc --env-manager local --experiment-name mental_health_classifier
docker compose exec -T trainer mlflow run /app -e train -P model=xgboost --env-manager local --experiment-name mental_health_classifier

# Promote the best one
docker compose exec -T trainer python /app/scripts/promote_model.py

# Restart FastAPI to load the new champion
docker compose restart fastapi
```

View all training runs at **http://localhost:5000**.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Streamlit shows "API unreachable" | FastAPI not running | `docker compose restart fastapi` |
| First prediction takes 5+ seconds | Model loading on first request | Wait for `/ready` to return 200 |
| Airflow DAG not visible | Scheduler permission issue | `docker compose restart airflow-scheduler` |
| Airflow `wait_for_csv` stuck in `up_for_reschedule` | No CSV in `data/incoming/` | Normal behaviour — drop a CSV to trigger |
| Grafana shows "No data" | No traffic yet | Generate a prediction first |
| Email alerts not arriving | SMTP not configured in `.env` | See `airflow/logs/notifications.log` for fallback |
| `dvc repro` fails with git-tracking error | A DVC output is committed to git | `git rm --cached <file>` then re-run |
| Drift score always > 0.3 | Rolling window smaller than baseline sample | Expected; see drift calibration note in HLD §8 |

---

## 10. Service URLs Reference

| Service | URL | Credentials |
|---|---|---|
| Streamlit (main UI) | http://localhost:8501 | None |
| FastAPI (direct) | http://localhost:8000 | None |
| FastAPI Swagger docs | http://localhost:8000/docs | None |
| MLflow | http://localhost:5000 | None |
| Airflow | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3001 | admin / admin123 |
| Prometheus | http://localhost:9090 | None |
| AlertManager | http://localhost:9093 | None |
