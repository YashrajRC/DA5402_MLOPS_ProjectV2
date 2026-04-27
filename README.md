# Mental Health Text Classifier — MLOps Project

## What this is

A text-classification web app wrapped in a full MLOps stack:
Streamlit UI → Nginx → 3× FastAPI → MLflow models · Airflow data pipeline · Prometheus + Grafana monitoring · AlertManager emails · DVC versioning.

---

## Services and their URLs

| Service | URL | Login |
|---|---|---|
| Streamlit (the app) | http://localhost:8501 | — |
| FastAPI (via Nginx) | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |
| AlertManager | http://localhost:9093 | — |

---

## Running it — READ EVERY STEP

### 0. Prerequisites

- WSL2 with Docker Desktop running
- `~16 GB` free RAM recommended (Airflow + Postgres + all others)
- Git and a Python 3.11 venv (for DVC/training on host)

### 1. Drop the files into your project folder

Every file below is inside this zip. Unzip into an empty folder and `cd` into it.

### 2. Get the dataset

Download the Kaggle "Sentiment Analysis for Mental Health" CSV
→ place it at `data/raw/data.csv`

(The column names are auto-detected — `statement`/`status` or `text`/`label`.)

### 3. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and fill in Mailtrap credentials + your alert email.
(Sign up at https://mailtrap.io — free. "Email Testing" → "Inboxes" → "SMTP Settings".)

Also edit `configs/alertmanager.yml` — replace the three `REPLACE_WITH_...` placeholders.

### 4. Install host venv and initialise DVC

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git init
dvc init
git add .
git commit -m "initial commit"

# Track raw data with DVC
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc data/.gitignore
git commit -m "track raw dataset"
git tag data-v0-raw
```

### 5. Start the infrastructure (ORDER MATTERS)

Start in this order — each wait ensures the next step works:

```bash
# A) MLflow + monitoring stack first (no app dependencies)
docker compose up -d mlflow prometheus node-exporter alertmanager grafana

# Wait ~20 seconds. Verify:
#   http://localhost:5000 loads
#   http://localhost:9090 loads (Status → Targets should show mlflow eventually; don't worry if fastapi is DOWN — we haven't started it)
#   http://localhost:3001 loads
```

### 6. Run the DVC pipeline (trains all 3 models)

This runs on your HOST venv, not in Docker. It logs to the MLflow container.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
dvc repro
```

You should see three training runs complete. Now check MLflow UI — all three appear.

### 7. Promote the best model to Production

```bash
python scripts/promote_model.py
```

Verify in MLflow UI → Models → `mental_health_classifier` → a version is tagged **Production**.

### 8. Commit DVC outputs

```bash
git add dvc.lock metrics/ params.yaml data/baseline_stats.json
git commit -m "trained 3 models"
git tag data-v1-processed
```

### 9. Start the app layer

```bash
docker compose up -d fastapi nginx streamlit
```

Wait ~30 seconds for FastAPI replicas to load the model from MLflow.

Verify:
- `curl http://localhost:8000/health` → `{"status":"ok",...}`
- `curl http://localhost:8000/ready` → `{"status":"ready",...}`
- Open http://localhost:8501 → type text → click Analyze

### 10. Start Airflow

```bash
docker compose up -d airflow-postgres
sleep 10
docker compose up -d airflow-init
# wait for airflow-init to exit with success
docker compose up -d airflow-webserver airflow-scheduler
```

Open http://localhost:8080 (admin / admin). Two DAGs should be listed and unpaused:
- `data_prep_pipeline` — FileSensor for incoming CSV batches, drift detection, notifications
- `retrain_pipeline` — polls feedback log every 2 hours; retrains when ≥ 10 user corrections accumulate

### 11. Trigger the Airflow pipeline

Drop a sample CSV into the incoming folder:

```bash
bash scripts/drop_sample_csv.sh
```

The FileSensor polls every 60s. Within ~2 minutes you should see the DAG run and an email arrive in your Mailtrap inbox.

---

## Proving each requirement works

### Load balancing across FastAPI replicas

```bash
python scripts/load_tester.py
```

Output shows requests distributed across different `container_id`s.

### Data drift detection & alert

```bash
python scripts/simulate_drift.py
```

Within 1-2 minutes:
- Grafana drift panel turns red
- AlertManager receives `DataDriftDetected`
- Mailtrap inbox gets an email

### Broken CSV alert

```bash
echo "this is not a valid csv" > data/incoming/broken.csv
```

Airflow DAG picks it up → `validate_csv` fails → `email_broken_csv` fires.

### Dry pipeline alert

Let the sensor time out (takes 12h by default — can be lowered in the DAG for testing).

### Feedback-triggered retraining

Use the Streamlit UI to submit ≥ 10 feedback corrections, then trigger `retrain_pipeline` manually in Airflow (or wait up to 2 hours for the scheduled run). The DAG appends feedback rows (80/20 split) to `data/processed/train.csv` and `test.csv`, retrains the champion model type, and promotes if the new version scores higher.

### Running tests

```bash
# Unit (no services needed)
pytest tests/unit/

# Integration (all services must be up)
pytest tests/integration/
```

---

## Stopping and cleaning up

```bash
docker compose down        # stop all, keep volumes
docker compose down -v     # stop + delete volumes (fresh start)
```

---

## Project structure

```
.
├── airflow/dags/
│   ├── data_prep_pipeline.py              # FileSensor → validate → clean → drift → archive → notify
│   └── retrain_pipeline.py                # feedback threshold → split → retrain → promote → notify
├── configs/
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   ├── recording_rules.yml
│   ├── alertmanager.yml
│   └── grafana/                           # auto-provisioned dashboard
├── docker/
│   ├── fastapi/    Dockerfile + requirements.txt
│   ├── streamlit/  Dockerfile + requirements.txt
│   ├── mlflow/     Dockerfile
│   ├── airflow/    Dockerfile (adds project deps)
│   └── nginx/      nginx.conf
├── docs/
│   ├── hld.md                             # High-level design
│   ├── lld.md                             # API spec
│   ├── test_plan.md
│   ├── test_cases.md
│   ├── test_report.md
│   ├── user_manual.md
│   └── experiment_report.md
├── src/
│   ├── api/     main.py schemas.py metrics.py drift.py model_client.py
│   ├── data/    clean.py prepare.py
│   ├── training/ features.py train.py
│   └── frontend/app.py
├── scripts/
│   ├── promote_model.py
│   ├── load_tester.py
│   ├── simulate_drift.py
│   ├── drop_sample_csv.sh
│   └── sample_incoming.csv
├── tests/
│   ├── unit/
│   └── integration/
├── dvc.yaml      # DVC pipeline
├── params.yaml   # hyperparameters
├── MLproject     # MLflow project file
├── docker-compose.yml
└── .env.example
```

---

## Troubleshooting

**FastAPI crashes with "No model available"**
→ You haven't run `dvc repro` + `promote_model.py` yet, or the MLflow artifact store isn't reachable from the FastAPI container. Check `docker logs fastapi-1`.

**Airflow webserver won't start**
→ airflow-init probably didn't finish. Run `docker compose logs airflow-init` and look for errors.

**Grafana dashboard is empty**
→ Prometheus isn't scraping FastAPI. Check http://localhost:9090/targets — fastapi job should be UP. If DOWN, the FastAPI containers aren't running.

**Mailtrap emails not arriving**
→ Check `configs/alertmanager.yml` has real credentials, not placeholders. Also check airflow's SMTP env vars in `docker-compose.yml`.

**`dvc repro` fails with MLflow connection error**
→ Make sure you did `export MLFLOW_TRACKING_URI=http://localhost:5000` in the same shell.

**Port already in use**
→ Something else is on 8000/8080/5000/etc. Stop it or change the port mapping in `docker-compose.yml`.

**`retrain_pipeline` task fails with PermissionError**
→ The Airflow user (UID 50000) can't write to files created by root-owned containers. Fix once after initial training:
```bash
docker exec trainer bash -c "chmod 666 /app/metrics/* /app/models/*.joblib /app/data/processed/*.csv 2>/dev/null; true"
docker exec project-fastapi-1 chmod 666 /app/data/feedback.log 2>/dev/null; true
```
