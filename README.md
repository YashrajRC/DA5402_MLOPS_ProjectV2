# Mental Health Text Classifier: DA5402 MLOps Project

**Yashraj Ramdas Chavan, DA25M031**

---

## What this is

A text-classification web app wrapped in a full MLOps stack:
Streamlit UI → Nginx → 3× FastAPI → MLflow models · Airflow data pipeline · Prometheus + Grafana monitoring · AlertManager emails · DVC versioning.

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
│   ├── alertmanager.yml                   # uses ${MAILTRAP_USER/PASS/ALERT_EMAIL} from .env
│   ├── alertmanager-entrypoint.sh
│   └── grafana/                           # auto-provisioned dashboard + datasource
├── docker/
│   ├── fastapi/    Dockerfile + requirements.txt   (also used by trainer)
│   ├── streamlit/  Dockerfile + requirements.txt
│   ├── mlflow/     Dockerfile
│   ├── airflow/    Dockerfile (adds project deps)
│   └── nginx/      nginx.conf
├── docs/
│   ├── architecture.md
│   ├── hld.md
│   ├── lld.md
│   ├── test_plan.md
│   ├── test_cases.md
│   ├── test_report.md
│   ├── user_manual.md
│   └── experiment_report.md
├── src/
│   ├── api/        main.py  schemas.py  metrics.py  drift.py  model_client.py
│   ├── data/       clean.py  prepare.py
│   ├── training/   features.py  train.py
│   ├── frontend/   app.py
│   └── utils/
├── scripts/
│   ├── promote_model.py
│   ├── load_tester.py
│   ├── simulate_drift.py
│   ├── drop_sample_csv.sh
│   └── sample_incoming.csv
├── tests/
│   ├── unit/       test_clean.py  test_drift.py  test_features.py
│   └── integration/ test_api.py
├── dvc.yaml        # pipeline: prepare → train×3 → promote (via docker exec trainer)
├── params.yaml     # hyperparameters
├── MLproject       # MLflow project entry points
├── docker-compose.yml
└── .env.example
```

---

## Services and their URLs

| Service | URL | Login |
|---|---|---|
| Streamlit (the app) | http://localhost:8501 | — |
| FastAPI (via Nginx) | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin123 |
| AlertManager | http://localhost:9093 | — |

---

## How to run

### 0. Prerequisites

- WSL2 with Docker Desktop running
- `~16 GB` free RAM recommended
- Git and a Python 3.11 venv (for DVC on host)

### 1. Get the dataset

Download the Kaggle "Sentiment Analysis for Mental Health" CSV → place it at `data/raw/data.csv`.

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` with your Mailtrap credentials and alert email (sign up free at https://mailtrap.io → "Email Testing" → "Inboxes" → "SMTP Settings"). The AlertManager container reads these at startup — no other file needs editing.

### 3. Install host venv and initialise DVC

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

git init && dvc init
git add . && git commit -m "initial commit"

dvc add data/raw/data.csv
git add data/raw/data.csv.dvc data/.gitignore
git commit -m "track raw dataset" && git tag data-v0-raw
```

### 4. Start Airflow (data ingestion)

```bash
docker compose up -d airflow-postgres
sleep 10
docker compose up -d airflow-init
# wait for airflow-init to exit successfully
docker compose up -d airflow-webserver airflow-scheduler
```

Open http://localhost:8080 (admin / admin). Two DAGs should be visible and unpaused:
- `data_prep_pipeline` — watches `data/incoming/` for new CSVs, validates, cleans, checks for drift, notifies
- `retrain_pipeline` — retrains automatically when ≥ 10 user feedback corrections accumulate

Trigger a test run by dropping a sample batch:

```bash
bash scripts/drop_sample_csv.sh
```

### 5. Start MLflow + monitoring, then train

```bash
docker compose up -d mlflow trainer prometheus node-exporter alertmanager grafana
```

Wait ~20 seconds, then run the DVC pipeline (runs inside the `trainer` container):

```bash
dvc repro
```

Stages run in order: **prepare** → **train_logreg** → **train_linearsvc** → **train_xgboost** → **promote** (best model registered as `Production` in MLflow).

Commit the outputs:

```bash
git add dvc.lock metrics/ params.yaml data/baseline_stats.json
git commit -m "trained 3 models" && git tag data-v1-processed
```

### 6. Start the app layer

```bash
docker compose up -d fastapi nginx streamlit
```

Wait ~30 seconds, then verify:

```bash
curl http://localhost:8000/health   # {"status":"ok",...}
```

Open http://localhost:8501 → type some text → click Analyze.

---

## Testing

```bash
pytest tests/unit/        # no services needed
pytest tests/integration/ # all services must be up
```

---

## Stopping and cleaning up

```bash
docker compose down        # stop all, keep volumes
docker compose down -v     # stop + delete volumes (fresh start)
```

---