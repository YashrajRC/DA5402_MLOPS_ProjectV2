# Architecture Diagrams

## 1. System Overview

```mermaid
graph TB
    User([End User])

    subgraph Ingest["Data Ingestion & Retraining — Airflow :8080"]
        AF[data_prep_pipeline]
        RT[retrain_pipeline]
    end

    subgraph Train["Training Pipeline — cached by DVC"]
        DVC[DVC dvc repro]
        MLB[MLflow :5000]
        DVC -->|log metrics + register| MLB
    end

    BASE[(baseline_stats.json)]

    subgraph Serve["Serving — Nginx :8000"]
        FA[FastAPI x3 replicas]
    end

    subgraph Observe["Observability"]
        PR[Prometheus :9090]
        GR[Grafana :3001]
        AM[AlertManager :9093]
        PR --> GR
        PR --> AM
    end

    ST[Streamlit :8501]

    DVC -->|produces| BASE
    BASE -->|batch drift baseline| AF
    BASE -->|live drift baseline| FA
    AF -->|cleaned batches| DVC
    FA -->|feedback.log| RT
    RT -->|append CSV + retrain| DVC
    MLB -->|champion bundle| FA
    FA -->|/metrics scrape| PR
    User --> ST
    ST -->|REST via Nginx| Serve
    User -->|REST| Serve
```

---

## 2. Data Flow

```mermaid
flowchart LR
    RAW[(data/raw/data.csv)]

    subgraph DVC["DVC Pipeline — stage-cached"]
        PREP[prepare]
        LR[train_logreg]
        SV[train_linearsvc]
        XG[train_xgboost]
        PRO[promote]
        PREP --> LR & SV & XG --> PRO
    end

    RAW --> PREP
    PRO -->|champion alias| MLB[(MLflow Registry)]
    PREP -->|baseline_stats.json| BASE[(baseline_stats.json)]

    INCOMING[(incoming/*.csv)]

    subgraph AF["Airflow — data_prep_pipeline (every 10 min)"]
        V[validate] --> C[clean] --> D[detect drift] --> A[archive]
    end

    INCOMING -->|FileSensor| AF
    BASE -->|drift baseline| D
    C -.->|cleaned batches| PREP

    MLB -->|champion bundle| FA2

    subgraph FA2["FastAPI x3 (Nginx :8000)"]
        P["/predict /feedback /metrics"]
    end

    BASE -->|live drift baseline| FA2
    User([User]) <-->|text in / class out| FA2
    FA2 -->|metrics| PR[Prometheus :9090]
    PR --> GR[Grafana :3001]
    PR -->|fire alerts| AM[AlertManager :9093]
```

---

## 3. Airflow DAG

```mermaid
flowchart TD
    START([DAG start / 10 min]) --> SENSOR

    SENSOR[wait_for_csv FileSensor]
    SENSOR -->|file found| FIND
    SENSOR -->|12h timeout| DRY

    FIND[find_latest_csv]
    FIND --> VALIDATE

    VALIDATE[validate_csv]
    VALIDATE -->|valid| CLEAN
    VALIDATE -->|invalid| BROKEN

    CLEAN[clean_and_stats]
    CLEAN --> DRIFT

    DRIFT[detect_drift JSD]
    DRIFT --> ARCHIVE

    ARCHIVE[archive]
    ARCHIVE --> NOTIFY

    NOTIFY[notify_stats]

    BROKEN[notify_broken_csv]
    DRY[notify_dry_pipeline]

    style SENSOR fill:#f0f4ff,stroke:#4a6cf7,color:#1a1a1a
    style VALIDATE fill:#f0f4ff,stroke:#4a6cf7,color:#1a1a1a
    style DRIFT fill:#fff3cd,stroke:#f0ad4e,color:#1a1a1a
    style BROKEN fill:#ffeaea,stroke:#dc3545,color:#1a1a1a
    style DRY fill:#ffeaea,stroke:#dc3545,color:#1a1a1a
    style NOTIFY fill:#e8f5e9,stroke:#28a745,color:#1a1a1a
```

---

## 3b. Airflow DAG — `retrain_pipeline`

```mermaid
flowchart TD
    START([DAG start / 2 hours]) --> CHECK

    CHECK[check_feedback_threshold]
    CHECK -->|under threshold| SKIP([DAG skipped])
    CHECK -->|threshold met| SPLIT

    SPLIT[split_and_append_feedback]
    SPLIT --> GETTYPE

    GETTYPE[get_champion_model_type]
    GETTYPE --> TRAIN

    TRAIN[run_training]
    TRAIN --> PROMOTE

    PROMOTE[promote_model]
    PROMOTE --> NOTIFY

    NOTIFY[notify_success]

    FAIL[notify_failure]

    SPLIT & GETTYPE & TRAIN & PROMOTE -->|any task fails| FAIL

    style CHECK fill:#f0f4ff,stroke:#4a6cf7,color:#1a1a1a
    style SKIP fill:#f5f5f5,stroke:#aaa,color:#1a1a1a
    style TRAIN fill:#fff3cd,stroke:#f0ad4e,color:#1a1a1a
    style PROMOTE fill:#e8f5e9,stroke:#28a745,color:#1a1a1a
    style NOTIFY fill:#e8f5e9,stroke:#28a745,color:#1a1a1a
    style FAIL fill:#ffeaea,stroke:#dc3545,color:#1a1a1a
```

---

## 4. DVC Pipeline

```mermaid
flowchart LR
    RAW[(data/raw/data.csv)]

    subgraph prepare["Stage: prepare"]
        P[prepare.py]
    end

    subgraph train_logreg["Stage: train_logreg"]
        LR[LogReg TF-IDF]
    end

    subgraph train_linearsvc["Stage: train_linearsvc"]
        SV[LinearSVC TF-IDF]
    end

    subgraph train_xgboost["Stage: train_xgboost"]
        XG[XGBoost TF-IDF]
    end

    subgraph promote["Stage: promote"]
        PRO[promote_model.py]
    end

    TRAIN[(train.csv)]
    TEST[(test.csv)]
    BASE[(baseline_stats.json)]

    RAW --> P --> TRAIN & TEST & BASE
    TRAIN & TEST --> LR & SV & XG
    LR -->|logreg metrics + bundle| PRO
    SV -->|linearsvc metrics + bundle| PRO
    XG -->|xgboost metrics + bundle| PRO
```

---

## 5. CI/CD Pipeline

```mermaid
flowchart TD
    PR([Push / Pull Request])

    subgraph CI["CI Workflow — ci.yml"]
        UT[Unit Tests 12]
        DB[Docker Build]
        UT & DB
    end

    subgraph INT["Integration Workflow — integration.yml"]
        SVC[docker compose up]
        WAIT[Wait /health 200]
        IT[Integration Tests 8]
        DOWN[docker compose down]
        SVC --> WAIT --> IT --> DOWN
    end

    PR --> CI
    PR -->|push to main only| INT

    CI -->|all pass| MERGE([Merge allowed])
    CI -->|any fail| BLOCK([Merge blocked])
    INT -->|results| LOG([Actions log])
```

---

## 6. Feature Engineering Pipeline

```mermaid
flowchart LR
    TEXT([Raw text]) --> CLEAN[clean_text]

    CLEAN --> W[Word TF-IDF]
    CLEAN --> C[Char TF-IDF]
    CLEAN --> H[Handcrafted 12]

    H --> H1[length / word count]
    H --> H2[punct / upper ratio]
    H --> H3[excl / question cnt]
    H --> H4[VADER sentiment]

    W --> HSTACK[hstack 40k features]
    C --> HSTACK
    H --> HSTACK

    HSTACK --> CLF[Classifier]
    CLF --> PRED([7-class prediction])
```
