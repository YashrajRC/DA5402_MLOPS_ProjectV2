# Architecture Diagrams

## 1. System Overview

```mermaid
graph TB
    User([End User])

    subgraph Ingest["Data Ingestion  —  Airflow :8080"]
        AF["FileSensor → validate → clean\n→ drift detect → archive → notify"]
    end

    subgraph Train["Training Pipeline  —  run once, cached by DVC"]
        DVC["DVC  (dvc repro)\nonly re-runs stages whose\ndeps or params changed"]
        MLB["MLflow  :5000\nexperiment tracking\nmodel registry"]
        DVC -->|"log metrics & bundles\nregister versions"| MLB
    end

    BASE[("baseline_stats.json\nshared drift baseline")]

    subgraph Serve["Serving  —  Nginx :8000"]
        FA["FastAPI × 3 replicas\npredict · feedback · metrics\nlive drift detection"]
    end

    subgraph Observe["Observability"]
        PR["Prometheus  :9090"]
        GR["Grafana  :3001"]
        AM["AlertManager  :9093"]
        PR --> GR
        PR --> AM
    end

    ST["Streamlit  :8501"]

    DVC -->|"produces"| BASE
    BASE -->|"batch drift baseline"| AF
    BASE -->|"live drift baseline"| FA
    AF -->|"cleaned batches\ntrigger retraining when needed"| DVC
    MLB -->|"champion model bundle\nloaded at startup"| FA
    FA -->|"/metrics scrape"| PR
    User --> ST
    ST -->|"REST via Nginx"| Serve
    User -->|"REST"| Serve
```

---

## 2. Data Flow

```mermaid
flowchart LR
    RAW([("data/raw/data.csv\nKaggle dataset")])

    subgraph DVC["DVC Pipeline  —  stage-cached, only reruns on change"]
        PREP["prepare\nclean · split · compute baseline\n─────────────────\nout: train.csv · test.csv\n     baseline_stats.json"]
        LR["train_logreg\nTF-IDF + handcrafted → LogReg\nval split 15% · logs to MLflow"]
        SV["train_linearsvc\nTF-IDF + handcrafted → SVC\nval split 15% · logs to MLflow"]
        XG["train_xgboost\nTF-IDF + handcrafted → XGBoost\nval split 15% · logs to MLflow"]
        PRO["promote\npick best macro-F1\nset champion alias"]
        PREP --> LR & SV & XG --> PRO
    end

    RAW --> PREP
    PRO -->|"champion alias"| MLB[(MLflow\nRegistry)]
    PREP -->|"baseline_stats.json"| BASE[("baseline_stats.json")]

    INCOMING([("data/incoming/*.csv\nnew data batches")])

    subgraph AF["Airflow  —  data_prep_pipeline  (every 10 min)"]
        V[validate] --> C[clean] --> D["detect drift\nJSD vs baseline"] --> A[archive]
    end

    INCOMING -->|"FileSensor"| AF
    BASE -->|"drift baseline"| D
    C -.->|"cleaned batches\nfor future retraining"| PREP

    MLB -->|"champion bundle"| FA2

    subgraph FA2["FastAPI × 3  (Nginx :8000)"]
        P["/predict · /feedback\n/metrics · /model_info"]
    end

    BASE -->|"live drift baseline"| FA2
    User([User]) <-->|"text in\nclass + confidence out"| FA2
    FA2 -->|"metrics"| PR[Prometheus\n:9090]
    PR --> GR[Grafana\n:3001]
    PR -->|"fire alerts"| AM[AlertManager\n:9093]
```

---

## 3. Airflow DAG

```mermaid
flowchart TD
    START([DAG start\nevery 10 min]) --> SENSOR

    SENSOR[wait_for_csv\nFileSensor\ndata/incoming/*.csv]
    SENSOR -->|file found| FIND
    SENSOR -->|12h timeout\nsoft_fail| DRY

    FIND[find_latest_csv\nPythonOperator]
    FIND --> VALIDATE

    VALIDATE[validate_csv\ncheck columns &\nrow count]
    VALIDATE -->|valid| CLEAN
    VALIDATE -->|invalid| BROKEN

    CLEAN[clean_and_stats\nlowercase, strip URLs\ncompute batch stats]
    CLEAN --> DRIFT

    DRIFT[detect_drift\nJensen-Shannon divergence\nvs baseline]
    DRIFT --> ARCHIVE

    ARCHIVE[archive\nmove to\nincoming_archive/]
    ARCHIVE --> NOTIFY

    NOTIFY[notify_stats\nemail / log:\nsamples, avg length,\nclass dist, drift score]

    BROKEN[notify_broken_csv\nemail / log:\nvalidation failure]
    DRY[notify_dry_pipeline\nemail / log:\nno data in 12h]

    style SENSOR fill:#f0f4ff,stroke:#4a6cf7
    style VALIDATE fill:#f0f4ff,stroke:#4a6cf7
    style DRIFT fill:#fff3cd,stroke:#f0ad4e
    style BROKEN fill:#ffeaea,stroke:#dc3545
    style DRY fill:#ffeaea,stroke:#dc3545
    style NOTIFY fill:#e8f5e9,stroke:#28a745
```

---

## 4. DVC Pipeline

```mermaid
flowchart LR
    RAW[("data/raw/data.csv\n.dvc pointer")]

    subgraph prepare["Stage: prepare"]
        P[src/data/prepare.py\nclean, split, baseline]
    end

    subgraph train_logreg["Stage: train_logreg"]
        LR[Logistic Regression\nC=2.0, liblinear\nval split 15%]
    end

    subgraph train_linearsvc["Stage: train_linearsvc"]
        SV[LinearSVC\nC=0.5 + CalibratedCV\nval split 15%]
    end

    subgraph train_xgboost["Stage: train_xgboost"]
        XG[XGBoost\nn_est=250, depth=4\nval split 15%]
    end

    subgraph promote["Stage: promote"]
        PRO[promote_model.py\npick max macro_f1\nset champion alias]
    end

    TRAIN[("data/processed/\ntrain.csv")]
    TEST[("data/processed/\ntest.csv")]
    BASE[("data/baseline_stats.json")]

    RAW --> P --> TRAIN & TEST & BASE
    TRAIN & TEST --> LR & SV & XG
    LR --> |"metrics/logreg_metrics.json\nmodels/logreg_bundle.joblib"| PRO
    SV --> |"metrics/linearsvc_metrics.json\nmodels/linearsvc_bundle.joblib"| PRO
    XG --> |"metrics/xgboost_metrics.json\nmodels/xgboost_bundle.joblib"| PRO
```

---

## 5. CI/CD Pipeline

```mermaid
flowchart TD
    PR([Push / Pull Request\nto main branch])

    subgraph CI["CI Workflow — ci.yml"]
        UT[Unit Tests\npytest tests/unit/\n12 tests]
        DB[Docker Build\nfastapi + airflow +\nstreamlit + mlflow]
        UT & DB
    end

    subgraph INT["Integration Workflow — integration.yml"]
        SVC[docker compose up\nmlflow + fastapi + nginx]
        WAIT[Wait for\n/health 200]
        IT[Integration Tests\npytest tests/integration/\n8 tests]
        DOWN[docker compose down -v]
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
    TEXT([Raw text]) --> CLEAN[clean_text\nlowercase, strip URLs\n& mentions]

    CLEAN --> W[Word TF-IDF\n1-3 grams\n30k features\nsublinear_tf]
    CLEAN --> C[Char TF-IDF\n3-4 char n-grams\n10k features\nanalyzer=char_wb]
    CLEAN --> H[Handcrafted\n12 features]

    H --> H1[text length\nword count\navg word length]
    H --> H2[punctuation density\nupper ratio\nfirst-person ratio]
    H --> H3[exclamation count\nquestion count]
    H --> H4[VADER sentiment\ncompound, pos\nneg, neu]

    W --> HSTACK[hstack\n40012 total features]
    C --> HSTACK
    H --> HSTACK

    HSTACK --> CLF[Classifier\nLogreg / LinearSVC / XGBoost]
    CLF --> PRED([7-class\nprediction])
```
