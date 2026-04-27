# Architecture Diagrams

## 1. System Overview

```mermaid
graph TB
    subgraph Host["Host Machine (WSL2 / Linux)"]

        subgraph DataLayer["Data & Training Layer"]

            subgraph Airflow["Airflow  :8080"]
                AF_SCH[Scheduler]
                AF_WEB[Webserver]
                AF_PG[(Postgres)]
                AF_SCH --- AF_WEB
                AF_SCH --- AF_PG
            end

            FS[("data/incoming/\n*.csv")]
            BATCHES[("data/batches/\ncleaned CSVs")]
            CLEAN_FN["src/data/clean.py\nclean_text()"]

            subgraph DVCPipeline["DVC Pipeline  (dvc repro)"]
                PREP["prepare\nclean + split + baseline"]
                TR_LR[train_logreg]
                TR_SV[train_linearsvc]
                TR_XG[train_xgboost]
                PROMO[promote]
                PREP --> TR_LR & TR_SV & TR_XG --> PROMO
            end

            RAW[("data/raw/data.csv")]
            PROC[("data/processed/\ntrain.csv  test.csv")]
            BASE[("data/baseline_stats.json")]

            RAW --> PREP
            PREP -->|"split"| PROC
            PREP -->|"500-text sample"| BASE
            CLEAN_FN -->|"used by"| PREP
            CLEAN_FN -->|"used by"| AF_SCH

            FS -->|"FileSensor"| AF_SCH
            AF_SCH -->|"validate → clean → drift → archive"| BATCHES
            BASE -->|"drift baseline\nfor detect_drift task"| AF_SCH
            PROC -->|"retraining data\nfor dvc repro"| DVCPipeline
            BATCHES -.->|"accumulated batches\nfeed next retrain"| DVCPipeline
        end

        subgraph Registry["Experiment & Model Registry  :5000"]
            MLFLOW[MLflow Server]
            MLDB[(mlflow.db)]
            MLRUNS[("mlruns/ artifacts")]
            MLFLOW --- MLDB
            MLFLOW --- MLRUNS
        end

        subgraph Serving["Serving Layer"]
            NGINX["Nginx\nLoad Balancer :8000"]
            API1[FastAPI Replica 1]
            API2[FastAPI Replica 2]
            API3[FastAPI Replica 3]
            NGINX --> API1 & API2 & API3
        end

        subgraph Frontend["Frontend  :8501"]
            ST[Streamlit UI]
        end

        subgraph Observability["Observability"]
            PROM["Prometheus :9090"]
            GRAF["Grafana :3001"]
            ALERT["AlertManager :9093"]
            NODE["Node Exporter :9100"]
            PROM --> GRAF
            PROM --> ALERT
            NODE --> PROM
        end
    end

    PROMO -->|"set champion alias"| MLFLOW
    MLFLOW -->|"download bundle at startup"| API1 & API2 & API3
    BASE -->|"baseline_stats.json\nfor live drift"| API1 & API2 & API3
    API1 & API2 & API3 -->|"/metrics"| PROM
    ST -->|"REST"| NGINX
    User([End User]) --> ST
    User --> NGINX
```

---

## 2. Data Flow

```mermaid
flowchart TD
    RAW([data/raw/data.csv]) -->|"dvc repro prepare\nsrc/data/prepare.py"| PREP

    subgraph PREP["prepare stage  —  both paths share clean_text()"]
        direction LR
        CT["clean_text()\nlowercase, strip URLs\nstrip mentions"]
    end

    PREP -->|"train.csv 42k\ntest.csv 10k"| PROC[("data/processed/")]
    PREP -->|"500-text stratified\nsample"| BASE[("data/baseline_stats.json\nword_freq_top1000")]

    PROC --> FEAT["Feature Engineering\nWord TF-IDF 30k\nChar TF-IDF 10k\nHandcrafted 12"]
    FEAT --> TRAIN["Model Training\nLogreg / LinearSVC / XGBoost\n15% val split"]
    TRAIN -->|"bundles + metrics"| MLFLOW[(MLflow Registry)]
    MLFLOW -->|"champion alias"| API["FastAPI × 3 Replicas"]

    INCOMING([data/incoming/*.csv]) -->|"Airflow FileSensor\nevery 10 min"| AFVAL

    subgraph AFPIPE["Airflow data_prep_pipeline  —  same clean_text()"]
        direction LR
        AFVAL["validate_csv\ncheck columns & rows"]
        AFCLEAN["clean_and_stats\nclean_text() + batch stats"]
        AFDRIFT["detect_drift\nJS divergence vs baseline"]
        AFARCH["archive"]
        AFVAL --> AFCLEAN --> AFDRIFT --> AFARCH
    end

    BASE -->|"drift baseline\nshared with Airflow"| AFDRIFT
    AFCLEAN -->|"cleaned CSV\ndata/batches/"| BATCHES[("data/batches/\naccumulated batches")]
    BATCHES -.->|"feed next\ndvc repro"| PROC
    AFDRIFT -->|"score > 0.3?"| NOTIFY{"Alert?"}
    NOTIFY -->|"Yes"| EMAIL["Email / Log\nnotification"]
    NOTIFY -->|"No"| STATS["Stats report\nemail / log"]

    API -->|"prediction + confidence"| USER([End User])
    USER -->|"text input"| API
    BASE -->|"baseline for\nlive drift"| API
    API -->|"rolling 200-text\nJSD score"| PROM[(Prometheus)]
    PROM --> GRAF[Grafana Dashboard]
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
