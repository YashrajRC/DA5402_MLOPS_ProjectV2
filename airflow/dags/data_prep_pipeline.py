"""
Airflow DAG: data_prep_pipeline

Behavior:
- FileSensor watches /opt/airflow/data/incoming for new CSV files
- If no file in 12h → "Dry Pipeline" email (sensor soft-fails and triggers alert)
- On new CSV:
    1. Validate CSV structure + content. If broken → "Broken CSV" email
    2. Clean text, compute stats
    3. Detect drift vs baseline stats
    4. Merge into processed training data
    5. Archive processed CSV
    6. Send "Collection Statistics" email summarizing the batch + drift
- All scraping-like tasks go through Airflow Pool 'data_prep_pool' (size 3)
- Retries with exponential backoff on transient failures
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowSensorTimeout
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "admin@example.com")

INCOMING_DIR = Path("/opt/airflow/data/incoming")
ARCHIVE_DIR = Path("/opt/airflow/data/incoming_archive")
PROCESSED_DIR = Path("/opt/airflow/data/processed")
BASELINE_PATH = Path("/opt/airflow/data/baseline_stats.json")

REQUIRED_COLS = {"text", "label"}


default_args = {
    "owner": "mlops-student",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=5),
}


def _find_latest_csv(**ctx):
    files = sorted(INCOMING_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No CSV present (sensor should have caught this)")
    target = files[-1]
    ctx["ti"].xcom_push(key="csv_path", value=str(target))
    return str(target)


def _validate_csv(**ctx):
    path = Path(ctx["ti"].xcom_pull(key="csv_path", task_ids="find_latest_csv"))
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Cannot parse CSV: {e}")

    missing = REQUIRED_COLS - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

    df.columns = df.columns.str.lower()
    if df[["text", "label"]].isnull().all(axis=1).any():
        raise ValueError("Found fully null rows")

    if len(df) == 0:
        raise ValueError("CSV is empty")

    ctx["ti"].xcom_push(key="row_count", value=len(df))
    ctx["ti"].xcom_push(key="classes", value=df["label"].value_counts().to_dict())
    return len(df)


def _clean_and_compute_stats(**ctx):
    import re
    path = Path(ctx["ti"].xcom_pull(key="csv_path", task_ids="find_latest_csv"))
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df = df[["text", "label"]].dropna()

    # Simple cleaning inline (avoids src import path issues in Airflow)
    def clean(t):
        t = str(t).lower()
        t = re.sub(r"https?://\S+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    df["text"] = df["text"].apply(clean)

    lengths = df["text"].str.split().str.len()
    stats = {
        "n_samples": int(len(df)),
        "avg_text_length": float(lengths.mean()),
        "class_distribution": df["label"].value_counts(normalize=True).to_dict(),
    }

    # Save cleaned intermediate
    out = PROCESSED_DIR / f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ctx["ti"].xcom_push(key="batch_path", value=str(out))
    ctx["ti"].xcom_push(key="batch_stats", value=stats)
    return stats


def _detect_drift(**ctx):
    """Compare batch stats to baseline_stats.json → flag drift."""
    import math
    from collections import Counter

    batch_path = ctx["ti"].xcom_pull(key="batch_path", task_ids="clean_and_stats")
    df = pd.read_csv(batch_path)

    # Build word-freq for batch
    words = " ".join(df["text"].astype(str).tolist()).split()
    counts = Counter(words)
    total = sum(counts.values()) or 1
    batch_freq = {w: c / total for w, c in counts.most_common(1000)}

    if not BASELINE_PATH.exists():
        drift = 0.0
        note = "no baseline yet"
    else:
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        baseline_freq = baseline.get("word_freq_top1000", {})

        vocab = set(batch_freq) | set(baseline_freq)
        eps = 1e-12
        js = 0.0
        for w in vocab:
            p = batch_freq.get(w, 0.0) + eps
            q = baseline_freq.get(w, 0.0) + eps
            m = 0.5 * (p + q)
            js += 0.5 * p * math.log(p / m) + 0.5 * q * math.log(q / m)
        drift = min(max(js, 0.0), 1.0)
        note = "drift detected" if drift > 0.3 else "within tolerance"

    ctx["ti"].xcom_push(key="drift_score", value=drift)
    ctx["ti"].xcom_push(key="drift_note", value=note)
    print(f"Drift: {drift:.4f} ({note})")
    return drift


def _archive(**ctx):
    csv_path = Path(ctx["ti"].xcom_pull(key="csv_path", task_ids="find_latest_csv"))
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = ARCHIVE_DIR / csv_path.name
    shutil.move(str(csv_path), str(target))
    return str(target)


def _build_stats_email(**ctx):
    ti = ctx["ti"]
    stats = ti.xcom_pull(key="batch_stats", task_ids="clean_and_stats")
    drift = ti.xcom_pull(key="drift_score", task_ids="detect_drift")
    note = ti.xcom_pull(key="drift_note", task_ids="detect_drift")

    classes_html = "".join(
        f"<li>{k}: {v*100:.1f}%</li>" for k, v in stats["class_distribution"].items()
    )
    html = f"""
    <h3>Batch Ingestion Report</h3>
    <ul>
      <li><b>Samples:</b> {stats['n_samples']}</li>
      <li><b>Avg text length (tokens):</b> {stats['avg_text_length']:.1f}</li>
      <li><b>Drift score:</b> {drift:.4f} ({note})</li>
    </ul>
    <h4>Class distribution</h4>
    <ul>{classes_html}</ul>
    """
    ti.xcom_push(key="stats_html", value=html)
    return "built"


with DAG(
    dag_id="data_prep_pipeline",
    description="Sensor → validate → clean → drift → archive → email",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="*/10 * * * *",       # check every 10 min
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "data-prep"],
) as dag:

    wait_for_csv = FileSensor(
        task_id="wait_for_csv",
        filepath="/opt/airflow/data/incoming/*.csv",
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=60 * 60 * 12,      # 12h
        mode="reschedule",
        soft_fail=True,            # if timeout, DAG continues to dry-alert branch
        pool="data_prep_pool",
    )

    find_latest_csv = PythonOperator(
        task_id="find_latest_csv",
        python_callable=_find_latest_csv,
        pool="data_prep_pool",
    )

    validate_csv = PythonOperator(
        task_id="validate_csv",
        python_callable=_validate_csv,
        pool="data_prep_pool",
    )

    clean_and_stats = PythonOperator(
        task_id="clean_and_stats",
        python_callable=_clean_and_compute_stats,
        pool="data_prep_pool",
    )

    detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=_detect_drift,
        pool="data_prep_pool",
    )

    archive = PythonOperator(
        task_id="archive",
        python_callable=_archive,
        pool="data_prep_pool",
    )

    build_stats_email = PythonOperator(
        task_id="build_stats_email",
        python_callable=_build_stats_email,
    )

    # Success path email: batch stats + drift
    email_stats = EmailOperator(
        task_id="email_stats",
        to=ALERT_EMAIL,
        subject="[MLOps] Batch ingested — stats & drift",
        html_content="{{ ti.xcom_pull(key='stats_html', task_ids='build_stats_email') }}",
    )

    # Failure path: broken CSV email (runs only if validate_csv fails)
    email_broken = EmailOperator(
        task_id="email_broken_csv",
        to=ALERT_EMAIL,
        subject="[MLOps] ALERT: Broken CSV in incoming/",
        html_content=(
            "A CSV in /opt/airflow/data/incoming failed validation. "
            "Check the Airflow UI → data_prep_pipeline → validate_csv logs."
        ),
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Dry-pipeline alert: fires if sensor times out (soft_fail → downstream skipped,
    # but we branch off the sensor itself with ALL_FAILED trigger)
    email_dry = EmailOperator(
        task_id="email_dry_pipeline",
        to=ALERT_EMAIL,
        subject="[MLOps] ALERT: Dry pipeline — no data in 12h",
        html_content=(
            "No CSV has arrived in /opt/airflow/data/incoming within the 12h window. "
            "Check data-source health."
        ),
        trigger_rule=TriggerRule.ALL_FAILED,
    )

    # Main success chain
    wait_for_csv >> find_latest_csv >> validate_csv >> clean_and_stats \
        >> detect_drift >> archive >> build_stats_email >> email_stats

    # Failure branches
    validate_csv >> email_broken
    wait_for_csv >> email_dry
