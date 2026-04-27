"""
Airflow DAG: data_prep_pipeline

Behavior:
- FileSensor watches /opt/airflow/data/incoming for new CSV files
- If no file in 12h → "Dry Pipeline" notification
- On new CSV:
    1. Validate structure (handles Kaggle 'statement'/'status' columns too)
    2. Clean text, compute batch statistics
    3. Detect drift vs baseline (Jensen-Shannon on top-1000 word frequencies)
    4. Archive the processed CSV
    5. Send batch stats report (email if SMTP configured, logs to file otherwise)
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "admin@example.com")
SMTP_CONFIGURED = bool(os.getenv("MAILTRAP_USER"))  # True only when credentials present

DATA_ROOT     = Path("/opt/airflow/data")
INCOMING_DIR  = DATA_ROOT / "incoming"
ARCHIVE_DIR   = DATA_ROOT / "incoming_archive"
BATCHES_DIR   = DATA_ROOT / "batches"   # separated from DVC-tracked processed/
BASELINE_PATH = DATA_ROOT / "baseline_stats.json"
NOTIFY_LOG    = Path("/opt/airflow/logs") / "notifications.log"

REQUIRED_COLS = {"text", "label"}
COL_ALIASES   = {"statement": "text", "status": "label"}   # Kaggle dataset columns


# ─────────────────────────── helpers ────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and apply Kaggle→canonical renaming."""
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df.rename(columns=COL_ALIASES)


def _notify(subject: str, html: str, **ctx):
    """Try to send an email; if SMTP isn't configured, write to notification log."""
    log_entry = (
        f"\n{'='*60}\n"
        f"TIMESTAMP : {datetime.utcnow().isoformat()}\n"
        f"SUBJECT   : {subject}\n"
        f"BODY      :\n{html}\n"
    )
    try:
        NOTIFY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(NOTIFY_LOG, "a") as f:
            f.write(log_entry)
    except Exception:
        pass  # best-effort logging

    if not SMTP_CONFIGURED:
        print(f"[notify] SMTP not configured — logged to {NOTIFY_LOG}")
        print(f"[notify] Subject: {subject}")
        return

    try:
        from airflow.utils.email import send_email
        send_email(to=ALERT_EMAIL, subject=subject, html_content=html)
        print(f"[notify] Email sent: {subject}")
    except Exception as e:
        print(f"[notify] Email failed ({e}) — already logged to file.")


# ─────────────────────────── task callables ─────────────────────────────────

def _find_latest_csv(**ctx):
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INCOMING_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No CSV in incoming/ (sensor should have caught this)")
    target = files[-1]
    ctx["ti"].xcom_push(key="csv_path", value=str(target))
    print(f"Found: {target.name} ({target.stat().st_size / 1024:.1f} KB)")
    return str(target)


def _validate_csv(**ctx):
    path = Path(ctx["ti"].xcom_pull(key="csv_path", task_ids="find_latest_csv"))

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Cannot parse CSV: {e}")

    df = _normalize_columns(df)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Got {list(df.columns)}. "
            f"Expected 'text'+'label' (or Kaggle aliases 'statement'+'status')."
        )

    if len(df) == 0:
        raise ValueError("CSV is empty after reading")

    df = df[["text", "label"]].dropna(subset=["text", "label"])
    if len(df) == 0:
        raise ValueError("No valid rows after dropping nulls in text/label")

    ctx["ti"].xcom_push(key="row_count", value=len(df))
    ctx["ti"].xcom_push(key="classes", value=df["label"].value_counts().to_dict())
    print(f"Validated: {len(df)} rows, {df['label'].nunique()} classes")
    return len(df)


def _clean_and_compute_stats(**ctx):
    import re

    path = Path(ctx["ti"].xcom_pull(key="csv_path", task_ids="find_latest_csv"))
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    df = df[["text", "label"]].dropna(subset=["text", "label"])

    def clean(t: str) -> str:
        t = str(t).lower()
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)
        t = re.sub(r"@\w+", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    df["text"] = df["text"].apply(clean)
    df = df[df["text"].str.len() >= 3]  # drop very short texts

    lengths = df["text"].str.split().str.len()
    stats = {
        "n_samples":          int(len(df)),
        "avg_text_length":    float(lengths.mean()),
        "class_distribution": df["label"].value_counts(normalize=True).to_dict(),
    }

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    out = BATCHES_DIR / f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out, index=False)

    ctx["ti"].xcom_push(key="batch_path",  value=str(out))
    ctx["ti"].xcom_push(key="batch_stats", value=stats)
    print(f"Cleaned batch → {out.name}  ({stats['n_samples']} rows)")
    return stats


def _detect_drift(**ctx):
    """Compare batch word frequencies to baseline using Jensen-Shannon divergence."""
    import math
    from collections import Counter

    batch_path = ctx["ti"].xcom_pull(key="batch_path", task_ids="clean_and_stats")
    df = pd.read_csv(batch_path)

    words = " ".join(df["text"].astype(str).tolist()).split()
    counts = Counter(words)
    top_words = dict(counts.most_common(1000))
    top_total = sum(top_words.values()) or 1
    batch_freq = {w: c / top_total for w, c in top_words.items()}

    if not BASELINE_PATH.exists():
        drift, note = 0.0, "no baseline available yet"
    else:
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        baseline_freq = baseline.get("word_freq_top1000", {})

        eps = 1e-12
        js = 0.0
        for w in set(batch_freq) | set(baseline_freq):
            p = batch_freq.get(w, 0.0) + eps
            q = baseline_freq.get(w, 0.0) + eps
            m = 0.5 * (p + q)
            js += 0.5 * p * math.log(p / m) + 0.5 * q * math.log(q / m)
        drift = float(min(max(js, 0.0), 1.0))
        note  = "⚠️ drift detected" if drift > 0.3 else "✅ within tolerance"

    ctx["ti"].xcom_push(key="drift_score", value=drift)
    ctx["ti"].xcom_push(key="drift_note",  value=note)
    print(f"Drift: {drift:.4f} ({note})")
    return drift


def _archive(**ctx):
    csv_path = Path(ctx["ti"].xcom_pull(key="csv_path", task_ids="find_latest_csv"))
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = ARCHIVE_DIR / csv_path.name
    shutil.move(str(csv_path), str(target))
    print(f"Archived: {csv_path.name} → incoming_archive/")
    return str(target)


def _send_stats_notification(**ctx):
    ti    = ctx["ti"]
    stats = ti.xcom_pull(key="batch_stats", task_ids="clean_and_stats")
    drift = ti.xcom_pull(key="drift_score", task_ids="detect_drift")
    note  = ti.xcom_pull(key="drift_note",  task_ids="detect_drift")

    classes_html = "".join(
        f"<li>{k}: {v*100:.1f}%</li>"
        for k, v in sorted(stats["class_distribution"].items(), key=lambda x: -x[1])
    )
    drift_color = "#d32f2f" if drift > 0.3 else "#388e3c"
    html = f"""
    <h2 style="font-family:sans-serif;">📦 Batch Ingestion Report</h2>
    <table style="font-family:sans-serif;border-collapse:collapse;width:400px;">
      <tr><td><b>Samples ingested</b></td><td>{stats['n_samples']:,}</td></tr>
      <tr><td><b>Avg text length (words)</b></td><td>{stats['avg_text_length']:.1f}</td></tr>
      <tr><td><b>Drift score</b></td>
          <td><span style="color:{drift_color};font-weight:700;">{drift:.4f} — {note}</span></td></tr>
    </table>
    <h3 style="font-family:sans-serif;">Class distribution</h3>
    <ul style="font-family:sans-serif;">{classes_html}</ul>
    """
    _notify(subject="[MLOps] Batch ingested — stats & drift report", html=html, **ctx)


def _send_broken_csv_notification(**ctx):
    _notify(
        subject="[MLOps] ALERT: Broken / invalid CSV in incoming/",
        html=(
            "<h2>⚠️ CSV Validation Failed</h2>"
            "<p>A CSV in <code>/opt/airflow/data/incoming</code> failed validation.</p>"
            "<p>Check Airflow UI → <b>data_prep_pipeline → validate_csv</b> logs for details.</p>"
        ),
        **ctx,
    )


def _send_dry_pipeline_notification(**ctx):
    _notify(
        subject="[MLOps] ALERT: Dry pipeline — no data in 12 hours",
        html=(
            "<h2>🔕 Dry Pipeline Alert</h2>"
            "<p>No CSV arrived in <code>/opt/airflow/data/incoming</code> within 12 hours.</p>"
            "<p>Check the data source health.</p>"
        ),
        **ctx,
    )


# ─────────────────────────── DAG definition ─────────────────────────────────

default_args = {
    "owner":                    "mlops-student",
    "depends_on_past":          False,
    "email_on_failure":         False,
    "email_on_retry":           False,
    "retries":                  2,
    "retry_delay":              timedelta(seconds=30),
    "retry_exponential_backoff": True,
    "max_retry_delay":          timedelta(minutes=3),
}

with DAG(
    dag_id="data_prep_pipeline",
    description="CSV sensor → validate → clean → drift → archive → notify",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="*/10 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "data-prep"],
) as dag:

    wait_for_csv = FileSensor(
        task_id="wait_for_csv",
        filepath=str(INCOMING_DIR / "*.csv"),
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=60 * 60 * 12,   # 12 h
        mode="reschedule",
        soft_fail=True,
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

    notify_stats = PythonOperator(
        task_id="notify_stats",
        python_callable=_send_stats_notification,
    )

    notify_broken_csv = PythonOperator(
        task_id="notify_broken_csv",
        python_callable=_send_broken_csv_notification,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    notify_dry_pipeline = PythonOperator(
        task_id="notify_dry_pipeline",
        python_callable=_send_dry_pipeline_notification,
        trigger_rule=TriggerRule.ALL_SKIPPED,   # fires when sensor soft-fails (→ skipped)
    )

    # ── Success path ──────────────────────────────────────────────────────────
    (
        wait_for_csv
        >> find_latest_csv
        >> validate_csv
        >> clean_and_stats
        >> detect_drift
        >> archive
        >> notify_stats
    )

    # ── Failure / alert branches ──────────────────────────────────────────────
    validate_csv     >> notify_broken_csv    # fires if validate_csv fails
    wait_for_csv     >> notify_dry_pipeline  # fires when sensor is skipped (12 h timeout)
