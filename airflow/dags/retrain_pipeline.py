"""
Airflow DAG: retrain_pipeline

Polls feedback.log every 2 hours. When FEEDBACK_THRESHOLD entries have
accumulated it:
  1. Splits feedback 80/20 (stratified) → appends to processed train/test CSVs
  2. Archives and clears feedback.log so entries aren't reused next cycle
  3. Retrains the current champion model type via train.py
  4. Runs promotion — new version wins only if its macro_f1 beats the incumbent
  5. Notifies via file log (and email if SMTP is configured)

Path notes:
  - Feedback log : /opt/airflow/data/feedback.log  (same bind-mount as FastAPI's /app/data)
  - Train CSV    : /opt/airflow/data/processed/train.csv
  - Test CSV     : /opt/airflow/data/processed/test.csv
  - params.yaml  : /opt/airflow/params.yaml
  - models/      : /opt/airflow/models
  - metrics/     : /opt/airflow/metrics
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

FEEDBACK_LOG      = Path("/opt/airflow/data/feedback.log")
FEEDBACK_ARCHIVE  = Path("/opt/airflow/data/feedback_archive")
TRAIN_CSV         = Path("/opt/airflow/data/processed/train.csv")
TEST_CSV          = Path("/opt/airflow/data/processed/test.csv")
NOTIFY_LOG        = Path("/opt/airflow/logs/notifications.log")
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME        = "mental_health_classifier"
CHAMPION_ALIAS    = "champion"
FEEDBACK_THRESHOLD = int(os.getenv("FEEDBACK_THRESHOLD", "10"))
ALERT_EMAIL       = os.getenv("ALERT_EMAIL", "admin@example.com")
SMTP_CONFIGURED   = bool(os.getenv("MAILTRAP_USER"))


# ─────────────────────────── helpers ────────────────────────────────────────

def _notify(subject: str, html: str, **ctx):
    """Write notification to log file; send email when SMTP is configured."""
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
        pass

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

def _check_feedback_threshold(**ctx):
    """Raise AirflowSkipException if feedback hasn't reached the threshold."""
    if not FEEDBACK_LOG.exists():
        raise AirflowSkipException(f"Feedback log not found: {FEEDBACK_LOG}")

    lines = [l for l in FEEDBACK_LOG.read_text().splitlines() if l.strip()]
    n = len(lines)
    if n < FEEDBACK_THRESHOLD:
        raise AirflowSkipException(
            f"Feedback threshold not met: {n}/{FEEDBACK_THRESHOLD} entries"
        )
    print(f"Feedback threshold met: {n} entries (>= {FEEDBACK_THRESHOLD})")
    return n


def _split_and_append_feedback(**ctx):
    """
    Parse feedback.log, split 80/20 (stratified where possible),
    append each slice to the matching processed CSV, then archive the log.
    Uses correct_label (human-verified ground truth) as the label.
    """
    from sklearn.model_selection import train_test_split

    lines = [l for l in FEEDBACK_LOG.read_text().splitlines() if l.strip()]
    records = []
    for line in lines:
        try:
            entry = json.loads(line)
            text  = str(entry.get("text", "")).strip()
            label = str(entry.get("correct_label", "")).strip()
            if text and label:
                records.append({"text": text, "label": label})
        except (json.JSONDecodeError, KeyError):
            continue

    if not records:
        raise AirflowSkipException("No valid feedback entries found in log")

    df = pd.DataFrame(records)

    # Stratified split when every class has ≥ 2 samples, otherwise random
    label_counts  = df["label"].value_counts()
    can_stratify  = (label_counts >= 2).all() and len(df) >= 5

    try:
        if can_stratify:
            train_fb, test_fb = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["label"]
            )
        else:
            train_fb, test_fb = train_test_split(df, test_size=0.2, random_state=42)
    except ValueError:
        train_fb, test_fb = df, pd.DataFrame(columns=df.columns)

    train_fb.to_csv(TRAIN_CSV, mode="a", header=False, index=False)
    if not test_fb.empty:
        test_fb.to_csv(TEST_CSV, mode="a", header=False, index=False)

    print(
        f"Appended {len(train_fb)} rows → train.csv, "
        f"{len(test_fb)} rows → test.csv"
    )

    # Archive consumed log, then truncate so FastAPI can keep writing
    FEEDBACK_ARCHIVE.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_path = FEEDBACK_ARCHIVE / f"feedback_{ts}.log"
    archive_path.write_text(FEEDBACK_LOG.read_text())
    FEEDBACK_LOG.unlink()  # delete so FastAPI recreates it fresh on next write
    print(f"Archived → {archive_path.name}, removed feedback.log")

    ctx["ti"].xcom_push(key="n_train", value=len(train_fb))
    ctx["ti"].xcom_push(key="n_test",  value=len(test_fb))
    return len(df)


def _get_champion_model_type(**ctx):
    """Query MLflow for the champion model type; fall back to logreg."""
    from mlflow import MlflowClient

    client     = MlflowClient(tracking_uri=MLFLOW_URI)
    model_type = "logreg"
    try:
        v          = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        run        = client.get_run(v.run_id)
        model_type = run.data.params.get("model_type", "logreg")
        print(f"Champion: v{v.version} ({model_type})")
    except Exception as e:
        print(f"Could not fetch champion ({e}). Defaulting to logreg.")

    ctx["ti"].xcom_push(key="model_type", value=model_type)
    return model_type


def _run_training(**ctx):
    """Retrain the champion model type on the updated dataset via train.py."""
    model_type = ctx["ti"].xcom_pull(key="model_type", task_ids="get_champion_model_type")

    env = os.environ.copy()
    env["PYTHONPATH"]           = "/opt/airflow"
    env["MLFLOW_TRACKING_URI"]  = MLFLOW_URI
    env["NLTK_DATA"]            = "/home/airflow/nltk_data"

    print(f"Training model: {model_type}")
    result = subprocess.run(
        ["python", "-m", "src.training.train", "--model", model_type],
        cwd="/opt/airflow",
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")
    print(f"Training complete: {model_type}")


def _promote_model(**ctx):
    """
    Compare all registered model versions by macro_f1; promote the best
    to the champion alias. Mirrors promote_model.py logic so it stays
    consistent with manual promotion runs.
    """
    from mlflow import MlflowClient

    client   = MlflowClient(tracking_uri=MLFLOW_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No registered versions. Skipping promotion.")
        return

    best = None
    for v in versions:
        run       = client.get_run(v.run_id)
        m         = run.data.metrics
        val_f1    = m.get("val_macro_f1", 0.0)
        f1        = m.get("macro_f1", 0.0)
        mtype     = run.data.params.get("model_type", "?")
        if val_f1 == 0.0:
            print(f"  v{v.version} ({mtype}) skipped — no val_macro_f1")
            continue
        print(f"  v{v.version} ({mtype}) macro_f1={f1:.4f}  val_macro_f1={val_f1:.4f}")
        if best is None or f1 > best[1]:
            best = (v, f1)

    if best is None:
        print("No eligible versions for promotion.")
        return

    winner, best_f1 = best
    run   = client.get_run(winner.run_id)
    mtype = run.data.params.get("model_type", "?")
    print(f"\nBest: v{winner.version} ({mtype}) macro_f1={best_f1:.4f}")

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        if current.version == winner.version:
            print(f"v{winner.version} is already champion. Nothing to do.")
            return
        client.set_registered_model_alias(MODEL_NAME, "challenger", current.version)
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
        print(f"Archived v{current.version} → challenger")
    except Exception:
        pass

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, winner.version)
    print(f"Promoted v{winner.version} ({mtype}) → '{CHAMPION_ALIAS}'")
    ctx["ti"].xcom_push(key="promoted_version", value=winner.version)
    ctx["ti"].xcom_push(key="promoted_type",    value=mtype)
    ctx["ti"].xcom_push(key="promoted_f1",      value=best_f1)


def _notify_success(**ctx):
    ti          = ctx["ti"]
    n_train     = ti.xcom_pull(key="n_train",           task_ids="split_and_append_feedback") or 0
    n_test      = ti.xcom_pull(key="n_test",            task_ids="split_and_append_feedback") or 0
    promoted_v  = ti.xcom_pull(key="promoted_version",  task_ids="promote_model")
    promoted_t  = ti.xcom_pull(key="promoted_type",     task_ids="promote_model")
    promoted_f  = ti.xcom_pull(key="promoted_f1",       task_ids="promote_model")

    if promoted_v:
        promo_msg = f"Promoted v{promoted_v} ({promoted_t}) — macro_f1={promoted_f:.4f}"
        promo_color = "#388e3c"
    else:
        promo_msg   = "No promotion — existing champion is still best"
        promo_color = "#f57c00"

    html = f"""
    <h2 style="font-family:sans-serif;">🔁 Retraining Complete</h2>
    <table style="font-family:sans-serif;border-collapse:collapse;width:420px;">
      <tr><td><b>Feedback rows → train.csv</b></td><td>{n_train}</td></tr>
      <tr><td><b>Feedback rows → test.csv</b></td><td>{n_test}</td></tr>
      <tr><td><b>Promotion</b></td>
          <td><span style="color:{promo_color};font-weight:700;">{promo_msg}</span></td></tr>
    </table>
    """
    _notify(subject="[MLOps] Retraining complete", html=html, **ctx)


def _notify_failure(**ctx):
    _notify(
        subject="[MLOps] ALERT: Retraining pipeline failed",
        html=(
            "<h2>⚠️ Retraining Failed</h2>"
            "<p>Check Airflow UI → <b>retrain_pipeline</b> for task details.</p>"
        ),
        **ctx,
    )


# ─────────────────────────── DAG definition ─────────────────────────────────

default_args = {
    "owner":            "mlops-student",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
}

with DAG(
    dag_id="retrain_pipeline",
    description="Polls feedback log every 2 h; retrains when threshold is met",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 */2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "retraining"],
) as dag:

    check_threshold = PythonOperator(
        task_id="check_feedback_threshold",
        python_callable=_check_feedback_threshold,
    )

    split_and_append = PythonOperator(
        task_id="split_and_append_feedback",
        python_callable=_split_and_append_feedback,
    )

    get_model_type = PythonOperator(
        task_id="get_champion_model_type",
        python_callable=_get_champion_model_type,
    )

    run_training = PythonOperator(
        task_id="run_training",
        python_callable=_run_training,
        execution_timeout=timedelta(minutes=35),
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_model,
    )

    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=_notify_success,
    )

    notify_failure = PythonOperator(
        task_id="notify_failure",
        python_callable=_notify_failure,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ── Happy path ────────────────────────────────────────────────────────────
    (
        check_threshold
        >> split_and_append
        >> get_model_type
        >> run_training
        >> promote
        >> notify_success
    )

    # ── Failure alert (fires if any core task fails, not just skips) ──────────
    [split_and_append, get_model_type, run_training, promote] >> notify_failure
