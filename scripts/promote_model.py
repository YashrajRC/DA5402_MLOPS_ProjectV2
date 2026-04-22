"""
After training all models, run this to promote the best one to Production.
Usage: python scripts/promote_model.py
"""
import os
from mlflow import MlflowClient

MODEL_NAME = "mental_health_classifier"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def main():
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No registered versions yet. Train first.")
        return

    # Pick highest macro_f1 from the parent run of each version
    best = None
    for v in versions:
        run = client.get_run(v.run_id)
        f1 = run.data.metrics.get("macro_f1", 0.0)
        if best is None or f1 > best[1]:
            best = (v, f1)

    version, f1 = best
    print(f"Best version: v{version.version} run={version.run_id} macro_f1={f1:.4f}")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted v{version.version} → Production")


if __name__ == "__main__":
    main()
