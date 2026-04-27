"""
After training all models, run this to promote the best one.
Uses MLflow model aliases (replaces deprecated stage-based promotion).

Usage: python scripts/promote_model.py
The winning version gets the alias 'champion'; previous champion is archived to 'challenger'.
"""
import os
from mlflow import MlflowClient

MODEL_NAME = "mental_health_classifier"
CHAMPION_ALIAS = "champion"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def main():
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No registered versions yet. Train first.")
        return

    # Pick the version with the highest macro_f1 on the test set
    best = None
    for v in versions:
        run = client.get_run(v.run_id)
        f1 = run.data.metrics.get("macro_f1", 0.0)
        model_type = run.data.params.get("model_type", "?")
        print(f"  v{v.version} ({model_type}) macro_f1={f1:.4f}")
        if best is None or f1 > best[1]:
            best = (v, f1)

    winner, best_f1 = best
    run = client.get_run(winner.run_id)
    model_type = run.data.params.get("model_type", "?")
    print(f"\nBest: v{winner.version} ({model_type}) macro_f1={best_f1:.4f}")

    # Archive existing champion as challenger before promoting new one
    try:
        current_champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        if current_champion.version == winner.version:
            print(f"v{winner.version} is already the champion. Nothing to do.")
            return
        # Rename old champion → challenger
        client.set_registered_model_alias(MODEL_NAME, "challenger", current_champion.version)
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
        print(f"  Archived old champion (v{current_champion.version}) → challenger")
    except Exception:
        pass  # No existing champion yet

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, winner.version)
    print(f"  Promoted v{winner.version} → '{CHAMPION_ALIAS}'")


if __name__ == "__main__":
    main()
