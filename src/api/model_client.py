import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack


class ModelClient:
    def __init__(self):
        self.bundle: Optional[dict] = None
        self.version: str = "unknown"
        self.stage: str = "unknown"

    def load(self):
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_name = os.getenv("MODEL_NAME", "mental_health_classifier")
        alias = os.getenv("MODEL_ALIAS", "champion")

        try:
            import mlflow
            from mlflow import MlflowClient

            mlflow.set_tracking_uri(mlflow_uri)
            client = MlflowClient(tracking_uri=mlflow_uri)

            v = client.get_model_version_by_alias(model_name, alias)
            run_id = v.run_id

            local_dir = client.download_artifacts(run_id, "model_bundle")
            bundle_files = list(Path(local_dir).glob("*_bundle.joblib"))
            if not bundle_files:
                raise RuntimeError("No bundle artifact found in MLflow run")
            self.bundle = joblib.load(bundle_files[0])
            self.version = v.version
            self.stage = alias
            print(f"[ModelClient] Loaded from MLflow: v{v.version} alias='{alias}'")
        except Exception as e:
            print(f"[ModelClient] MLflow load failed: {e}. Trying local fallback...")
            models_dir = Path("/app/models")
            bundles = list(models_dir.glob("*_bundle.joblib"))
            if not bundles:
                raise RuntimeError(
                    "No model available. Run: dvc repro && docker compose exec -T trainer python /app/scripts/promote_model.py"
                ) from e
            priority = {"linearsvc": 0, "xgboost": 1, "logreg": 2}
            bundles.sort(key=lambda p: priority.get(p.stem.split("_")[0], 99))
            self.bundle = joblib.load(bundles[0])
            self.version = "local"
            self.stage = "fallback"
            print(f"[ModelClient] Loaded local fallback: {bundles[0].name}")

    def _vectorize(self, text: str):
        b = self.bundle
        w = b["tfidf_word"].transform([text])
        c = b["tfidf_char"].transform([text])
        h = b["handcrafted"].transform([text])
        return hstack([w, c, csr_matrix(h)])

    def predict(self, text: str):
        if self.bundle is None:
            raise RuntimeError("Model not loaded")

        X = self._vectorize(text)
        clf = self.bundle["classifier"]
        labels = self.bundle["labels"]

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
        else:
            pred = clf.predict(X)[0]
            probs = np.zeros(len(labels))
            probs[pred] = 1.0

        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {lbl: float(p) for lbl, p in zip(labels, probs)},
            "model_version": self.version,
            "model_stage": self.stage,
        }