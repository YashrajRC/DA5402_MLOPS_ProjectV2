"""
Loads the model bundle from MLflow registry, with local fallback.
The bundle contains TF-IDF, handcrafted features, classifier, and label encoder.
"""
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.training.features import combine_features


class ModelClient:
    def __init__(self):
        self.bundle: Optional[dict] = None
        self.version: str = "unknown"
        self.stage: str = "unknown"

    def load(self):
        """Try MLflow registry first, fall back to local models/ folder."""
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_name = os.getenv("MODEL_NAME", "mental_health_classifier")
        stage = os.getenv("MODEL_STAGE", "Production")

        try:
            import mlflow
            from mlflow import MlflowClient

            mlflow.set_tracking_uri(mlflow_uri)
            client = MlflowClient(tracking_uri=mlflow_uri)

            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise RuntimeError(f"No versions at stage {stage}")
            v = versions[0]
            run_id = v.run_id

            # Download the bundle artifact logged during training
            local_dir = client.download_artifacts(run_id, "model_bundle")
            bundle_files = list(Path(local_dir).glob("*_bundle.joblib"))
            if not bundle_files:
                raise RuntimeError("No bundle in artifacts")
            self.bundle = joblib.load(bundle_files[0])
            self.version = v.version
            self.stage = stage
            print(f"[ModelClient] Loaded from MLflow: v{v.version} ({stage})")
        except Exception as e:
            print(f"[ModelClient] MLflow load failed: {e}. Trying local fallback...")
            models_dir = Path("/app/models")
            bundles = list(models_dir.glob("*_bundle.joblib"))
            if not bundles:
                raise RuntimeError(
                    "No model available. Train first, then promote to Production."
                ) from e
            # Prefer logreg as default local fallback
            priority = {"logreg": 0, "linearsvc": 1, "xgboost": 2}
            bundles.sort(key=lambda p: priority.get(p.stem.split("_")[0], 99))
            self.bundle = joblib.load(bundles[0])
            self.version = "local"
            self.stage = "fallback"
            print(f"[ModelClient] Loaded local fallback: {bundles[0].name}")

    def predict(self, text: str):
        if self.bundle is None:
            raise RuntimeError("Model not loaded")

        tfidf = self.bundle["tfidf"]
        hf = self.bundle["handcrafted"]
        clf = self.bundle["classifier"]
        le = self.bundle["label_encoder"]
        labels = self.bundle["labels"]

        tfidf_vec = tfidf.transform([text])
        hf_vec = hf.transform([text])
        X = combine_features(tfidf_vec, hf_vec)

        # Probabilities
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
        else:
            # Shouldn't happen since we wrap LinearSVC in CalibratedClassifierCV
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
