"""
Training entry point for all 3 models.
Logs parameters, metrics, and model artifacts to MLflow.
Also writes a metrics JSON for DVC.

Usage:
    python src/training/train.py --model logreg
    python src/training/train.py --model linearsvc
    python src/training/train.py --model xgboost
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.training.features import HandcraftedFeatures, build_tfidf, combine_features

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "processed" / "train.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
METRICS_DIR = ROOT / "metrics"
MODELS_DIR = ROOT / "models"


def load_params():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def make_classifier(kind: str, params: dict):
    if kind == "logreg":
        return LogisticRegression(
            C=params["C"], max_iter=params["max_iter"],
            class_weight="balanced", n_jobs=-1,
        )
    if kind == "linearsvc":
        base = LinearSVC(C=params["C"], max_iter=params["max_iter"], class_weight="balanced")
        return CalibratedClassifierCV(base, cv=3)
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_jobs=-1, eval_metric="mlogloss",
        )
    raise ValueError(f"Unknown model: {kind}")


def plot_confusion(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["logreg", "linearsvc", "xgboost"])
    args = parser.parse_args()

    params = load_params()
    feat_params = params["features"]
    model_params = params["train"][args.model]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("mental_health_classifier")

    print(f"Loading data...")
    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    y_test = le.transform(test_df["label"].astype(str))
    labels = list(le.classes_)

    print(f"Building features...")
    tfidf = build_tfidf(feat_params["max_tfidf_features"], feat_params["ngram_range"])
    hf = HandcraftedFeatures()

    tfidf_train = tfidf.fit_transform(train_df["text"].astype(str))
    tfidf_test = tfidf.transform(test_df["text"].astype(str))
    hf_train = hf.fit_transform(train_df["text"].tolist())
    hf_test = hf.transform(test_df["text"].tolist())

    X_train = combine_features(tfidf_train, hf_train)
    X_test = combine_features(tfidf_test, hf_test)

    clf = make_classifier(args.model, model_params)

    with mlflow.start_run(run_name=f"{args.model}") as run:
        mlflow.log_params({**model_params, **feat_params, "model_type": args.model})

        print(f"Training {args.model}...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy: {acc:.4f}  Macro-F1: {f1_macro:.4f}  Weighted-F1: {f1_weighted:.4f}")

        mlflow.log_metrics({
            "accuracy": acc,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
        })

        # Per-class F1
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
        for label in labels:
            mlflow.log_metric(f"f1_{label}", report[label]["f1-score"])

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_path = METRICS_DIR / f"{args.model}_confusion.png"
        plot_confusion(cm, labels, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Save & log the full pipeline artifacts
        bundle = {
            "tfidf": tfidf,
            "handcrafted": hf,
            "classifier": clf,
            "label_encoder": le,
            "labels": labels,
            "model_type": args.model,
        }
        bundle_path = MODELS_DIR / f"{args.model}_bundle.joblib"
        joblib.dump(bundle, bundle_path)
        mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")

        # Also register as MLflow sklearn model for model-registry workflow
        if args.model == "xgboost":
            mlflow.xgboost.log_model(clf, artifact_path="model",
                                     registered_model_name="mental_health_classifier")
        else:
            mlflow.sklearn.log_model(clf, artifact_path="model",
                                     registered_model_name="mental_health_classifier")

        # DVC metric JSON
        metrics_out = METRICS_DIR / f"{args.model}_metrics.json"
        with open(metrics_out, "w") as f:
            json.dump({
                "accuracy": acc,
                "macro_f1": f1_macro,
                "weighted_f1": f1_weighted,
                "run_id": run.info.run_id,
            }, f, indent=2)

        print(f"Done. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
