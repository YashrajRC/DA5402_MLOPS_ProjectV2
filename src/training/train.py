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
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.training.features import HandcraftedFeatures

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
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params.get("solver", "liblinear"),
            penalty=params.get("penalty", "l2"),
            class_weight="balanced",
        )
    if kind == "linearsvc":
        base = LinearSVC(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight="balanced",
        )
        return CalibratedClassifierCV(base, cv=3)
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            reg_alpha=params.get("reg_alpha", 0),
            reg_lambda=params.get("reg_lambda", 1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            min_child_weight=params.get("min_child_weight", 1),
            tree_method=params.get("tree_method", "hist"),
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unknown model: {kind}")


def plot_confusion(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
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
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
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
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp_name = "mental_health_classifier"

    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(exp_name)

    print("Loading data...")
    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)

    # Split off a validation set from training data (15%)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                 random_state=params["prepare"]["random_state"])
    idx_train, idx_val = next(sss.split(train_df, train_df["label"]))
    inner_df = train_df.iloc[idx_train].reset_index(drop=True)
    val_df   = train_df.iloc[idx_val].reset_index(drop=True)
    print(f"Inner train: {len(inner_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    le = LabelEncoder()
    y_inner = le.fit_transform(inner_df["label"].astype(str))
    y_val   = le.transform(val_df["label"].astype(str))
    y_test  = le.transform(test_df["label"].astype(str))
    labels  = list(le.classes_)

    print("Building features (fit on inner train)...")
    tfidf_word = TfidfVectorizer(
        max_features=feat_params["max_tfidf_features"],
        ngram_range=tuple(feat_params["ngram_range"]),
        min_df=feat_params["min_df"],
        max_df=feat_params["max_df"],
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_char = TfidfVectorizer(
        max_features=feat_params["max_tfidf_char_features"],
        ngram_range=tuple(feat_params["char_ngram_range"]),
        min_df=feat_params["min_df"],
        max_df=feat_params["max_df"],
        sublinear_tf=True,
        analyzer="char_wb",
    )

    hf = HandcraftedFeatures()

    word_inner = tfidf_word.fit_transform(inner_df["text"].astype(str))
    word_val   = tfidf_word.transform(val_df["text"].astype(str))
    word_test  = tfidf_word.transform(test_df["text"].astype(str))

    char_inner = tfidf_char.fit_transform(inner_df["text"].astype(str))
    char_val   = tfidf_char.transform(val_df["text"].astype(str))
    char_test  = tfidf_char.transform(test_df["text"].astype(str))

    hf_inner = hf.fit_transform(inner_df["text"].tolist())
    hf_val   = hf.transform(val_df["text"].tolist())
    hf_test  = hf.transform(test_df["text"].tolist())

    X_inner = hstack([word_inner, char_inner, csr_matrix(hf_inner)])
    X_val   = hstack([word_val,   char_val,   csr_matrix(hf_val)])
    X_test  = hstack([word_test,  char_test,  csr_matrix(hf_test)])

    print(f"Feature matrix: {X_inner.shape}")

    clf = make_classifier(args.model, model_params)

    with mlflow.start_run(run_name=args.model) as run:
        mlflow.log_params({**model_params, **feat_params, "model_type": args.model})
        mlflow.log_param("n_features", X_inner.shape[1])
        mlflow.log_param("n_train_inner", X_inner.shape[0])
        mlflow.log_param("n_val", X_val.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])

        print(f"Training {args.model} on inner split ({X_inner.shape[0]} samples)...")
        clf.fit(X_inner, y_inner)

        y_val_pred = clf.predict(X_val)
        val_acc      = accuracy_score(y_val, y_val_pred)
        val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
        val_f1_w     = f1_score(y_val, y_val_pred, average="weighted")
        print(f"  Val  → acc={val_acc:.4f}  macro-F1={val_f1_macro:.4f}  weighted-F1={val_f1_w:.4f}")
        mlflow.log_metrics({
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1_macro,
            "val_weighted_f1": val_f1_w,
        })

        y_pred        = clf.predict(X_test)
        acc           = accuracy_score(y_test, y_pred)
        f1_macro      = f1_score(y_test, y_pred, average="macro")
        f1_weighted   = f1_score(y_test, y_pred, average="weighted")
        print(f"  Test → acc={acc:.4f}  macro-F1={f1_macro:.4f}  weighted-F1={f1_weighted:.4f}")
        mlflow.log_metrics({
            "accuracy": acc,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
        })

        report = classification_report(
            y_test, y_pred, target_names=labels, output_dict=True, zero_division=0
        )
        for label in labels:
            mlflow.log_metric(f"f1_{label.replace(' ', '_')}", report[label]["f1-score"])

        cm = confusion_matrix(y_test, y_pred)
        cm_path = METRICS_DIR / f"{args.model}_confusion.png"
        plot_confusion(cm, labels, cm_path)
        mlflow.log_artifact(str(cm_path))

        bundle = {
            "tfidf_word": tfidf_word,
            "tfidf_char": tfidf_char,
            "handcrafted": hf,
            "classifier": clf,
            "label_encoder": le,
            "labels": labels,
            "model_type": args.model,
        }
        bundle_path = MODELS_DIR / f"{args.model}_bundle.joblib"
        joblib.dump(bundle, bundle_path)
        mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")

        if args.model == "xgboost":
            mlflow.xgboost.log_model(
                clf, artifact_path="model",
                registered_model_name="mental_health_classifier",
            )
        else:
            mlflow.sklearn.log_model(
                clf, artifact_path="model",
                registered_model_name="mental_health_classifier",
            )

        # DVC metric JSON (uses test metrics for model comparison)
        metrics_out = METRICS_DIR / f"{args.model}_metrics.json"
        with open(metrics_out, "w") as f:
            json.dump({
                "accuracy": acc,
                "macro_f1": f1_macro,
                "weighted_f1": f1_weighted,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1_macro,
                "run_id": run.info.run_id,
            }, f, indent=2)

        print(f"Done. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()