import json
import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.clean import clean_text

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "data.csv"
TRAIN_OUT = ROOT / "data" / "processed" / "train.csv"
TEST_OUT = ROOT / "data" / "processed" / "test.csv"
BASELINE_OUT = ROOT / "data" / "baseline_stats.json"


def load_params():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def compute_baseline_stats(df: pd.DataFrame, sample_size: int = 500) -> dict:
    texts = df["text"].astype(str)
    lengths = texts.str.split().str.len()

    # Stratified sample for word-frequency baseline
    sample_df = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(
            min(len(g), max(1, int(sample_size * len(g) / len(df)))),
            random_state=42,
        ),
        include_groups=False,
    ).reset_index(level=0)
    sample_texts = sample_df["text"].astype(str)

    from collections import Counter
    all_words = " ".join(sample_texts.tolist()).split()
    word_counts = Counter(all_words)
    top_words = dict(word_counts.most_common(1000))
    total = sum(top_words.values()) or 1
    word_freq = {w: c / total for w, c in top_words.items()}

    return {
        "avg_text_length": float(lengths.mean()),
        "std_text_length": float(lengths.std()),
        "class_distribution": df["label"].value_counts(normalize=True).to_dict(),
        "vocab_size": len(set(all_words)),
        "word_freq_top1000": word_freq,
        "total_samples": int(len(df)),
    }


def main():
    params = load_params()["prepare"]
    print(f"Reading {RAW}")
    df = pd.read_csv(RAW)

    if "statement" in df.columns and "status" in df.columns:
        df = df.rename(columns={"statement": "text", "status": "label"})
    elif "text" not in df.columns or "label" not in df.columns:
        # Fallback: assume first non-id column is text, last is label
        cols = [c for c in df.columns if c.lower() not in ("id", "unnamed: 0")]
        df = df[cols].copy()
        df.columns = ["text", "label"] if len(cols) == 2 else df.columns

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() >= params["min_text_length"]]

    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["label"],
    )

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    stats = compute_baseline_stats(train)
    with open(BASELINE_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Classes: {dict(train['label'].value_counts())}")
    print(f"Baseline stats → {BASELINE_OUT}")


if __name__ == "__main__":
    main()
