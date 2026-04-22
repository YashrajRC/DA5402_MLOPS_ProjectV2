# Experiment Report

## Objective

Compare three classical ML approaches for mental health text classification. Same features (TF-IDF + handcrafted), different classifiers.

## Dataset

Kaggle "Sentiment Analysis for Mental Health" (~53K Reddit posts, 7 classes). 80/20 stratified split.

## Features

- **TF-IDF**: max 10,000 features, n-grams (1, 2), English stopwords removed
- **Handcrafted (12)**: text length, token count, avg word length, punct density, uppercase ratio, first-person ratio, `!` count, `?` count, VADER (compound/pos/neg/neu)

## Models & Results

_(Fill in actual numbers after running `dvc repro`)_

| Model | Accuracy | Macro-F1 | Weighted-F1 | Training time |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| LinearSVC (calibrated) | — | — | — | — |
| XGBoost | — | — | — | — |

## How to reproduce

```bash
dvc repro
dvc metrics show
dvc plots diff
```

## Chosen model

The model with the highest macro-F1 is promoted to MLflow `Production` stage via:
```bash
python scripts/promote_model.py
```

## Observations

_(fill after running)_
- Which classes are hardest?
- Where do the models disagree?
- How much does adding handcrafted features help vs TF-IDF alone?

## Future work

- Fine-tune a transformer (DistilBERT) if GPU becomes available
- Active learning using the `/feedback` log
- Class-weighted sampling for the minority class (Personality Disorder)
