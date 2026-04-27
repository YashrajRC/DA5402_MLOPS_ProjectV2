# Experiment Report
## Mental Health Text Classification — Model Comparison

---

## 1. Objective

Compare three classical ML classifier families for 7-class mental health text classification. All models share the same feature extraction pipeline (TF-IDF + handcrafted features). The goal is to identify the best model by macro-F1 for promotion to the MLflow `champion` alias.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Kaggle "Sentiment Analysis for Mental Health" |
| Origin | Reddit posts from mental health subreddits |
| Total samples | 52,675 |
| Train set | 42,140 (80%, stratified) |
| Inner train | 35,819 (85% of train, for fitting) |
| Validation set | 6,321 (15% of train, held out during fitting) |
| Test set | 10,535 (20%, never seen during training) |
| Random seed | 42 |

### Class Distribution (Training Set)

| Class | Samples | % |
|---|---|---|
| Normal | 13,070 | 31.0% |
| Depression | 12,323 | 29.2% |
| Suicidal | 8,521 | 20.2% |
| Anxiety | 3,073 | 7.3% |
| Bipolar | 2,222 | 5.3% |
| Stress | 2,069 | 4.9% |
| Personality Disorder | 862 | 2.0% |

---

## 3. Feature Engineering

All three models use the same 40,012-dimensional feature vector:

| Feature Group | Detail | Dimensions |
|---|---|---|
| Word-level TF-IDF | 1–3 n-grams, 30k vocab, sublinear TF, min_df=2, max_df=0.95 | 30,000 |
| Char-level TF-IDF | 3–4 char n-grams, 10k vocab, `analyzer=char_wb` | 10,000 |
| Handcrafted | Character count, word count, avg word length, punctuation density, uppercase ratio, first-person ratio, `!` count, `?` count, VADER compound/pos/neg/neu | 12 |
| **Total** | | **40,012** |

Feature fitting is performed on the inner-train split only (35,819 samples). Vectorizers are then applied to validation and test sets to prevent data leakage.

---

## 4. Models and Hyperparameters

### 4.1 Logistic Regression (MLflow v7)

| Parameter | Value |
|---|---|
| C | 2.0 |
| Solver | liblinear |
| Penalty | L2 |
| Max iterations | 2,000 |
| Class weight | balanced |

**Rationale:** Strong probabilistic baseline for text; liblinear is well-suited for high-dimensional sparse features. Balanced class weights compensate for the Personality Disorder minority class (2%).

### 4.2 LinearSVC with Calibration (MLflow v9)

| Parameter | Value |
|---|---|
| C | 0.5 |
| Max iterations | 2,000 |
| Class weight | balanced |
| Calibration | CalibratedClassifierCV (cv=3) |

**Rationale:** LinearSVC finds maximum-margin decision boundaries. CalibratedClassifierCV wraps it to produce valid probabilities (required for the `/predict` endpoint which returns per-class scores).

### 4.3 XGBoost (MLflow v8 — Champion)

| Parameter | Value |
|---|---|
| n_estimators | 250 |
| max_depth | 4 |
| learning_rate | 0.15 |
| reg_alpha | 0.1 (L1) |
| reg_lambda | 1.0 (L2) |
| subsample | 0.85 |
| colsample_bytree | 0.85 |
| min_child_weight | 3 |
| tree_method | hist |
| eval_metric | mlogloss |

**Rationale:** Gradient-boosted trees capture non-linear feature interactions that linear classifiers miss. Regularisation (L1 + L2) and subsampling prevent overfitting on the high-dimensional sparse input.

---

## 5. Results

### 5.1 Overall Performance

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|---|---|---|---|---|---|
| Logistic Regression (v7) | 0.7928 | 0.7672 | 0.7978 | 0.7729 | 0.7969 |
| **XGBoost (v8) ← Champion** | **0.8070** | **0.7767** | **0.8042** | **0.7803** | **0.8025** |
| LinearSVC (v9) | 0.6921 | 0.5237 | 0.6868 | 0.5176 | 0.6643 |

### 5.2 Per-Class F1 — Champion Model (XGBoost v8)

| Class | Test F1 | Training samples | Notes |
|---|---|---|---|
| Normal | **0.929** | 13,070 | Largest class; very high F1 |
| Bipolar | **0.847** | 2,222 | Strong despite small class size |
| Anxiety | **0.833** | 3,073 | Clear lexical signal |
| Personality Disorder | **0.759** | 862 | Good given only 2% of training data |
| Depression | **0.756** | 12,323 | Some confusion with Suicidal |
| Suicidal | **0.697** | 8,521 | Overlap with Depression vocabulary |
| Stress | **0.641** | 2,069 | Lowest F1; vocabulary overlaps with Normal |

### 5.3 Key Observations

1. **XGBoost outperforms Logistic Regression** by 0.74pp macro-F1 on the test set. Non-linear feature interactions (between word frequency and sentiment scores, for example) provide meaningful signal.

2. **LinearSVC underperforms significantly** (macro-F1 0.518). Despite being theoretically appropriate for high-dimensional text, the CalibratedClassifierCV wrapper with 3-fold CV appears to overfit to the calibration folds given the imbalanced class distribution.

3. **Normal and Bipolar achieve the highest F1** despite very different class sizes. Normal has abundant training data; Bipolar's vocabulary (mania, mood swings, cycling) appears highly discriminative.

4. **Stress is the hardest class** (F1 = 0.641). Work/deadline language overlaps substantially with Normal-class descriptions of everyday life. This is an inherent dataset ambiguity.

5. **Depression and Suicidal are frequently confused** (both depressive-spectrum). The model correctly distinguishes them most of the time (F1 ≥ 0.70 for both) but some suicidal content is labelled as Depression.

6. **Validation split reveals no overfitting**: XGBoost val macro-F1 (0.777) is very close to test macro-F1 (0.780), confirming the model generalises well.

---

## 6. Promotion Decision

The `promote` DVC stage runs `scripts/promote_model.py` which:
1. Queries all registered MLflow versions
2. Retrieves each run's `macro_f1` metric
3. Sets the `champion` alias on the highest-scoring version
4. Archives the previous champion as `challenger`

XGBoost v8 was manually force-promoted as champion because it was trained with the proper validation split (v4, the previous champion, was trained without a validation split and lacks validation metrics for display).

---

## 7. Reproducing the Experiments

```bash
# Retrain all models
docker compose exec -T trainer mlflow run /app -e train -P model=logreg --env-manager local --experiment-name mental_health_classifier
docker compose exec -T trainer mlflow run /app -e train -P model=linearsvc --env-manager local --experiment-name mental_health_classifier
docker compose exec -T trainer mlflow run /app -e train -P model=xgboost --env-manager local --experiment-name mental_health_classifier

# Promote best model
docker compose exec -T trainer python /app/scripts/promote_model.py

# View metrics comparison
# Open http://localhost:5000 → mental_health_classifier experiment
# Click "Chart view" → select macro_f1 metric to compare all runs
```

---

## 8. Future Work

| Direction | Expected Benefit |
|---|---|
| Fine-tune DistilBERT or RoBERTa | Transformer models capture long-range context; likely +5-8pp macro-F1 |
| Oversample Personality Disorder | 862 samples is very few; synthetic oversampling (SMOTE on embeddings) could help |
| Ensemble logreg + xgboost | Soft-voting ensemble may stabilise predictions across class types |
| Add character-level CNN features | Capture misspellings and informal writing style common in Reddit data |
| Active learning from feedback logs | Route low-confidence predictions to human reviewers; re-train on corrected labels |
| Separate Suicidal from Depression model | A binary classifier specifically for crisis detection would be higher precision |
