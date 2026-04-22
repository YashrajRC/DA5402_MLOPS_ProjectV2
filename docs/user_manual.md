# User Manual — Mental Health Text Classifier

## Who is this for?

This manual is for a non-technical user who just wants to try the classifier.

## Opening the application

1. Open your web browser
2. Go to **http://localhost:8501**
3. You should see the "Mental Health Text Classification" page

## Using the classifier

### Step 1 — Paste your text

Type or paste any text into the large box labeled "Text to analyze". You can use:
- a journal entry
- a social media post
- any written expression

The character counter below shows how long your text is (max 10,000).

### Step 2 — Click Analyze

Click the **Analyze** button. It becomes active once you've typed something.

### Step 3 — Read the result

Within a second you'll see:
- **Predicted** category (one of: Normal, Depression, Anxiety, Bipolar, PTSD, Stress, Personality Disorder)
- **Confidence** — how sure the model is (higher is better)
- A **probability chart** showing how the model scored all 7 categories

### Step 4 — Give feedback

Below the chart you can tell the system if the prediction was correct or not. If wrong, pick the correct label. This helps improve future versions.

## Important disclaimer

> This is a pattern-detection tool, **not a clinical diagnosis**.
> Results should not replace professional mental health evaluation.
> If you or someone you know is in crisis, please contact a licensed professional or a crisis helpline.

## What to do if something breaks

- **Page doesn't load** → check that `docker compose up` finished without errors
- **"API unreachable"** shown in sidebar → the FastAPI backend isn't running; restart with `docker compose restart fastapi`
- **Prediction takes > 5 seconds** → first request loads the model into memory; subsequent ones are fast

## Other dashboards (for curious users)

| URL | What it shows |
|---|---|
| http://localhost:3000 | Grafana — live system metrics |
| http://localhost:5000 | MLflow — all trained model runs |
| http://localhost:8080 | Airflow — data pipeline runs |
| http://localhost:9090 | Prometheus — raw metric queries |
