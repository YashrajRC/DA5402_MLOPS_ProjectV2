"""Streamlit UI — 2 screens: Analyze + Result (with feedback)."""
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Mental Health Text Classifier", page_icon="🧠", layout="wide")

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# Sidebar
with st.sidebar:
    st.title("🧠 MH Classifier")
    st.markdown("---")
    st.markdown("**Disclaimer**")
    st.info(
        "This is a pattern-detection tool, not a clinical diagnosis. "
        "Results should not replace professional mental health evaluation."
    )
    st.markdown("---")
    st.caption(f"API: `{API_URL}`")
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API: {h.get('status')} · container `{h.get('container_id','?')[:12]}`")
    except Exception:
        st.error("API unreachable")
    st.markdown("---")
    st.markdown(
        "**Other Dashboards**\n"
        "- [MLflow](http://localhost:5000)\n"
        "- [Grafana](http://localhost:3000)\n"
        "- [Prometheus](http://localhost:9090)\n"
        "- [Airflow](http://localhost:8080)"
    )

st.title("Mental Health Text Classification")
st.caption("Paste text — the model predicts one of 7 categories.")

col_input, col_meta = st.columns([3, 1])

with col_input:
    text = st.text_area(
        "Text to analyze",
        height=180,
        placeholder="e.g. I've been feeling overwhelmed and can't sleep lately...",
        key="text_input",
    )
    st.caption(f"{len(text)} / 10000 characters")

    if st.button("Analyze", type="primary", disabled=(len(text.strip()) == 0)):
        try:
            with st.spinner("Analyzing..."):
                r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
                r.raise_for_status()
                st.session_state.result = r.json()
                st.session_state.last_text = text
        except Exception as e:
            st.error(f"Request failed: {e}")

with col_meta:
    if st.session_state.result:
        r = st.session_state.result
        st.metric("Predicted", r["predicted_class"])
        st.metric("Confidence", f"{r['confidence']*100:.1f}%")
        st.progress(r["confidence"])
        st.caption(f"Model v{r['model_version']} ({r['model_stage']})")
        st.caption(f"Container: `{r['container_id'][:12]}`")
        st.caption(f"Drift score: {r['drift_score']:.3f}")

# Result panel
if st.session_state.result:
    st.markdown("---")
    st.subheader("Class Probabilities")

    probs = st.session_state.result["probabilities"]
    df = (
        pd.DataFrame({"class": list(probs.keys()), "probability": list(probs.values())})
        .sort_values("probability", ascending=True)
    )
    fig = px.bar(
        df, x="probability", y="class", orientation="h",
        text=df["probability"].apply(lambda x: f"{x*100:.1f}%"),
        color="probability", color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feedback")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        correct = st.radio("Was this correct?", ["Correct", "Incorrect"], horizontal=True)

    with col_b:
        correct_label = None
        if correct == "Incorrect":
            correct_label = st.selectbox("What's the correct label?", list(probs.keys()))

    if st.button("Submit feedback"):
        try:
            requests.post(
                f"{API_URL}/feedback",
                json={
                    "text": st.session_state.last_text,
                    "predicted_label": st.session_state.result["predicted_class"],
                    "correct_label": correct_label,
                    "was_correct": (correct == "Correct"),
                },
                timeout=5,
            )
            st.success("Feedback logged. Thanks!")
        except Exception as e:
            st.error(f"Failed: {e}")
