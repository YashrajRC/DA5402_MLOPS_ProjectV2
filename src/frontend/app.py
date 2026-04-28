import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL        = os.getenv("API_URL",        "http://localhost:8000")
MLFLOW_URL     = os.getenv("MLFLOW_URL",     "http://localhost:5000")
GRAFANA_URL    = os.getenv("GRAFANA_URL",    "http://localhost:3001")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
AIRFLOW_URL    = os.getenv("AIRFLOW_URL",    "http://localhost:8080")

CAT = {
    "Normal":               {"bg": "#e8f5e9", "fg": "#1b5e20", "bar": "#2e7d32", "icon": "😊"},
    "Depression":           {"bg": "#e8eaf6", "fg": "#1a237e", "bar": "#283593", "icon": "😔"},
    "Anxiety":              {"bg": "#fff3e0", "fg": "#bf360c", "bar": "#e64a19", "icon": "😰"},
    "Bipolar":              {"bg": "#f3e5f5", "fg": "#4a148c", "bar": "#6a1b9a", "icon": "🔄"},
    "Suicidal":             {"bg": "#ffebee", "fg": "#b71c1c", "bar": "#c62828", "icon": "🆘"},
    "PTSD":                 {"bg": "#ffebee", "fg": "#b71c1c", "bar": "#c62828", "icon": "⚠️"},
    "Stress":               {"bg": "#fffde7", "fg": "#f57f17", "bar": "#f9a825", "icon": "😤"},
    "Personality Disorder": {"bg": "#e0f2f1", "fg": "#004d40", "bar": "#00695c", "icon": "🔹"},
    "Personality disorder": {"bg": "#e0f2f1", "fg": "#004d40", "bar": "#00695c", "icon": "🔹"},
}
DEFAULT_CAT = {"bg": "#e3f2fd", "fg": "#0d47a1", "bar": "#1565c0", "icon": "🧠"}


def cat_style(label: str) -> dict:
    for k, v in CAT.items():
        if k.lower() in label.lower():
            return v
    return DEFAULT_CAT


st.set_page_config(
    page_title="MH Text Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Global font size bump */
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, div {
    font-size: 1.12rem;
}

/* Tab labels — target the <p> inside each tab button */
button[data-baseweb="tab"] p,
div[data-testid="stTabs"] button[role="tab"] p,
[role="tab"] p {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}

/* Compact top padding */
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Result card — inline styles set both bg and fg, no theme conflict */
.result-card {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    margin-bottom: 0.8rem;
}
.result-card .cat-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    opacity: 0.8;
}
.result-card .cat-name  { font-size: 2rem; font-weight: 800; margin: 0; line-height: 1.1; }
.result-card .cat-conf  { font-size: 1rem; margin-top: 0.4rem; opacity: 0.85; }

/* Confidence bar */
.conf-bar-wrap { background: rgba(0,0,0,0.08); border-radius: 6px; height: 10px; margin: 0.5rem 0 1rem; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 6px; transition: width .4s ease; }

/* Tool card */
.tool-card {
    border: 1px solid #dde3ea;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    background: #ffffff;
}
.tool-card a { text-decoration: none; font-weight: 700; font-size: 1rem; }
.tool-card p { font-size: 0.8rem; color: #555; margin: 0.3rem 0 0; }

/* Category guide row */
.cat-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0.8rem;
    border-radius: 9px;
    margin-bottom: 0.4rem;
    border-left: 4px solid;
}
.cat-row .cat-title { font-weight: 700; font-size: 0.88rem; }
.cat-row .cat-desc  { font-size: 0.78rem; opacity: 0.8; }

/* Meta pills */
.pill {
    display: inline-block;
    background: #f0f4f8;
    border: 1px solid #dde3ea;
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.76rem;
    color: #445;
    margin: 0.15rem 0.15rem 0.15rem 0;
}

/* Feedback box */
.fb-done {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)


for k, v in [("result", None), ("last_text", ""), ("feedback_given", False)]:
    if k not in st.session_state:
        st.session_state[k] = v


with st.sidebar:
    st.markdown("## 🧠 MH Classifier")
    st.caption("Mental Health Text Analysis System")

    st.divider()

    # API health
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        cid = h.get("container_id", "?")[:12]
        model_loaded = h.get("model_loaded", False)
        if model_loaded:
            st.success(f"**API online**\nContainer: `{cid}`")
        else:
            st.warning("API up — model not loaded yet")
    except Exception:
        st.error("**API unreachable**")

    st.divider()

    st.markdown("**🔗 MLOps Tools**")

    tools = [
        ("📊 MLflow",     MLFLOW_URL,     "Experiments & model registry"),
        ("📈 Grafana",    GRAFANA_URL,     "Live metrics & alerts"),
        ("🔭 Prometheus", PROMETHEUS_URL,  "Raw metrics & query explorer"),
        ("🔄 Airflow",    AIRFLOW_URL,     "Data pipeline management"),
    ]
    for name, url, tip in tools:
        st.markdown(f"[{name}]({url}) — <small>{tip}</small>", unsafe_allow_html=True)

    st.divider()
    st.caption(
        "⚠️ This tool detects patterns in text. "
        "**It is not a clinical diagnosis.** "
        "If you're in crisis, please contact a licensed mental-health professional."
    )


tab_analyze, tab_dashboard, tab_about = st.tabs([
    "🔍 Analyze Text",
    "📊 Model & System Dashboard",
    "📚 About & Help",
])


with tab_analyze:
    st.markdown("### Paste any text to get a mental-health classification")
    st.warning(
        "⚠️ **Important:** This is a pattern-detection tool, not a clinical diagnosis. "
        "If you or someone you know is struggling, please reach out to a professional or helpline.",
        icon=None,
    )

    input_col, result_col = st.columns([3, 2], gap="large")

    with input_col:
        text = st.text_area(
            "Your text",
            height=200,
            placeholder=(
                "Paste a journal entry, social media post, or any personal writing here.\n\n"
                "Example: \"I've been feeling really overwhelmed lately and can't seem to sleep. "
                "Every small thing makes me anxious and I don't know how to cope...\""
            ),
            label_visibility="collapsed",
        )
        char_count = len(text)

        btn_col, char_col = st.columns([2, 5])
        with btn_col:
            analyze = st.button(
                "Analyze →",
                type="primary",
                use_container_width=True,
            )
        with char_col:
            if char_count > 9500:
                st.error(f"{char_count:,} / 10,000 characters — almost at limit")
            else:
                st.caption(f"{char_count:,} / 10,000 characters")

        if analyze:
            if char_count < 3:
                st.warning("Please enter at least 3 characters before analyzing.")
            else:
                st.session_state.feedback_given = False
                with st.spinner("Analyzing text..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/predict", json={"text": text}, timeout=15
                        )
                        resp.raise_for_status()
                        st.session_state.result    = resp.json()
                        st.session_state.last_text = text
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach the API. Check that Docker Compose is running.")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"API error {e.response.status_code}: {e.response.text}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    with result_col:
        if st.session_state.result:
            r    = st.session_state.result
            s    = cat_style(r["predicted_class"])
            conf = r["confidence"] * 100

            st.markdown(
                f"<div class='result-card' style='background:{s['bg']};color:{s['fg']};'>"
                f"<div class='cat-eyebrow'>Predicted Category</div>"
                f"<div class='cat-name'>{s['icon']} {r['predicted_class']}</div>"
                f"<div class='cat-conf'>{conf:.1f}% confidence</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='conf-bar-wrap'>"
                f"<div class='conf-bar-fill' style='width:{conf:.1f}%;background:{s['bar']};'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='pill'>Model v{r['model_version']}</span>"
                f"<span class='pill'>Stage: {r['model_stage']}</span>"
                f"<span class='pill'>Drift: {r['drift_score']:.3f}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("📝 Your classification result will appear here after you click Analyze.", icon=None)

    if st.session_state.result:
        st.divider()
        chart_col, fb_col = st.columns([5, 2], gap="large")

        with chart_col:
            st.markdown("#### 📊 Probability breakdown")
            probs = st.session_state.result["probabilities"]
            sorted_items = sorted(probs.items(), key=lambda x: x[1])
            labels_sorted = [x[0] for x in sorted_items]
            values_sorted = [x[1] for x in sorted_items]
            bar_colors    = [cat_style(l)["bar"] for l in labels_sorted]

            fig = go.Figure(go.Bar(
                x=values_sorted,
                y=labels_sorted,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v*100:.1f}%" for v in values_sorted],
                textposition="outside",
                textfont=dict(color="#1e2d3d", size=12),
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=150, r=90, t=10, b=30),
                xaxis=dict(
                    tickformat=".0%", range=[0, 1.2],
                    showgrid=True, gridcolor="#ddd", zeroline=False,
                    tickfont=dict(color="#1e2d3d", size=12),
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="#1e2d3d", size=13),
                    automargin=True,
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=13, color="#1e2d3d"),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with fb_col:
            st.markdown("#### 💬 Was this correct?")
            if st.session_state.feedback_given:
                st.markdown(
                    "<div class='fb-done'>"
                    "<div style='font-size:1.5rem;'>✅</div>"
                    "<div style='font-weight:700;margin-top:0.3rem;'>Thank you!</div>"
                    "<div style='font-size:0.82rem;margin-top:0.2rem;'>"
                    "Your feedback helps improve the model.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Give different feedback", use_container_width=True):
                    st.session_state.feedback_given = False
                    st.rerun()
            else:
                correct = st.radio(
                    "Prediction quality",
                    ["✓ Correct", "✗ Incorrect"],
                    horizontal=True,
                    label_visibility="collapsed",
                )
                correct_label = None
                if correct == "✗ Incorrect":
                    correct_label = st.selectbox(
                        "What is the correct category?",
                        list(probs.keys()),
                    )
                if st.button("Submit", type="primary", use_container_width=True):
                    try:
                        requests.post(
                            f"{API_URL}/feedback",
                            json={
                                "text":            st.session_state.last_text,
                                "predicted_label": st.session_state.result["predicted_class"],
                                "correct_label":   correct_label,
                                "was_correct":     (correct == "✓ Correct"),
                            },
                            timeout=5,
                        )
                        st.session_state.feedback_given = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not submit: {e}")
                st.caption(
                    "Incorrect predictions are logged and used to retrain the model."
                )


with tab_dashboard:

    st.markdown("### 🤖 Active Model")
    try:
        info = requests.get(f"{API_URL}/model_info", timeout=5).json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version",    f"v{info.get('version','?')}")
        c2.metric("Stage",      info.get("stage", "?").upper())
        c3.metric("Model Type", info.get("model_type", "?").upper())
        c4.metric("Classes",    len(info.get("labels", [])))

        m = info.get("metrics", {})
        if m and m.get("accuracy"):
            st.markdown("#### 📈 Performance (held-out test set)")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Accuracy",     f"{m.get('accuracy', 0)*100:.1f}%")
            p2.metric("Macro-F1",     f"{m.get('macro_f1', 0):.3f}")
            p3.metric("Weighted-F1",  f"{m.get('weighted_f1', 0):.3f}")
            p4.metric("Val Macro-F1", f"{m.get('val_macro_f1', 0):.3f}"
                      if m.get("val_macro_f1") else "—")
    except Exception as e:
        st.warning(f"Could not fetch model info from API: {e}")

    st.divider()

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("#### 📊 Training Data — Class Distribution")
        try:
            info = requests.get(f"{API_URL}/model_info", timeout=5).json()
            d    = info.get("class_distribution", {})
            if d:
                labels_d = list(d.keys())
                values_d = list(d.values())
                colors_d = [cat_style(l)["bar"] for l in labels_d]
                fig2 = go.Figure(go.Pie(
                    labels=labels_d,
                    values=values_d,
                    hole=0.42,
                    marker_colors=colors_d,
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{percent}<extra></extra>",
                ))
                fig2.update_layout(
                    height=340,
                    margin=dict(l=0, r=0, t=10, b=10),
                    showlegend=False,
                    paper_bgcolor="white",
                    font=dict(size=12, color="#1e2d3d"),
                )
                st.plotly_chart(fig2, use_container_width=True, theme=None)
            else:
                st.info("Class distribution unavailable — model may not be loaded.")
        except Exception:
            st.info("Class distribution will appear once the model is loaded.")

    with right_col:
        st.markdown("#### ⚡ Live System Metrics")
        try:
            stats = requests.get(f"{API_URL}/live_stats", timeout=5).json()
            s1, s2 = st.columns(2)
            s1.metric("Total Predictions", f"{stats.get('total_predictions', 0):,}")
            s2.metric("Feedback Received", f"{stats.get('feedback_count', 0):,}")
            s3, s4 = st.columns(2)
            s3.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.0f} ms")
            drift_val = stats.get("drift_score", 0)
            drift_delta = "⚠️ HIGH" if drift_val > 0.3 else ("↗ rising" if drift_val > 0.2 else None)
            s4.metric("Drift Score", f"{drift_val:.3f}", delta=drift_delta,
                      delta_color="inverse" if drift_val > 0.2 else "off")
        except Exception:
            st.info("Live stats will populate after predictions are made.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔗 External Tools")
        tool_cols = st.columns(2)
        ext_tools = [
            ("📊", "MLflow",     MLFLOW_URL,     "#1565c0"),
            ("📈", "Grafana",    GRAFANA_URL,     "#e65100"),
            ("🔭", "Prometheus", PROMETHEUS_URL,  "#cc3300"),
            ("🔄", "Airflow",    AIRFLOW_URL,     "#017cee"),
        ]
        for i, (icon, name, url, color) in enumerate(ext_tools):
            with tool_cols[i % 2]:
                st.markdown(
                    f"<div class='tool-card' style='margin-bottom:0.6rem;'>"
                    f"<div style='font-size:1.6rem;'>{icon}</div>"
                    f"<a href='{url}' target='_blank' style='color:{color};'>{name} ↗</a>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


with tab_about:
    about_left, about_right = st.columns([3, 2], gap="large")

    with about_left:
        st.markdown("### What is this tool?")
        st.markdown(
            "This application uses **machine learning** to read a piece of text "
            "and predict which of 7 mental-health categories it is most likely associated with. "
            "It was trained on thousands of real posts from people discussing their mental-health experiences."
        )

        st.markdown("### What it can and cannot do")
        st.error("❌  It **cannot** diagnose you. It has no access to your medical history.")
        st.warning("⚠️  It **can** make mistakes — roughly 1 in 5 predictions may be wrong.")
        st.success("✅  It **can** give a useful first signal and help you reflect on your writing.")

        st.markdown("### How to use it — step by step")
        steps = [
            ("1️⃣  Write or paste", "any text — a journal entry, a message you drafted, or something you read."),
            ("2️⃣  Click Analyze →", "the model reads your text and returns a category + confidence score."),
            ("3️⃣  Read the result", "the predicted category and the probability bar chart show how certain the model is."),
            ("4️⃣  Give feedback", "tell the system if the prediction was right or wrong — this trains future versions."),
        ]
        for icon_label, desc in steps:
            st.markdown(f"**{icon_label}** — {desc}")

        st.markdown("### How feedback improves the model")
        st.markdown(
            "Every time you flag an incorrect prediction, the text and correct label are logged "
            "securely. These logs feed the model retraining pipeline — the more feedback, the more "
            "accurately the model learns over time."
        )

        st.markdown("### 🔒 Privacy")
        st.info(
            "Text you enter is sent to the prediction server. "
            "It is **not stored** unless you submit feedback. "
            "Feedback logs are used only for model improvement."
        )

    with about_right:
        st.markdown("### The 7 categories")
        cats_info = [
            ("Normal",               "Everyday writing with no strong mental-health signals."),
            ("Depression",           "Persistent low mood, hopelessness, loss of interest."),
            ("Anxiety",              "Worry, restlessness, panic, physical symptoms of stress."),
            ("Bipolar",              "Mood swings between depression and elevated/manic states."),
            ("Suicidal",             "Self-harm thoughts, trauma re-experiencing, crisis signals."),
            ("Stress",               "Acute pressure-related symptoms, burnout, overwhelm."),
            ("Personality Disorder", "Long-term patterns affecting identity and relationships."),
        ]
        for label, desc in cats_info:
            s = cat_style(label)
            bg, bar, fg, icon = s["bg"], s["bar"], s["fg"], s["icon"]
            st.markdown(
                f"<div class='cat-row' style='background:{bg};border-color:{bar};'>"
                f"<span style='font-size:1.3rem;'>{icon}</span>"
                f"<div>"
                f"<div class='cat-title' style='color:{fg};'>{label}</div>"
                f"<div class='cat-desc' style='color:{fg};'>{desc}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 🆘 Crisis resources")
        st.error(
            "If you or someone you know is in immediate danger, please reach out now:\n\n"
            "🇮🇳 **India — iCall:** +91 9152987821 (Mon–Sat, 8 am–10 pm IST)\n\n"
            "🇺🇸 **US — 988 Lifeline:** Call or text **988**\n\n"
            "🌍 **International:** [findahelpline.com](https://findahelpline.com)",
            icon=None,
        )
