"""
streamlit_app.py — Spam Email Classifier UI

Calls backend/core.py directly (no HTTP/FastAPI needed for local use).
To run:  streamlit run streamlit_app.py
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path bootstrap ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.core import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD_CONFIG_PATH,
    bert_predict_single_email,
    evaluate,
    predict_csv_batch,
    predict_single_email,
    train,
    tune_threshold,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Checker",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #0f0f1a; }
    .stTextArea textarea { font-family: monospace; font-size: 14px; }
    .metric-card { background: #1a1a2e; border-radius: 12px; padding: 16px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .spam-badge { background: #ff4b4b; color: white; border-radius: 8px;
                  padding: 8px 20px; font-size: 1.4rem; font-weight: 700; }
    .ham-badge  { background: #21c55d; color: white; border-radius: 8px;
                  padding: 8px 20px; font-size: 1.4rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    threshold_config = st.text_input("Threshold config", value=DEFAULT_THRESHOLD_CONFIG_PATH)

    st.subheader("🧠 Model type")
    model_type = st.radio(
        "Choose algorithm",
        options=["tfidf_lr", "count_nb", "bert_tiny"],
        format_func=lambda x: {
            "tfidf_lr":   "TF-IDF + Logistic Regression (recommended)",
            "count_nb":   "CountVectorizer + Naive Bayes (~98% on SMS spam)",
            "bert_tiny":  "🤗 BERT-tiny (HuggingFace, pre-trained)",
        }[x],
        index=0,
        help="BERT downloads ~17MB on first use and is cached. No GPU needed.",
    )
    if model_type == "bert_tiny":
        st.info("⚡ BERT mode: no training needed. First run downloads the model (~17MB).")

    st.subheader("📐 Review Threshold")
    use_manual = st.checkbox("Override threshold manually", value=False)
    manual_threshold: float | None = None
    if use_manual:
        manual_threshold = st.slider("Threshold", 0.05, 0.99, 0.65, 0.01,
                                     help="Predictions below this confidence will be flagged for review.")

    st.divider()
    st.caption("Built with 🐍 scikit-learn + 🤗 HuggingFace + Streamlit")

# ── Main header ──────────────────────────────────────────────────────────────
st.title("📧 Email Spam Checker")
st.caption("Paste an email, upload a CSV, or train a new model — all in one place.")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_train = st.tabs(
    ["🔍 Single Email", "📂 Batch CSV", "🎓 Train & Evaluate"]
)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Single Email                                                        #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_single:
    st.subheader("Check a single email")
    email_text = st.text_area(
        "Paste email body here",
        height=250,
        placeholder="Congratulations! You have won a free iPhone. Click here to claim...",
    )

    if st.button("🔎 Check Email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please paste some email text first.")
        elif model_type != "bert_tiny" and not Path(model_path).exists():
            st.error("⚠️ Model not found. Go to **Train & Evaluate** tab and train a model first.")
        else:
            spinner_msg = "Running BERT inference... (first run downloads ~17MB)" if model_type == "bert_tiny" else "Classifying..."
            with st.spinner(spinner_msg):
                try:
                    if model_type == "bert_tiny":
                        result = bert_predict_single_email(
                            text=email_text,
                            review_threshold=manual_threshold or 0.6,
                        )
                        result["ml_spam_probability"] = result.pop("bert_spam_probability", None)
                    else:
                        result = predict_single_email(
                            text=email_text,
                            model_path=model_path,
                            review_threshold=manual_threshold,
                            threshold_config_path=threshold_config or None,
                        )
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    result = None

            if result:
                prediction = result["prediction"]
                confidence = float(result["confidence"])
                needs_review = bool(result["needs_review"])
                active_threshold = float(result["review_threshold"])
                ml_prob = result.get("ml_spam_probability", None)
                signals = result.get("email_signals_detected", [])

                st.divider()
                if prediction == "spam":
                    st.markdown('<div class="spam-badge">🚨 SPAM</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ham-badge">✅ NOT SPAM</div>', unsafe_allow_html=True)

                st.write("")
                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{confidence:.1%}")
                c2.metric("Threshold", f"{active_threshold:.2f}")
                c3.metric("Needs Review", "⚠️ Yes" if needs_review else "✅ No")

                # Hybrid breakdown panel
                if signals:
                    ml_label = "spam" if (ml_prob or 0) >= 0.5 else "not spam"
                    st.warning(
                        f"⚡ **Email signal boost applied** — "
                        f"ML alone said **{ml_label}** ({ml_prob:.1%} spam prob). "
                        f"**{len(signals)} marketing signal(s)** detected → boosted to spam."
                    )
                    with st.expander(f"📋 Why was this flagged? ({len(signals)} signals)", expanded=True):
                        for sig in signals:
                            st.markdown(
                                f"- **`{sig['name']}`** *(boost: {sig['weight']:.0%})* — {sig['description']}"
                            )
                elif ml_prob is not None:
                    st.caption(f"🤖 ML spam probability: {ml_prob:.1%} | No marketing email signals detected.")

                with st.expander("Raw result JSON"):
                    st.json(result)


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Batch CSV                                                           #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_batch:
    st.subheader("Classify a CSV file")
    st.caption("Upload a CSV with a column containing email text. Each row gets a prediction.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    text_col = st.text_input("Text column name", value="Message",
                              help="The column in your CSV that contains the email body.")

    if st.button("▶️ Run Batch Prediction", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a CSV file.")
        elif not Path(model_path).exists():
            st.error("⚠️ Model not found. Train a model first.")
        else:
            # Save uploaded file to a temp location inside project
            tmp_input = PROJECT_ROOT / "data" / "processed" / "batch_upload.csv"
            tmp_input.parent.mkdir(parents=True, exist_ok=True)
            tmp_input.write_bytes(uploaded_file.read())

            tmp_output = PROJECT_ROOT / "data" / "processed" / "batch_predictions.csv"

            with st.spinner(f"Running predictions on {uploaded_file.name}..."):
                try:
                    batch_result = predict_csv_batch(
                        input_csv=str(tmp_input),
                        output_csv=str(tmp_output),
                        model_path=model_path,
                        text_column=text_col,
                        review_threshold=manual_threshold,
                        threshold_config_path=threshold_config or None,
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    batch_result = None

            if batch_result:
                st.success(
                    f"✅ Done! {batch_result['rows']} rows classified. "
                    f"{batch_result['rows_needing_review']} flagged for review."
                )

                result_df = pd.read_csv(tmp_output)
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv_bytes,
                    file_name="spam_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Train & Evaluate                                                    #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_train:
    st.subheader("Train a new model")

    train_data_path = st.text_input("Training data path", value=DEFAULT_DATA_PATH)

    col_train, col_eval, col_tune = st.columns(3)

    # ── Train ────────────────────────────────────────────────────────────────
    with col_train:
        if st.button("🎓 Train Model", use_container_width=True):
            if not Path(train_data_path).exists():
                st.error(f"File not found: {train_data_path}")
            else:
                with st.spinner(f"Training {model_type} model..."):
                    try:
                        report = train(
                            data_path=train_data_path,
                            model_path=model_path,
                            model_type=model_type,
                        )
                    except Exception as exc:
                        st.error(f"Training failed: {exc}")
                        report = None

                if report:
                    st.success("✅ Model trained and saved!")
                    _tm = report.get("test_metrics", {})
                    st.metric("Test Accuracy", f"{_tm.get('accuracy', 0):.1%}")
                    st.metric("F1 (Spam)", f"{_tm.get('f1_spam', 0):.3f}")
                    with st.expander("Full training report"):
                        st.json(report)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    with col_eval:
        if st.button("📊 Evaluate", use_container_width=True):
            if not Path(model_path).exists():
                st.error("Train a model first.")
            elif not Path(train_data_path).exists():
                st.error(f"File not found: {train_data_path}")
            else:
                with st.spinner("Evaluating..."):
                    try:
                        eval_result = evaluate(
                            data_path=train_data_path,
                            model_path=model_path,
                        )
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
                        eval_result = None

                if eval_result:
                    st.success("✅ Evaluation complete!")
                    _m = eval_result.get("metrics", {})

                    # Bar chart
                    chart_data = {
                        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
                        "Score": [
                            _m.get("accuracy", 0),
                            _m.get("precision_spam", 0),
                            _m.get("recall_spam", 0),
                            _m.get("f1_spam", 0),
                        ],
                    }
                    chart_df = pd.DataFrame(chart_data).set_index("Metric")
                    st.bar_chart(chart_df, use_container_width=True, color="#4f8ef7")

                    with st.expander("Full evaluation report"):
                        st.json(eval_result)

    # ── Tune Threshold ────────────────────────────────────────────────────────
    with col_tune:
        target_rate = st.number_input(
            "Target review rate", min_value=0.01, max_value=0.99, value=0.20, step=0.05,
            help="What fraction of predictions should be flagged for manual review?",
        )
        if st.button("🎯 Tune Threshold", use_container_width=True):
            if not Path(model_path).exists():
                st.error("Train a model first.")
            elif not Path(train_data_path).exists():
                st.error(f"File not found: {train_data_path}")
            else:
                with st.spinner("Finding optimal threshold..."):
                    try:
                        tune_result = tune_threshold(
                            data_path=train_data_path,
                            model_path=model_path,
                            target_review_rate=target_rate,
                            output_path=threshold_config,
                        )
                    except Exception as exc:
                        st.error(f"Tuning failed: {exc}")
                        tune_result = None

                if tune_result:
                    st.success("✅ Threshold tuned and saved!")
                    st.metric("Suggested Threshold", f"{tune_result['suggested_threshold']:.3f}")
                    st.metric("Actual Review Rate", f"{tune_result['actual_review_rate']:.1%}")
                    with st.expander("Full tuning report"):
                        st.json(tune_result)
