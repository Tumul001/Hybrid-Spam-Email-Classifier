"""
streamlit_app.py — Spam Email Classifier UI
Hybrid system: BERT-tiny (HuggingFace) + Keyword Rules Engine
No training required. No Kaggle dataset.
Run: streamlit run streamlit_app.py
"""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path bootstrap ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.core import (
    DEFAULT_TEST_DATA_PATH,
    evaluate,
    get_rules_info,
    predict_csv_batch,
    predict_single_email,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0f0f1a; }
.stTextArea textarea { font-family: monospace; font-size: 14px; }
div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.spam-badge {
    background: linear-gradient(135deg, #ff4b4b, #c0392b);
    color: white; border-radius: 10px;
    padding: 10px 24px; font-size: 1.5rem; font-weight: 700;
    display: inline-block; margin-bottom: 8px;
    box-shadow: 0 4px 15px rgba(255,75,75,0.4);
}
.ham-badge {
    background: linear-gradient(135deg, #21c55d, #16a34a);
    color: white; border-radius: 10px;
    padding: 10px 24px; font-size: 1.5rem; font-weight: 700;
    display: inline-block; margin-bottom: 8px;
    box-shadow: 0 4px 15px rgba(33,197,93,0.4);
}
.rule-chip {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 6px; padding: 4px 10px;
    font-size: 0.8rem; color: #94a3b8;
    display: inline-block; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📧 Spam Classifier")
    st.caption("BERT + Keyword Rules · No training needed")
    st.divider()

    st.subheader("⚙️ Settings")
    review_threshold = st.slider(
        "Review threshold",
        min_value=0.05, max_value=0.99, value=0.60, step=0.01,
        help="Predictions below this confidence are flagged for human review.",
    )

    st.divider()
    st.subheader("🔬 System Info")
    st.markdown("""
- 🤗 **BERT-tiny** (HuggingFace)
- 📋 **48 keyword rules** across 6 categories
- 🚫 No Kaggle data used
- ✅ Custom hand-crafted test set
    """)

    # Show keyword rules in expander
    with st.expander("📋 All Keyword Rules"):
        rules_info = get_rules_info()
        for cat in rules_info["categories"]:
            st.markdown(f"**{cat.upper()}**")
            for r in rules_info["rules_by_category"][cat]:
                st.markdown(
                    f"<span class='rule-chip'>{r['name']} ({r['weight']:.0%})</span>",
                    unsafe_allow_html=True,
                )
            st.write("")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📧 Email Spam Classifier")
st.caption("Hybrid system: BERT deep learning + hand-crafted keyword rules. No Kaggle dataset.")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_evaluate = st.tabs(
    ["🔍 Single Email", "📂 Batch CSV", "📊 Evaluate & Rules"]
)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Single Email                                                         #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_single:
    st.subheader("Check a single email")
    email_text = st.text_area(
        "Paste email body here",
        height=260,
        placeholder="Paste any email content here — newsletter, phishing attempt, work email...",
    )

    if st.button("🔎 Classify Email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please paste some email text first.")
        else:
            with st.spinner("Running BERT inference + keyword analysis..."):
                try:
                    result = predict_single_email(email_text, review_threshold=review_threshold)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    result = None

            if result:
                prediction  = result["prediction"]
                confidence  = float(result["confidence"])
                needs_review = bool(result["needs_review"])
                bert_prob   = result.get("bert_spam_probability", 0.0)
                signals     = result.get("email_signals_detected", [])

                st.divider()

                # Verdict badge
                if prediction == "spam":
                    st.markdown('<div class="spam-badge">🚨 SPAM</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ham-badge">✅ NOT SPAM</div>', unsafe_allow_html=True)

                st.write("")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Confidence", f"{confidence:.1%}")
                c2.metric("BERT Score", f"{bert_prob:.1%}")
                c3.metric("Signals Found", len(signals))
                c4.metric("Needs Review", "⚠️ Yes" if needs_review else "✅ No")

                # Keyword signal breakdown
                if signals:
                    cats = sorted({s["category"] for s in signals})
                    st.warning(
                        f"⚡ **Keyword boost applied** — BERT alone: {bert_prob:.1%}. "
                        f"**{len(signals)} signal(s)** from: {', '.join(cats)}."
                    )
                    with st.expander(f"📋 Why is this spam? ({len(signals)} keyword signals)", expanded=True):
                        # Group by category
                        by_cat: dict[str, list] = {}
                        for s in signals:
                            by_cat.setdefault(s["category"], []).append(s)
                        for cat, rules in by_cat.items():
                            st.markdown(f"**{cat.upper()}**")
                            for r in rules:
                                st.markdown(
                                    f"- **`{r['name']}`** *(boost: {r['weight']:.0%})* — {r['description']}"
                                )
                else:
                    st.info(f"🤖 BERT-only prediction (no keyword signals detected). Spam probability: {bert_prob:.1%}")

                with st.expander("Raw JSON result"):
                    st.json(result)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Batch CSV                                                            #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_batch:
    st.subheader("Classify a CSV file")
    st.caption("Upload a CSV with an email column. We'll classify every row.")

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Must contain a column named 'message', 'text', 'email', or 'body'."
    )

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"**{len(df_preview)} rows detected.** Preview:")
        st.dataframe(df_preview.head(5), use_container_width=True)
        uploaded_file.seek(0)

        if st.button("🚀 Classify All Rows", type="primary", use_container_width=True):
            tmp_in  = PROJECT_ROOT / "data" / "processed" / "_batch_input.csv"
            tmp_out = PROJECT_ROOT / "data" / "processed" / "_batch_output.csv"
            tmp_in.parent.mkdir(parents=True, exist_ok=True)
            tmp_in.write_bytes(uploaded_file.read())

            with st.spinner(f"Classifying {len(df_preview)} emails with BERT..."):
                try:
                    summary = predict_csv_batch(
                        input_csv=str(tmp_in),
                        output_csv=str(tmp_out),
                        review_threshold=review_threshold,
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    summary = None

            if summary:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Emails", summary["rows"])
                c2.metric("🚨 Spam", summary["spam_count"])
                c3.metric("✅ Ham", summary["ham_count"])

                result_df = pd.read_csv(tmp_out)
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_bytes,
                    file_name="spam_results.csv",
                    mime="text/csv",
                )

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Evaluate & Rules                                                     #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_evaluate:
    st.subheader("📊 System Evaluation")
    st.caption(
        "Evaluate the BERT + keyword system against our **custom hand-crafted test set** "
        "(50 emails — not from Kaggle, written from scratch)."
    )

    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Running BERT + keyword rules on 50 test emails..."):
            try:
                report = evaluate(review_threshold=review_threshold)
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")
                report = None

        if report:
            st.divider()
            st.subheader("Results")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{report['accuracy']:.1%}")
            c2.metric("Precision", f"{report['precision']:.1%}")
            c3.metric("Recall",    f"{report['recall']:.1%}")
            c4.metric("F1 Score",  f"{report['f1']:.1%}")
            c5.metric("Correct",   f"{report['correct']}/{report['total']}")

            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = report["confusion"]
            cm_df = pd.DataFrame(
                [[cm["tp"], cm["fp"]], [cm["fn"], cm["tn"]]],
                index=["Actual: Spam", "Actual: Ham"],
                columns=["Predicted: Spam", "Predicted: Ham"],
            )
            st.table(cm_df)

            # Per-row results
            st.subheader("Per-Email Results")
            rows_df = pd.DataFrame(report["per_row"])
            rows_df["✓"] = rows_df["correct"].apply(lambda x: "✅" if x else "❌")
            rows_df["bert_prob"] = rows_df["bert_prob"].apply(lambda x: f"{x:.1%}")
            rows_df["confidence"] = rows_df["confidence"].apply(lambda x: f"{x:.1%}")
            st.dataframe(
                rows_df[["✓", "text", "true_label", "predicted", "confidence", "bert_prob", "keyword_signals"]],
                use_container_width=True,
            )

    st.divider()
    st.subheader("📋 Keyword Rules Engine")
    st.caption(f"48 rules across 6 categories. All hand-crafted. No training data needed.")

    rules_info = get_rules_info()
    cols = st.columns(2)
    for i, cat in enumerate(rules_info["categories"]):
        with cols[i % 2]:
            with st.expander(f"**{cat.upper()}** ({len(rules_info['rules_by_category'][cat])} rules)"):
                for r in rules_info["rules_by_category"][cat]:
                    st.markdown(f"**`{r['name']}`** *(weight: {r['weight']:.0%})*")
                    st.caption(r["description"])
