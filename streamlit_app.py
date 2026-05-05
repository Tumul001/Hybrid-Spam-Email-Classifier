"""
streamlit_app.py — Spam Email Classifier UI
Hybrid system: BERT-tiny + 106 Keyword Rules
Premium aesthetics: Glassmorphism, Plotly, Lottie.
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

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
    page_title="Novalantis Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Assets ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# High-quality Lottie animations
lottie_shield = load_lottieurl("https://lottie.host/805ba213-9a3b-482a-bc91-2dc0da2d137f/X4uNlS2YnC.json")
lottie_scan = load_lottieurl("https://lottie.host/bbd964cb-dd74-4b45-93df-4efbd47c7c34/2O8AOFJdOo.json")

# ── Custom Glassmorphism CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { 
    font-family: 'Outfit', sans-serif; 
}

/* Glassmorphism containers */
.glass-panel {
    background: rgba(17, 24, 39, 0.65);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 1.5rem;
}

/* Verdict Badges */
.verdict-badge {
    text-align: center;
    border-radius: 16px;
    padding: 20px 24px; 
    font-size: 2.2rem; 
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    margin-bottom: 24px;
    backdrop-filter: blur(8px);
}
.verdict-spam {
    background: linear-gradient(135deg, rgba(239,68,68,0.9), rgba(153,27,27,0.9));
    border: 1px solid rgba(248,113,113,0.3);
    color: white;
}
.verdict-ham {
    background: linear-gradient(135deg, rgba(16,185,129,0.9), rgba(6,95,70,0.9));
    border: 1px solid rgba(52,211,153,0.3);
    color: white;
}

/* Big Metrics */
div[data-testid="stMetricValue"] { 
    font-size: 2.8rem !important; 
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #00F0FF, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
div[data-testid="stMetricLabel"] {
    font-size: 1.1rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Rule chips */
.rule-chip {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(51, 65, 85, 0.8);
    border-radius: 20px; 
    padding: 6px 14px;
    font-size: 0.85rem; 
    color: #cbd5e1;
    display: inline-block; 
    margin: 4px;
    backdrop-filter: blur(4px);
    transition: all 0.2s ease;
}
.rule-chip:hover {
    background: rgba(0, 240, 255, 0.1);
    border-color: #00F0FF;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if lottie_shield:
        st_lottie(lottie_shield, height=120, key="shield")
    
    st.markdown("<h2 style='text-align: center;'>Novalantis Shield</h2>", unsafe_allow_html=True)
    st.caption("<p style='text-align: center;'>Enterprise-Grade Spam AI</p>", unsafe_allow_html=True)
    st.divider()

    st.subheader("⚙️ Engine Tuning")
    review_threshold = st.slider(
        "Review threshold",
        min_value=0.05, max_value=0.99, value=0.60, step=0.01,
        help="Predictions below this confidence are flagged for human review.",
    )

    st.divider()
    st.subheader("🛡️ Defense Layers")
    st.markdown("""
- 🧠 **BERT-tiny (Semantic Core)**
- 🎯 **106 Keyword Signals**
- 🚫 **Zero Kaggle Data**
    """)

    # Show keyword rules in expander
    with st.expander("📋 Browse All 106 Rules"):
        rules_info = get_rules_info()
        for cat in rules_info["categories"]:
            st.markdown(f"<div style='color: #00F0FF; margin-top:10px; font-weight:600;'>{cat.upper()}</div>", unsafe_allow_html=True)
            for r in rules_info["rules_by_category"][cat]:
                st.markdown(
                    f"<span class='rule-chip'>{r['name']} ({r['weight']:.0%})</span>",
                    unsafe_allow_html=True,
                )
            st.write("")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Email Threat Analysis")
st.caption("Hybrid Zero-Shot Deep Learning + Heuristic Rules Engine")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_evaluate = st.tabs(
    ["🔍 Live Inspector", "📂 Batch Scanner", "📊 Performance Dashboard"]
)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Single Email                                                         #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_single:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    email_text = st.text_area(
        "Paste email payload here",
        height=240,
        placeholder="Paste full email body here to analyze for phishing, scams, or marketing spam...",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🛡️ Initiate Threat Scan", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please paste an email payload first.")
        else:
            # ── Staged progress bar simulation for premium feel ──
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("1. Tokenizing input sequence...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            status_text.text("2. Running BERT semantic analysis...")
            progress_bar.progress(60)
            
            try:
                result = predict_single_email(email_text, review_threshold=review_threshold)
            except Exception as exc:
                st.error(f"Error: {exc}")
                result = None

            status_text.text("3. Executing 106 heuristic rules...")
            progress_bar.progress(90)
            time.sleep(0.3)
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            st.toast("Threat scan completed successfully!", icon="✅")

            if result:
                prediction   = result["prediction"]
                confidence   = float(result["confidence"])
                needs_review = bool(result["needs_review"])
                bert_prob    = result.get("bert_spam_probability", 0.0)
                signals      = result.get("email_signals_detected", [])
                safe_signals = result.get("safe_signals_detected", [])

                st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
                
                # Verdict badge
                if prediction == "spam":
                    st.markdown('<div class="verdict-badge verdict-spam">🚨 THREAT DETECTED: SPAM</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict-badge verdict-ham">✅ SAFE: NOT SPAM</div>', unsafe_allow_html=True)

                st.write("")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Confidence", f"{confidence:.1%}")
                c2.metric("BERT Base Score", f"{bert_prob:.1%}")
                c3.metric("Heuristics Triggered", len(signals) + len(safe_signals))
                c4.metric("Action Required", "⚠️ REVIEW" if needs_review else "✅ NONE")

                # Keyword signal breakdown
                if signals or safe_signals:
                    st.markdown("### Threat & Safe Signatures Detected")
                    
                    if signals:
                        st.warning(f"🚨 **Threat Heuristics Applied** — Found **{len(signals)} signals**.")
                        for s in signals:
                            st.markdown(
                                f"- <span style='color:#ef4444;font-weight:bold;'>[{s['category'].upper()}]</span> "
                                f"**`{s['name']}`** *(boost: +{s['weight']:.0%})* — {s['description']}",
                                unsafe_allow_html=True
                            )
                            
                    if safe_signals:
                        st.success(f"🛡️ **Safe Heuristics Applied** — Reduced spam probability to prevent false positive.")
                        for s in safe_signals:
                            st.markdown(
                                f"- <span style='color:#10b981;font-weight:bold;'>[{s['category'].upper()}]</span> "
                                f"**`{s['name']}`** *(discount: -{s['discount']:.0%})* — {s['description']}",
                                unsafe_allow_html=True
                            )
                else:
                    st.info(f"🤖 Pure Semantic Verdict. No explicit heuristic signatures triggered. BERT probability: {bert_prob:.1%}")

                st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Batch CSV                                                            #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_batch:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("Bulk Threat Scanner")
    st.caption("Upload a CSV with an email column. Processes thousands of rows via BERT + Rules.")

    uploaded_file = st.file_uploader(
        "Upload dataset (.csv)", type=["csv"],
        help="Must contain a column named 'message', 'text', 'email', or 'body'."
    )

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"**{len(df_preview)} records queued.**")
        st.dataframe(df_preview.head(5), use_container_width=True)
        uploaded_file.seek(0)

        if st.button("🚀 Execute Batch Scan", type="primary", use_container_width=True):
            tmp_in  = PROJECT_ROOT / "data" / "processed" / "_batch_input.csv"
            tmp_out = PROJECT_ROOT / "data" / "processed" / "_batch_output.csv"
            tmp_in.parent.mkdir(parents=True, exist_ok=True)
            tmp_in.write_bytes(uploaded_file.read())

            with st.spinner(f"Processing {len(df_preview)} records..."):
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
                st.success("Batch scan complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Processed", summary["rows"])
                c2.metric("🚨 Spam Identified", summary["spam_count"])
                c3.metric("✅ Safe Mails", summary["ham_count"])

                result_df = pd.read_csv(tmp_out)
                
                # Interactive Plotly Donut Chart
                fig = px.pie(
                    result_df, names="prediction", 
                    hole=0.6, 
                    color="prediction",
                    color_discrete_map={"spam": "#ef4444", "not spam": "#10b981"},
                    title="Threat Distribution"
                )
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Secure Export (.csv)",
                    data=csv_bytes,
                    file_name="threat_scan_results.csv",
                    mime="text/csv",
                )
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Evaluate & Rules                                                     #
# ════════════════════════════════════════════════════════════════════════════ #
with tab_evaluate:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Engine Telemetry & Dashboard")
    st.caption("Validating against the secure 50-email synthetic benchmark suite.")

    if st.button("▶️ Generate Telemetry Report", type="primary", use_container_width=True):
        with st.spinner("Compiling system telemetry..."):
            try:
                report = evaluate(review_threshold=review_threshold)
            except Exception as exc:
                st.error(f"Telemetry error: {exc}")
                report = None

        if report:
            # ── TOP METRICS ROW ──
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("System Accuracy",  f"{report['accuracy']:.1%}")
            c2.metric("Precision (Quality)", f"{report['precision']:.1%}")
            c3.metric("Recall (Coverage)",    f"{report['recall']:.1%}")
            c4.metric("F1 Score",  f"{report['f1']:.1%}")

            st.write("")
            col_left, col_right = st.columns([1, 1])

            # ── RADAR CHART (Left) ──
            with col_left:
                st.markdown("### Performance Vector")
                radar_fig = go.Figure(data=go.Scatterpolar(
                  r=[
                      report['accuracy'], report['precision'], 
                      report['recall'], report['specificity'], 
                      report['mcc']
                  ],
                  theta=['Accuracy', 'Precision', 'Recall', 'Specificity', 'MCC'],
                  fill='toself',
                  fillcolor='rgba(0, 240, 255, 0.2)',
                  line=dict(color='#00F0FF')
                ))
                radar_fig.update_layout(
                  polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
                    bgcolor='rgba(0,0,0,0)'
                  ),
                  showlegend=False,
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(radar_fig, use_container_width=True)

            # ── BAR CHART: CATEGORY HITS (Right) ──
            with col_right:
                st.markdown("### Heuristic Triggers by Category")
                hits = report["category_hits"]
                df_hits = pd.DataFrame(list(hits.items()), columns=["Category", "Hits"])
                df_hits = df_hits[df_hits["Hits"] > 0].sort_values("Hits", ascending=True)
                
                bar_fig = px.bar(
                    df_hits, x="Hits", y="Category", orientation='h',
                    color="Hits", color_continuous_scale="Teal"
                )
                bar_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Emails triggered",
                    yaxis_title=""
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            st.divider()

            # ── DETAILED RESULTS ──
            st.markdown("### Benchmark Trace")
            rows_df = pd.DataFrame(report["per_row"])
            
            # Format dataframe for display
            def style_correct(val):
                color = '#10b981' if val else '#ef4444'
                return f'color: {color}; font-weight: bold;'
                
            st.dataframe(
                rows_df[["correct", "text", "true_label", "predicted", "confidence", "keyword_signals"]].style.applymap(
                    style_correct, subset=['correct']
                ),
                use_container_width=True,
                height=400
            )

    st.markdown("</div>", unsafe_allow_html=True)
