"""
backend/core.py — Direct-import backend.
Architecture: BERT-tiny (HuggingFace) + 106 Keyword Rules. No training needed.
"""

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spam_detector.predict import predict_text, predict_batch, DEFAULT_REVIEW_THRESHOLD
from spam_detector.keyword_rules import SPAM_RULES, CATEGORIES, RULE_COUNT

DEFAULT_TEST_DATA_PATH = str(_PROJECT_ROOT / "data" / "raw" / "custom_test_set.csv")
DEFAULT_RESULTS_PATH   = str(_PROJECT_ROOT / "data" / "processed" / "evaluation_results.csv")


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_single_email(
    text: str,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """Classify a single email. Returns prediction, confidence, keyword signals."""
    return predict_text(text, review_threshold=review_threshold)


def predict_csv_batch(
    input_csv: str,
    output_csv: str = DEFAULT_RESULTS_PATH,
    text_column: str = "message",
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """Classify all rows in a CSV. Saves results to output_csv."""
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    return predict_batch(
        input_csv=input_csv,
        output_csv=output_csv,
        text_column=text_column,
        review_threshold=review_threshold,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    test_data_path: str = DEFAULT_TEST_DATA_PATH,
    label_column: str = "label",
    text_column: str = "message",
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """
    Full evaluation of the BERT + keyword system against our custom test set.

    Metrics returned:
        accuracy, precision, recall, f1
        specificity (true negative rate)
        false positive rate, false negative rate
        matthews correlation coefficient (MCC)
        cohen's kappa
        average confidence, average BERT probability
        per-category keyword hit statistics
        per-row results
    """
    import math
    import pandas as pd

    df = pd.read_csv(test_data_path)

    lower_map = {c.lower(): c for c in df.columns}
    lbl_col = lower_map.get(label_column.lower())
    txt_col = lower_map.get(text_column.lower()) or lower_map.get("message") or lower_map.get("text")
    if not lbl_col or not txt_col:
        raise ValueError(f"Could not find label/text columns in {test_data_path}. Columns: {list(df.columns)}")

    results = []
    category_hits: dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for _, row in df.iterrows():
        text       = str(row[txt_col])
        true_label = "spam" if str(row[lbl_col]).strip().lower() == "spam" else "not spam"
        pred       = predict_text(text, review_threshold=review_threshold)

        predicted   = pred["prediction"]
        signals     = pred["email_signals_detected"]
        correct     = predicted == true_label

        for s in signals:
            if s["category"] in category_hits:
                category_hits[s["category"]] += 1

        results.append({
            "text":            text[:100] + "..." if len(text) > 100 else text,
            "true_label":      true_label,
            "predicted":       predicted,
            "correct":         correct,
            "confidence":      pred["confidence"],
            "bert_prob":       pred["bert_spam_probability"],
            "keyword_signals": len(signals),
            "signal_names":    ", ".join(s["name"] for s in signals) if signals else "—",
        })

    res_df = pd.DataFrame(results)
    n = len(res_df)

    tp = int(((res_df["predicted"] == "spam")     & (res_df["true_label"] == "spam")).sum())
    fp = int(((res_df["predicted"] == "spam")     & (res_df["true_label"] == "not spam")).sum())
    fn = int(((res_df["predicted"] == "not spam") & (res_df["true_label"] == "spam")).sum())
    tn = int(((res_df["predicted"] == "not spam") & (res_df["true_label"] == "not spam")).sum())

    precision   = tp / (tp + fp)   if (tp + fp) > 0   else 0.0
    recall      = tp / (tp + fn)   if (tp + fn) > 0   else 0.0
    specificity = tn / (tn + fp)   if (tn + fp) > 0   else 0.0
    fpr         = fp / (fp + tn)   if (fp + tn) > 0   else 0.0
    fnr         = fn / (fn + tp)   if (fn + tp) > 0   else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy    = (tp + tn) / n    if n > 0            else 0.0

    # Matthews Correlation Coefficient
    denom_mcc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else 0.0

    # Cohen's Kappa
    p_o = (tp + tn) / n
    p_e = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    # Average scores
    avg_confidence = float(res_df["confidence"].mean())
    avg_bert_prob  = float(res_df["bert_prob"].mean())
    pct_signals    = float((res_df["keyword_signals"] > 0).mean())

    return {
        "total":            n,
        "correct":          int(res_df["correct"].sum()),
        # Core metrics
        "accuracy":         round(accuracy,    4),
        "precision":        round(precision,   4),
        "recall":           round(recall,      4),
        "f1":               round(f1,          4),
        # Extended metrics
        "specificity":      round(specificity, 4),
        "false_positive_rate": round(fpr,      4),
        "false_negative_rate": round(fnr,      4),
        "mcc":              round(mcc,         4),
        "cohens_kappa":     round(kappa,       4),
        # Confidence stats
        "avg_confidence":   round(avg_confidence, 4),
        "avg_bert_prob":    round(avg_bert_prob,  4),
        "pct_with_signals": round(pct_signals,    4),
        # Confusion matrix
        "confusion":        {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        # Keyword category hits
        "category_hits":    category_hits,
        # Per-row details
        "per_row":          results,
    }


# ── Rules Info ────────────────────────────────────────────────────────────────

def get_rules_info() -> dict[str, Any]:
    """Return all keyword rules grouped by category for display in the UI."""
    grouped: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    for rule in SPAM_RULES:
        grouped[rule.category].append({
            "name":        rule.name,
            "description": rule.description,
            "weight":      round(rule.weight, 2),
        })
    return {
        "total_rules":        RULE_COUNT,
        "categories":         CATEGORIES,
        "rules_by_category":  grouped,
    }
