"""
predict.py — Hybrid spam classifier: BERT + Keyword Rules.

Architecture:
    1. BERT-tiny (HuggingFace, pre-trained) provides semantic spam probability.
    2. Keyword rules engine boosts probability for explicit, predictable patterns.
    3. Final score = weighted blend of BERT + keyword boost (noisy-OR).

No sklearn model or training data required.
"""

from __future__ import annotations
from typing import Any

import pandas as pd
from pathlib import Path

from .bert_model import bert_predict_text
from .keyword_rules import match_rules, combined_keyword_boost, SpamRule


DEFAULT_REVIEW_THRESHOLD = 0.6


def _hybrid_score(bert_prob: float, text: str) -> tuple[float, list[SpamRule]]:
    """
    Combine BERT probability with keyword rule boost.

    Strategy (noisy-OR blend):
    - Keywords run on RAW text (before cleaning strips signal words).
    - If signals found: final = max(bert_prob, 0.6*keyword_boost + 0.4*bert_prob)
    - If no signals:   final = bert_prob (BERT alone)

    This ensures we never downgrade a confident BERT spam call, but we can
    upgrade newsletters/phishing that BERT rates as borderline.
    """
    matched = match_rules(text)
    if not matched:
        return bert_prob, []

    keyword_boost = combined_keyword_boost(matched)
    blended = 0.6 * keyword_boost + 0.4 * bert_prob
    final = max(bert_prob, blended)
    return final, matched


def predict_text(
    text: str,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """
    Classify a single email using BERT + keyword rules.

    Returns:
        prediction          : "spam" | "not spam"
        confidence          : float — how confident we are in the prediction
        needs_review        : bool  — True if confidence < review_threshold
        review_threshold    : float
        bert_spam_probability   : raw BERT output
        email_signals_detected  : list of matched keyword rule dicts
    """
    bert_result = bert_predict_text(text)
    bert_prob = bert_result["spam_probability"]

    final_prob, matched_rules = _hybrid_score(bert_prob, text)

    if final_prob >= 0.5:
        prediction = "spam"
        confidence = final_prob
    else:
        prediction = "not spam"
        confidence = 1.0 - final_prob

    return {
        "text": text,
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "needs_review": confidence < review_threshold,
        "review_threshold": review_threshold,
        "bert_spam_probability": round(bert_prob, 6),
        "email_signals_detected": [
            {
                "name": r.name,
                "category": r.category,
                "description": r.description,
                "weight": round(r.weight, 2),
            }
            for r in matched_rules
        ],
    }


def predict_batch(
    input_csv: str,
    output_csv: str,
    text_column: str = "message",
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """
    Classify all rows in a CSV. Saves results to output_csv.
    Auto-detects common text column names (message, text, email, body).
    """
    df = pd.read_csv(Path(input_csv))

    # Auto-detect text column
    lower_map = {c.lower(): c for c in df.columns}
    resolved_col = None
    for candidate in [text_column, "message", "text", "email", "body", "content"]:
        if candidate.lower() in lower_map:
            resolved_col = lower_map[candidate.lower()]
            break
    if resolved_col is None:
        raise ValueError(f"Could not find a text column in {input_csv}. Columns: {list(df.columns)}")

    raw_texts = df[resolved_col].fillna("").astype(str).tolist()

    predictions, confidences, needs_review_list, signal_counts = [], [], [], []

    for raw in raw_texts:
        result = predict_text(raw, review_threshold=review_threshold)
        predictions.append(result["prediction"])
        confidences.append(result["confidence"])
        needs_review_list.append(result["needs_review"])
        signal_counts.append(len(result["email_signals_detected"]))

    out = df.copy()
    out["prediction"] = predictions
    out["confidence"] = confidences
    out["needs_review"] = needs_review_list
    out["keyword_signals"] = signal_counts

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    return {
        "rows": len(out),
        "spam_count": predictions.count("spam"),
        "ham_count": predictions.count("not spam"),
        "rows_needing_review": sum(needs_review_list),
        "review_threshold": review_threshold,
        "output_csv": str(out_path),
    }
