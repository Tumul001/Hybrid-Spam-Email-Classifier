import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import load_model
from .preprocess import clean_text, email_spam_signals

DEFAULT_REVIEW_THRESHOLD = 0.6


def _validate_review_threshold(review_threshold: float) -> None:
    if not 0.0 < review_threshold < 1.0:
        raise ValueError("review_threshold must be between 0 and 1.")


def _load_threshold_from_config(threshold_config_path: str | None) -> float | None:
    if not threshold_config_path:
        return None

    config_path = Path(threshold_config_path)
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid threshold config JSON: {threshold_config_path}") from exc

    if "suggested_threshold" not in payload:
        raise ValueError(f"Missing suggested_threshold in threshold config: {threshold_config_path}")

    try:
        threshold = float(payload["suggested_threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("suggested_threshold in threshold config must be numeric.") from exc

    _validate_review_threshold(threshold)
    return threshold


def resolve_review_threshold(
    review_threshold: float | None = None,
    threshold_config_path: str | None = "models/review_threshold.json",
) -> float:
    if review_threshold is not None:
        _validate_review_threshold(review_threshold)
        return review_threshold

    configured_threshold = _load_threshold_from_config(threshold_config_path)
    if configured_threshold is not None:
        return configured_threshold

    return DEFAULT_REVIEW_THRESHOLD


def _ml_spam_probability(model: Any, text: str) -> float:
    """Return the ML model's probability that text is spam (0.0–1.0)."""
    if not hasattr(model, "predict_proba"):
        # No probability support — use hard prediction
        prediction = model.predict([text])[0]
        return 1.0 if prediction == "spam" else 0.0

    probabilities = model.predict_proba([text])[0]
    class_to_idx = {label: idx for idx, label in enumerate(model.classes_)}
    return float(probabilities[class_to_idx["spam"]])


def _hybrid_spam_probability(ml_prob: float, text: str) -> tuple[float, list[dict]]:
    """
    Combine ML probability with rule-based email signal detection.

    Strategy:
    - Detect structural marketing-email signals (unsubscribe, newsletter, etc.)
    - Each signal contributes a weighted boost toward spam
    - Final probability = max(ml_prob, boosted_prob) so we never downgrade a
      confident ML spam call, but we can upgrade a missed email newsletter.

    Returns:
        (final_probability, list of detected signal dicts)
    """
    signals = email_spam_signals(text)
    if not signals:
        return ml_prob, []

    # Aggregate signal boost — signals are partially independent, so we
    # use a "noisy-OR" combination: each signal independently adds evidence.
    # boost = 1 - product(1 - w_i)
    combined_boost = 1.0
    for sig in signals:
        combined_boost *= (1.0 - sig.weight)
    signal_boost = 1.0 - combined_boost  # 0.0–1.0

    # Blend: if signals are strong, override ML; otherwise take the max.
    # Weight: 60% signal, 40% ML when signals present (signals are very reliable).
    boosted_prob = 0.6 * signal_boost + 0.4 * ml_prob
    final_prob = max(ml_prob, boosted_prob)

    signal_dicts = [
        {"name": s.name, "description": s.description, "weight": round(s.weight, 2)}
        for s in signals
    ]
    return final_prob, signal_dicts


def _prediction_confidences(model: Any, texts: list[str], predictions: list[str]) -> list[float]:
    if not hasattr(model, "predict_proba"):
        return [1.0 for _ in texts]

    probabilities = model.predict_proba(texts)
    class_to_idx = {label: index for index, label in enumerate(model.classes_)}

    confidences: list[float] = []
    for idx, predicted_label in enumerate(predictions):
        label_idx = class_to_idx[predicted_label]
        confidences.append(float(probabilities[idx][label_idx]))
    return confidences


def _resolve_text_column(df: pd.DataFrame, text_column: str) -> str:
    if text_column in df.columns:
        return text_column

    lower_map = {column.lower(): column for column in df.columns}
    candidates = [text_column, "text", "message", "body", "email", "content"]
    for candidate in candidates:
        key = str(candidate).lower()
        if key in lower_map:
            return lower_map[key]

    raise ValueError(f"Missing text column: {text_column}")


def predict_text(
    text: str,
    model_path: str = "models/spam_model.joblib",
    review_threshold: float | None = None,
    threshold_config_path: str | None = "models/review_threshold.json",
) -> dict[str, Any]:
    resolved_threshold = resolve_review_threshold(review_threshold, threshold_config_path)
    model = load_model(Path(model_path))
    prepared_text = clean_text(text)

    # Step 1: ML probability
    ml_spam_prob = _ml_spam_probability(model, prepared_text)

    # Step 2: Hybrid — boost with email signal rules (run on RAW text so we
    # catch "Unsubscribe", "View in browser" before they're stripped)
    final_spam_prob, detected_signals = _hybrid_spam_probability(ml_spam_prob, text)

    # Step 3: Derive label and confidence from final probability
    if final_spam_prob >= 0.5:
        prediction = "spam"
        confidence = final_spam_prob
    else:
        prediction = "not spam"
        confidence = 1.0 - final_spam_prob

    needs_review = confidence < resolved_threshold

    return {
        "text": text,
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "needs_review": needs_review,
        "review_threshold": resolved_threshold,
        "ml_spam_probability": round(ml_spam_prob, 6),
        "email_signals_detected": detected_signals,
    }


def predict_batch(
    input_csv: str,
    output_csv: str,
    model_path: str = "models/spam_model.joblib",
    text_column: str = "text",
    review_threshold: float | None = None,
    threshold_config_path: str | None = "models/review_threshold.json",
) -> dict[str, Any]:
    resolved_threshold = resolve_review_threshold(review_threshold, threshold_config_path)
    df = pd.read_csv(Path(input_csv))
    resolved_text_column = _resolve_text_column(df, text_column)

    model = load_model(Path(model_path))
    raw_texts = df[resolved_text_column].fillna("").astype(str).tolist()
    cleaned_texts = [clean_text(t) for t in raw_texts]

    predictions_out = []
    confidences_out = []
    needs_review_out = []
    signals_count_out = []

    for raw, cleaned in zip(raw_texts, cleaned_texts):
        ml_prob = _ml_spam_probability(model, cleaned)
        final_prob, signals = _hybrid_spam_probability(ml_prob, raw)

        if final_prob >= 0.5:
            pred = "spam"
            conf = final_prob
        else:
            pred = "not spam"
            conf = 1.0 - final_prob

        predictions_out.append(pred)
        confidences_out.append(round(conf, 6))
        needs_review_out.append(conf < resolved_threshold)
        signals_count_out.append(len(signals))

    output = df.copy()
    output["prediction"] = predictions_out
    output["confidence"] = confidences_out
    output["needs_review"] = needs_review_out
    output["email_signals"] = signals_count_out

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    return {
        "rows": int(len(output)),
        "rows_needing_review": int(sum(needs_review_out)),
        "review_threshold": resolved_threshold,
        "output_csv": str(output_path),
    }
