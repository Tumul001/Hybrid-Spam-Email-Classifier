from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import load_model, save_json
from .preprocess import clean_text, prepare_dataframe


def _validate_target_review_rate(target_review_rate: float) -> None:
    if not 0.0 < target_review_rate < 1.0:
        raise ValueError("target_review_rate must be between 0 and 1.")


def _validate_search_args(min_threshold: float, max_threshold: float, step: float) -> None:
    if not 0.0 < min_threshold < 1.0:
        raise ValueError("min_threshold must be between 0 and 1.")
    if not 0.0 < max_threshold < 1.0:
        raise ValueError("max_threshold must be between 0 and 1.")
    if min_threshold >= max_threshold:
        raise ValueError("min_threshold must be less than max_threshold.")
    if step <= 0.0:
        raise ValueError("step must be greater than 0.")


def _prediction_confidences(model: Any, texts: list[str]) -> list[float]:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba; cannot tune threshold.")

    predictions = model.predict(texts).tolist()
    probabilities = model.predict_proba(texts)
    class_to_idx = {label: index for index, label in enumerate(model.classes_)}

    confidences: list[float] = []
    for idx, predicted_label in enumerate(predictions):
        label_idx = class_to_idx[predicted_label]
        confidences.append(float(probabilities[idx][label_idx]))
    return confidences


def _review_rate(confidences: list[float], threshold: float) -> float:
    if not confidences:
        return 0.0
    review_count = sum(confidence < threshold for confidence in confidences)
    return float(review_count / len(confidences))


def _threshold_candidates(min_threshold: float, max_threshold: float, step: float) -> list[float]:
    candidates: list[float] = []
    current = min_threshold
    while current <= max_threshold + 1e-12:
        candidates.append(round(current, 4))
        current += step
    return candidates


def tune_review_threshold(
    data_path: str,
    model_path: str = "models/spam_model.joblib",
    text_column: str | None = None,  # kept for API compat but ignored; auto-detected
    target_review_rate: float = 0.2,
    output_path: str | None = "models/review_threshold.json",
    min_threshold: float = 0.05,
    max_threshold: float = 0.99,
    step: float = 0.01,
) -> dict[str, Any]:
    _validate_target_review_rate(target_review_rate)
    _validate_search_args(min_threshold, max_threshold, step)

    raw_df = pd.read_csv(Path(data_path))
    prepared_df = prepare_dataframe(raw_df)  # auto-detects Message/text, Category/label
    texts = prepared_df["text"].tolist()
    if not texts:
        raise ValueError("Dataset has no rows for threshold tuning.")

    model = load_model(Path(model_path))
    confidences = _prediction_confidences(model, texts)
    candidates = _threshold_candidates(min_threshold, max_threshold, step)

    best_threshold = candidates[0]
    best_rate = _review_rate(confidences, best_threshold)

    for threshold in candidates[1:]:
        rate = _review_rate(confidences, threshold)
        diff = abs(rate - target_review_rate)
        best_diff = abs(best_rate - target_review_rate)

        better_match = diff < best_diff
        tie_but_safer = diff == best_diff and rate >= target_review_rate and best_rate < target_review_rate

        if better_match or tie_but_safer:
            best_threshold = threshold
            best_rate = rate

    rows_needing_review = int(sum(confidence < best_threshold for confidence in confidences))

    report: dict[str, Any] = {
        "dataset_rows": int(len(texts)),
        "target_review_rate": float(target_review_rate),
        "suggested_threshold": float(best_threshold),
        "actual_review_rate": float(best_rate),
        "rows_needing_review": rows_needing_review,
        "search": {
            "min_threshold": float(min_threshold),
            "max_threshold": float(max_threshold),
            "step": float(step),
            "candidates_evaluated": int(len(candidates)),
        },
        "confidence_stats": {
            "min": float(min(confidences)),
            "max": float(max(confidences)),
            "mean": float(sum(confidences) / len(confidences)),
        },
        "artifacts": {
            "model_path": str(model_path),
            "data_path": str(data_path),
            "output_path": str(output_path) if output_path else None,
        },
    }

    if output_path:
        save_json(report, Path(output_path))

    return report
