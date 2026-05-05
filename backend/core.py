"""
backend/core.py — Direct-import backend (no HTTP, no FastAPI needed).

Streamlit imports these functions directly instead of calling the REST API.
This makes local development simpler: one process, no ports, no uvicorn.

The FastAPI app in main.py is still available if you want to expose the API
to other clients (mobile, Chrome extension, etc.) in the future.
"""

import sys
from pathlib import Path
from typing import Any

# Make sure src/ is on sys.path when imported from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spam_detector.evaluate import run_evaluation
from spam_detector.model import MODEL_TYPE_TFIDF_LR
from spam_detector.predict import predict_batch, predict_text, resolve_review_threshold
from spam_detector.threshold import tune_review_threshold
from spam_detector.train import run_training
from spam_detector.bert_model import bert_predict_text
from spam_detector.preprocess import email_spam_signals, clean_text

# --------------------------------------------------------------------------- #
# Paths & defaults                                                             #
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / "models" / "spam_model.joblib")
DEFAULT_METRICS_PATH = str(_PROJECT_ROOT / "models" / "metrics.json")
DEFAULT_THRESHOLD_CONFIG_PATH = str(_PROJECT_ROOT / "models" / "review_threshold.json")
DEFAULT_EVALUATION_PATH = str(_PROJECT_ROOT / "models" / "evaluation.json")
DEFAULT_DATA_PATH = str(_PROJECT_ROOT / "data" / "raw" / "spam.csv")


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def train(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    metrics_path: str = DEFAULT_METRICS_PATH,
    model_type: str = MODEL_TYPE_TFIDF_LR,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a model and save artifacts. Returns the metrics report."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    return run_training(
        data_path=data_path,
        model_output=model_path,
        metrics_output=metrics_path,
        random_state=random_state,
        model_type=model_type,
    )


def evaluate(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    report_path: str = DEFAULT_EVALUATION_PATH,
) -> dict[str, Any]:
    """Evaluate the model and return metrics."""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    return run_evaluation(
        data_path=data_path,
        model_path=model_path,
        report_output=report_path,
    )


def predict_single_email(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    review_threshold: float | None = None,
    threshold_config_path: str | None = DEFAULT_THRESHOLD_CONFIG_PATH,
) -> dict[str, Any]:
    """Classify a single email text. Returns prediction, confidence, needs_review."""
    return predict_text(
        text=text,
        model_path=model_path,
        review_threshold=review_threshold,
        threshold_config_path=threshold_config_path,
    )


def predict_csv_batch(
    input_csv: str,
    output_csv: str,
    model_path: str = DEFAULT_MODEL_PATH,
    text_column: str = "Message",
    review_threshold: float | None = None,
    threshold_config_path: str | None = DEFAULT_THRESHOLD_CONFIG_PATH,
) -> dict[str, Any]:
    """Classify all rows in a CSV. Saves results to output_csv."""
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    return predict_batch(
        input_csv=input_csv,
        output_csv=output_csv,
        model_path=model_path,
        text_column=text_column,
        review_threshold=review_threshold,
        threshold_config_path=threshold_config_path,
    )


def tune_threshold(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    target_review_rate: float = 0.2,
    output_path: str = DEFAULT_THRESHOLD_CONFIG_PATH,
) -> dict[str, Any]:
    """Find the confidence threshold that achieves ~target_review_rate."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return tune_review_threshold(
        data_path=data_path,
        model_path=model_path,
        target_review_rate=target_review_rate,
        output_path=output_path,
    )


def bert_predict_single_email(
    text: str,
    review_threshold: float = 0.6,
) -> dict[str, Any]:
    """
    Classify a single email using BERT-tiny (HuggingFace pre-trained model).
    Also applies the hybrid email signal layer on top — same as the ML model.
    """
    # BERT prediction
    bert_result = bert_predict_text(text)
    bert_spam_prob = bert_result["spam_probability"]

    # Apply email signal rules on top (same hybrid layer as the ML model)
    signals = email_spam_signals(text)
    if signals:
        combined_boost = 1.0
        for sig in signals:
            combined_boost *= (1.0 - sig.weight)
        signal_boost = 1.0 - combined_boost
        boosted_prob = 0.6 * signal_boost + 0.4 * bert_spam_prob
        final_prob = max(bert_spam_prob, boosted_prob)
    else:
        final_prob = bert_spam_prob

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
        "bert_spam_probability": round(bert_spam_prob, 6),
        "email_signals_detected": [
            {"name": s.name, "description": s.description, "weight": round(s.weight, 2)}
            for s in signals
        ],
        "model": "bert-tiny (mrm8488/bert-tiny-finetuned-sms-spam-detection)",
    }
