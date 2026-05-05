"""
test_predict.py — Tests for the hybrid prediction pipeline (BERT + keywords).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from spam_detector.predict import predict_text, DEFAULT_REVIEW_THRESHOLD


NEWSLETTER = (
    "The HireReady Newsletter Vol. 87. Unsubscribe. View in browser. "
    "Manage preferences. P.S. Forward this newsletter. "
    "Jobscan, 1518 1st Ave S, Seattle WA 98134."
)
PHISHING = (
    "Your account has been suspended. Verify your identity immediately "
    "by clicking the link below. Action required."
)
LEGIT = "Hi, can we move tomorrow's standup to 3pm? Let me know if that works."


def test_predict_returns_required_keys():
    result = predict_text(LEGIT)
    required = {"text", "prediction", "confidence", "needs_review",
                "review_threshold", "bert_spam_probability", "email_signals_detected"}
    assert required.issubset(result.keys())


def test_prediction_is_valid_label():
    result = predict_text(LEGIT)
    assert result["prediction"] in ("spam", "not spam")


def test_confidence_in_range():
    result = predict_text(LEGIT)
    assert 0.0 <= result["confidence"] <= 1.0


def test_newsletter_classified_as_spam():
    result = predict_text(NEWSLETTER)
    assert result["prediction"] == "spam", (
        f"Newsletter should be spam but got: {result['prediction']} "
        f"(confidence: {result['confidence']:.1%}, bert: {result['bert_spam_probability']:.1%})"
    )


def test_newsletter_has_keyword_signals():
    result = predict_text(NEWSLETTER)
    assert len(result["email_signals_detected"]) > 0


def test_phishing_classified_as_spam():
    result = predict_text(PHISHING)
    assert result["prediction"] == "spam"


def test_legit_classified_as_not_spam():
    result = predict_text(LEGIT)
    assert result["prediction"] == "not spam", (
        f"Legit email should be not spam but got: {result['prediction']}"
    )


def test_custom_threshold_respected():
    result_low  = predict_text(LEGIT, review_threshold=0.99)
    result_high = predict_text(LEGIT, review_threshold=0.01)
    assert result_low["needs_review"] is True
    assert result_high["needs_review"] is False


def test_signal_dicts_have_required_keys():
    result = predict_text(NEWSLETTER)
    for sig in result["email_signals_detected"]:
        assert "name"        in sig
        assert "category"    in sig
        assert "description" in sig
        assert "weight"      in sig
