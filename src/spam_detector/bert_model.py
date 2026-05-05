"""
bert_model.py — BERT-based spam classifier using a pre-trained HuggingFace model.

Model: mrm8488/bert-tiny-finetuned-sms-spam-detection
- 17MB, CPU-friendly, no GPU needed
- Already fine-tuned for spam detection — no training required
- Downloads once, then cached locally (~/.cache/huggingface/)

Labels from this model:
  LABEL_0 → ham (not spam)
  LABEL_1 → spam
"""

from __future__ import annotations

from typing import Any

_pipeline = None  # lazy-loaded on first call


def _get_pipeline():
    """Load the HuggingFace pipeline once and cache it in memory."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            tokenizer="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            truncation=True,
            max_length=512,
        )
    return _pipeline


def bert_predict_text(text: str) -> dict[str, Any]:
    """
    Classify a single text using the pre-trained BERT-tiny spam model.

    Returns:
        {
            "label": "LABEL_0" | "LABEL_1",
            "spam_probability": float,   # 0.0 - 1.0
        }
    """
    pipe = _get_pipeline()

    # Run the model — returns [{"label": "LABEL_X", "score": float}]
    # We ask for all scores so we get both classes
    results = pipe(text, top_k=None)  # top_k=None → all labels

    spam_prob = 0.0
    for r in results:
        if r["label"] == "LABEL_1":  # LABEL_1 = spam
            spam_prob = float(r["score"])
            break

    return {
        "label": "spam" if spam_prob >= 0.5 else "not spam",
        "spam_probability": round(spam_prob, 6),
    }
