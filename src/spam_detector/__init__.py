"""Spam detector package."""

from .evaluate import run_evaluation
from .predict import predict_batch, predict_text
from .threshold import tune_review_threshold
from .train import run_training

__all__ = ["run_training", "run_evaluation", "predict_text", "predict_batch", "tune_review_threshold"]
