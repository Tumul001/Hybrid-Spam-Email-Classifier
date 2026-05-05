"""Spam detector package — BERT + Keyword Rules hybrid classifier."""

from .bert_model import bert_predict_text
from .keyword_rules import match_spam_rules, match_ham_rules, combined_keyword_boost, SPAM_RULES, HAM_RULES, CATEGORIES, RULE_COUNT
from .predict import predict_text, predict_batch
from .preprocess import clean_text

__all__ = [
    "bert_predict_text",
    "match_spam_rules",
    "match_ham_rules",
    "combined_keyword_boost",
    "SPAM_RULES",
    "HAM_RULES",
    "CATEGORIES",
    "RULE_COUNT",
    "predict_text",
    "predict_batch",
    "clean_text",
]
