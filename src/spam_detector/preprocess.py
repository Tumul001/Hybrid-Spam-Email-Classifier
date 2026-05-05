"""
preprocess.py — Text cleaning utilities.
Only clean_text() is used by the prediction pipeline.
"""

import re
from typing import Any


def clean_text(text: Any) -> str:
    """
    Normalise raw email text for display / logging purposes.
    NOTE: keyword rules run on RAW text (before cleaning) so that
    legal/structural signals like 'Unsubscribe' are not stripped.
    """
    content = str(text or "")
    content = re.sub(r"<[^>]+>", " ", content)          # strip HTML tags
    content = re.sub(r"https?://\S+|www\.\S+", " ", content)  # strip URLs
    content = re.sub(r"\S+@\S+", " ", content)           # strip email addrs
    content = re.sub(r"[^a-zA-Z0-9\s]", " ", content)   # strip punctuation
    content = re.sub(r"\s+", " ", content).strip().lower()
    return content
