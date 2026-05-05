import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

SPAM_LABELS = {"spam", "1", "true", "yes", "junk"}
HAM_LABELS = {"ham", "not spam", "not_spam", "notspam", "0", "false", "no", "legit"}
TEXT_COLUMN_CANDIDATES = ("text", "message", "body", "email", "content")
LABEL_COLUMN_CANDIDATES = ("label", "category", "class", "target")


def normalize_label(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in SPAM_LABELS:
        return "spam"
    if raw in HAM_LABELS:
        return "not spam"
    raise ValueError(f"Unsupported label: {value}")


def clean_text(text: Any) -> str:
    content = str(text or "")
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"https?://\S+|www\.\S+", " ", content)
    content = re.sub(r"\S+@\S+", " ", content)
    content = re.sub(r"[^a-zA-Z0-9\s]", " ", content)
    content = re.sub(r"\s+", " ", content).strip().lower()
    return content


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(
        f"Missing required {kind} column. Expected one of: {', '.join(candidates)}."
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _resolve_column(df, TEXT_COLUMN_CANDIDATES, "text")
    label_col = _resolve_column(df, LABEL_COLUMN_CANDIDATES, "label")

    cleaned = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna().copy()
    if cleaned.empty:
        raise ValueError("Dataset has no usable rows after dropping empty values.")

    cleaned["text"] = cleaned["text"].apply(clean_text)
    cleaned = cleaned[cleaned["text"].str.len() > 0]

    normalized_labels = []
    for idx, label in cleaned["label"].items():
        try:
            normalized_labels.append(normalize_label(label))
        except ValueError as exc:
            raise ValueError(f"Invalid label at row {idx}: {label}") from exc

    cleaned["label"] = normalized_labels

    if cleaned["label"].nunique() < 2:
        raise ValueError("Dataset must contain at least two classes after normalization.")

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Email spam signal detector
# ─────────────────────────────────────────────────────────────────────────────
# Marketing emails must include certain elements by law (CAN-SPAM Act, GDPR).
# The SMS-trained ML model has never seen these patterns, so we detect them
# separately and use them to boost the spam probability (hybrid scoring).

@dataclass
class EmailSignal:
    name: str
    description: str
    weight: float  # how much to boost spam probability (0.0–1.0)


# Ordered from strongest to weakest signal
_EMAIL_SIGNAL_RULES: list[tuple[str, str, float, re.Pattern]] = [
    (
        "unsubscribe_link",
        "Contains 'Unsubscribe' link/text (legally required in all marketing emails)",
        0.55,
        re.compile(r"\bunsubscribe\b", re.IGNORECASE),
    ),
    (
        "view_in_browser",
        "Contains 'View in browser' (newsletter/email-client pattern)",
        0.45,
        re.compile(r"\bview\s+in\s+(browser|your\s+browser)\b", re.IGNORECASE),
    ),
    (
        "manage_preferences",
        "Contains 'Manage preferences/settings' (marketing email footer)",
        0.40,
        re.compile(r"\bmanage\s+(preferences|settings|subscription)\b", re.IGNORECASE),
    ),
    (
        "newsletter_label",
        "Self-identifies as a Newsletter",
        0.40,
        re.compile(r"\bnewsletter\b", re.IGNORECASE),
    ),
    (
        "company_address",
        "Contains a physical mailing address (CAN-SPAM requirement for commercial email)",
        0.35,
        re.compile(
            r"\b\d{2,5}\s+\w[\w\s]{2,30}(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Pl|Ct|Sq|Pkwy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "promotional_ps",
        "Contains P.S. / P.P.S. (classic marketing email pattern)",
        0.25,
        re.compile(r"\bP\.?P?\.?S\.?\b", re.IGNORECASE),
    ),
    (
        "forwarding_cta",
        "Asks recipients to forward to others (viral marketing)",
        0.20,
        re.compile(r"\bforward\s+this\b|\bshare\s+this\b", re.IGNORECASE),
    ),
    (
        "subscribe_cta",
        "Asks non-subscribers to subscribe",
        0.20,
        re.compile(r"\bclick\s+here\s+to\s+subscribe\b|\bsubscribe\s+to\s+(our|this|the)\b", re.IGNORECASE),
    ),
]


def email_spam_signals(text: str) -> list[EmailSignal]:
    """
    Detect structural signals that indicate a commercial/marketing email.
    Returns a list of matched EmailSignal objects (empty list = no signals found).

    These signals are independent of the ML model — they catch newsletter spam
    that the SMS-trained model misses entirely.
    """
    raw = str(text or "")
    matched: list[EmailSignal] = []
    for name, description, weight, pattern in _EMAIL_SIGNAL_RULES:
        if pattern.search(raw):
            matched.append(EmailSignal(name=name, description=description, weight=weight))
    return matched
