from typing import Any, Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import clean_text

LABEL_ORDER = ["not spam", "spam"]

MODEL_TYPE_TFIDF_LR = "tfidf_lr"
MODEL_TYPE_COUNT_NB = "count_nb"
VALID_MODEL_TYPES = {MODEL_TYPE_TFIDF_LR, MODEL_TYPE_COUNT_NB}


def build_pipeline(random_state: int = 42, model_type: str = MODEL_TYPE_TFIDF_LR) -> Pipeline:
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {sorted(VALID_MODEL_TYPES)}")

    if model_type == MODEL_TYPE_COUNT_NB:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB

        return Pipeline(
            steps=[
                (
                    "vectorizer",
                    CountVectorizer(
                        preprocessor=clean_text,
                        ngram_range=(1, 2),
                        max_features=8000,
                    ),
                ),
                ("classifier", MultinomialNB()),
            ]
        )

    # Default: TF-IDF + Logistic Regression
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=8000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=600,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _scores(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_ORDER,
    }


def evaluate_pipeline(pipeline: Pipeline, texts: pd.Series, labels: pd.Series) -> Dict[str, Any]:
    preds = pipeline.predict(texts)
    return _scores(labels.tolist(), preds.tolist())


def train_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = MODEL_TYPE_TFIDF_LR,
) -> Dict[str, Any]:
    x = df["text"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(random_state=random_state, model_type=model_type)
    pipeline.fit(x_train, y_train)

    return {
        "model": pipeline,
        "model_type": model_type,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_metrics": evaluate_pipeline(pipeline, x_train, y_train),
        "test_metrics": evaluate_pipeline(pipeline, x_test, y_test),
    }
