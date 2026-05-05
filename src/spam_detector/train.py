from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import file_sha256, save_json, save_model
from .model import MODEL_TYPE_TFIDF_LR, train_pipeline
from .preprocess import prepare_dataframe


def run_training(
    data_path: str,
    model_output: str = "models/spam_model.joblib",
    metrics_output: str = "models/metrics.json",
    random_state: int = 42,
    model_type: str = MODEL_TYPE_TFIDF_LR,
) -> dict[str, Any]:
    data_file = Path(data_path)
    raw_df = pd.read_csv(data_file)
    prepared_df = prepare_dataframe(raw_df)

    result = train_pipeline(prepared_df, random_state=random_state, model_type=model_type)

    model_path = Path(model_output)
    metrics_path = Path(metrics_output)
    save_model(result["model"], model_path)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "dataset": {
            "path": str(data_file),
            "rows": int(len(prepared_df)),
            "sha256": file_sha256(data_file),
            "class_distribution": {
                key: int(value)
                for key, value in prepared_df["label"].value_counts().to_dict().items()
            },
        },
        "split": {"train_size": result["train_size"], "test_size": result["test_size"]},
        "train_metrics": result["train_metrics"],
        "test_metrics": result["test_metrics"],
        "artifacts": {"model": str(model_path), "metrics": str(metrics_path)},
    }

    save_json(report, metrics_path)
    return report
