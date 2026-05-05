from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import load_model, save_json
from .model import evaluate_pipeline
from .preprocess import prepare_dataframe


def run_evaluation(
    data_path: str,
    model_path: str = "models/spam_model.joblib",
    report_output: str | None = "models/evaluation.json",
) -> dict[str, Any]:
    df = pd.read_csv(Path(data_path))
    prepared = prepare_dataframe(df)

    model = load_model(Path(model_path))
    metrics = evaluate_pipeline(model, prepared["text"], prepared["label"])

    report: dict[str, Any] = {
        "dataset_rows": int(len(prepared)),
        "metrics": metrics,
        "model_path": model_path,
    }

    if report_output:
        save_json(report, Path(report_output))

    return report
