from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    model_path: str = "models/spam_model.joblib"
    metrics_path: str = "models/metrics.json"
    evaluation_path: str = "models/evaluation.json"
    threshold_config_path: str = "models/review_threshold.json"
    prediction_output_path: str = "data/processed/predictions.csv"


DEFAULT_CONFIG = AppConfig()


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)
