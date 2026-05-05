from typing import Any

from pydantic import BaseModel, Field


class PredictTextRequest(BaseModel):
    text: str = Field(min_length=1)
    model_path: str | None = None
    review_threshold: float | None = None
    threshold_config_path: str | None = None


class PredictBatchRequest(BaseModel):
    input_csv: str
    output_csv: str | None = None
    model_path: str | None = None
    text_column: str = "text"
    review_threshold: float | None = None
    threshold_config_path: str | None = None


class EvaluateRequest(BaseModel):
    data_path: str
    model_path: str | None = None
    report_output: str | None = None


class TrainRequest(BaseModel):
    data_path: str
    model_output: str | None = None
    metrics_output: str | None = None
    random_state: int = 42


class TuneThresholdRequest(BaseModel):
    data_path: str
    model_path: str | None = None
    text_column: str = "text"
    target_review_rate: float = 0.2
    output_path: str | None = None
    min_threshold: float = 0.05
    max_threshold: float = 0.99
    step: float = 0.01


class ApiResult(BaseModel):
    status: str = "ok"
    data: dict[str, Any]
