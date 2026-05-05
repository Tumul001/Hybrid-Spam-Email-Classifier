import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from .schemas import (
    ApiResult,
    EvaluateRequest,
    PredictBatchRequest,
    PredictTextRequest,
    TrainRequest,
    TuneThresholdRequest,
)
from .service import evaluate_model, predict_csv, predict_single, train_model, tune_threshold

app = FastAPI(title="Spam Detector API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/train", response_model=ApiResult)
def train_endpoint(payload: TrainRequest) -> ApiResult:
    try:
        result = train_model(
            data_path=payload.data_path,
            model_output=payload.model_output,
            metrics_output=payload.metrics_output,
            random_state=payload.random_state,
        )
        return ApiResult(data=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/evaluate", response_model=ApiResult)
def evaluate_endpoint(payload: EvaluateRequest) -> ApiResult:
    try:
        result = evaluate_model(
            data_path=payload.data_path,
            model_path=payload.model_path,
            report_output=payload.report_output,
        )
        return ApiResult(data=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/predict/text", response_model=ApiResult)
def predict_text_endpoint(payload: PredictTextRequest) -> ApiResult:
    try:
        result = predict_single(
            text=payload.text,
            model_path=payload.model_path,
            review_threshold=payload.review_threshold,
            threshold_config_path=payload.threshold_config_path,
        )
        return ApiResult(data=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/predict/batch", response_model=ApiResult)
def predict_batch_endpoint(payload: PredictBatchRequest) -> ApiResult:
    try:
        result = predict_csv(
            input_csv=payload.input_csv,
            output_csv=payload.output_csv,
            model_path=payload.model_path,
            text_column=payload.text_column,
            review_threshold=payload.review_threshold,
            threshold_config_path=payload.threshold_config_path,
        )
        return ApiResult(data=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/threshold/tune", response_model=ApiResult)
def tune_threshold_endpoint(payload: TuneThresholdRequest) -> ApiResult:
    try:
        result = tune_threshold(
            data_path=payload.data_path,
            model_path=payload.model_path,
            text_column=payload.text_column,
            target_review_rate=payload.target_review_rate,
            output_path=payload.output_path,
            min_threshold=payload.min_threshold,
            max_threshold=payload.max_threshold,
            step=payload.step,
        )
        return ApiResult(data=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
