from .config import DEFAULT_CONFIG, ensure_parent
from spam_detector.evaluate import run_evaluation
from spam_detector.predict import predict_batch, predict_text
from spam_detector.threshold import tune_review_threshold
from spam_detector.train import run_training


def train_model(data_path: str, model_output: str | None, metrics_output: str | None, random_state: int) -> dict:
    model_path = model_output or DEFAULT_CONFIG.model_path
    metrics_path = metrics_output or DEFAULT_CONFIG.metrics_path
    ensure_parent(model_path)
    ensure_parent(metrics_path)
    return run_training(data_path=data_path, model_output=model_path, metrics_output=metrics_path, random_state=random_state)


def evaluate_model(data_path: str, model_path: str | None, report_output: str | None) -> dict:
    chosen_model = model_path or DEFAULT_CONFIG.model_path
    chosen_report = report_output if report_output is not None else DEFAULT_CONFIG.evaluation_path
    if chosen_report:
        ensure_parent(chosen_report)
    return run_evaluation(data_path=data_path, model_path=chosen_model, report_output=chosen_report)


def predict_single(text: str, model_path: str | None, review_threshold: float | None, threshold_config_path: str | None) -> dict:
    chosen_model = model_path or DEFAULT_CONFIG.model_path
    chosen_threshold_config = threshold_config_path if threshold_config_path is not None else DEFAULT_CONFIG.threshold_config_path
    return predict_text(
        text=text,
        model_path=chosen_model,
        review_threshold=review_threshold,
        threshold_config_path=chosen_threshold_config,
    )


def predict_csv(
    input_csv: str,
    output_csv: str | None,
    model_path: str | None,
    text_column: str,
    review_threshold: float | None,
    threshold_config_path: str | None,
) -> dict:
    chosen_output = output_csv or DEFAULT_CONFIG.prediction_output_path
    chosen_model = model_path or DEFAULT_CONFIG.model_path
    chosen_threshold_config = threshold_config_path if threshold_config_path is not None else DEFAULT_CONFIG.threshold_config_path
    ensure_parent(chosen_output)
    return predict_batch(
        input_csv=input_csv,
        output_csv=chosen_output,
        model_path=chosen_model,
        text_column=text_column,
        review_threshold=review_threshold,
        threshold_config_path=chosen_threshold_config,
    )


def tune_threshold(
    data_path: str,
    model_path: str | None,
    text_column: str,
    target_review_rate: float,
    output_path: str | None,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> dict:
    chosen_model = model_path or DEFAULT_CONFIG.model_path
    chosen_output = output_path if output_path is not None else DEFAULT_CONFIG.threshold_config_path
    if chosen_output:
        ensure_parent(chosen_output)
    return tune_review_threshold(
        data_path=data_path,
        model_path=chosen_model,
        text_column=text_column,
        target_review_rate=target_review_rate,
        output_path=chosen_output,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
    )
