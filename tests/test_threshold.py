import json
from pathlib import Path

import pytest

from spam_detector.predict import DEFAULT_REVIEW_THRESHOLD, predict_text
from spam_detector.threshold import tune_review_threshold
from spam_detector.train import run_training


def test_tune_review_threshold_generates_report(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "spam.csv"

    model_path = tmp_path / "spam_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    run_training(str(data_path), str(model_path), str(metrics_path), random_state=42)

    report_path = tmp_path / "review_threshold.json"
    report = tune_review_threshold(
        data_path=str(data_path),
        model_path=str(model_path),
        target_review_rate=0.2,
        output_path=str(report_path),
    )

    assert report_path.exists()
    assert 0.0 < report["suggested_threshold"] < 1.0
    assert 0.0 <= report["actual_review_rate"] <= 1.0
    assert report["dataset_rows"] > 0
    assert abs(report["actual_review_rate"] - 0.2) <= 0.25

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["suggested_threshold"] == report["suggested_threshold"]


def test_predict_text_auto_loads_tuned_threshold(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "spam.csv"

    model_path = tmp_path / "spam_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    run_training(str(data_path), str(model_path), str(metrics_path), random_state=42)

    report_path = tmp_path / "review_threshold.json"
    report = tune_review_threshold(
        data_path=str(data_path),
        model_path=str(model_path),
        target_review_rate=0.2,
        output_path=str(report_path),
    )

    prediction = predict_text(
        "Claim your free reward now",
        model_path=str(model_path),
        threshold_config_path=str(report_path),
    )

    assert abs(prediction["review_threshold"] - report["suggested_threshold"]) < 1e-12


def test_predict_text_falls_back_to_default_threshold_when_config_missing(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "spam.csv"

    model_path = tmp_path / "spam_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    run_training(str(data_path), str(model_path), str(metrics_path), random_state=42)

    missing_config = tmp_path / "does_not_exist.json"
    prediction = predict_text(
        "This is a normal meeting invite",
        model_path=str(model_path),
        threshold_config_path=str(missing_config),
    )

    assert prediction["review_threshold"] == DEFAULT_REVIEW_THRESHOLD


def test_tune_review_threshold_rejects_invalid_target() -> None:
    with pytest.raises(ValueError):
        tune_review_threshold(
            data_path="ignored.csv",
            target_review_rate=0.0,
            output_path=None,
        )

