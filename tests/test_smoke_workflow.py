import json
from pathlib import Path

import pandas as pd

from spam_detector.evaluate import run_evaluation
from spam_detector.predict import predict_batch, predict_text
from spam_detector.train import run_training


def test_train_evaluate_predict_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "spam.csv"

    model_path = tmp_path / "spam_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    report = run_training(str(data_path), str(model_path), str(metrics_path), random_state=42)

    assert model_path.exists()
    assert metrics_path.exists()
    assert "test_metrics" in report
    assert "f1_spam" in report["test_metrics"]

    evaluation = run_evaluation(str(data_path), str(model_path), str(tmp_path / "evaluation.json"))
    assert "metrics" in evaluation
    assert "accuracy" in evaluation["metrics"]

    single = predict_text("You have won a free prize", str(model_path), review_threshold=0.6)
    assert single["prediction"] in {"spam", "not spam"}
    assert 0.0 <= float(single["confidence"]) <= 1.0
    assert isinstance(single["needs_review"], bool)
    assert single["review_threshold"] == 0.6

    batch_output = tmp_path / "predictions.csv"
    batch = predict_batch(
        str(data_path),
        str(batch_output),
        str(model_path),
        text_column="Message",
        review_threshold=0.65,
    )
    assert batch_output.exists()
    assert batch["rows"] > 0
    assert "rows_needing_review" in batch
    assert batch["review_threshold"] == 0.65

    batch_df = pd.read_csv(batch_output)
    assert "needs_review" in batch_df.columns

    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "dataset" in metrics_json

