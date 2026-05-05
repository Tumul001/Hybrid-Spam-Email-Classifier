from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from spam_detector.train import run_training


def _ensure_model() -> None:
    model_path = Path("models/spam_model.joblib")
    if model_path.exists():
        return
    run_training(
        data_path="data/raw/spam.csv",
        model_output=str(model_path),
        metrics_output="models/metrics.json",
        random_state=42,
    )


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_text_endpoint() -> None:
    _ensure_model()
    client = TestClient(app)
    response = client.post(
        "/api/v1/predict/text",
        json={
            "text": "Reminder: standup is at 10 AM",
            "model_path": "models/spam_model.joblib",
        },
    )
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["prediction"] in {"spam", "not spam"}
    assert 0.0 <= float(payload["confidence"]) <= 1.0


def test_evaluate_endpoint() -> None:
    _ensure_model()
    client = TestClient(app)
    response = client.post(
        "/api/v1/evaluate",
        json={
            "data_path": "data/raw/spam.csv",
            "model_path": "models/spam_model.joblib",
            "report_output": None,
        },
    )
    assert response.status_code == 200
    payload = response.json()["data"]
    assert "metrics" in payload
    assert "accuracy" in payload["metrics"]

