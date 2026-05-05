import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_detector.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spam classifier.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV with text and label columns.")
    parser.add_argument("--model", default="models/spam_model.joblib", help="Output model artifact path.")
    parser.add_argument("--metrics", default="models/metrics.json", help="Output metrics JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    args = parser.parse_args()

    report = run_training(
        data_path=args.data,
        model_output=args.model,
        metrics_output=args.metrics,
        random_state=args.seed,
    )

    print("Training complete")
    print("Test metrics:")
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Model artifact: {report['artifacts']['model']}")
    print(f"Metrics report: {report['artifacts']['metrics']}")


if __name__ == "__main__":
    main()
