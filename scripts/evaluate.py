import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_detector.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained spam classifier.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV with text and label columns.")
    parser.add_argument("--model", default="models/spam_model.joblib", help="Model artifact path.")
    parser.add_argument("--report", default="models/evaluation.json", help="Output report path.")
    args = parser.parse_args()

    report = run_evaluation(data_path=args.data, model_path=args.model, report_output=args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
