import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_detector.predict import predict_batch, predict_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spam predictions.")
    parser.add_argument("--model", default="models/spam_model.joblib", help="Model artifact path.")
    parser.add_argument("--text", help="Single message for prediction.")
    parser.add_argument("--input-csv", help="Batch input CSV path.")
    parser.add_argument("--output-csv", default="data/processed/predictions.csv", help="Batch output CSV path.")
    parser.add_argument("--text-column", default="text", help="Input text column for batch mode.")
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Manual threshold override. If omitted, threshold config is used when available.",
    )
    parser.add_argument(
        "--threshold-config",
        default="models/review_threshold.json",
        help="JSON file path that contains suggested_threshold from tuning.",
    )
    args = parser.parse_args()

    threshold_config = args.threshold_config if args.threshold_config else None

    if args.text:
        result = predict_text(
            text=args.text,
            model_path=args.model,
            review_threshold=args.review_threshold,
            threshold_config_path=threshold_config,
        )
        print(json.dumps(result, indent=2))
        return

    if args.input_csv:
        result = predict_batch(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            model_path=args.model,
            text_column=args.text_column,
            review_threshold=args.review_threshold,
            threshold_config_path=threshold_config,
        )
        print(json.dumps(result, indent=2))
        return

    parser.error("Provide either --text or --input-csv")


if __name__ == "__main__":
    main()
