import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_detector.predict import predict_batch, predict_text, DEFAULT_REVIEW_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hybrid spam predictions (BERT-tiny + keyword rules)."
    )
    parser.add_argument("--text", help="Single message for prediction.")
    parser.add_argument("--input-csv", help="Batch input CSV path.")
    parser.add_argument(
        "--output-csv",
        default="data/processed/predictions.csv",
        help="Batch output CSV path.",
    )
    parser.add_argument(
        "--text-column",
        default="message",
        help="Input text column for batch mode (auto-detects common names if missing).",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=DEFAULT_REVIEW_THRESHOLD,
        help="Confidence threshold for needs_review (default: 0.60).",
    )
    args = parser.parse_args()

    if args.text:
        result = predict_text(
            text=args.text,
            review_threshold=args.review_threshold,
        )
        print(json.dumps(result, indent=2))
        return

    if args.input_csv:
        result = predict_batch(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            text_column=args.text_column,
            review_threshold=args.review_threshold,
        )
        print(json.dumps(result, indent=2))
        return

    parser.error("Provide either --text or --input-csv")


if __name__ == "__main__":
    main()
