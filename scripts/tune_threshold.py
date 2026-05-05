import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_detector.threshold import tune_review_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune review threshold for spam predictions.")
    parser.add_argument("--data", required=True, help="Path to CSV with text column.")
    parser.add_argument("--model", default="models/spam_model.joblib", help="Model artifact path.")
    parser.add_argument("--text-column", default="text", help="Input text column for tuning data.")
    parser.add_argument(
        "--target-review-rate",
        type=float,
        default=0.2,
        help="Target fraction of rows to flag for manual review.",
    )
    parser.add_argument(
        "--output",
        default="models/review_threshold.json",
        help="Output JSON path for threshold report. Use empty string to skip file output.",
    )
    parser.add_argument("--min-threshold", type=float, default=0.05, help="Minimum threshold in search range.")
    parser.add_argument("--max-threshold", type=float, default=0.99, help="Maximum threshold in search range.")
    parser.add_argument("--step", type=float, default=0.01, help="Search step size.")
    args = parser.parse_args()

    output_path = args.output if args.output else None

    report = tune_review_threshold(
        data_path=args.data,
        model_path=args.model,
        text_column=args.text_column,
        target_review_rate=args.target_review_rate,
        output_path=output_path,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
    )

    print("Threshold tuning complete")
    print(f"Suggested threshold: {report['suggested_threshold']}")
    print(f"Actual review rate: {report['actual_review_rate']:.4f}")
    print(f"Rows needing review: {report['rows_needing_review']} / {report['dataset_rows']}")
    if output_path:
        print(f"Report saved: {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
