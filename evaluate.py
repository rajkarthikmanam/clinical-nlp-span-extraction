import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parent / "src").resolve()))

from clinical_nlp_span_extraction.data import load_jsonl_examples
from clinical_nlp_span_extraction.metrics import compute_span_classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate span extraction predictions.")
    parser.add_argument("--data", required=True, help="Path to gold JSONL data.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_examples = load_jsonl_examples(args.data)

    pred_path = Path(args.predictions)
    predictions = {}
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        predictions[row["id"]] = row["labels"]

    gold = []
    pred = []
    missing_prediction_ids = []
    for example in gold_examples:
        gold.append(example.labels)
        if example.id not in predictions:
            missing_prediction_ids.append(example.id)
            pred.append(["O"] * len(example.labels))
            continue
        pred.append(predictions[example.id])

    report = compute_span_classification_report(gold, pred)
    if missing_prediction_ids:
        report["missing_prediction_ids"] = missing_prediction_ids
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
