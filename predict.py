import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parent / "src").resolve()))

from clinical_nlp_span_extraction.training import predict_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run token classification inference on raw text.")
    parser.add_argument("--model-path", required=True, help="Directory containing the trained model.")
    parser.add_argument("--text", required=True, help="Input text to tag.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = predict_text(Path(args.model_path), args.text)
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
