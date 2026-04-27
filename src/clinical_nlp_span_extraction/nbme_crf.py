from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import sklearn_crfsuite

from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans
from clinical_nlp_span_extraction.nbme_training import extract_spans_from_labels


WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CRF sequence tagger for NBME span extraction.")
    parser.add_argument("--train", required=True, help="Prepared train JSONL.")
    parser.add_argument("--valid", required=True, help="Prepared valid JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory to save artifacts.")
    parser.add_argument("--limit-train", type=int, default=0, help="Optional row limit for training.")
    parser.add_argument("--limit-valid", type=int, default=0, help="Optional row limit for validation.")
    parser.add_argument("--c1", type=float, default=0.05, help="L1 regularization strength.")
    parser.add_argument("--c2", type=float, default=0.05, help="L2 regularization strength.")
    parser.add_argument("--max-iterations", type=int, default=200, help="Maximum optimizer iterations.")
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def feature_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def token_features(tokens: list[str], feature_words: set[str], index: int) -> dict[str, object]:
    token = tokens[index]
    token_lower = token.lower()
    prev_token = tokens[index - 1].lower() if index > 0 else "<BOS>"
    next_token = tokens[index + 1].lower() if index + 1 < len(tokens) else "<EOS>"
    feature_positions = [pos for pos, current in enumerate(tokens) if current.lower() in feature_words]
    min_distance = min((abs(index - pos) for pos in feature_positions), default=99)

    return {
        "bias": 1.0,
        "token.lower": token_lower,
        "token.is_upper": token.isupper(),
        "token.is_title": token.istitle(),
        "token.is_digit": token.isdigit(),
        "token.len": min(len(token), 20),
        "token.prefix2": token_lower[:2],
        "token.prefix3": token_lower[:3],
        "token.suffix2": token_lower[-2:],
        "token.suffix3": token_lower[-3:],
        "context.prev": prev_token,
        "context.next": next_token,
        "context.bigram_prev": f"{prev_token}_{token_lower}",
        "context.bigram_next": f"{token_lower}_{next_token}",
        "feature.contains_token": token_lower in feature_words,
        "feature.distance_bucket": min(min_distance, 4),
        "position.is_first": index == 0,
        "position.is_last": index == len(tokens) - 1,
    }


def row_to_features(row: dict) -> list[dict[str, object]]:
    tokens = row["tokens"]
    feature_words = set(feature_tokens(row["feature_text"]))
    return [token_features(tokens, feature_words, index) for index in range(len(tokens))]


def build_datasets(rows: list[dict]) -> tuple[list[list[dict[str, object]]], list[list[str]]]:
    x_rows = [row_to_features(row) for row in rows]
    y_rows = [row["labels"] for row in rows]
    return x_rows, y_rows


def evaluate_rows(model, rows: list[dict]) -> tuple[dict, list[dict]]:
    gold_spans = []
    pred_spans = []
    prediction_rows = []

    for row in rows:
        predicted_labels = model.predict_single(row_to_features(row))
        spans = extract_spans_from_labels(row["tokens"], predicted_labels, row["note_text"])
        gold_spans.append([tuple(span) for span in row["spans"]])
        pred_spans.append(spans)
        prediction_rows.append({"id": row["id"], "predicted_spans": spans})

    metrics = micro_f1_from_spans(gold_spans, pred_spans)
    return metrics, prediction_rows


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train, args.limit_train)
    valid_rows = read_jsonl(args.valid, args.limit_valid)
    train_x, train_y = build_datasets(train_rows)

    model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
        all_possible_transitions=True,
    )
    model.fit(train_x, train_y)

    metrics, prediction_rows = evaluate_rows(model, valid_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "crf_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "valid_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prediction_rows),
        encoding="utf-8",
    )
    run_summary = {
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "c1": args.c1,
        "c2": args.c2,
        "max_iterations": args.max_iterations,
        "metrics": metrics,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
