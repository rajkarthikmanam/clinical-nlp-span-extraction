from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from clinical_nlp_span_extraction.nbme import tokenize_with_offsets
from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans
from clinical_nlp_span_extraction.nbme_training import extract_spans_from_labels


WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
LABELS = ["O", "B-SPAN", "I-SPAN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear token classifier for NBME span extraction.")
    parser.add_argument("--train", required=True, help="Prepared train JSONL.")
    parser.add_argument("--valid", required=True, help="Prepared valid JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory to save artifacts.")
    parser.add_argument("--limit-train", type=int, default=0, help="Optional row limit for training.")
    parser.add_argument("--limit-valid", type=int, default=0, help="Optional row limit for validation.")
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Train only on examples containing at least one positive span.",
    )
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum solver iterations.")
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def normalize_feature_words(text: str) -> set[str]:
    return set(WORD_PATTERN.findall(text.lower()))


def token_features(tokens: list[str], feature_words: set[str], index: int) -> dict[str, object]:
    token = tokens[index]
    token_lower = token.lower()
    prev_token = tokens[index - 1].lower() if index > 0 else "<BOS>"
    next_token = tokens[index + 1].lower() if index + 1 < len(tokens) else "<EOS>"

    features: dict[str, object] = {
        "token": token_lower,
        "token.is_upper": token.isupper(),
        "token.is_title": token.istitle(),
        "token.is_digit": token.isdigit(),
        "token.len": min(len(token), 20),
        "token.prefix2": token_lower[:2],
        "token.prefix3": token_lower[:3],
        "token.suffix2": token_lower[-2:],
        "token.suffix3": token_lower[-3:],
        "position.is_first": index == 0,
        "position.is_last": index == len(tokens) - 1,
        "context.prev": prev_token,
        "context.next": next_token,
        "context.bigram_prev": f"{prev_token}_{token_lower}",
        "context.bigram_next": f"{token_lower}_{next_token}",
        "feature.contains_token": token_lower in feature_words,
        "feature.has_digit": any(char.isdigit() for char in "".join(feature_words)),
    }

    for word in sorted(feature_words)[:8]:
        features[f"feature.word={word}"] = True

    return features


def build_token_rows(rows: list[dict]) -> tuple[list[dict[str, object]], list[str]]:
    examples = []
    labels = []
    for row in rows:
        feature_words = normalize_feature_words(row["feature_text"])
        tokens = row["tokens"]
        for index, label in enumerate(row["labels"]):
            examples.append(token_features(tokens, feature_words, index))
            labels.append(label)
    return examples, labels


def predict_spans_for_rows(model: Pipeline, rows: list[dict]) -> tuple[dict, list[dict]]:
    gold_spans = []
    pred_spans = []
    prediction_rows = []

    for row in rows:
        feature_words = normalize_feature_words(row["feature_text"])
        token_dicts = [token_features(row["tokens"], feature_words, index) for index in range(len(row["tokens"]))]
        predicted_labels = model.predict(token_dicts).tolist()
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
    if args.positive_only:
        train_rows = [row for row in train_rows if row["spans"]]

    train_x, train_y = build_token_rows(train_rows)

    model = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)

    metrics, prediction_rows = predict_spans_for_rows(model, valid_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "linear_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "valid_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prediction_rows),
        encoding="utf-8",
    )
    run_summary = {
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "positive_only": args.positive_only,
        "max_iter": args.max_iter,
        "metrics": metrics,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
