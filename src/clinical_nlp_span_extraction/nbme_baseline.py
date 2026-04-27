from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from clinical_nlp_span_extraction.nbme import parse_location_field
from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans


WORD_PATTERN = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple NBME baseline.")
    parser.add_argument("--data", required=True, help="Prepared NBME JSONL validation file.")
    parser.add_argument("--output", required=True, help="Path to save baseline predictions.")
    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def feature_keywords(feature_text: str) -> list[str]:
    stopwords = {"of", "or", "and", "on", "in", "the", "a", "an", "to", "with", "for", "history"}
    words = [word for word in normalize_words(feature_text) if word not in stopwords]
    return sorted(words, key=len, reverse=True)


def predict_spans(feature_text: str, note_text: str) -> list[tuple[int, int]]:
    keywords = feature_keywords(feature_text)
    lowered_note = note_text.lower()

    spans: list[tuple[int, int]] = []
    for keyword in keywords[:4]:
        for match in re.finditer(re.escape(keyword), lowered_note):
            spans.append((match.start(), match.end()))

    merged = merge_overlapping_spans(spans)
    return merged[:3]


def merge_overlapping_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]

    output_rows = []
    gold = []
    predicted = []
    for row in rows:
        pred_spans = predict_spans(row["feature_text"], row["note_text"])
        output_rows.append({"id": row["id"], "predicted_spans": pred_spans})
        gold.append([tuple(span) for span in row["spans"]])
        predicted.append(pred_spans)

    report = micro_f1_from_spans(gold, predicted)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row) for row in output_rows), encoding="utf-8")
    (output_path.parent / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
