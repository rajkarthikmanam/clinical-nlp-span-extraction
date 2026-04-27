from __future__ import annotations

from collections import Counter


def extract_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = None
    current_type = None

    for index, label in enumerate(labels):
        if label == "O":
            if start is not None and current_type is not None:
                spans.append((current_type, start, index - 1))
            start = None
            current_type = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            if start is not None and current_type is not None:
                spans.append((current_type, start, index - 1))
            start = index
            current_type = entity_type
            continue

        if prefix == "I":
            if start is None or current_type != entity_type:
                start = index
                current_type = entity_type

    if start is not None and current_type is not None:
        spans.append((current_type, start, len(labels) - 1))

    return spans


def compute_span_classification_report(
    gold_sequences: list[list[str]],
    predicted_sequences: list[list[str]],
) -> dict[str, float | dict[str, float]]:
    gold_spans = [span for seq in gold_sequences for span in extract_spans(seq)]
    pred_spans = [span for seq in predicted_sequences for span in extract_spans(seq)]

    gold_counter = Counter(gold_spans)
    pred_counter = Counter(pred_spans)
    true_positive = sum((gold_counter & pred_counter).values())
    false_positive = sum((pred_counter - gold_counter).values())
    false_negative = sum((gold_counter - pred_counter).values())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "counts": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }
