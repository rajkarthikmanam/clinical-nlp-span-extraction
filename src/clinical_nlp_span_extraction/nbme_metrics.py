from __future__ import annotations


def spans_to_char_set(spans: list[tuple[int, int]]) -> set[int]:
    char_positions: set[int] = set()
    for start, end in spans:
        char_positions.update(range(start, end))
    return char_positions


def micro_f1_from_spans(
    gold: list[list[tuple[int, int]]],
    predicted: list[list[tuple[int, int]]],
) -> dict[str, float | dict[str, int]]:
    tp = 0
    fp = 0
    fn = 0

    for gold_spans, pred_spans in zip(gold, predicted, strict=False):
        gold_chars = spans_to_char_set(gold_spans)
        pred_chars = spans_to_char_set(pred_spans)
        tp += len(gold_chars & pred_chars)
        fp += len(pred_chars - gold_chars)
        fn += len(gold_chars - pred_chars)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "counts": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        },
    }
