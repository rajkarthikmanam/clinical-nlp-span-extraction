from clinical_nlp_span_extraction.metrics import compute_span_classification_report, extract_spans


def test_extract_spans_reads_bio_sequences() -> None:
    labels = ["O", "B-PROBLEM", "I-PROBLEM", "O", "B-TEST", "I-TEST"]
    assert extract_spans(labels) == [("PROBLEM", 1, 2), ("TEST", 4, 5)]


def test_span_report_scores_exact_matches() -> None:
    gold = [["O", "B-PROBLEM", "I-PROBLEM"]]
    pred = [["O", "B-PROBLEM", "I-PROBLEM"]]
    report = compute_span_classification_report(gold, pred)
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0
