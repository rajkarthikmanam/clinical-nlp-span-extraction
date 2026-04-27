from clinical_nlp_span_extraction.nbme import build_bio_labels, parse_location_field, tokenize_with_offsets
from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans


def test_parse_location_field_handles_multiple_segments() -> None:
    raw = "['10 15;20 25', '30 32']"
    assert parse_location_field(raw) == [(10, 15), (20, 25), (30, 32)]


def test_build_bio_labels_marks_overlapping_tokens() -> None:
    tokens = tokenize_with_offsets("patient reports chest pain")
    labels = build_bio_labels(tokens, [(16, 26)])
    assert labels == ["O", "O", "B-SPAN", "I-SPAN"]


def test_char_level_micro_f1_is_exact() -> None:
    report = micro_f1_from_spans([[(0, 4)]], [[(0, 4)]])
    assert report["f1"] == 1.0
