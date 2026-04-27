from clinical_nlp_span_extraction.nbme import build_bio_labels, parse_location_field, tokenize_with_offsets
from clinical_nlp_span_extraction.nbme_crf import row_to_features
from clinical_nlp_span_extraction.nbme_linear import decode_labels_with_constraints
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


def test_linear_decoder_limits_predictions_to_feature_neighborhood() -> None:
    labels = decode_labels_with_constraints(
        probabilities=[
            [0.05, 0.90, 0.05],
            [0.10, 0.15, 0.75],
            [0.05, 0.90, 0.05],
        ],
        classes=["O", "B-SPAN", "I-SPAN"],
        tokens=["severe", "chest", "pain"],
        feature_words={"chest"},
        candidate_window=0,
        positive_threshold=0.55,
    )
    assert labels == ["O", "B-SPAN", "O"]


def test_crf_features_capture_feature_proximity() -> None:
    feature_rows = row_to_features(
        {
            "feature_text": "chest pain",
            "tokens": ["patient", "has", "chest", "pain"],
        }
    )
    assert feature_rows[2]["feature.contains_token"] is True
    assert feature_rows[0]["feature.distance_bucket"] == 2
