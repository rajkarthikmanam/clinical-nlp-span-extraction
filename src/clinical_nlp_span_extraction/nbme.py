from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(slots=True)
class NbmeToken:
    text: str
    start: int
    end: int


@dataclass(slots=True)
class NbmeExample:
    id: str
    case_num: str
    pn_num: str
    feature_num: str
    feature_text: str
    note_text: str
    spans: list[tuple[int, int]]
    tokens: list[str]
    labels: list[str]


def tokenize_with_offsets(text: str) -> list[NbmeToken]:
    return [
        NbmeToken(text=match.group(0), start=match.start(), end=match.end())
        for match in TOKEN_PATTERN.finditer(text)
    ]


def parse_location_field(raw_value: str) -> list[tuple[int, int]]:
    if not raw_value:
        return []

    parsed = ast.literal_eval(raw_value)
    spans: list[tuple[int, int]] = []
    for item in parsed:
        if not item:
            continue
        for segment in str(item).split(";"):
            segment = segment.strip()
            if not segment:
                continue
            parts = segment.split()
            if len(parts) != 2:
                continue
            start, end = int(parts[0]), int(parts[1])
            spans.append((start, end))
    return spans


def build_bio_labels(tokens: list[NbmeToken], spans: list[tuple[int, int]]) -> list[str]:
    labels = ["O"] * len(tokens)
    for span_start, span_end in spans:
        active_indices = [
            idx
            for idx, token in enumerate(tokens)
            if not (token.end <= span_start or token.start >= span_end)
        ]
        for position, token_index in enumerate(active_indices):
            labels[token_index] = "B-SPAN" if position == 0 else "I-SPAN"
    return labels


def normalize_feature_text(text: str) -> str:
    return text.replace("-OR-", " or ").replace("-", " ").strip()


def load_nbme_tables(dataset_dir: str | Path) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], list[dict[str, str]]]:
    dataset_dir = Path(dataset_dir)

    features: dict[tuple[str, str], str] = {}
    with (dataset_dir / "features.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["case_num"], row["feature_num"])
            features[key] = normalize_feature_text(row["feature_text"])

    notes: dict[tuple[str, str], str] = {}
    with (dataset_dir / "patient_notes.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["case_num"], row["pn_num"])
            notes[key] = row["pn_history"]

    train_rows: list[dict[str, str]] = []
    with (dataset_dir / "train.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        train_rows.extend(reader)

    return features, notes, train_rows


def row_to_example(
    row: dict[str, str],
    features: dict[tuple[str, str], str],
    notes: dict[tuple[str, str], str],
) -> NbmeExample:
    case_num = row["case_num"]
    pn_num = row["pn_num"]
    feature_num = row["feature_num"]
    feature_text = features[(case_num, feature_num)]
    note_text = notes[(case_num, pn_num)]
    spans = parse_location_field(row["location"])
    token_objects = tokenize_with_offsets(note_text)
    labels = build_bio_labels(token_objects, spans)
    return NbmeExample(
        id=row["id"],
        case_num=case_num,
        pn_num=pn_num,
        feature_num=feature_num,
        feature_text=feature_text,
        note_text=note_text,
        spans=spans,
        tokens=[token.text for token in token_objects],
        labels=labels,
    )


def write_examples(path: str | Path, examples: list[NbmeExample]) -> None:
    rows = [
        {
            "id": example.id,
            "case_num": example.case_num,
            "pn_num": example.pn_num,
            "feature_num": example.feature_num,
            "feature_text": example.feature_text,
            "note_text": example.note_text,
            "spans": example.spans,
            "tokens": example.tokens,
            "labels": example.labels,
        }
        for example in examples
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def split_examples(
    rows: list[dict[str, str]],
    features: dict[tuple[str, str], str],
    notes: dict[tuple[str, str], str],
    valid_mod: int,
    valid_fold: int,
) -> tuple[list[NbmeExample], list[NbmeExample]]:
    train_examples: list[NbmeExample] = []
    valid_examples: list[NbmeExample] = []

    for row in rows:
        example = row_to_example(row, features, notes)
        fold_bucket = int(example.pn_num) % valid_mod
        if fold_bucket == valid_fold:
            valid_examples.append(example)
        else:
            train_examples.append(example)

    return train_examples, valid_examples


def parse_prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NBME data for token classification.")
    parser.add_argument("--dataset-dir", required=True, help="Folder containing Kaggle CSV files.")
    parser.add_argument("--output-dir", required=True, help="Folder to write train/valid JSONL files.")
    parser.add_argument("--valid-mod", type=int, default=5, help="Modulo base for deterministic split.")
    parser.add_argument("--valid-fold", type=int, default=0, help="Which modulo bucket becomes validation.")
    return parser.parse_args()


def main_prepare_nbme() -> None:
    args = parse_prepare_args()
    features, notes, train_rows = load_nbme_tables(args.dataset_dir)
    train_examples, valid_examples = split_examples(
        train_rows,
        features,
        notes,
        valid_mod=args.valid_mod,
        valid_fold=args.valid_fold,
    )

    output_dir = Path(args.output_dir)
    write_examples(output_dir / "train.jsonl", train_examples)
    write_examples(output_dir / "valid.jsonl", valid_examples)

    summary = {
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "cases": sorted({row["case_num"] for row in train_rows}),
        "features": len(features),
        "notes": len(notes),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
