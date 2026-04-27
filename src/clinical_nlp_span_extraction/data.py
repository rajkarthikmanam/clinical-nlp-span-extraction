from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ClinicalExample:
    id: str
    tokens: list[str]
    labels: list[str]


def load_jsonl_examples(path: str | Path) -> list[ClinicalExample]:
    examples: list[ClinicalExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(
            ClinicalExample(
                id=row["id"],
                tokens=row["tokens"],
                labels=row["labels"],
            )
        )
    return examples


def build_label_vocabulary(examples: list[ClinicalExample]) -> list[str]:
    labels = sorted({label for example in examples for label in example.labels})
    if "O" in labels:
        labels.remove("O")
        labels.insert(0, "O")
    return labels
