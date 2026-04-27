from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans


PROMPT_TEMPLATE = """You are extracting evidence spans from a clinical note.

Feature:
{feature}

Patient note:
{note}

Return a JSON object with this schema:
{{"spans": ["exact text span 1", "exact text span 2"]}}

Only include exact substrings from the patient note that express the feature. If not present, return an empty list.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an optional LLM comparison on an NBME validation subset.")
    parser.add_argument("--data", required=True, help="Prepared validation JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--limit", type=int, default=50, help="Number of rows to score.")
    return parser.parse_args()


def find_substring_spans(note_text: str, snippets: list[str]) -> list[tuple[int, int]]:
    lowered = note_text.lower()
    spans = []
    for snippet in snippets:
        snippet = snippet.strip()
        if not snippet:
            continue
        start = lowered.find(snippet.lower())
        if start >= 0:
            spans.append((start, start + len(snippet)))
    return spans


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. This script is optional and only runs when an API key is available.")

    from openai import OpenAI

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]
    client = OpenAI(api_key=api_key)

    output_rows = []
    gold = []
    predicted = []

    for row in rows:
        prompt = PROMPT_TEMPLATE.format(feature=row["feature_text"], note=row["note_text"])
        response = client.responses.create(model=args.model, input=prompt)
        text = response.output_text
        payload = json.loads(text)
        pred_spans = find_substring_spans(row["note_text"], payload.get("spans", []))
        output_rows.append({"id": row["id"], "predicted_spans": pred_spans, "raw_response": payload})
        gold.append([tuple(span) for span in row["spans"]])
        predicted.append(pred_spans)

    metrics = micro_f1_from_spans(gold, predicted)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row) for row in output_rows), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
