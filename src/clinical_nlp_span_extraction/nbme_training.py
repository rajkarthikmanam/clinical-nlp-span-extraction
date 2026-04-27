from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from datasets import Dataset
import torch
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from clinical_nlp_span_extraction.nbme import tokenize_with_offsets
from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans

WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NBME token classification model.")
    parser.add_argument("--train", required=True, help="Prepared train.jsonl.")
    parser.add_argument("--valid", required=True, help="Prepared valid.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory to save artifacts.")
    parser.add_argument("--model-name", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0, help="Optional row limit for training.")
    parser.add_argument("--limit-valid", type=int, default=0, help="Optional row limit for validation.")
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Train only on rows with at least one gold span to reduce class imbalance.",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def to_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


def tokenize_batch(batch: dict, tokenizer, label_to_id: dict[str, int], max_length: int) -> dict:
    feature_tokens = [WORD_PATTERN.findall(text.lower()) for text in batch["feature_text"]]
    tokenized = tokenizer(
        feature_tokens,
        batch["tokens"],
        truncation="only_second",
        is_split_into_words=True,
        max_length=max_length,
    )

    aligned_labels = []
    for row_index, labels in enumerate(batch["labels"]):
        word_ids = tokenized.word_ids(batch_index=row_index)
        sequence_ids = tokenized.sequence_ids(batch_index=row_index)
        previous_word_idx = None
        label_ids = []

        for word_idx, sequence_id in zip(word_ids, sequence_ids):
            if sequence_id != 1 or word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[labels[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def extract_spans_from_labels(tokens: list[str], labels: list[str], note_text: str) -> list[tuple[int, int]]:
    token_offsets = tokenize_with_offsets(note_text)
    spans: list[tuple[int, int]] = []
    start = None
    end = None
    for idx, label in enumerate(labels):
        if label == "B-SPAN":
            if start is not None and end is not None:
                spans.append((start, end))
            start = token_offsets[idx].start
            end = token_offsets[idx].end
        elif label == "I-SPAN" and start is not None:
            end = token_offsets[idx].end
        else:
            if start is not None and end is not None:
                spans.append((start, end))
            start = None
            end = None
    if start is not None and end is not None:
        spans.append((start, end))
    return spans


def build_label_weights(tokenized_dataset: Dataset, num_labels: int) -> torch.Tensor:
    counts = [0] * num_labels
    for labels in tokenized_dataset["labels"]:
        for label_id in labels:
            if label_id != -100:
                counts[int(label_id)] += 1

    total = sum(counts)
    weights = []
    for count in counts:
        safe_count = max(count, 1)
        weights.append(total / (num_labels * safe_count))
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, label_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            weight=self.label_weights.to(logits.device) if self.label_weights is not None else None,
        )
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_rows = read_jsonl(args.train, args.limit_train)
    valid_rows = read_jsonl(args.valid, args.limit_valid)
    if args.positive_only:
        train_rows = [row for row in train_rows if row["spans"]]

    labels = ["O", "B-SPAN", "I-SPAN"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_dataset = to_dataset(train_rows)
    valid_dataset = to_dataset(valid_rows)
    train_tokenized = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, label_to_id, args.max_length),
        batched=True,
    )
    valid_tokenized = valid_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, label_to_id, args.max_length),
        batched=True,
    )
    label_weights = build_label_weights(train_tokenized, len(labels))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        label_weights=label_weights,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    predictions, _, _ = trainer.predict(valid_tokenized)
    predicted_label_ids = predictions.argmax(axis=-1)

    pred_spans = []
    gold_spans = []
    prediction_rows = []
    for row, aligned_label_ids, predicted_ids in zip(
        valid_rows,
        valid_tokenized["labels"],
        predicted_label_ids,
        strict=False,
    ):
        note_labels = []
        for gold_label_id, predicted_label_id in zip(aligned_label_ids, predicted_ids, strict=False):
            if gold_label_id == -100:
                continue
            note_labels.append(id_to_label[int(predicted_label_id)])
        predicted_spans = extract_spans_from_labels(row["tokens"], note_labels, row["note_text"])
        pred_spans.append(predicted_spans)
        gold_spans.append([tuple(span) for span in row["spans"]])
        prediction_rows.append({"id": row["id"], "predicted_spans": predicted_spans})

    metrics = micro_f1_from_spans(gold_spans, pred_spans)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "valid_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prediction_rows),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    run_summary = {
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "positive_only": args.positive_only,
        "label_weights": [round(float(weight), 6) for weight in label_weights.tolist()],
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "metrics": metrics,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
