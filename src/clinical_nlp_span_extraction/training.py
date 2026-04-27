from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

from clinical_nlp_span_extraction.config import TrainingConfig
from clinical_nlp_span_extraction.data import build_label_vocabulary, load_jsonl_examples
from clinical_nlp_span_extraction.metrics import compute_span_classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clinical token classification model.")
    parser.add_argument("--train", required=True, help="Path to train JSONL.")
    parser.add_argument("--valid", required=True, help="Path to validation JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model artifacts.")
    parser.add_argument("--model-name", default="emilyalsentzer/Bio_ClinicalBERT", help="HF model name.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    return parser.parse_args()


def examples_to_dataset(path: str) -> Dataset:
    examples = load_jsonl_examples(path)
    return Dataset.from_list(
        [
            {
                "id": example.id,
                "tokens": example.tokens,
                "labels": example.labels,
            }
            for example in examples
        ]
    )


def tokenize_and_align_labels(dataset: Dataset, tokenizer, label_to_id: dict[str, int], max_length: int) -> Dataset:
    def tokenize_batch(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )

        aligned_labels = []
        for row_index, labels in enumerate(batch["labels"]):
            word_ids = tokenized.word_ids(batch_index=row_index)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[labels[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    return dataset.map(tokenize_batch, batched=True)


def train_model(config: TrainingConfig) -> None:
    set_seed(config.seed)

    train_examples = load_jsonl_examples(config.train_path)
    valid_examples = load_jsonl_examples(config.valid_path)
    labels = build_label_vocabulary(train_examples + valid_examples)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_dataset = examples_to_dataset(config.train_path)
    valid_dataset = examples_to_dataset(config.valid_path)
    train_tokenized = tokenize_and_align_labels(train_dataset, tokenizer, label_to_id, config.max_length)
    valid_tokenized = tokenize_and_align_labels(valid_dataset, tokenizer, label_to_id, config.max_length)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    inference = pipeline(
        "token-classification",
        model=config.output_dir,
        tokenizer=config.output_dir,
        aggregation_strategy="simple",
    )

    predictions_path = Path(config.output_dir) / "predictions.jsonl"
    rows = []
    for example in valid_examples:
        text = " ".join(example.tokens)
        entities = inference(text)
        predicted_labels = project_entities_to_tokens(example.tokens, entities)
        rows.append({"id": example.id, "labels": predicted_labels})

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    gold = [example.labels for example in valid_examples]
    predicted = [row["labels"] for row in rows]
    report = compute_span_classification_report(gold, predicted)
    (Path(config.output_dir) / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def project_entities_to_tokens(tokens: list[str], entities: list[dict]) -> list[str]:
    labels = ["O"] * len(tokens)
    token_text = " ".join(tokens)
    offsets = []
    position = 0
    for token in tokens:
        start = position
        end = start + len(token)
        offsets.append((start, end))
        position = end + 1

    for entity in entities:
        entity_label = entity["entity_group"]
        span_start = entity["start"]
        span_end = entity["end"]
        active = []
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_end <= span_start or token_start >= span_end:
                continue
            active.append(idx)

        for position_idx, token_index in enumerate(active):
            prefix = "B" if position_idx == 0 else "I"
            labels[token_index] = f"{prefix}-{entity_label}"

    return labels


def predict_text(model_path: Path, text: str) -> dict:
    inference = pipeline(
        "token-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        aggregation_strategy="simple",
    )
    return {
        "text": text,
        "entities": inference(text),
    }


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        train_path=args.train,
        valid_path=args.valid,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    train_model(config)


if __name__ == "__main__":
    main()
