from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans
from clinical_nlp_span_extraction.nbme_training import extract_spans_from_labels


WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SEP_TOKEN = "[SEP]"


def tokenize_feature_text(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BiLSTM baseline for NBME span extraction.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-valid", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_vocab(rows: list[dict], min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(token.lower() for token in row["tokens"])
        counter.update(tokenize_feature_text(row["feature_text"]))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SEP_TOKEN: 2}
    for token, count in counter.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_label_weights(rows: list[dict]) -> torch.Tensor:
    counter = Counter()
    for row in rows:
        counter.update(row["labels"])
    total = sum(counter.values())
    weights = []
    for label in ["O", "B-SPAN", "I-SPAN"]:
        count = max(counter.get(label, 1), 1)
        weights.append(total / (len(LABEL_TO_ID) * count))
    return torch.tensor(weights, dtype=torch.float32)


LABEL_TO_ID = {"O": 0, "B-SPAN": 1, "I-SPAN": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(slots=True)
class EncodedExample:
    id: str
    input_ids: list[int]
    labels: list[int]
    note_token_count: int
    tokens: list[str]
    note_text: str
    spans: list[list[int]]


def encode_row(row: dict, vocab: dict[str, int]) -> EncodedExample:
    feature_tokens = tokenize_feature_text(row["feature_text"])
    note_tokens = [token.lower() for token in row["tokens"]]
    input_tokens = feature_tokens + [SEP_TOKEN] + note_tokens
    input_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in input_tokens]
    labels = [-100] * (len(feature_tokens) + 1) + [LABEL_TO_ID[label] for label in row["labels"]]
    return EncodedExample(
        id=row["id"],
        input_ids=input_ids,
        labels=labels,
        note_token_count=len(note_tokens),
        tokens=row["tokens"],
        note_text=row["note_text"],
        spans=row["spans"],
    )


class NbmeBiLstmDataset(Dataset):
    def __init__(self, rows: list[dict], vocab: dict[str, int]) -> None:
        self.examples = [encode_row(row, vocab) for row in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def collate_batch(batch: list[EncodedExample]) -> dict[str, torch.Tensor | list]:
    max_len = max(len(example.input_ids) for example in batch)
    input_ids = []
    labels = []
    attention_mask = []
    metadata = []

    for example in batch:
        pad_len = max_len - len(example.input_ids)
        input_ids.append(example.input_ids + [0] * pad_len)
        labels.append(example.labels + [-100] * pad_len)
        attention_mask.append([1] * len(example.input_ids) + [0] * pad_len)
        metadata.append(example)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "metadata": metadata,
    }


class BiLstmTagger(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_labels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        encoded = self.dropout(encoded)
        return self.classifier(encoded)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device, loss_fn: nn.Module) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    all_gold = []
    all_pred = []
    prediction_rows = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            predictions = logits.argmax(dim=-1).cpu().tolist()

            for example, label_ids in zip(batch["metadata"], predictions, strict=False):
                note_label_ids = [label_id for gold_id, label_id in zip(example.labels, label_ids, strict=False) if gold_id != -100]
                note_labels = [ID_TO_LABEL[label_id] for label_id in note_label_ids[: example.note_token_count]]
                if len(note_labels) < example.note_token_count:
                    note_labels.extend(["O"] * (example.note_token_count - len(note_labels)))

                pred_spans = extract_spans_from_labels(example.tokens, note_labels, example.note_text)
                gold_spans = [tuple(span) for span in example.spans]
                all_gold.append(gold_spans)
                all_pred.append(pred_spans)
                prediction_rows.append({"id": example.id, "predicted_spans": pred_spans})

    return micro_f1_from_spans(all_gold, all_pred), prediction_rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_rows = read_jsonl(args.train, args.limit_train)
    valid_rows = read_jsonl(args.valid, args.limit_valid)
    vocab = build_vocab(train_rows)
    label_weights = build_label_weights(train_rows).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = NbmeBiLstmDataset(train_rows, vocab)
    valid_dataset = NbmeBiLstmDataset(valid_rows, vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLstmTagger(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_labels=len(LABEL_TO_ID),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=label_weights)

    best_metrics = None
    best_predictions = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, loss_fn)
        metrics, prediction_rows = evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_predictions = prediction_rows
        print(json.dumps({"epoch": epoch, "train_loss": round(train_loss, 6), **metrics}, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "bilstm_model.pt")
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    (output_dir / "valid_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (best_predictions or [])),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
