# Clinical NLP Span Extraction

Clinical NLP project centered on the NBME clinical patient notes competition, built to compare rule-based baselines, a BiLSTM sequence tagger, transformer token classification, and an optional LLM-based comparison.

## Project Summary

This repository focuses on extracting evidence spans from patient notes for a given clinical feature. The core task is not simple classification. For each note-feature pair, the system must identify the exact text span or spans in the note that express the target clinical concept.

That makes the project a strong fit for:

- deep learning
- transformers
- evaluation-heavy NLP
- healthcare AI
- portfolio and resume use

## Competition Context

The current implementation is adapted to the Kaggle NBME clinical notes setup, where each example includes:

- a `feature_text`
- a `patient note`
- gold span locations inside the note

The project uses the competition-style character-level micro F1 metric for evaluation.

## Methods Implemented

The repository now supports these paths directly:

- keyword-style baseline based on feature text matching
- BiLSTM sequence tagging baseline in PyTorch
- transformer token classification workflow for ClinicalBERT-style models
- optional LLM subset comparison using prompt-based span extraction

## Repository Layout

```text
clinical-nlp-span-extraction/
|-- data/
|-- artifacts/
|-- src/
|   `-- clinical_nlp_span_extraction/
|-- tests/
|-- prepare_nbme_data.py
|-- baseline_nbme.py
|-- train_nbme_bilstm.py
|-- train_nbme.py
|-- llm_nbme_compare.py
|-- train.py
|-- evaluate.py
|-- predict.py
|-- requirements.txt
`-- README.md
```

## Data Preparation

If the Kaggle NBME files are stored locally, prepare the joined training data first:

```bash
python prepare_nbme_data.py \
  --dataset-dir ../nbme-score-clinical-patient-notes \
  --output-dir data/nbme
```

This creates:

- `data/nbme/train.jsonl`
- `data/nbme/valid.jsonl`
- `data/nbme/summary.json`

## NBME Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a local package:

```bash
pip install -e .
```

Run the baseline:

```bash
python baseline_nbme.py \
  --data data/nbme/valid.jsonl \
  --output artifacts/nbme-baseline/predictions.jsonl
```

The current keyword-style baseline already runs end to end and produced an initial validation F1 around `0.347` on the prepared split.

Train the BiLSTM baseline:

```bash
python train_nbme_bilstm.py \
  --train data/nbme/train.jsonl \
  --valid data/nbme/valid.jsonl \
  --output-dir artifacts/nbme-bilstm
```

Quick CPU smoke test:

```bash
python train_nbme_bilstm.py \
  --train data/nbme/train.jsonl \
  --valid data/nbme/valid.jsonl \
  --output-dir artifacts/nbme-bilstm-smoke \
  --limit-train 512 \
  --limit-valid 128 \
  --epochs 2
```

Train the transformer token-classification model:

```bash
python train_nbme.py \
  --train data/nbme/train.jsonl \
  --valid data/nbme/valid.jsonl \
  --output-dir artifacts/nbme-clinicalbert
```

Optional LLM comparison on a small validation subset:

```bash
python llm_nbme_compare.py \
  --data data/nbme/valid.jsonl \
  --output artifacts/nbme-llm/predictions.jsonl \
  --limit 50
```

This script is optional and only runs when `OPENAI_API_KEY` is available.

## What Is Already Working

- Kaggle CSV ingestion for NBME
- joined training example generation
- deterministic train/validation split
- competition-style micro F1 at character span level
- baseline prediction pipeline
- BiLSTM training pipeline in PyTorch
- transformer training pipeline structure
- optional LLM comparison scaffold
- package metadata for local install and test discovery
- evaluation fallback for partial prediction files

## Current Results

- keyword-style baseline validation F1: about `0.347`
- BiLSTM full-run validation F1: about `0.075`
- weighted DistilBERT smoke run validation F1: about `0.092`

The current transformer result comes from a small CPU-friendly smoke configuration with `distilbert-base-uncased`, weighted loss, and positive-only training rows. It is meant to prove the full token-classification path works end to end. The next practical step is to run a stronger biomedical encoder on a longer schedule.

## Result Snapshot

| Method | Split / setup | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Keyword baseline | prepared validation split | 0.497 | 0.267 | 0.347 |
| BiLSTM | full prepared split, 2 epochs | 0.039 | 0.857 | 0.075 |
| DistilBERT token classifier | smoke subset, weighted loss, positive-only | 0.050 | 0.626 | 0.092 |

This is a believable project story for class and portfolio use because it shows:

- a simple baseline that is not trivial to beat
- a first deep learning model that over-predicts spans
- a transformer path that needed explicit imbalance handling to avoid all-`O` collapse

## Report and Presentation Support

The repository includes a ready-made result template at:

- `artifacts/report-template.md`
- `artifacts/results-summary.md`
- `artifacts/course-readiness.md`

This can be used to track:

- baseline vs BiLSTM vs transformer results
- ablations
- error analysis
- optional LLM comparison

## Why This Project Is Strong

This project demonstrates more than a standard text classification workflow:

- token- or span-level prediction
- paraphrase-heavy medical language
- multi-span extraction
- healthcare-domain evaluation
- direct comparison across simple baselines, deep learning, transformers, and optional LLM methods

## Next Steps

1. run a biomedical transformer on the full split
2. tune the BiLSTM to reduce false positives
3. compare baseline vs BiLSTM vs transformer in the result table
4. optionally score a small validation subset with an LLM for comparison

## Testing

Run the test suite with:

```bash
pytest
```
