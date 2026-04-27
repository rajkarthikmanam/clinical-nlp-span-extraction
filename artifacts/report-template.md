# NBME Project Result Template

## Competition

- NBME - Score Clinical Patient Notes
- Task: extract evidence spans from patient notes for a given clinical feature
- Evaluation metric: micro F1

## Methods Compared

| Method | Type | Validation F1 | Notes |
|---|---|---:|---|
| Keyword / fuzzy baseline | rule-based baseline | 0.3472 | current implemented baseline |
| BiLSTM sequence tagger | deep learning | 0.0753 | full split run; currently over-predicts spans |
| DistilBERT token classifier | transformer smoke run | 0.0919 | weighted loss + positive-only smoke experiment |
| ClinicalBERT / BioClinicalBERT | transformer full run | TBD | `train_nbme.py` next target |
| Optional LLM subset comparison | zero-shot / few-shot | TBD | `llm_nbme_compare.py` |

## Ablations

- feature text included vs omitted
- model size comparison
- sequence length comparison
- class weighting or sampling strategy
- with and without LLM-assisted augmentation or validation

## Error Analysis

- paraphrase misses
- multi-span misses
- boundary errors
- feature confusion across similar symptoms
