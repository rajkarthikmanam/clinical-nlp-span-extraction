# NBME Results Summary

This file tracks the current project state for the class report and presentation.

## Prepared Data

- Training rows: `11,185`
- Validation rows: `3,115`
- Positive-span training rows: `7,736`
- No-span training rows: `3,449`

## Current Experiments

| Method | Data scope | Main configuration | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| Keyword baseline | full validation split | feature keyword matching | 0.497264 | 0.266662 | 0.347158 |
| BiLSTM sequence tagger | full prepared split | 2 epochs, batch size 32 | 0.039358 | 0.857226 | 0.075260 |
| DistilBERT token classifier | smoke subset | 2 epochs, weighted loss, positive-only | 0.049604 | 0.626066 | 0.091925 |

## Interpretation

- The keyword baseline is currently the strongest score because it is conservative and precise on direct lexical overlap.
- The BiLSTM and smoke-transformer runs both recover many spans, but they currently over-predict and lose precision.
- The main technical lesson so far is that NBME has strong class imbalance, so weighted loss or sampling strategy is necessary.

## Qualitative Error Examples

### 1. Baseline wins on direct lexical demographics

- Feature: `44 year`
- Example ID: `20040_216`
- Gold span: `[[5, 7]]`
- Baseline: exact match on the age mention
- BiLSTM: predicted dozens of extra spans across the note

Takeaway:
When the feature maps almost directly to a short lexical cue, the baseline is hard to beat. The BiLSTM currently behaves like a high-recall tagger with weak precision control.

### 2. Baseline misses broader paraphrased history, BiLSTM partially recovers it

- Feature: `Increased frequency recently`
- Example ID: `50615_509`
- Gold span: `[[147, 219]]`
- Baseline: no span predicted
- BiLSTM: captured a broad region overlapping the correct note segment, but with extra surrounding text

Takeaway:
The learned model is beginning to detect longer contextual evidence that simple keyword matching cannot capture, but span boundaries remain loose.

### 3. Learned model captures multi-part stress evidence better than a short keyword match

- Feature: `Stress due to caring for elderly parents`
- Example ID: `41130_402`
- Gold spans: `[[233, 242], [252, 266], [275, 293]]`
- Baseline: only a short nearby lexical hit
- BiLSTM: recovered a much larger overlapping region across the main stress narrative

Takeaway:
This is the upside of the sequence model: it can connect broader semantics, but it still needs better calibration to avoid over-expanding spans.

## Best Immediate Upgrade Path

1. Run `Bio_ClinicalBERT` or another biomedical encoder on the full prepared split.
2. Add a smaller learning rate sweep for the transformer.
3. Reduce BiLSTM false positives with weaker positive weighting or post-processing.
4. Add an optional zero-shot LLM comparison on `25-50` validation rows if API access is available.

## Presentation Note

For a course presentation, the clean story is:

- the baseline establishes a strong lexical floor
- the BiLSTM demonstrates that deep learning can recover broader context
- the transformer smoke run proves the token-classification pipeline works end to end
- class imbalance and span-boundary precision are the main reasons the learned models currently underperform
