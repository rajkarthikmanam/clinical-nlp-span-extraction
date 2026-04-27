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

## Best Immediate Upgrade Path

1. Run `Bio_ClinicalBERT` or another biomedical encoder on the full prepared split.
2. Add a smaller learning rate sweep for the transformer.
3. Reduce BiLSTM false positives with weaker positive weighting or post-processing.
4. Add an optional zero-shot LLM comparison on `25-50` validation rows if API access is available.
