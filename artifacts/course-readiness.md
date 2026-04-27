# Course Readiness Checklist

## Already Covered

- Kaggle competition dataset prepared locally
- Clear problem statement and business relevance
- Several methods, not just one model
- At least one real PyTorch deep learning method
- Competition-style evaluation metric
- Reproducible scripts and README
- Report template and result summary

## Minimum Remaining Work

1. Run one stronger transformer experiment beyond the smoke subset.
2. Add the final comparison table to the report.
3. Prepare 3-5 qualitative error analysis examples.
4. Decide whether to include the optional LLM comparison.

## Natural Presentation Story

1. Problem: automate clinical note evidence extraction.
2. Data: note-feature pairs with exact gold evidence spans.
3. Baseline: lexical matching gives a surprisingly strong floor.
4. Deep learning: BiLSTM recovers many positives but over-predicts.
5. Transformer: pretrained token classifier needs imbalance-aware training.
6. Conclusion: span extraction is harder than classification, and domain-aware models are the natural next step.
