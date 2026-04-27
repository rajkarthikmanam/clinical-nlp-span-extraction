# Final Submission Checklist

Use this checklist before submitting the project for the course or publishing it online.

## Repository

- [ ] `README.md` explains the problem, methods, results, and next steps clearly
- [ ] all core scripts run from the command line
- [ ] `requirements.txt` and `pyproject.toml` are present
- [ ] tests or smoke checks are included
- [ ] bulky generated artifacts are ignored unless intentionally committed

## Experiments

- [ ] baseline run completed and metric recorded
- [ ] BiLSTM run completed and metric recorded
- [ ] transformer run completed and metric recorded
- [ ] result table in `artifacts/results-summary.md` matches the latest runs
- [ ] at least one paragraph explains why the baseline is currently strongest

## Report

- [ ] include task definition and why span extraction is harder than classification
- [ ] describe dataset preparation from NBME Kaggle CSV files
- [ ] explain evaluation metric as character-level micro F1
- [ ] compare baseline vs BiLSTM vs transformer honestly
- [ ] include failure analysis or error examples

## Presentation

- [ ] one slide for problem statement
- [ ] one slide for data and label format
- [ ] one slide for model comparison
- [ ] one slide for results table
- [ ] one slide for lessons learned and future work

## GitHub Publish

- [ ] create public repo `clinical-nlp-span-extraction`
- [ ] add `origin` remote
- [ ] push `main`
- [ ] verify README renders correctly on GitHub
