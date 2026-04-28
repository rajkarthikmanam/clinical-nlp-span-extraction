#!/usr/bin/env python
"""Improve baseline predictions with post-processing."""

import json
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str((Path(__file__).resolve().parent / "src").resolve()))

from clinical_nlp_span_extraction.nbme_metrics import micro_f1_from_spans


def improve_baseline(baseline_file, valid_file, output_file):
    """Post-process baseline predictions to improve F1."""
    
    # Load baseline predictions
    baseline_preds = {}
    with open(baseline_file) as f:
        for line in f:
            row = json.loads(line)
            doc_id = row.get("id")
            baseline_preds[doc_id] = row.get("predicted_spans", [])
    
    # Load validation data
    valid_data = {}
    with open(valid_file) as f:
        for line in f:
            row = json.loads(line)
            doc_id = row.get("id")
            valid_data[doc_id] = {
                "text": row.get("text", ""),
                "spans": row.get("spans", [])
            }
    
    # Post-process predictions - filter very short spans
    improved_preds = {}
    for doc_id, spans in baseline_preds.items():
        # Filter very short spans (< 2 chars) - likely noise
        filtered_spans = []
        for span in spans:
            if isinstance(span, (list, tuple)) and len(span) == 2:
                start, end = span
                if end - start >= 2:  # Keep spans of at least 2 characters
                    filtered_spans.append([start, end])
        improved_preds[doc_id] = filtered_spans
    
    # Calculate metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for doc_id in valid_data:
        gold_spans = set(tuple(s) for s in valid_data[doc_id]["spans"] if len(s) == 2)
        pred_spans = set(tuple(s) for s in improved_preds.get(doc_id, []))
        
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "counts": {
            "true_positive": true_positives,
            "false_positive": false_positives,
            "false_negative": false_negatives
        }
    }
    
    print("\nImproved Baseline Metrics (with min-length filtering):")
    print(json.dumps(metrics, indent=2))
    
    # Save metrics
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics_improved.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improve baseline predictions")
    parser.add_argument("--baseline", required=True, help="Baseline prediction file")
    parser.add_argument("--valid", required=True, help="Validation data file")
    parser.add_argument("--output", required=True, help="Output file")
    
    args = parser.parse_args()
    
    improve_baseline(args.baseline, args.valid, args.output)
