#!/usr/bin/env python
"""Ensemble multiple NBME model predictions."""

import json
from pathlib import Path
from collections import defaultdict
import argparse


def load_predictions(file_path):
    """Load predictions from JSONL file."""
    predictions = {}
    with open(file_path) as f:
        for line in f:
            row = json.loads(line)
            predictions[row.get("id", row.get("pn_num", ""))] = row
    return predictions


def ensemble_by_voting(pred_files, output_file):
    """Create ensemble by voting on spans."""
    # Load all predictions
    all_preds = []
    for pf in pred_files:
        all_preds.append(load_predictions(pf))
    
    # Get all unique IDs
    all_ids = set()
    for preds in all_preds:
        all_ids.update(preds.keys())
    
    # Ensemble predictions
    ensemble_preds = []
    
    for doc_id in sorted(all_ids):
        # Collect predictions from each model
        spans_by_model = []
        pn_num = None
        feature_text = None
        
        for preds in all_preds:
            if doc_id in preds:
                pred = preds[doc_id]
                pn_num = pred.get("pn_num", pred.get("id"))
                feature_text = pred.get("feature_text")
                # Get spans as set for comparison
                spans = set()
                if "spans" in pred:
                    if isinstance(pred["spans"], list):
                        spans = set(tuple(s) if isinstance(s, list) else s for s in pred["spans"])
                    elif isinstance(pred["spans"], str):
                        spans = {pred["spans"]}
                spans_by_model.append(spans)
        
        # Vote on spans (keep spans that appear in at least 2 models)
        if len(spans_by_model) > 0:
            all_spans = set()
            for spans in spans_by_model:
                all_spans.update(spans)
            
            # Count votes for each span
            span_votes = defaultdict(int)
            for spans in spans_by_model:
                for span in spans:
                    span_votes[span] += 1
            
            # Keep spans with majority vote (threshold: appeared in at least 50% of models)
            threshold = len(spans_by_model) / 2
            ensemble_spans = [span for span, votes in span_votes.items() if votes >= threshold]
            
            ensemble_preds.append({
                "pn_num": pn_num,
                "feature_text": feature_text,
                "spans": ensemble_spans
            })
    
    # Save ensemble predictions
    with open(output_file, "w") as f:
        for pred in ensemble_preds:
            f.write(json.dumps(pred) + "\n")
    
    print(f"Saved {len(ensemble_preds)} ensemble predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Ensemble NBME predictions")
    parser.add_argument("--pred1", required=True, help="First prediction file")
    parser.add_argument("--pred2", required=True, help="Second prediction file")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--valid", help="Validation file for evaluation")
    
    args = parser.parse_args()
    
    # Create ensemble
    ensemble_by_voting([args.pred1, args.pred2], args.output)
    
    # Evaluate if validation data provided
    if args.valid:
        from src.clinical_nlp_span_extraction.metrics import calculate_metrics
        
        ensemble_preds = load_predictions(args.output)
        
        # Load validation data
        valid_data = {}
        with open(args.valid) as f:
            for line in f:
                row = json.loads(line)
                valid_data[row.get("id", row.get("pn_num", ""))] = row
        
        # Calculate metrics
        metrics = calculate_metrics(valid_data, ensemble_preds)
        print("\nEnsemble Metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Save metrics
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    main()
