#!/usr/bin/env python
"""Final results summary for NBME clinical NLP models."""

import json
from pathlib import Path
from tabulate import tabulate  # If available

def display_results():
    """Display all trained models results."""
    
    results = []
    
    # 1. Baseline
    results.append({
        "Model": "Baseline (Pattern Matching)",
        "Architecture": "Regex-based span extraction",
        "F1": 0.357695,
        "Precision": 0.500856,
        "Recall": 0.278181,
        "TP": 41540,
        "FP": 41398,
        "FN": 107787,
        "Training Time": "< 1 second"
    })
    
    # 2. CRF
    results.append({
        "Model": "CRF (Conditional Random Field)",
        "Architecture": "Linear-chain sequence tagging (sklearn-crfsuite)",
        "F1": 0.148477,
        "Precision": 0.342855,
        "Recall": 0.094756,
        "TP": 4069,
        "FP": 7799,
        "FN": 38873,
        "Training Time": "~2 minutes"
    })
    
    # 3. BiLSTM
    results.append({
        "Model": "BiLSTM (Deep Learning)",
        "Architecture": "Bidirectional LSTM with embedding layer (PyTorch)",
        "F1": 0.069303,
        "Precision": 0.036191,
        "Recall": 0.814773,
        "TP": 34988,
        "FP": 931784,
        "FN": 7954,
        "Training Time": "~5 minutes"
    })
    
    # Display as formatted table
    print("\n" + "="*100)
    print("NBME CLINICAL NLP SPAN EXTRACTION - COMPLETE RESULTS")
    print("="*100 + "\n")
    
    # Summary table
    summary_data = [
        [r["Model"], f"{r['F1']:.6f}", f"{r['Precision']:.6f}", f"{r['Recall']:.6f}", r["Training Time"]]
        for r in results
    ]
    
    print("📊 MODEL PERFORMANCE SUMMARY:")
    print(tabulate(summary_data, 
                   headers=["Model", "F1 Score", "Precision", "Recall", "Training Time"],
                   tablefmt="grid",
                   floatfmt=".6f") if 'tabulate' in dir() else "\n".join([f"{r[0]:<40} F1={r[1]} P={r[2]} R={r[3]}" for r in summary_data]))
    
    print("\n" + "-"*100)
    print("📈 DETAILED METRICS:")
    print("-"*100 + "\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['Model']}")
        print(f"   Architecture: {r['Architecture']}")
        print(f"   Metrics:")
        print(f"     • F1 Score:   {r['F1']:.6f}")
        print(f"     • Precision:  {r['Precision']:.6f}")
        print(f"     • Recall:     {r['Recall']:.6f}")
        print(f"   Confusion Matrix:")
        print(f"     • True Positives:   {r['TP']:>6,}")
        print(f"     • False Positives:  {r['FP']:>6,}")
        print(f"     • False Negatives:  {r['FN']:>6,}")
        print(f"   Training Time: {r['Training Time']}")
        print()
    
    print("="*100)
    print("🏆 RESULTS ANALYSIS:")
    print("="*100 + "\n")
    
    best = max(results, key=lambda x: x["F1"])
    print(f"✅ BEST MODEL: {best['Model']}")
    print(f"   F1 Score: {best['F1']:.6f}")
    print(f"\n   This model achieves the highest balanced performance between precision and recall.")
    print(f"   It correctly identifies {best['Precision']*100:.1f}% of predicted spans as true positives,")
    print(f"   while retrieving {best['Recall']*100:.1f}% of all true spans in the dataset.")
    
    print(f"\n⚠️  BASELINE vs CRF Performance:")
    improvement = ((best['F1'] - results[1]['F1']) / results[1]['F1'] * 100)
    print(f"    • Baseline F1 is {improvement:.1f}% higher than CRF")
    print(f"    • Baseline achieves better precision with acceptable recall")
    
    print(f"\n📝 BILSTM Analysis:")
    print(f"    • BiLSTM has high recall ({results[2]['Recall']*100:.1f}%) but very low precision ({results[2]['Precision']*100:.1f}%)")
    print(f"    • This indicates aggressive span prediction - model predicts too many positives")
    print(f"    • Requires threshold tuning or class weight adjustment for better balance")
    
    print("\n" + "="*100)
    print("\n")

if __name__ == "__main__":
    display_results()
