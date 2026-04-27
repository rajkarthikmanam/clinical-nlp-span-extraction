import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_handles_missing_prediction_ids(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"

    gold_rows = [
        {"id": "a", "tokens": ["patient", "pain"], "labels": ["O", "B-PAIN"]},
        {"id": "b", "tokens": ["no", "fever"], "labels": ["O", "B-FEVER"]},
    ]
    pred_rows = [
        {"id": "a", "labels": ["O", "B-PAIN"]},
    ]

    gold_path.write_text("\n".join(json.dumps(row) for row in gold_rows), encoding="utf-8")
    pred_path.write_text("\n".join(json.dumps(row) for row in pred_rows), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "evaluate.py", "--data", str(gold_path), "--predictions", str(pred_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    report = json.loads(result.stdout)
    assert report["missing_prediction_ids"] == ["b"]
