from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    train_path: str
    valid_path: str
    output_dir: str
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 3
    weight_decay: float = 0.01
    max_length: int = 256
    seed: int = 42
