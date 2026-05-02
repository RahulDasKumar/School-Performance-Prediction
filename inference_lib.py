"""Shared inference helpers for the trained SchoolLSTM model.

Mirrors the feature ordering used by training.py's VectorAssembler so that
single-row predictions made from the dashboard match what the model saw at
training time.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

NUMERIC_COLS = [
    "year",
    "ratio",
    "Violent Crime Total",
    "Property Crime Total",
    "Burglary",
    "Larceny-theft",
    "pct_notprof",
    "Motor vehicle theft",
]
CATEGORICAL_COLS = ["grade", "subgroup", "county", "city", "subject"]
TARGET = "pct_ccr"

MODEL_PATH = "model.pth"
SCALER_PATH = "scaler_stats.pkl"
ENCODERS_PATH = "encoders.pkl"


class SchoolLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def load_artifacts(
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    encoders_path: str = ENCODERS_PATH,
) -> dict[str, Any]:
    """Load model, scaler stats, and encoders. Returns a dict.

    Encoders file is optional — if missing, categoricals fall back to a
    deterministic hash-free ordering driven entirely by the unknown bucket.
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    encoders: dict[str, list[str]] = {}
    feature_dim = len(NUMERIC_COLS) + len(CATEGORICAL_COLS)
    enc_path = Path(encoders_path)
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            enc_blob = pickle.load(f)
        encoders = enc_blob["categorical"]
        feature_dim = enc_blob.get("feature_dim", feature_dim)

    model = SchoolLSTM(input_size=feature_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_dim": feature_dim,
    }


def _encode_categorical(value: Any, labels: list[str]) -> float:
    """Map a category to its StringIndexer-equivalent index.

    Matches StringIndexer(handleInvalid="keep"): unknown values land in the
    sentinel bucket at index len(labels).
    """
    if not labels:
        return 0.0
    s = "" if value is None else str(value)
    try:
        return float(labels.index(s))
    except ValueError:
        return float(len(labels))


def build_feature_vector(
    form: dict[str, Any],
    encoders: dict[str, list[str]],
    scaler: dict[str, Any],
) -> np.ndarray:
    """Construct a single feature row in training-assembler order."""
    x_mean = np.asarray(scaler["X_mean"], dtype=np.float32)

    numeric_vals = []
    for i, col in enumerate(NUMERIC_COLS):
        v = form.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            numeric_vals.append(float(x_mean[i]))
        else:
            numeric_vals.append(float(v))

    cat_vals = []
    for col in CATEGORICAL_COLS:
        labels = encoders.get(col, [])
        cat_vals.append(_encode_categorical(form.get(col), labels))

    return np.asarray(numeric_vals + cat_vals, dtype=np.float32)


def predict_one(form: dict[str, Any], artifacts: dict[str, Any]) -> float:
    """Run the model on a single user-supplied feature row.

    Sequence length = 1 (matches streaming.py). The form is treated as one
    timestep at the user-chosen year.
    """
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoders = artifacts["encoders"]

    raw = build_feature_vector(form, encoders, scaler)

    x_mean = np.asarray(scaler["X_mean"], dtype=np.float32)
    x_std = np.asarray(scaler["X_std"], dtype=np.float32)
    scaled = (raw - x_mean) / x_std

    x_tensor = torch.tensor(scaled, dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        pred_norm = model(x_tensor).squeeze().item()

    return float(pred_norm * scaler["y_std"] + scaler["y_mean"])
