"""One-shot script: persist categorical -> index mappings.

Replicates the StringIndexer ordering used in training.py without requiring
Spark in the dashboard environment. Reads the same training CSVs, applies
the same pct_ccr filter, then ranks each categorical column by descending
frequency (ties broken alphabetically ascending) — matching Spark's default
`stringOrderType="frequencyDesc"` behavior.

Caveat: the existing model.pth was trained on indexers fit during a previous
run, and that mapping was not persisted. The mapping produced here is stable
going forward, but does not retroactively fix the trained weights. A retrain
using these encoders is the proper fix.

Run:
    python build_encoders.py
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

from inference_lib import CATEGORICAL_COLS, NUMERIC_COLS, TARGET

YEARS = [2021, 2022, 2023]
DATA_DIR = "training-data"
OUT_PATH = "encoders.pkl"


def load_year(year: int) -> pd.DataFrame:
    csv_path = Path(DATA_DIR) / f"{year}-data.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    df["year"] = year
    df[TARGET] = df[TARGET].astype(str).map(lambda v: re.sub(r"<|>", "", v))
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def rank_labels(series: pd.Series) -> list[str]:
    """Mirror Spark's StringIndexer frequencyDesc ordering."""
    s = series.fillna("").astype(str)
    counts = s.value_counts()
    ordered = counts.reset_index()
    ordered.columns = ["label", "count"]
    ordered = ordered.sort_values(["count", "label"], ascending=[False, True])
    return ordered["label"].tolist()


def main() -> None:
    frames = [load_year(y) for y in YEARS]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows across {YEARS}")

    categorical: dict[str, list[str]] = {}
    for c in CATEGORICAL_COLS:
        labels = rank_labels(df[c])
        categorical[c] = labels
        print(f"  {c}: {len(labels)} labels (top: {labels[:3]})")

    feature_dim = len(NUMERIC_COLS) + len(CATEGORICAL_COLS)
    blob = {
        "categorical": categorical,
        "feature_dim": feature_dim,
        "years_fit_on": YEARS,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(blob, f)
    print(f"\nWrote {OUT_PATH} (feature_dim={feature_dim})")


if __name__ == "__main__":
    main()
