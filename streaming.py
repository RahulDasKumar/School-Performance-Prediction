import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH = "model.pth"
SCALER_PATH = "scaler_stats.pkl"
INFERENCE_YEARS = [2024]
DATA_DIR = "training-data"

numeric_cols = ["year", "ratio", "Violent Crime Total", "Property Crime Total",
                "Burglary", "Larceny-theft", "pct_notprof", "Motor vehicle theft"]
categorical_cols = ["grade", "subgroup", "county", "city", "subject"]
target = "pct_ccr"

GROUP_KEYS = ["school_code", "grade", "subgroup"]

class SchoolLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def build_features(years=INFERENCE_YEARS):
    spark = (SparkSession.builder
             .appName("SchoolCrimeInference")
             .config("spark.driver.memory", "8g")
             .config("spark.driver.maxResultSize", "4g")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

    dfs = [
        spark.read.parquet(f"{DATA_DIR}/{year}-data.parquet")
             .withColumn("year", F.lit(year))
        for year in years
    ]
    df = reduce(lambda a, b: a.union(b), dfs)
    df = df.withColumn(target, F.regexp_replace(col(target), r"<|>", "").cast("float"))
    df = df.filter(col(target).isNotNull())

    for c in numeric_cols:
        df = df.withColumn(c, col(c).cast("float"))

    indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
                for c in categorical_cols]
    imputer  = Imputer(inputCols=numeric_cols,
                       outputCols=[c+"_imp" for c in numeric_cols])
    assembler = VectorAssembler(
        inputCols=[c+"_imp" for c in numeric_cols] + [c+"_idx" for c in categorical_cols],
        outputCol="features"
    )

    pipeline = Pipeline(stages=indexers + [imputer, assembler])
    prepped  = pipeline.fit(df).transform(df)
    prepped  = prepped.withColumn("features", vector_to_array("features"))

    select_cols = ["features", target, "year", "school_code"] + categorical_cols
    pdf = prepped.select(*select_cols).toPandas()
    pdf["features"] = pdf["features"].apply(np.asarray)
    pdf = pdf.sort_values(["school_code", "year"])
    return pdf

def pad_sequence(group, all_years, feature_dim):
    year_map = dict(zip(group["year"], group["features"]))
    return np.stack([year_map.get(y, np.zeros(feature_dim)) for y in all_years])

def run_inference():
    print("Building features...")
    pdf = build_features()

    feature_dim = len(pdf["features"].iloc[0])

    grouped      = list(pdf.groupby(GROUP_KEYS))
    group_keys   = [keys for keys, _ in grouped]

    X = np.stack([pad_sequence(g, INFERENCE_YEARS, feature_dim) for _, g in grouped])

    actuals = np.array([g[target].values[-1] for _, g in grouped], dtype=np.float32)

    print(f"Loading scaler stats from {SCALER_PATH}...")
    with open(SCALER_PATH, "rb") as f:
        stats = pickle.load(f)
    X_mean = stats["X_mean"]
    X_std  = stats["X_std"]
    y_mean = stats["y_mean"]
    y_std  = stats["y_std"]

    X_flat   = X.reshape(-1, feature_dim)
    X_scaled = ((X_flat - X_mean) / X_std).reshape(X.shape)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    print(f"Loading model from {MODEL_PATH}...")
    model = SchoolLSTM(input_size=feature_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds_norm = model(X_tensor).squeeze(1).numpy()

    preds = preds_norm * y_std + y_mean

    keys_df = pd.DataFrame(group_keys, columns=GROUP_KEYS)
    results = keys_df.copy()
    results["actual_pct_ccr"]    = actuals
    results["predicted_pct_ccr"] = preds
    results["residual"]          = results["actual_pct_ccr"] - results["predicted_pct_ccr"]

    out_path = "inference_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} predictions → {out_path}")
    print("\nSample results:")
    print(results.head(20).to_string(index=False))

    mae  = np.abs(results["residual"]).mean()
    rmse = np.sqrt((results["residual"] ** 2).mean())
    print(f"\nMAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return results

if __name__ == "__main__":
    run_inference()