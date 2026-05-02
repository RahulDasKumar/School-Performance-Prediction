import torch
import torch.nn as nn
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

spark = (SparkSession.builder
         .appName("SchoolCrimePrediction")
         .config("spark.driver.memory", "8g")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

numeric_cols = ["year", "ratio", "Violent Crime Total", "Property Crime Total",
                "Burglary", "Larceny-theft", "pct_notprof", "Motor vehicle theft"]
categorical_cols = ["grade", "subgroup", "county", "city", "subject"]
target = "pct_ccr"

years = list(range(2021, 2024))
dfs = [
    spark.read.parquet(f"training-data/{year}-data.parquet")
         .withColumn("year", F.lit(year))
    for year in years if year != 2019
]
df = reduce(lambda a, b: a.union(b), dfs)
df = df.withColumn(target, F.regexp_replace(col(target), r"<|>", "").cast("float"))
df = df.filter(col(target).isNotNull())

for c in numeric_cols:
    df = df.withColumn(c, col(c).cast("float"))

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
imputer = Imputer(inputCols=numeric_cols, outputCols=[c+"_imp" for c in numeric_cols])
assembler = VectorAssembler(
    inputCols=[c+"_imp" for c in numeric_cols] + [c+"_idx" for c in categorical_cols],
    outputCol="features"
)

pipeline = Pipeline(stages=indexers + [imputer, assembler])
prepped = pipeline.fit(df).transform(df)
prepped = prepped.withColumn("features", vector_to_array("features"))

pdf = prepped.select("features", target, "year", "school_code").toPandas()
pdf["features"] = pdf["features"].apply(np.asarray)
pdf = pdf.sort_values(["school_code", "year"])
feature_dim = len(pdf["features"].iloc[0])

def pad_sequence(group, all_years, feature_dim):
    year_map = dict(zip(group["year"], group["features"]))
    return np.stack([year_map.get(y, np.zeros(feature_dim)) for y in all_years])

grouped = list(pdf.groupby("school_code"))

X = np.stack([pad_sequence(g, years, feature_dim) for _, g in grouped])
y = np.array([g[target].values[-1] for _, g in grouped], dtype=np.float32)


X_flat = X.reshape(-1, feature_dim)
X_mean = X_flat.mean(axis=0)
X_std  = X_flat.std(axis=0) + 1e-8     
X_scaled = ((X_flat - X_mean) / X_std).reshape(X.shape)


y_mean = y.mean()
y_std  = y.std() + 1e-8
y_norm = (y - y_mean) / y_std

with open("scaler_stats.pkl", "wb") as f:
    pickle.dump({"X_mean": X_mean, "X_std": X_std,
                 "y_mean": float(y_mean), "y_std": float(y_std)}, f)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_norm,   dtype=torch.float32).unsqueeze(1)

class SchoolLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model    = SchoolLSTM(input_size=feature_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn  = nn.MSELoss()

epochs = 2000
loss_values = []
print("running epochs")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss  = loss_fn(preds, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        loss_values.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")

x_values = [i * 100 + 1 for i in range(len(loss_values))]
plt.plot(x_values, loss_values)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (normalized)")
plt.title("Training Loss")
plt.savefig("test_results.png")