import torch
import torch.nn as nn
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, Imputer
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.functions import col

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
num_assembler = VectorAssembler(
    inputCols=[c+"_imp" for c in numeric_cols],
    outputCol="num_vec"
)
scaler = StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=True, withStd=True)
assembler = VectorAssembler(
    inputCols=["num_scaled"] + [c+"_idx" for c in categorical_cols],
    outputCol="features"
)

pipeline = Pipeline(stages=indexers + [imputer, num_assembler, scaler, assembler])
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

y_mean = y.mean()
y_std = y.std() + 1e-8
y_norm = (y - y_mean) / y_std

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)

class SchoolLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x) 
        return self.fc(hn[-1])

model = SchoolLSTM(input_size=feature_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 2000
print("running epoch")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = loss_fn(preds, y_tensor)
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 50 == 0:
        rmse_pp = loss.item() ** 0.5 * y_std
        print(f"Epoch {epoch+1}/{epochs} — Loss(norm): {loss.item():.4f} — RMSE(pp): {rmse_pp:.2f}")

torch.save(model.state_dict(), 'model.pth')
np.save('y_stats.npy', np.array([y_mean, y_std]))