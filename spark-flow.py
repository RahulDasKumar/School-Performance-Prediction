import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import pandas as pd
from collections import Counter
spark = SparkSession.builder.appName("SchoolCrimePrediction").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
years = [ 2021, 2022, 2023,2024]  # 2019 skipped

numeric_cols = ["year", "ratio", "Violent Crime Total", "Property Crime Total",
                "Burglary", "Larceny-theft", "pct_notprof", "Motor vehicle theft"]
categorical_cols = ["grade", "subgroup", "county", "city", "subject"]

dfs = {}
for year in years:
    csv_path = f"training-data/{year}-data.csv"
    parquet_path = f"training-data/{year}-data.parquet"

    if not os.path.exists(parquet_path):
        print(f"Converting {year}-data.csv -> parquet (one-time)...")
        tmp = pd.read_csv(csv_path, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        if "Year" in tmp.columns:
            tmp = tmp.drop(columns=["Year"])

        for c in numeric_cols:
            if c in tmp.columns:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        for c in categorical_cols:
            if c in tmp.columns:
                tmp[c] = tmp[c].astype("string")
        for c in tmp.select_dtypes(include="object").columns:
            tmp[c] = tmp[c].astype("string")

        tmp["year"] = year
        tmp.to_parquet(parquet_path, index=False)

    df = spark.read.parquet(parquet_path)

    print(f"\nYear {year} — distinct Area Type values:")
    df.select("Area Type").distinct().show(truncate=False)

    df = df.filter(F.trim(col("Area Type")) == "Municipality")

    pdf = pd.read_parquet(parquet_path)
    data = pdf[pdf["Area Type"] == "Municipality"].dropna()
    print(f"Pandas rows: {len(data)}")
    print(f"Spark rows:  {df.count()}")

    dfs[year] = df
# create pandas equavalant

