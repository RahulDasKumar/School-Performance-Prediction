from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
from pyspark.sql.types import StructType,StructField
from pyspark.sql import functions as sf
from pyspark.sql.functions import col,avg,round
spark = SparkSession.builder.appName("MyAppName").getOrCreate()

dataset = sns.load_dataset("titanic")

df = spark.createDataFrame(dataset)

df.createOrReplaceTempView("titanic")

print(df.columns)
df.show()


gender_survival = df.groupBy(["pclass","sex"]).agg({"fare":"avg"})
# how can we see the gender differences for each class
# which town had the most passengers
most_popular_towns = df.groupBy("embarked").agg({"embarked":"count"})

gender_survival = df.groupBy("sex").agg({"survived":"avg"}).withColumn("survival_rate",round(col("avg(survived)") * 100, 2))

# which deck had the highest avg fair

deck_ranking = df.groupBy("deck").agg({"fare":"avg"}).sort(sf.desc("avg(fare)"))


# rank passeger classes
passenger_rank = df.where(df.survived == 1).dropna(subset=['age']).groupBy("pclass").agg({"age":"avg"}).sort(sf.desc("avg(age)"))

passenger_rank.show()