from pyspark.sql import SparkSession
from pyspark.sql.functions import floor, col, count

spark = SparkSession.builder \
    .appName("Time Analysis") \
    .getOrCreate()

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

# Convert seconds to hours
df = df.withColumn("Hour", floor(col("Time") / 3600))

print("\n===== FRAUDS BY HOUR =====")

df.filter(col("Class") == 1) \
    .groupBy("Hour") \
    .agg(count("*").alias("Fraud_Count")) \
    .orderBy("Hour") \
    .show(50, False)

spark.stop()