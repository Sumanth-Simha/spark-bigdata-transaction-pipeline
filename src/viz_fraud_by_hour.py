from pyspark.sql import SparkSession
from pyspark.sql.functions import floor, col, count
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("Fraud By Hour Visualization") \
    .getOrCreate()

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

hourly_fraud = df.withColumn(
    "Hour",
    floor(col("Time") / 3600)
).filter(
    col("Class") == 1
).groupBy(
    "Hour"
).agg(
    count("*").alias("Fraud_Count")
).orderBy(
    "Hour"
).collect()

hours = [row["Hour"] for row in hourly_fraud]
counts = [row["Fraud_Count"] for row in hourly_fraud]

plt.figure(figsize=(12, 6))
plt.bar(hours, counts)

plt.title("Fraud Transactions by Hour")
plt.xlabel("Hour")
plt.ylabel("Fraud Count")

plt.savefig("outputs/fraud_by_hour.png")
plt.show()

spark.stop()

print("Saved: outputs/fraud_by_hour.png")