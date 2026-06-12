from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, max

spark = SparkSession.builder \
    .appName("Fraud EDA") \
    .getOrCreate()

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

print("\n===== BASIC INFORMATION =====")

total = df.count()

fraud = df.filter(col("Class") == 1).count()

normal = df.filter(col("Class") == 0).count()

print("Total Transactions:", total)
print("Legitimate Transactions:", normal)
print("Fraud Transactions:", fraud)

print("\n===== FRAUD PERCENTAGE =====")

fraud_percentage = (fraud / total) * 100

print(f"Fraud Percentage: {fraud_percentage:.4f}%")

print("\n===== AMOUNT ANALYSIS =====")

df.select(
    avg("Amount").alias("Average Amount"),
    max("Amount").alias("Maximum Amount")
).show()

spark.stop()