from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import os

spark = SparkSession.builder \
    .appName("Class Distribution Visualization") \
    .getOrCreate()

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

legitimate = df.filter(col("Class") == 0).count()
fraud = df.filter(col("Class") == 1).count()

classes = ["Legitimate", "Fraud"]
counts = [legitimate, fraud]

plt.figure(figsize=(6, 5))
plt.bar(classes, counts)

plt.title("Transaction Class Distribution")
plt.ylabel("Number of Transactions")

plt.savefig("outputs/class_distribution.png")
plt.show()

spark.stop()

print("Saved: outputs/class_distribution.png")