from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("Amount Visualization") \
    .getOrCreate()

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

results = df.groupBy("Class") \
    .agg(avg("Amount").alias("avg_amount")) \
    .collect()

amounts = {}

for row in results:
    amounts[row["Class"]] = row["avg_amount"]

labels = ["Legitimate", "Fraud"]
values = [amounts[0], amounts[1]]

plt.figure(figsize=(6, 5))
plt.bar(labels, values)

plt.title("Average Transaction Amount by Class")
plt.ylabel("Average Amount ($)")

plt.savefig("outputs/amount_comparison.png")
plt.show()

spark.stop()

print("Saved: outputs/amount_comparison.png")