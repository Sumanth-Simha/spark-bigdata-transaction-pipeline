from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, col

spark = SparkSession.builder \
    .appName("Fraud Analysis") \
    .getOrCreate()

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

print("\n===== AMOUNT COMPARISON =====")

df.groupBy("Class").agg(
    avg("Amount").alias("Average Amount"),
    max("Amount").alias("Maximum Amount")
).show()

spark.stop()
