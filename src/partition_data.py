from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Partitioning Demo") \
    .getOrCreate()

# Read Parquet
df = spark.read.parquet("data/parquet/creditcard")

# Check distribution
df.groupBy("Class").count().show()

# Write partitioned parquet
df.write \
    .mode("overwrite") \
    .partitionBy("Class") \
    .parquet("data/parquet/partitioned_creditcard")

print("Partitioning complete!")

spark.stop()