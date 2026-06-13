from pyspark.sql import SparkSession
import time

spark = SparkSession.builder \
    .appName("Caching Demo") \
    .getOrCreate()

# Read partitioned parquet dataset
df = spark.read.parquet(
    "data/parquet/partitioned_creditcard"
)

print("\n===== WITHOUT CACHE =====")

start = time.time()

df.groupBy("Class").count().show()

df.groupBy("Class").avg("Amount").show()

df.filter("Class = 1").count()

without_cache = time.time() - start

print(f"Time without cache: {without_cache:.2f} seconds")


print("\n===== WITH CACHE =====")

df.cache()

# Materialize cache
df.count()

start = time.time()

df.groupBy("Class").count().show()

df.groupBy("Class").avg("Amount").show()

df.filter("Class = 1").count()

with_cache = time.time() - start

print(f"Time with cache: {with_cache:.2f} seconds")

print("\n===== PERFORMANCE IMPROVEMENT =====")

improvement = without_cache - with_cache

print(f"Improvement: {improvement:.2f} seconds")

df.unpersist()

spark.stop()