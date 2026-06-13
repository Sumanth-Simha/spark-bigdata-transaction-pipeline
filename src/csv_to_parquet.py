from pyspark.sql import SparkSession

# Create Spark Session
spark = SparkSession.builder \
    .appName("CSV to Parquet Conversion") \
    .getOrCreate()

# Read CSV
df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

# Write as Parquet
df.write.mode("overwrite").parquet("data/parquet/creditcard")

# Read back the Parquet file
parquet_df = spark.read.parquet("data/parquet/creditcard")

# Verify row counts
print("CSV rows:", df.count())
print("Parquet rows:", parquet_df.count())

# Stop Spark
spark.stop()