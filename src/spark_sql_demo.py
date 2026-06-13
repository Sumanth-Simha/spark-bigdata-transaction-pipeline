from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Spark SQL Demo") \
    .getOrCreate()

# Read partitioned parquet
df = spark.read.parquet(
    "data/parquet/partitioned_creditcard"
)

# Create SQL view
df.createOrReplaceTempView("transactions")

print("\nFraud Distribution:")
spark.sql("""
    SELECT Class,
           COUNT(*) AS total_transactions
    FROM transactions
    GROUP BY Class
""").show()

print("\nFraud Amount Statistics:")
spark.sql("""
    SELECT Class,
           ROUND(AVG(Amount), 2) AS avg_amount,
           ROUND(MAX(Amount), 2) AS max_amount,
           ROUND(MIN(Amount), 2) AS min_amount
    FROM transactions
    GROUP BY Class
""").show()

print("\nTop 5 Highest Transaction Amounts:")
spark.sql("""
    SELECT Time,
           Amount,
           Class
    FROM transactions
    ORDER BY Amount DESC
    LIMIT 5
""").show()

spark.stop()