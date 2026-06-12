from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Fraud Detection").getOrCreate()

df = spark.read.csv(
"data/creditcard.csv",
header=True,
inferSchema=True
)

print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

df.printSchema()

df.show(5)

spark.stop()