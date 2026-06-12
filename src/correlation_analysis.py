from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create Spark Session
spark = SparkSession.builder \
    .appName("Correlation Analysis") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

# Select only Class and a few important features
selected_columns = [
    "V1", "V2", "V3", "V4", "V10",
    "V11", "V12", "V14", "V16", "V17",
    "Amount", "Class"
]

# Convert to Pandas
pdf = df.select(selected_columns).toPandas()

# Correlation matrix
corr = pdf.corr()

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Plot heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")

plt.show()

spark.stop()

print("Saved: outputs/correlation_heatmap.png")