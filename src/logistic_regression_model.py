from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Create Spark Session
spark = SparkSession.builder \
    .appName("Fraud Detection using Logistic Regression") \
    .getOrCreate()

# Load Dataset
df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

print("Dataset Loaded Successfully")

# Features (everything except Class)
feature_columns = df.columns[:-1]

# Convert features into a single vector
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

data = assembler.transform(df)

# Select only required columns
final_data = data.select("features", "Class")

# Rename Class → label
final_data = final_data.withColumnRenamed("Class", "label")

print("Feature Vector Created")

# Train-Test Split
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

print("Training Records:", train_data.count())
print("Testing Records:", test_data.count())

# Logistic Regression Model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=10
)

# Train Model
model = lr.fit(train_data)

print("Model Training Completed")

# Predictions
predictions = model.transform(test_data)

print("\n===== Sample Predictions =====")

predictions.select(
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)

# Evaluation Metrics
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

print("\n===== MODEL PERFORMANCE =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Confusion Matrix
print("\n===== CONFUSION MATRIX =====")

predictions.groupBy(
    "label",
    "prediction"
).count().orderBy(
    "label",
    "prediction"
).show()

spark.stop()