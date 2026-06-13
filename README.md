# Spark Big Data Transaction Pipeline

## Overview

This project explores how Apache Spark can be used to build an end-to-end analytical pipeline on transactional data. Using a real-world credit card transaction dataset, the project combines Big Data processing concepts with fraud analytics to demonstrate how raw data can be transformed, optimized, analyzed, and used for predictive modeling.

What started as a fraud detection exercise gradually evolved into a Spark-based data pipeline that focuses not only on machine learning, but also on efficient data storage, query optimization, and performance improvements.

---

## Objectives

* Process transaction data using Apache Spark.
* Explore and analyze fraud patterns within the dataset.
* Build a baseline fraud detection model.
* Demonstrate Big Data concepts such as Parquet storage, partitioning, Spark SQL, and caching.
* Understand and document the practical challenges of developing Spark applications on a local environment.

---

## Dataset

The project uses the publicly available Credit Card Fraud Detection dataset.

**Dataset Characteristics:**

* Total Transactions: **284,807**
* Fraudulent Transactions: **492**
* Genuine Transactions: **284,315**
* Features:

  * `Time`
  * `Amount`
  * `V1` to `V28` (PCA-transformed features)
  * `Class` (0 = Genuine, 1 = Fraud)

---

## Project Journey

The project progressed through the following phases:

### Phase 1: Data Exploration and Fraud Analytics

* Exploratory Data Analysis (EDA)
* Fraud distribution analysis
* Time-based fraud analysis
* Correlation analysis
* Visualizations to understand transaction behavior

### Phase 2: Baseline Fraud Detection

* Data preparation using Spark
* Logistic Regression using Spark MLlib
* Model evaluation and interpretation of results

### Phase 3: Storage Optimization

* CSV ingestion using PySpark
* Conversion of raw CSV data into Parquet format
* Validation of transformed data

### Phase 4: Partitioned Data Organization

* Partitioned Parquet storage using the `Class` attribute
* Demonstration of partition pruning concepts
* Efficient access to fraud-specific records

### Phase 5: Analytical Querying

* Spark SQL using temporary views
* Aggregation and analytical queries on partitioned datasets
* Comparison of SQL-based and DataFrame-based approaches

### Phase 6: Performance Optimization

* Spark caching experiments
* Performance comparison with and without caching
* Demonstration of Spark's in-memory computation capabilities

---

## Architecture

```text
Credit Card Dataset (CSV)
            │
            ▼
     Spark Ingestion
            │
            ▼
    Exploratory Analysis
            │
            ▼
 Fraud Detection Modeling
            │
            ▼
    CSV to Parquet ETL
            │
            ▼
   Partitioned Storage
            │
            ▼
      Spark SQL
            │
            ▼
    Caching Experiments
            │
            ▼
 Fraud Analytics Insights
```

---

## Technologies Used

* Python
* Apache Spark (PySpark)
* Spark SQL
* Spark MLlib
* Parquet
* Pandas
* Matplotlib
* Scikit-learn

---

## Key Outcomes

* Developed an end-to-end Spark-based analytical pipeline on transaction data.
* Implemented a baseline fraud detection model using Logistic Regression.
* Converted raw datasets into Parquet format for optimized storage.
* Applied partitioning strategies to improve query efficiency.
* Performed analytical querying using Spark SQL.
* Demonstrated the impact of Spark caching through performance experiments.
* Gained practical experience with Spark execution concepts and filesystem integration.

---

## Challenges Encountered

One of the primary challenges involved configuring Spark on a Windows environment. Filesystem operations such as Parquet writes required additional Hadoop-related configuration, including setting up compatible binaries and environment variables.

Resolving these issues provided a deeper understanding of how Spark interacts with underlying storage systems.

---



## Future Scope

Possible extensions to this project include:

* Real-time fraud detection using streaming concepts
* Dashboard-based visualization of analytical results
* Deployment using containerized environments
* Integration with cloud storage backends

---

## Conclusion

This project demonstrates that building a Big Data application involves more than training a machine learning model. It highlights the complete journey of working with data using Apache Spark, from ingestion and exploration to storage optimization, analytical querying, and performance tuning.

The focus throughout the project has been to understand not only how these techniques are implemented, but also why they are important in practical data engineering and analytics workflows.
