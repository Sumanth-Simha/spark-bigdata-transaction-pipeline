# Phase 1: CSV to Parquet Conversion

## Objective

The objective of this phase was to transform the raw transaction dataset from CSV format into Parquet format using Apache Spark. This step represents the beginning of the ETL layer of the Big Data pipeline and prepares the data for efficient analytical workloads.

---

## Why Parquet?

The original dataset was stored as a CSV file, which is simple and human-readable but inefficient for large-scale analytics.

Parquet was chosen because it provides:

* **Columnar Storage:** Stores data by columns instead of rows, allowing analytical queries to read only the required columns.
* **Compression:** Reduces storage requirements compared to plain-text CSV files.
* **Improved Query Performance:** Minimizes I/O during analysis.
* **Predicate Pushdown Support:** Enables Spark to skip irrelevant data during filtering operations.
* **Industry Adoption:** Commonly used in modern data engineering and Big Data ecosystems.

---

## Implementation

Using PySpark, the following steps were performed:

1. Loaded the raw credit card transaction dataset from CSV.
2. Inferred the schema and validated ingestion.
3. Converted the dataset into Parquet format.
4. Stored the generated Parquet files locally.
5. Reloaded the Parquet dataset.
6. Verified the integrity of the transformation.

### Pipeline Flow

Raw CSV Dataset

↓

Spark Ingestion

↓

CSV to Parquet Transformation

↓

Parquet Storage

↓

Validation

---

## Validation

Data integrity was verified by comparing record counts before and after conversion.

| Dataset         | Record Count |
| --------------- | -----------: |
| CSV Dataset     |      284,807 |
| Parquet Dataset |      284,807 |

Since both counts matched, the conversion was considered successful.

---

## Challenges Encountered

Development was performed on a Windows environment using Apache Spark 4.0.0.

Several filesystem-related issues were encountered during Parquet writes, including:

* Missing `HADOOP_HOME` configuration.
* Missing `winutils.exe`.
* Missing `hadoop.dll`.
* Native Hadoop library compatibility issues.

These challenges were resolved by configuring the required Hadoop binaries and environment variables compatible with the Spark installation.

---

## Outcome

This phase transformed the project from a simple fraud detection exercise into the initial stage of a Big Data processing pipeline.

Key outcomes include:

* Successful implementation of Spark-based ETL processing.
* Understanding of the advantages of Parquet over CSV.
* Practical experience with Spark filesystem dependencies.
* Validation techniques to ensure transformation correctness.
* Preparation for subsequent optimization phases.

---

## Deliverables

* CSV ingestion using PySpark.
* Parquet dataset generation.
* Parquet dataset validation.
* Documentation of implementation decisions and challenges.

---

## Phase Status

**Status:** Completed ✅

This phase establishes the foundation for the remaining Big Data pipeline components, including partitioning, Spark SQL, caching experiments, and analytical workloads.
