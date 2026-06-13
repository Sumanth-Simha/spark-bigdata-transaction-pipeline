# Phase 3: Spark SQL Analytics

## Objective

The objective of this phase was to perform analytical queries on the processed transaction dataset using Spark SQL and demonstrate how SQL can be integrated with Spark's distributed execution engine.

---

## Implementation

The partitioned Parquet dataset was loaded into Spark and converted into a temporary SQL view.

```python
df.createOrReplaceTempView("transactions")
```

SQL queries were then executed using Spark SQL.

---

## Analytical Queries Performed

### 1. Fraud Distribution

Calculated the number of transactions in each class.

```sql
SELECT Class,
       COUNT(*) AS total_transactions
FROM transactions
GROUP BY Class;
```

Result:

| Class | Transactions |
| ----- | -----------: |
| 0     |      284,315 |
| 1     |          492 |

---

### 2. Transaction Amount Statistics

Compared transaction amounts between fraud and non-fraud classes.

```sql
SELECT Class,
       AVG(Amount),
       MAX(Amount),
       MIN(Amount)
FROM transactions
GROUP BY Class;
```

This helped understand differences in transaction behavior.

---

### 3. Highest Value Transactions

Identified the top five transactions by amount.

```sql
SELECT Time,
       Amount,
       Class
FROM transactions
ORDER BY Amount DESC
LIMIT 5;
```

---

## Key Concepts Learned

* Spark SQL enables SQL-based analytics on distributed datasets.
* Temporary Views allow SQL queries on DataFrames.
* SQL and DataFrame APIs can be used interchangeably.
* Spark SQL combines the familiarity of SQL with Spark's execution capabilities.

---

## Outcome

This phase demonstrated how structured analytical queries can be executed efficiently using Spark SQL while leveraging the underlying distributed processing engine.

**Status:** Completed ✅



# Phase 4: Caching and Performance Optimization

## Objective

The objective of this phase was to demonstrate Spark's in-memory processing capabilities by comparing execution times with and without caching.

---

## Implementation

The partitioned Parquet dataset was queried multiple times.

First, the queries were executed normally.

Then, the dataset was cached using:

```python
df.cache()
```

The cache was materialized using:

```python
df.count()
```

The same queries were executed again, and execution times were recorded.

---

## Queries Executed

* Transaction count by class
* Average transaction amount by class
* Fraud transaction count

---

## Performance Results

| Scenario      | Execution Time |
| ------------- | -------------: |
| Without Cache |   5.03 seconds |
| With Cache    |   1.11 seconds |

### Improvement

Execution time improved by approximately:

```text
77.9%
```

---

## Why Did Performance Improve?

Without caching:

```text
Read Data
↓
Compute Query
↓
Recompute for Next Query
```

With caching:

```text
Read Data Once
↓
Store in Memory
↓
Reuse Cached Data
```

Spark avoided repeated recomputation of the same dataset.

---

## Key Concepts Learned

* Spark uses lazy evaluation and recomputes lineage when required.
* Caching stores frequently accessed datasets in memory.
* Reusing cached datasets significantly reduces execution time.
* In-memory computation is one of Spark's major advantages over traditional disk-based approaches.

---

## Outcome

This phase validated Spark's performance optimization capabilities through an experimental comparison, demonstrating measurable improvements in repeated analytical workloads.

**Status:** Completed ✅
