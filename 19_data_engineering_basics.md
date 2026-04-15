[← Back to Table of Contents](./README.md)

# Chapter 19 — Data Engineering Basics

> "Data is the new oil — but like oil, it's useless if unrefined." — Clive Humby (adapted)

ML models are only as good as the data they're trained on. Data engineering — the discipline of building reliable pipelines that collect, clean, validate, and serve data — is the foundation of every successful ML system. This chapter covers the essential data engineering skills every ML engineer needs.

---

## 19.1 The ML Data Pipeline

<div class="diagram">
  <div class="diagram-title">End-to-End ML Data Pipeline</div>
  <div class="flow">
    <div class="flow-node cyan">Data Sources (APIs, DBs, Logs, Streams)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Ingestion (Extract)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node blue">Validation (Quality Gates)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">Transformation (Clean, Featurize)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Storage (Data Lake / Warehouse)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">Serving (Feature Store, Training Sets)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node teal">Monitoring (Quality, Freshness, Drift)</div>
  </div>
</div>

Each stage has distinct failure modes:

| Stage | Common Failures | Mitigation |
|---|---|---|
| Ingestion | API rate limits, schema changes | Retries, schema contracts |
| Validation | Null values, out-of-range data | Great Expectations, Pandera |
| Transformation | Incorrect joins, data leakage | Unit tests, train/test split audits |
| Storage | Corruption, inconsistent formats | Checksums, schema enforcement |
| Serving | Training/serving skew | Feature stores, shared transforms |
| Monitoring | Silent degradation | Automated quality checks, alerts |

---

## 19.2 ETL vs ELT

| Aspect | ETL | ELT |
|---|---|---|
| Full name | Extract, Transform, Load | Extract, Load, Transform |
| Transform location | Before loading (pipeline) | After loading (in warehouse) |
| Best for | Structured, known schemas | Exploratory, schema-on-read |
| Tools | Airflow + custom code | dbt + BigQuery/Snowflake |
| ML use case | Feature pipelines with fixed schema | Ad-hoc analysis, new feature exploration |
| Data freshness | Batch (hourly/daily) | Near-real-time possible |

```python
# ETL Example: Extract from API, transform, load to Parquet
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

def extract_transactions(api_url: str, date: str) -> list[dict]:
    response = requests.get(f"{api_url}/transactions", params={"date": date})
    response.raise_for_status()
    return response.json()["transactions"]

def transform_transactions(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["amount", "user_id"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    return df

def load_to_parquet(df: pd.DataFrame, output_dir: str, date: str) -> Path:
    path = Path(output_dir) / f"transactions_{date}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    return path

# Run ETL
raw = extract_transactions("https://api.example.com", "2025-01-15")
clean = transform_transactions(raw)
output = load_to_parquet(clean, "data/processed", "2025-01-15")
```

```sql
-- ELT Example: Transform in the warehouse using dbt
-- models/features/user_transaction_features.sql

WITH daily_stats AS (
    SELECT
        user_id,
        DATE(timestamp) AS txn_date,
        COUNT(*) AS txn_count,
        SUM(amount) AS total_amount,
        AVG(amount) AS avg_amount,
        MAX(amount) AS max_amount
    FROM {{ ref('raw_transactions') }}
    GROUP BY user_id, DATE(timestamp)
)

SELECT
    user_id,
    txn_date,
    txn_count,
    total_amount,
    avg_amount,
    max_amount,
    -- Rolling features
    AVG(txn_count) OVER (
        PARTITION BY user_id ORDER BY txn_date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS avg_daily_txns_7d,
    AVG(total_amount) OVER (
        PARTITION BY user_id ORDER BY txn_date
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) AS avg_daily_spend_30d
FROM daily_stats
```

---

## 19.3 Data Validation

### Great Expectations

```python
import great_expectations as gx

context = gx.get_context()

# Define expectations for training data
validator = context.sources.pandas_default.read_dataframe(training_df)

validator.expect_column_to_exist("user_id")
validator.expect_column_to_exist("label")
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=100_000)
validator.expect_column_values_to_be_in_set("label", [0, 1])
validator.expect_column_mean_to_be_between("amount", min_value=10, max_value=500)
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=10_000_000)

results = validator.validate()
if not results.success:
    failed = [r for r in results.results if not r.success]
    raise ValueError(f"Data validation failed: {len(failed)} checks failed")
```

### Pandera — Schema Validation for DataFrames

```python
import pandera as pa
from pandera.typing import DataFrame, Series
import pandas as pd

class TrainingDataSchema(pa.DataFrameModel):
    """Schema for fraud detection training data."""
    user_id: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    amount: Series[float] = pa.Field(ge=0, le=100_000)
    hour_of_day: Series[int] = pa.Field(ge=0, le=23)
    is_weekend: Series[bool]
    label: Series[int] = pa.Field(isin=[0, 1])
    transaction_count_7d: Series[int] = pa.Field(ge=0)

    class Config:
        strict = True  # No extra columns allowed
        coerce = True  # Auto-coerce types

    @pa.check("amount")
    def amount_distribution_check(cls, series: Series[float]) -> bool:
        """Ensure amount distribution hasn't shifted dramatically."""
        return 10 < series.mean() < 500

@pa.check_types
def prepare_training_data(raw_df: pd.DataFrame) -> DataFrame[TrainingDataSchema]:
    """Pandera validates the output automatically."""
    processed = feature_pipeline(raw_df)
    return processed  # Raises SchemaError if validation fails
```

---

## 19.4 Data Versioning with DVC

```bash
# Initialize DVC in your ML project
pip install dvc dvc-s3
dvc init

# Track a large dataset
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data v1"

# Configure remote storage
dvc remote add -d myremote s3://ml-data-bucket/dvc-store
dvc push

# Reproduce a specific data version
git checkout v2.0  # checkout the code version
dvc checkout      # DVC restores the matching data version
```

```yaml
# dvc.yaml — Define reproducible pipeline stages
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/transactions.csv
    outs:
      - data/processed/features.parquet
    params:
      - preprocess.min_amount
      - preprocess.max_date_range

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/features.parquet
    outs:
      - models/fraud_detector.pkl
    params:
      - train.learning_rate
      - train.n_estimators
    metrics:
      - metrics/train_metrics.json:
          cache: false
```

```bash
# Reproduce the full pipeline
dvc repro

# Compare metrics across experiments
dvc metrics diff

# Show pipeline DAG
dvc dag
```

---

## 19.5 Data Storage Patterns

<div class="diagram">
  <div class="diagram-title">Data Storage Architecture for ML</div>
  <div class="flow">
    <div class="flow-node cyan">Raw Data (Data Lake — S3, GCS)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Cleaned Data (Data Warehouse — BigQuery, Snowflake)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node blue">Features (Feature Store — Feast, Tecton)</div>
    <div class="flow-arrow">→ offline</div>
    <div class="flow-node purple">Training Sets (Parquet on S3)</div>
  </div>
  <div class="flow" style="margin-top: 8px;">
    <div class="flow-node blue">Features (Feature Store)</div>
    <div class="flow-arrow">→ online</div>
    <div class="flow-node orange">Low-Latency Store (Redis, DynamoDB)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">Model Serving</div>
  </div>
</div>

| Pattern | Use Case | Characteristics |
|---|---|---|
| Data Lake | Raw data archive | Schema-on-read, cheap storage, S3/GCS |
| Data Warehouse | Cleaned, structured analytics | Schema-on-write, SQL queries, BigQuery |
| Feature Store | ML-specific features | Online + offline, point-in-time correctness |
| Data Lakehouse | Unified lake + warehouse | Delta Lake, Iceberg, Hudi |

---

## 19.6 File Formats for ML

| Format | Best For | Compression | Column Pruning | Language Support |
|---|---|---|---|---|
| Parquet | Tabular ML data | Excellent (Snappy, Zstd) | ✅ Yes | Python, Java, Spark |
| Arrow (Feather) | In-memory interchange | Good | ✅ Yes | Python, R, Julia |
| HDF5 | Large numerical arrays | Good | Partial | Python, C++, Java |
| TFRecord | TensorFlow pipelines | Good | ❌ No | Python (TF only) |
| CSV | Quick prototyping | ❌ None | ❌ No | Universal |
| JSON Lines | Semi-structured logs | ❌ None | ❌ No | Universal |

```python
# Parquet: the go-to format for tabular ML data
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Write with compression and partitioning
df = pd.DataFrame({
    "user_id": ["u1", "u2", "u3"],
    "features": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    "label": [0, 1, 0],
    "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
})

table = pa.Table.from_pandas(df)
pq.write_to_dataset(
    table,
    root_path="data/training",
    partition_cols=["date"],
    compression="snappy",
)

# Read only the columns you need — huge performance win on wide tables
df_subset = pd.read_parquet(
    "data/training",
    columns=["user_id", "label"],  # Only reads these columns from disk
    filters=[("date", "=", "2025-01-01")],  # Partition pruning
)
```

```python
# Arrow for zero-copy data sharing between processes
import pyarrow as pa
import pyarrow.ipc as ipc

# Write to memory-mapped file (zero-copy read)
table = pa.table({"features": pa.array([1.0, 2.0, 3.0]), "label": pa.array([0, 1, 0])})
writer = ipc.new_file("data/features.arrow", table.schema)
writer.write_table(table)
writer.close()

# Read with memory mapping — no deserialization overhead
source = pa.memory_map("data/features.arrow", "r")
reader = ipc.open_file(source)
loaded_table = reader.read_all()  # Virtually instant for large files
```

---

## 19.7 Streaming vs Batch Processing

| Aspect | Batch | Streaming |
|---|---|---|
| Latency | Minutes to hours | Seconds to minutes |
| Throughput | Very high | Moderate |
| Complexity | Lower | Higher |
| Tools | Spark, Airflow, dbt | Kafka, Flink, Spark Streaming |
| ML use case | Daily retraining, batch features | Real-time features, online learning |
| Cost | Lower (spot instances) | Higher (always-on infra) |

```python
# Batch processing with PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

# Read partitioned Parquet data
transactions = spark.read.parquet("s3://data-lake/transactions/")

# Compute aggregated features
user_features = (
    transactions
    .groupBy("user_id")
    .agg(
        F.count("*").alias("total_transactions"),
        F.avg("amount").alias("avg_amount"),
        F.stddev("amount").alias("std_amount"),
        F.max("amount").alias("max_amount"),
        F.countDistinct("merchant_id").alias("unique_merchants"),
    )
)

# Write features to feature store's offline layer
user_features.write.mode("overwrite").parquet("s3://feature-store/user_features/")
```

```python
# Stream processing: real-time feature updates
from kafka import KafkaConsumer
import json
import redis

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers=["kafka:9092"],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)
cache = redis.Redis(host="redis", port=6379)

for message in consumer:
    txn = message.value
    user_id = txn["user_id"]

    # Update running statistics in Redis
    pipe = cache.pipeline()
    pipe.incr(f"user:{user_id}:txn_count")
    pipe.incrbyfloat(f"user:{user_id}:total_amount", txn["amount"])
    pipe.lpush(f"user:{user_id}:recent_amounts", txn["amount"])
    pipe.ltrim(f"user:{user_id}:recent_amounts", 0, 99)
    pipe.execute()
```

---

## 19.8 Data Quality Monitoring

```python
# Automated data quality monitoring
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    timestamp: datetime
    row_count: int
    null_rates: dict[str, float]
    numeric_stats: dict[str, dict]
    alerts: list[str]

def monitor_data_quality(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    null_threshold: float = 0.05,
    drift_threshold: float = 2.0,
) -> DataQualityReport:
    alerts = []

    # Check for unusual null rates
    null_rates = current.isnull().mean().to_dict()
    for col, rate in null_rates.items():
        if rate > null_threshold:
            alerts.append(f"High null rate in '{col}': {rate:.2%}")

    # Check for distribution drift (z-score of mean shift)
    numeric_stats = {}
    for col in current.select_dtypes(include=[np.number]).columns:
        ref_mean = reference[col].mean()
        ref_std = reference[col].std()
        curr_mean = current[col].mean()

        if ref_std > 0:
            z_score = abs(curr_mean - ref_mean) / ref_std
            if z_score > drift_threshold:
                alerts.append(
                    f"Distribution drift in '{col}': "
                    f"z-score={z_score:.2f} (mean {ref_mean:.2f} → {curr_mean:.2f})"
                )

        numeric_stats[col] = {
            "mean": float(curr_mean),
            "std": float(current[col].std()),
            "min": float(current[col].min()),
            "max": float(current[col].max()),
        }

    # Check row count
    expected_rows = len(reference)
    actual_rows = len(current)
    if actual_rows < expected_rows * 0.5:
        alerts.append(f"Row count dropped: {expected_rows} → {actual_rows}")

    report = DataQualityReport(
        timestamp=datetime.now(),
        row_count=actual_rows,
        null_rates=null_rates,
        numeric_stats=numeric_stats,
        alerts=alerts,
    )

    for alert in alerts:
        logger.warning(f"DATA QUALITY ALERT: {alert}")

    return report
```

---

## 19.9 Summary

> **The data engineering hierarchy of needs:** Before investing in fancy model architectures, ensure your data pipeline is reliable, validated, and reproducible. A simple model on clean data will outperform a complex model on messy data every time.

| Practice | Tool | Priority |
|---|---|---|
| Data validation | Great Expectations, Pandera | Critical |
| Data versioning | DVC | High |
| Column-oriented storage | Parquet, Arrow | High |
| Pipeline orchestration | Airflow, Prefect | High |
| Feature stores | Feast, Tecton | Medium (at scale) |
| Stream processing | Kafka, Flink | Medium (if real-time needed) |
| Data quality monitoring | Custom + Evidently | High |
| Data catalogs | DataHub, Amundsen | Medium (at scale) |

---

*Last updated: April 2026*
