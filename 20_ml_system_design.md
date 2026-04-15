[← Back to Table of Contents](./README.md)

# Chapter 20 — ML System Design

> "The hard part of ML isn't the model — it's everything around the model." — D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems"

Building ML systems that work reliably in production requires far more than training a good model. This chapter covers the end-to-end design of ML systems — from experiment tracking through deployment, monitoring, and the organizational practices that make it all work at scale.

---

## 20.1 The ML System Landscape

<div class="diagram">
  <div class="diagram-title">Comprehensive ML System Architecture</div>
  <div class="flow">
    <div class="flow-node cyan">Data Sources</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Data Pipeline (Ingestion, Validation)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node blue">Feature Store (Offline + Online)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">Training Pipeline (Experiments, HPO)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Model Registry (Versioning, Approval)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">Serving Infrastructure (API, Batch)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node teal">Monitoring (Drift, Performance, Alerts)</div>
  </div>
</div>

> **Key insight:** The model code is typically less than 5% of a production ML system. The rest is data pipelines, feature engineering, serving infrastructure, monitoring, and operational tooling.

---

## 20.2 Experiment Tracking

Experiment tracking records every training run's parameters, metrics, artifacts, and environment — enabling reproducibility and systematic comparison.

### Weights & Biases (W&B)

```python
import wandb
import torch

wandb.init(
    project="fraud-detection",
    config={
        "learning_rate": 1e-3,
        "batch_size": 256,
        "architecture": "transformer-small",
        "dataset_version": "v2.3",
        "epochs": 50,
    },
)

for epoch in range(wandb.config.epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_metrics = evaluate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/accuracy": val_metrics["accuracy"],
        "val/precision": val_metrics["precision"],
        "val/recall": val_metrics["recall"],
        "val/auc_roc": val_metrics["auc_roc"],
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    if val_metrics["auc_roc"] > best_auc:
        best_auc = val_metrics["auc_roc"]
        torch.save(model.state_dict(), "best_model.pt")
        wandb.save("best_model.pt")

wandb.finish()
```

### MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="gbm-baseline"):
    # Log parameters
    params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
    mlflow.log_params(params)

    # Train
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate and log metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
    })

    # Log model artifact
    mlflow.sklearn.log_model(model, "model")

    # Log training data hash for reproducibility
    mlflow.log_param("data_hash", hashlib.md5(X_train.tobytes()).hexdigest())
```

| Feature | W&B | MLflow |
|---|---|---|
| Hosting | Cloud (free tier) or self-hosted | Self-hosted or Databricks |
| Visualization | Excellent built-in dashboards | Good, extensible |
| Collaboration | Team features, reports | Basic |
| Model registry | ✅ Yes | ✅ Yes |
| Artifact tracking | ✅ Yes | ✅ Yes |
| Cost | Free for individuals | Open source |

---

## 20.3 Model Registry Patterns

A model registry is a centralized store for trained models with versioning, metadata, and promotion workflows.

```python
# MLflow Model Registry workflow
import mlflow

# Register a model from an experiment run
model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(model_uri, "fraud-detector")

# Transition through stages
client = mlflow.tracking.MlflowClient()

# Stage 1: Staging — for integration testing
client.transition_model_version_stage(
    name="fraud-detector",
    version=model_version.version,
    stage="Staging",
)

# Stage 2: Production — after validation
client.transition_model_version_stage(
    name="fraud-detector",
    version=model_version.version,
    stage="Production",
)

# Load the production model for serving
model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")
```

<div class="diagram">
  <div class="diagram-title">Model Lifecycle in the Registry</div>
  <div class="flow">
    <div class="flow-node green">Training Complete</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node blue">Registered (None)</div>
    <div class="flow-arrow">→ auto-eval</div>
    <div class="flow-node purple">Staging</div>
    <div class="flow-arrow">→ human approval</div>
    <div class="flow-node orange">Production (Canary)</div>
    <div class="flow-arrow">→ metrics OK</div>
    <div class="flow-node accent">Production (Full)</div>
    <div class="flow-arrow">→ new version</div>
    <div class="flow-node teal">Archived</div>
  </div>
</div>

---

## 20.4 Feature Stores

Feature stores solve the **training/serving skew** problem by providing a single source of truth for feature definitions and computation.

```python
# Feast feature store definition
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Define entity
user = Entity(name="user_id", join_keys=["user_id"])

# Define data source
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view
user_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_transactions_7d", dtype=Int64),
        Field(name="avg_amount_30d", dtype=Float32),
        Field(name="max_amount_30d", dtype=Float32),
        Field(name="unique_merchants_7d", dtype=Int64),
    ],
    source=user_stats_source,
    online=True,  # Materialize to online store for serving
)
```

```python
# Training: get historical features (point-in-time correct)
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

training_data = store.get_historical_features(
    entity_df=entity_df,  # DataFrame with user_id + event_timestamp
    features=[
        "user_transaction_features:total_transactions_7d",
        "user_transaction_features:avg_amount_30d",
        "user_transaction_features:max_amount_30d",
        "user_transaction_features:unique_merchants_7d",
    ],
).to_df()

# Serving: get latest features (low-latency)
online_features = store.get_online_features(
    features=[
        "user_transaction_features:total_transactions_7d",
        "user_transaction_features:avg_amount_30d",
    ],
    entity_rows=[{"user_id": "user_12345"}],
).to_dict()
```

---

## 20.5 Model Serving Patterns

| Pattern | Latency | Throughput | Use Case |
|---|---|---|---|
| Batch inference | Minutes–hours | Very high | Recommendations, risk scores |
| Real-time (REST) | 10–100ms | Moderate | Fraud detection, search ranking |
| Real-time (gRPC) | 1–50ms | High | Latency-critical services |
| Streaming | Seconds | High | Continuous monitoring, alerts |
| Edge/embedded | < 10ms | Low | On-device ML, IoT |

```python
# Batch inference with Spark
def batch_predict(model_path: str, input_path: str, output_path: str):
    from pyspark.sql import SparkSession
    import mlflow

    spark = SparkSession.builder.getOrCreate()
    model = mlflow.pyfunc.load_model(model_path)

    input_df = spark.read.parquet(input_path).toPandas()
    predictions = model.predict(input_df)
    input_df["prediction"] = predictions

    spark.createDataFrame(input_df).write.parquet(output_path)
```

```python
# Real-time serving with FastAPI + batching
from fastapi import FastAPI
import asyncio
import numpy as np
from collections import deque

app = FastAPI()
request_queue: deque = deque()
BATCH_SIZE = 32
MAX_WAIT_MS = 10

async def batch_inference_loop():
    """Background task that batches requests for efficient GPU inference."""
    while True:
        if len(request_queue) >= BATCH_SIZE or (
            len(request_queue) > 0 and request_queue[0]["wait_ms"] > MAX_WAIT_MS
        ):
            batch = [request_queue.popleft() for _ in range(min(BATCH_SIZE, len(request_queue)))]
            features = np.array([r["features"] for r in batch])
            predictions = model.predict(features)
            for req, pred in zip(batch, predictions):
                req["future"].set_result(float(pred))
        await asyncio.sleep(0.001)

@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_inference_loop())

@app.post("/predict")
async def predict(features: list[float]):
    future = asyncio.get_event_loop().create_future()
    request_queue.append({"features": features, "future": future, "wait_ms": 0})
    result = await future
    return {"prediction": result}
```

---

## 20.6 A/B Testing & Canary Deployments

```python
# Traffic splitting for A/B testing models
import hashlib
import random

class ModelRouter:
    """Routes traffic between model versions for A/B testing."""

    def __init__(self, models: dict[str, object], weights: dict[str, float]):
        self.models = models
        self.weights = weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def route(self, user_id: str) -> tuple[str, object]:
        """Deterministic routing based on user_id for consistent experience."""
        hash_val = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        bucket = (hash_val % 10000) / 10000.0

        cumulative = 0.0
        for name, weight in self.weights.items():
            cumulative += weight
            if bucket < cumulative:
                return name, self.models[name]

        last_name = list(self.models.keys())[-1]
        return last_name, self.models[last_name]

# Canary deployment: 5% traffic to new model
router = ModelRouter(
    models={"champion_v3": model_v3, "challenger_v4": model_v4},
    weights={"champion_v3": 0.95, "challenger_v4": 0.05},
)

# Usage
model_name, model = router.route(user_id="user_12345")
prediction = model.predict(features)
log_prediction(user_id, model_name, prediction)  # Log for analysis
```

---

## 20.7 Model Monitoring

<div class="diagram">
  <div class="diagram-title">Model Monitoring Dimensions</div>
  <div class="diagram-grid">
    <div class="diagram-card green">
      <strong>Data Drift</strong>
      <p>Input feature distributions shift from training data</p>
    </div>
    <div class="diagram-card blue">
      <strong>Concept Drift</strong>
      <p>Relationship between features and target changes</p>
    </div>
    <div class="diagram-card purple">
      <strong>Performance Degradation</strong>
      <p>Accuracy, precision, recall decline over time</p>
    </div>
    <div class="diagram-card orange">
      <strong>Operational Health</strong>
      <p>Latency, throughput, error rates, resource usage</p>
    </div>
  </div>
</div>

```python
# Data drift detection using Population Stability Index (PSI)
import numpy as np

def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.
    PSI < 0.1: No significant change
    PSI 0.1-0.25: Moderate change — investigate
    PSI > 0.25: Significant change — likely need to retrain
    """
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    curr_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    # Avoid division by zero
    ref_counts = np.clip(ref_counts, 1e-6, None)
    curr_counts = np.clip(curr_counts, 1e-6, None)

    psi = np.sum((curr_counts - ref_counts) * np.log(curr_counts / ref_counts))
    return float(psi)

# Monitor all features
def monitor_features(reference_data, current_data, threshold=0.25):
    alerts = []
    for column in reference_data.columns:
        psi = calculate_psi(
            reference_data[column].values,
            current_data[column].values,
        )
        if psi > threshold:
            alerts.append(f"DRIFT ALERT: {column} PSI={psi:.3f}")
    return alerts
```

```python
# Comprehensive model monitoring with Prometheus metrics
from prometheus_client import Histogram, Counter, Gauge, start_http_server

# Define metrics
prediction_latency = Histogram(
    "model_prediction_latency_seconds",
    "Time to generate prediction",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
prediction_counter = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["model_version", "prediction_class"],
)
feature_drift_gauge = Gauge(
    "feature_drift_psi",
    "Population Stability Index per feature",
    ["feature_name"],
)
model_accuracy_gauge = Gauge(
    "model_accuracy",
    "Rolling model accuracy",
    ["model_version"],
)

@prediction_latency.time()
def predict_with_monitoring(model, features, model_version="v3"):
    prediction = model.predict(features)
    prediction_counter.labels(
        model_version=model_version,
        prediction_class=str(int(prediction[0])),
    ).inc()
    return prediction
```

---

## 20.8 ML Pipeline Orchestration

| Tool | Best For | Complexity | Cloud Native |
|---|---|---|---|
| Airflow | General DAG orchestration | Medium | No (any cloud) |
| Kubeflow Pipelines | Kubernetes-native ML | High | Yes (K8s) |
| Prefect | Modern Python-first orchestration | Low | Hybrid |
| Dagster | Data-asset-centric pipelines | Medium | Hybrid |
| Vertex AI Pipelines | Google Cloud ML | Medium | Yes (GCP) |
| SageMaker Pipelines | AWS ML | Medium | Yes (AWS) |

```python
# Prefect: modern ML pipeline orchestration
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def extract_data(date: str) -> pd.DataFrame:
    return pd.read_parquet(f"s3://data-lake/transactions/{date}.parquet")

@task
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    schema = TrainingDataSchema()
    schema.validate(df)
    return df

@task
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    return feature_pipeline.transform(df)

@task
def train_model(features: pd.DataFrame) -> object:
    model = XGBClassifier(n_estimators=200, max_depth=6)
    X, y = features.drop("label", axis=1), features["label"]
    model.fit(X, y)
    return model

@task
def evaluate_model(model, test_data: pd.DataFrame) -> dict:
    X_test, y_test = test_data.drop("label", axis=1), test_data["label"]
    predictions = model.predict_proba(X_test)[:, 1]
    return {
        "auc_roc": roc_auc_score(y_test, predictions),
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
    }

@flow(name="weekly-fraud-model-training")
def training_pipeline(date: str):
    raw = extract_data(date)
    validated = validate_data(raw)
    features = compute_features(validated)
    model = train_model(features)
    metrics = evaluate_model(model, test_data)

    if metrics["auc_roc"] > 0.95:
        register_model(model, metrics)
        deploy_canary(model)
    else:
        alert_team(f"Model underperformed: AUC={metrics['auc_roc']:.3f}")
```

---

## 20.9 MLOps Maturity Levels

<div class="diagram">
  <div class="diagram-title">MLOps Maturity Model</div>
  <div class="layer-stack">
    <div class="layer purple">Level 4 — Full Automation: Automated retraining, deployment, monitoring, and rollback. Self-healing pipelines.</div>
    <div class="layer blue">Level 3 — Automated Training: Scheduled retraining pipelines, automated evaluation gates, canary deployments.</div>
    <div class="layer green">Level 2 — ML Pipeline: Reproducible pipelines, experiment tracking, model registry, basic monitoring.</div>
    <div class="layer yellow">Level 1 — DevOps for ML: Version control, CI/CD for model code, automated testing, containerization.</div>
    <div class="layer orange">Level 0 — Manual: Jupyter notebooks, manual deployment, no versioning, no monitoring.</div>
  </div>
</div>

| Level | Training | Deployment | Monitoring | Team Size |
|---|---|---|---|---|
| 0 — Manual | Jupyter notebooks | `scp model.pkl server:` | None | 1–2 |
| 1 — DevOps | Scripts + Git | Docker + CI/CD | Basic logs | 3–5 |
| 2 — Pipeline | Orchestrated pipeline | Model registry + staging | Metrics dashboard | 5–10 |
| 3 — Automated | Scheduled + triggered | Canary + automated rollback | Drift detection | 10–20 |
| 4 — Full Auto | Self-improving | Zero-downtime, multi-region | Self-healing | 20+ |

> **Where most teams should aim:** Level 2 is the sweet spot for most ML teams. It provides reproducibility and reliability without the operational overhead of full automation. Move to Level 3 only when you have the team and scale to justify it.

---

## 20.10 ML at Scale: Lessons from Top Labs

Practices observed at leading ML organizations:

**Experiment management:**
- Every training run is logged with full reproducibility metadata
- Experiments are organized by hypothesis, not by date
- Failed experiments are documented as thoroughly as successful ones

**Infrastructure:**
- Training and serving are completely separate systems with different scaling properties
- Feature stores ensure training/serving consistency
- Model artifacts are immutable and checksummed

**Deployment:**
- Models go through staging → canary → production promotion
- Automated evaluation gates prevent regressions
- Shadow deployments run new models alongside production for comparison

**Monitoring:**
- Data drift is monitored per-feature with automated alerts
- Model performance is tracked against holdout sets refreshed regularly
- Incident response playbooks exist for model failures

```yaml
# Example: GitHub Actions CI/CD for ML models
name: ML Model CI/CD
on:
  push:
    paths:
      - "src/models/**"
      - "src/features/**"
      - "configs/**"

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  train-and-evaluate:
    needs: test
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: dvc pull
      - run: python src/train.py --config configs/production.yaml
      - run: python src/evaluate.py --model models/latest --threshold 0.95
      - run: |
          if [ $? -eq 0 ]; then
            python src/register_model.py --stage staging
          fi

  deploy-canary:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    steps:
      - run: python src/deploy.py --stage canary --traffic-pct 5
```

---

## 20.11 Summary

Building production ML systems requires integrating many components beyond the model itself. The key principles:

1. **Track everything** — experiments, data versions, model artifacts, and deployment configs
2. **Validate at every boundary** — data quality, model performance, serving correctness
3. **Automate gradually** — start manual, add automation where it reduces risk
4. **Monitor continuously** — data drift, model performance, and operational health
5. **Design for failure** — rollback mechanisms, shadow deployments, circuit breakers

| Component | Start With | Scale To |
|---|---|---|
| Experiment tracking | MLflow (local) | W&B (team) |
| Feature store | Pandas + Parquet | Feast / Tecton |
| Model registry | MLflow | Custom + S3 |
| Serving | FastAPI | Triton / TorchServe |
| Monitoring | Custom + Grafana | Evidently + PagerDuty |
| Orchestration | Cron + scripts | Prefect / Airflow |
| CI/CD | GitHub Actions | Kubeflow Pipelines |

---

*Last updated: April 2026*
