[← Back to Table of Contents](./README.md)

# Chapter 16 — Software Architecture

> "Architecture is the decisions you wish you could get right early in a project." — Ralph Johnson

Good software architecture is especially critical in ML systems, where the interplay between data, training, and serving creates complexity that can cripple teams if not managed deliberately. This chapter covers architectural patterns through the lens of building ML-powered products.

---

## 16.1 Architecture vs Design

**Architecture** defines the high-level structure — how major components interact, where boundaries lie, and which trade-offs are locked in. **Design** operates within those boundaries — class hierarchies, function signatures, and algorithmic choices.

| Aspect | Architecture | Design |
|---|---|---|
| Scope | System-wide | Component-level |
| Changed by | Team consensus / ADR | Individual developer |
| Cost to change | High | Low–medium |
| Example | "We use microservices with gRPC" | "The DataLoader uses a thread pool" |
| ML example | "Training and serving are separate services" | "We batch predictions in groups of 32" |

In ML systems, architectural decisions include: where models are trained (cloud vs on-prem), how features are computed (online vs offline), and how models are deployed (embedded vs served).

---

## 16.2 Monolith vs Microservices

<div class="diagram">
  <div class="diagram-title">Monolith vs Microservices for ML</div>
  <div class="compare">
    <div class="compare-side">
      <h4>Monolith</h4>
      <div class="layer-stack">
        <div class="layer accent">Web API</div>
        <div class="layer green">Feature Engineering</div>
        <div class="layer blue">Model Training</div>
        <div class="layer purple">Model Serving</div>
        <div class="layer orange">Data Storage</div>
      </div>
      <p><em>Single deployable unit</em></p>
    </div>
    <div class="compare-side">
      <h4>Microservices</h4>
      <div class="flow">
        <div class="flow-node green">Feature Service</div>
        <div class="flow-node blue">Training Service</div>
        <div class="flow-node purple">Serving Service</div>
        <div class="flow-node orange">Data Service</div>
        <div class="flow-node accent">API Gateway</div>
      </div>
      <p><em>Independent services communicating via APIs</em></p>
    </div>
  </div>
</div>

### When to Use Each

| Factor | Monolith | Microservices |
|---|---|---|
| Team size | < 10 engineers | > 10 engineers |
| Model count | 1–3 models | Many models, different cadences |
| Deployment frequency | Weekly | Daily per service |
| ML maturity | Early exploration | Production at scale |
| Data coupling | Shared database is fine | Services own their data |
| Latency budget | Tight (no network hops) | Can tolerate inter-service calls |

> **Practical advice:** Start monolithic. Extract services when pain emerges — not before. Most ML startups do fine with a well-structured monolith for their first 1–2 years.

```python
# Monolith: everything in one Flask app
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/fraud_detector_v3.pkl")
feature_pipeline = joblib.load("pipelines/feature_pipeline.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    raw_data = request.json
    features = feature_pipeline.transform(pd.DataFrame([raw_data]))
    prediction = model.predict_proba(features)[0, 1]
    return jsonify({"fraud_probability": float(prediction)})
```

```python
# Microservice: dedicated model-serving service with gRPC
import grpc
from concurrent import futures
import model_serving_pb2
import model_serving_pb2_grpc
import torch

class ModelServer(model_serving_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        self.model = torch.jit.load("models/fraud_detector.pt")
        self.model.eval()

    def Predict(self, request, context):
        tensor = torch.tensor(request.features).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(self.model(tensor)).item()
        return model_serving_pb2.PredictionResponse(
            fraud_probability=score
        )

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
model_serving_pb2_grpc.add_ModelServiceServicer_to_server(
    ModelServer(), server
)
server.add_insecure_port("[::]:50051")
server.start()
```

---

## 16.3 Layered Architecture

The classic layered (or "n-tier") architecture organizes code into horizontal layers where each layer depends only on the layer below it.

<div class="diagram">
  <div class="diagram-title">Layered Architecture for an ML Application</div>
  <div class="layer-stack">
    <div class="layer accent">Presentation Layer — REST API, Web UI, CLI</div>
    <div class="layer green">Application Layer — Orchestration, Use Cases</div>
    <div class="layer blue">Domain Layer — Business Rules, Model Selection Logic</div>
    <div class="layer purple">Infrastructure Layer — Database, Model Store, Message Queue</div>
  </div>
</div>

```python
# domain/model_selector.py — Domain layer (no framework dependencies)
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    name: str
    version: str
    accuracy: float
    is_champion: bool

class ModelSelector:
    """Selects which model to use based on business rules."""

    def select(self, candidates: list[ModelMetadata]) -> ModelMetadata:
        champion = [m for m in candidates if m.is_champion]
        if not champion:
            raise ValueError("No champion model registered")
        return max(champion, key=lambda m: m.accuracy)
```

```python
# application/prediction_service.py — Application layer
class PredictionService:
    def __init__(self, model_repo, feature_store, model_selector):
        self.model_repo = model_repo
        self.feature_store = feature_store
        self.model_selector = model_selector

    def predict(self, user_id: str) -> dict:
        candidates = self.model_repo.list_models("fraud_detection")
        chosen = self.model_selector.select(candidates)
        model = self.model_repo.load(chosen.name, chosen.version)
        features = self.feature_store.get_features(user_id)
        score = model.predict(features)
        return {"model": chosen.name, "version": chosen.version, "score": score}
```

---

## 16.4 Hexagonal Architecture (Ports & Adapters)

Hexagonal architecture decouples core logic from external systems through **ports** (interfaces) and **adapters** (implementations). This is especially powerful in ML, where you might swap model storage from local files to S3, or switch from SQLite to BigQuery.

```python
# ports.py — Abstract interfaces
from abc import ABC, abstractmethod
import numpy as np

class ModelRepository(ABC):
    @abstractmethod
    def load_model(self, name: str, version: str) -> object:
        ...

    @abstractmethod
    def save_model(self, name: str, version: str, model: object) -> None:
        ...

class FeatureStore(ABC):
    @abstractmethod
    def get_features(self, entity_id: str) -> np.ndarray:
        ...
```

```python
# adapters/s3_model_repo.py — S3 adapter
import boto3
import joblib
import io

class S3ModelRepository(ModelRepository):
    def __init__(self, bucket: str):
        self.s3 = boto3.client("s3")
        self.bucket = bucket

    def load_model(self, name: str, version: str) -> object:
        key = f"models/{name}/{version}/model.joblib"
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return joblib.load(io.BytesIO(response["Body"].read()))

    def save_model(self, name: str, version: str, model: object) -> None:
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        key = f"models/{name}/{version}/model.joblib"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer)
```

---

## 16.5 Event-Driven Architecture

In event-driven systems, components communicate through events rather than direct calls. This pattern excels in ML pipelines where training completion should trigger evaluation, which should trigger deployment.

<div class="diagram">
  <div class="diagram-title">Event-Driven ML Pipeline</div>
  <div class="flow">
    <div class="flow-node green">New Data Arrives</div>
    <div class="flow-arrow">→ event</div>
    <div class="flow-node blue">Training Job Triggered</div>
    <div class="flow-arrow">→ event</div>
    <div class="flow-node purple">Evaluation Runs</div>
    <div class="flow-arrow">→ event</div>
    <div class="flow-node orange">Model Registered</div>
    <div class="flow-arrow">→ event</div>
    <div class="flow-node accent">Canary Deploy Started</div>
    <div class="flow-arrow">→ event</div>
    <div class="flow-node teal">Monitoring Begins</div>
  </div>
</div>

```python
# Simple event bus for ML pipeline orchestration
from collections import defaultdict
from typing import Callable
import logging

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event_type: str, payload: dict) -> None:
        logger.info(f"Publishing event: {event_type}")
        for handler in self._handlers[event_type]:
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Handler failed for {event_type}: {e}")

# Usage
bus = EventBus()

def on_training_complete(payload):
    model_path = payload["model_path"]
    metrics = evaluate_model(model_path)
    bus.publish("evaluation_complete", {"model_path": model_path, "metrics": metrics})

def on_evaluation_complete(payload):
    if payload["metrics"]["accuracy"] > 0.95:
        register_model(payload["model_path"], payload["metrics"])
        bus.publish("model_registered", payload)

bus.subscribe("training_complete", on_training_complete)
bus.subscribe("evaluation_complete", on_evaluation_complete)
```

---

## 16.6 ML Pipeline Architecture Patterns

### DAG Pipelines

Most ML pipelines are directed acyclic graphs (DAGs), where each node is a computation step with defined inputs and outputs.

```python
# Airflow-style DAG definition for an ML pipeline
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG("fraud_model_training", start_date=datetime(2025, 1, 1),
         schedule_interval="@weekly") as dag:

    extract = PythonOperator(
        task_id="extract_transactions",
        python_callable=extract_raw_transactions,
    )
    validate = PythonOperator(
        task_id="validate_data",
        python_callable=run_great_expectations_suite,
    )
    features = PythonOperator(
        task_id="compute_features",
        python_callable=compute_feature_vectors,
    )
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_xgboost_model,
    )
    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_compare,
    )

    extract >> validate >> features >> train >> evaluate
```

### Feature Store Pattern

Feature stores provide a centralized layer for feature computation, storage, and serving — ensuring training/serving consistency.

| Component | Role | Example |
|---|---|---|
| Feature definitions | Declarative transformations | Feast feature views |
| Offline store | Historical features for training | BigQuery, S3 + Parquet |
| Online store | Low-latency features for serving | Redis, DynamoDB |
| Feature registry | Metadata and lineage | Feast registry, Tecton |

---

## 16.7 ML System Separation of Concerns

<div class="diagram">
  <div class="diagram-title">Complete ML System Architecture</div>
  <div class="flow">
    <div class="flow-node cyan">Data Sources (APIs, DBs, Streams)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Data Layer (Ingestion, Validation, Storage)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node blue">Feature Layer (Feature Store, Transforms)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">Training Layer (Experiment Tracking, HPO)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Model Registry (Versioning, Metadata)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">Serving Layer (API, Batch, Streaming)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node teal">Monitoring Layer (Drift, Latency, Accuracy)</div>
  </div>
</div>

Each layer has distinct responsibilities:

| Layer | Responsibility | Tooling Examples |
|---|---|---|
| Data | Ingest, clean, validate, version | DVC, Great Expectations, dbt |
| Feature | Transform, store, serve features | Feast, Tecton, Hopsworks |
| Training | Experiment tracking, hyperparameter tuning | W&B, MLflow, Optuna |
| Registry | Model versioning, approval workflows | MLflow Registry, Vertex AI |
| Serving | Inference at required latency/throughput | TorchServe, Triton, BentoML |
| Monitoring | Detect drift, track metrics, alert | Evidently, WhyLabs, Prometheus |

---

## 16.8 Architecture Decision Records (ADRs)

ADRs document significant architectural decisions with their context and consequences. They prevent revisiting settled debates and help onboard new team members.

```markdown
# ADR-007: Use Separate Services for Training and Serving

## Status
Accepted

## Context
Our fraud detection model is retrained weekly but serves predictions
at 10k QPS. Training requires GPU instances; serving runs on CPU.
Coupling these creates deployment risk — a training update could
break the serving path.

## Decision
We will deploy training and serving as separate services:
- Training service: runs on GPU spot instances, triggered weekly
- Serving service: runs on CPU instances behind a load balancer
- Communication via model registry (S3 + metadata DB)

## Consequences
- **Positive:** Independent scaling, isolated failure domains
- **Positive:** Can update serving without retraining
- **Negative:** Must ensure feature consistency across services
- **Negative:** Increased operational complexity (two deployments)
```

> **Key practice at top ML labs:** Google DeepMind and Anthropic use ADRs extensively. When a system's architecture seems "obvious," it's often because someone documented the reasoning well.

---

## 16.9 Summary

| Pattern | Best For | ML Use Case |
|---|---|---|
| Monolith | Small teams, early stage | MVP with single model |
| Microservices | Large teams, many models | Multi-model platform |
| Layered | Clear separation of concerns | Standard ML application |
| Hexagonal | Testability, swappable infra | Switching cloud providers |
| Event-driven | Async workflows, pipelines | Training → eval → deploy |
| DAG pipeline | Reproducible ML workflows | Weekly retraining jobs |

**Rules of thumb:**
1. Start simple — extract complexity only when pain demands it
2. Separate what changes at different rates (data schema vs model code vs serving logic)
3. Document decisions with ADRs — your future self will thank you
4. Design for the team you have, not the team you wish you had

---

*Last updated: April 2026*
