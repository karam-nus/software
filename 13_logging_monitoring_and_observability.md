[← Back to Table of Contents](./README.md)

# Chapter 13 — Logging, Monitoring & Observability

> "You can't improve what you can't measure, and you can't debug what you can't observe. In ML systems, observability is the difference between guessing and knowing."

ML systems fail in subtle ways that traditional software doesn't. A model might silently degrade because of data drift, a training run might slow down due to a straggling GPU, or inference latency might spike because of batch padding. Without proper logging, monitoring, and observability, these issues go undetected until they cause real damage.

This chapter covers the three pillars of observability — logs, metrics, and traces — applied specifically to ML workflows.

---

## 13.1 The Three Pillars of Observability

<div class="diagram">
  <div class="diagram-title">The Three Pillars of Observability</div>
  <div class="diagram-grid">
    <div class="diagram-card accent">
      <strong>Logs</strong><br/>
      Discrete events with context<br/>
      "What happened?"<br/>
      <em>Training started, epoch completed, error occurred</em>
    </div>
    <div class="diagram-card green">
      <strong>Metrics</strong><br/>
      Numeric measurements over time<br/>
      "How is it performing?"<br/>
      <em>Loss=0.23, latency_p99=45ms, GPU util=87%</em>
    </div>
    <div class="diagram-card blue">
      <strong>Traces</strong><br/>
      Request flows across services<br/>
      "Where did time go?"<br/>
      <em>API→preprocess→model→postprocess: 120ms total</em>
    </div>
  </div>
</div>

| Pillar | Tool Examples | ML Use Case |
|---|---|---|
| **Logs** | Python logging, structlog, Loguru | Training events, errors, data issues |
| **Metrics** | Prometheus, Grafana, StatsD, Datadog | Loss curves, throughput, GPU utilization |
| **Traces** | OpenTelemetry, Jaeger, Zipkin | Inference pipeline latency breakdown |

---

## 13.2 Python's Logging Module

### Basic Setup

```python
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Usage
logger.info("Starting training run")
logger.debug("Batch %d: loss=%.4f, lr=%.2e", step, loss, lr)
logger.warning("GPU memory usage above 90%%: %.1f%%", gpu_mem_pct)
logger.error("Checkpoint save failed: %s", str(e))
```

### Log Levels

| Level | Value | Use Case |
|---|---|---|
| `DEBUG` | 10 | Detailed diagnostic info (tensor shapes, gradient norms) |
| `INFO` | 20 | General progress (epoch completed, checkpoint saved) |
| `WARNING` | 30 | Something unexpected but not fatal (NaN in metrics, slow data loading) |
| `ERROR` | 40 | Something failed (OOM, file not found, API timeout) |
| `CRITICAL` | 50 | System is unusable (all GPUs failed, data corruption) |

### Multi-Handler Setup

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    """Configure logging with console and file handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler — human readable
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(console)

    # File handler — detailed, rotating
    file_handler = RotatingFileHandler(
        f"{log_dir}/training.log",
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
    ))
    root_logger.addHandler(file_handler)

    # JSON file handler — for log aggregation systems
    json_handler = RotatingFileHandler(
        f"{log_dir}/training.jsonl",
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(json_handler)

    return root_logger
```

---

## 13.3 Structured Logging with structlog

Structured logging produces machine-parseable log entries, essential for log aggregation and analysis:

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Structured log entries with context
logger.info(
    "training_step",
    step=1000,
    loss=0.234,
    learning_rate=3e-4,
    gpu_memory_mb=15234,
    tokens_per_second=45000,
)
# Output: {"event": "training_step", "step": 1000, "loss": 0.234, ...}

# Bind context that persists across log calls
log = logger.bind(run_id="exp-2024-03-15", model="llama-7b")
log.info("epoch_complete", epoch=1, val_loss=0.198)
log.info("epoch_complete", epoch=2, val_loss=0.187)
```

### Logging in Distributed Training

```python
import structlog
import torch.distributed as dist

def get_rank_logger():
    """Create a logger that includes distributed training context."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    logger = structlog.get_logger()
    return logger.bind(rank=rank, world_size=world_size)

def log_on_main_only(logger, event, **kwargs):
    """Only log from rank 0 to avoid duplicate messages."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.info(event, **kwargs)

# Usage in training loop
logger = get_rank_logger()

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)

    # All ranks log (useful for debugging stragglers)
    if step % 100 == 0:
        logger.debug("step_complete", step=step, loss=loss.item())

    # Only rank 0 logs progress
    log_on_main_only(logger, "training_progress",
        step=step, loss=loss.item(), lr=scheduler.get_last_lr()[0])
```

---

## 13.4 Experiment Tracking

<div class="diagram">
  <div class="diagram-title">Experiment Tracking Data Flow</div>
  <div class="flow">
    <div class="flow-node accent">Training Script</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Tracking Client (W&amp;B / MLflow)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node blue">Backend Storage</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">Dashboard UI</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Team Analysis &amp; Comparison</div>
  </div>
</div>

### Weights & Biases Integration

```python
import wandb
import torch

def train_with_wandb(config: dict):
    # Initialize run
    run = wandb.init(
        project="llm-pretraining",
        name=f"{config['model_name']}-{config['learning_rate']}",
        config=config,
        tags=["baseline", "v2"],
    )

    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    for step in range(config["max_steps"]):
        loss = train_step(model, optimizer, next(dataloader))

        # Log metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/grad_norm": get_grad_norm(model),
            "system/gpu_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "system/gpu_utilization": get_gpu_utilization(),
        }, step=step)

        # Log evaluation metrics periodically
        if step % config["eval_every"] == 0:
            val_metrics = evaluate(model, val_dataloader)
            wandb.log({
                "val/loss": val_metrics["loss"],
                "val/perplexity": val_metrics["perplexity"],
                "val/accuracy": val_metrics["accuracy"],
            }, step=step)

        # Save model artifact
        if step % config["save_every"] == 0:
            artifact = wandb.Artifact(
                name=f"model-{run.id}",
                type="model",
                metadata={"step": step, "val_loss": val_metrics["loss"]},
            )
            artifact.add_file(f"checkpoints/step_{step}.pt")
            run.log_artifact(artifact)

    run.finish()
```

### MLflow Integration

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("llm-pretraining")

with mlflow.start_run(run_name="transformer-base-lr3e4"):
    # Log parameters
    mlflow.log_params({
        "model_name": "transformer-base",
        "hidden_size": 768,
        "learning_rate": 3e-4,
        "batch_size": 32,
    })

    for step in range(max_steps):
        loss = train_step(model, optimizer, batch)

        # Log metrics with step
        mlflow.log_metrics({
            "train_loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
        }, step=step)

    # Log model artifact
    mlflow.pytorch.log_model(model, "model")

    # Log training artifacts
    mlflow.log_artifact("configs/train.yaml")
    mlflow.log_artifact("logs/training.log")
```

### Tool Comparison

| Feature | W&B | MLflow | TensorBoard |
|---|---|---|---|
| **Hosting** | Cloud (free tier) or self-hosted | Self-hosted or managed | Local or TensorBoard.dev |
| **Experiment comparison** | Excellent | Good | Basic |
| **Hyperparameter sweeps** | Built-in (W&B Sweeps) | Via plugins | No |
| **Model registry** | Yes | Yes | No |
| **Team collaboration** | Excellent | Good | Limited |
| **Artifacts** | Yes (datasets, models) | Yes | No |
| **Cost** | Free for individuals | Free (OSS) | Free |

---

## 13.5 Metrics Collection

### Prometheus Metrics for Model Serving

```python
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, start_http_server,
)

# Define metrics
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["model_name", "model_version"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_duration_seconds",
    "Prediction latency in seconds",
    ["model_name"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

BATCH_SIZE_GAUGE = Gauge(
    "model_current_batch_size",
    "Current dynamic batch size",
)

GPU_MEMORY_USAGE = Gauge(
    "gpu_memory_usage_bytes",
    "GPU memory usage in bytes",
    ["gpu_id"],
)

# Use in application
import time

def predict(inputs: list[str]) -> list[dict]:
    BATCH_SIZE_GAUGE.set(len(inputs))

    start = time.perf_counter()
    results = model.predict(inputs)
    duration = time.perf_counter() - start

    PREDICTION_LATENCY.labels(model_name="gpt2").observe(duration)
    PREDICTION_COUNT.labels(
        model_name="gpt2", model_version="v2.1"
    ).inc(len(inputs))

    return results

# Start metrics server on port 9090
start_http_server(9090)
```

---

## 13.6 Health Checks and Alerting

```python
from fastapi import FastAPI, Response
import torch

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic liveness check."""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    """Check if model is loaded and GPU is available."""
    checks = {
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_ok": (
            torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() < 0.95
            if torch.cuda.is_available() else True
        ),
    }

    all_ok = all(checks.values())
    return Response(
        content=json.dumps({"status": "ready" if all_ok else "not_ready", "checks": checks}),
        status_code=200 if all_ok else 503,
        media_type="application/json",
    )

@app.get("/metrics/model")
async def model_metrics():
    """Custom model-specific metrics."""
    return {
        "model_name": "gpt2-large",
        "model_version": "v2.1",
        "total_predictions": PREDICTION_COUNT._value.get(),
        "avg_latency_ms": PREDICTION_LATENCY._sum.get() / max(PREDICTION_LATENCY._count.get(), 1) * 1000,
        "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
    }
```

---

## 13.7 Observability for Model Serving

<div class="diagram">
  <div class="diagram-title">Model Serving Observability Stack</div>
  <div class="layer-stack">
    <div class="layer accent">Alerting (PagerDuty, Slack, OpsGenie)</div>
    <div class="layer orange">Dashboards (Grafana)</div>
    <div class="layer yellow">Metrics Store (Prometheus / Datadog)</div>
    <div class="layer green">Application Metrics (latency, throughput, errors)</div>
    <div class="layer blue">Log Aggregation (ELK / Loki / CloudWatch)</div>
    <div class="layer purple">Distributed Tracing (OpenTelemetry / Jaeger)</div>
  </div>
</div>

### Key Metrics to Track

| Category | Metric | Alert Threshold |
|---|---|---|
| **Latency** | p50, p95, p99 response time | p99 > 500ms |
| **Throughput** | Requests per second | < 80% of baseline |
| **Error rate** | 4xx and 5xx responses | > 1% of requests |
| **GPU utilization** | GPU compute % | < 20% (wasting $) or > 95% (saturated) |
| **Memory** | GPU memory usage | > 90% of capacity |
| **Queue depth** | Pending inference requests | > 100 requests |
| **Model freshness** | Time since last model update | > 7 days |
| **Data drift** | Distribution shift score | Score > threshold |

---

## 13.8 Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup tracing
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

async def inference_pipeline(request):
    with tracer.start_as_current_span("inference_pipeline") as span:
        span.set_attribute("model.name", "gpt2-large")
        span.set_attribute("batch.size", len(request.inputs))

        # Preprocessing
        with tracer.start_as_current_span("preprocess"):
            tokens = tokenizer(request.inputs, padding=True, return_tensors="pt")

        # Model inference
        with tracer.start_as_current_span("model_forward") as model_span:
            outputs = model.generate(**tokens)
            model_span.set_attribute("output.tokens", outputs.shape[-1])

        # Postprocessing
        with tracer.start_as_current_span("postprocess"):
            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return results
```

---

## 13.9 Logging Pipeline Architecture

<div class="diagram">
  <div class="diagram-title">Production Logging Pipeline</div>
  <div class="flow">
    <div class="flow-node blue">Application (structlog)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Log Shipper (Fluent Bit)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">Stream Processing (Kafka)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">Index &amp; Store (Elasticsearch / Loki)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Visualize (Kibana / Grafana)</div>
  </div>
</div>

---

## 13.10 Key Takeaways

1. **Implement all three pillars** — logs for events, metrics for numbers, traces for request flows.
2. **Use structured logging** (structlog or JSON) — plain text logs don't scale.
3. **Log on rank 0 only** in distributed training to avoid N-way duplicate messages.
4. **Track every experiment** with W&B or MLflow — you will need to compare runs later.
5. **Expose Prometheus metrics** from model-serving endpoints for real-time monitoring.
6. **Set up health checks** (`/health`, `/ready`) for load balancers and orchestrators.
7. **Alert on the right things** — latency percentiles, error rates, GPU utilization, and data drift.
8. **Use distributed tracing** to find bottlenecks in multi-stage inference pipelines.

---

*Last updated: April 2026*
