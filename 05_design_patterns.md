[← Back to Table of Contents](./README.md)

# Chapter 5 — Design Patterns

> *"Design patterns are not about code — they are about communication. They give engineers a shared vocabulary for recurring solutions."* — Adapted from *Design Patterns: Elements of Reusable Object-Oriented Software* (GoF)

Design patterns are proven solutions to common software design problems. In ML, certain patterns appear repeatedly: strategies for swapping training algorithms, factories for creating models, registries for tracking experiments, and pipelines for processing data. This chapter covers the patterns most relevant to ML engineering with practical, production-ready examples.

---

## 5.1 Gang of Four Pattern Categories

<div class="diagram">
<div class="diagram-title">Design Pattern Categories</div>
<div class="diagram-grid">
<div class="diagram-card green">
<strong>Creational</strong><br/>
How objects are created<br/>
• Factory Method<br/>
• Abstract Factory<br/>
• Singleton<br/>
• Builder<br/>
• Prototype
</div>
<div class="diagram-card blue">
<strong>Structural</strong><br/>
How objects are composed<br/>
• Adapter<br/>
• Decorator<br/>
• Facade<br/>
• Composite<br/>
• Proxy
</div>
<div class="diagram-card purple">
<strong>Behavioral</strong><br/>
How objects communicate<br/>
• Strategy<br/>
• Observer<br/>
• Template Method<br/>
• Iterator<br/>
• Command
</div>
</div>
</div>

### Most Relevant Patterns for ML

| Pattern | ML Use Case | Frequency |
|---|---|---|
| **Strategy** | Swap training algorithms, optimizers, schedulers | ⭐⭐⭐⭐⭐ |
| **Factory** | Create models, datasets, tokenizers from config | ⭐⭐⭐⭐⭐ |
| **Registry** | Model registry, experiment catalog | ⭐⭐⭐⭐⭐ |
| **Pipeline** | Data processing, feature engineering | ⭐⭐⭐⭐⭐ |
| **Observer** | Callbacks, logging, monitoring | ⭐⭐⭐⭐ |
| **Template Method** | Training loops, evaluation loops | ⭐⭐⭐⭐ |
| **Singleton** | Config management, device management | ⭐⭐⭐ |
| **Adapter** | Dataset format conversion, API compatibility | ⭐⭐⭐ |

---

## 5.2 Strategy Pattern — Training Strategies

The Strategy pattern defines a family of interchangeable algorithms. In ML, this is ubiquitous: you swap optimizers, learning rate schedulers, loss functions, and augmentation policies without changing the training loop.

<div class="diagram">
<div class="diagram-title">Strategy Pattern for Training</div>
<div class="flow">
<div class="flow-node accent">Trainer<br/><small>Uses strategy interface</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node green">TrainingStrategy (ABC)<br/><small>configure_optimizers()<br/>compute_loss()</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node blue">SupervisedStrategy</div>
<div class="flow-arrow">|</div>
<div class="flow-node purple">ContrastiveStrategy</div>
<div class="flow-arrow">|</div>
<div class="flow-node orange">RLHFStrategy</div>
</div>
</div>

```python
# strategies.py — Training strategies as interchangeable components
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TrainingStrategy(ABC):
    """Abstract base for training strategies."""

    @abstractmethod
    def configure_optimizers(
        self, model: nn.Module
    ) -> tuple[Optimizer, LRScheduler]:
        """Create optimizer and scheduler for this strategy."""
        ...

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute training loss for this strategy."""
        ...


class SupervisedStrategy(TrainingStrategy):
    """Standard supervised training with cross-entropy loss."""

    def __init__(self, lr: float = 3e-4, weight_decay: float = 0.01):
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self, model):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000
        )
        return optimizer, scheduler

    def compute_loss(self, model, batch):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            batch["labels"].view(-1),
        )
        return {"loss": loss, "logits": outputs.logits}


class ContrastiveStrategy(TrainingStrategy):
    """Contrastive learning (e.g., SimCLR, CLIP-style)."""

    def __init__(self, lr: float = 1e-4, temperature: float = 0.07):
        self.lr = lr
        self.temperature = temperature

    def configure_optimizers(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=500
        )
        return optimizer, scheduler

    def compute_loss(self, model, batch):
        z_i = model.encode(batch["view_1"])
        z_j = model.encode(batch["view_2"])

        z_i = nn.functional.normalize(z_i, dim=-1)
        z_j = nn.functional.normalize(z_j, dim=-1)

        similarity = torch.mm(z_i, z_j.T) / self.temperature
        labels = torch.arange(z_i.size(0), device=z_i.device)
        loss = (
            nn.functional.cross_entropy(similarity, labels)
            + nn.functional.cross_entropy(similarity.T, labels)
        ) / 2

        return {"loss": loss, "similarity": similarity}


# --- Usage: The Trainer doesn't know which strategy it uses ---
class Trainer:
    """Generic trainer that delegates to a strategy."""

    def __init__(self, model: nn.Module, strategy: TrainingStrategy):
        self.model = model
        self.strategy = strategy
        self.optimizer, self.scheduler = strategy.configure_optimizers(model)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.optimizer.zero_grad()
        result = self.strategy.compute_loss(self.model, batch)
        result["loss"].backward()
        self.optimizer.step()
        self.scheduler.step()
        return {k: v.item() for k, v in result.items() if v.dim() == 0}
```

---

## 5.3 Factory Pattern — Model & Dataset Creation

The Factory pattern creates objects without specifying their exact class. This is essential when model/dataset selection is driven by configuration files.

```python
# factories.py — Create models and datasets from config strings
from __future__ import annotations
import torch.nn as nn
from typing import Any


class ModelFactory:
    """Factory for creating ML models from configuration."""

    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: type[nn.Module]) -> type[nn.Module]:
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Create a model instance by name."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)


# --- Register models using the decorator ---
@ModelFactory.register("transformer")
class TransformerModel(nn.Module):
    def __init__(self, d_model: int = 512, n_layers: int = 6, **kwargs):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=n_layers,
        )

    def forward(self, x):
        return self.encoder(x)


@ModelFactory.register("mlp")
class MLPModel(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# --- Usage: Create models from YAML config ---
# config.yaml:
#   model:
#     name: transformer
#     d_model: 256
#     n_layers: 4

def build_from_config(config: dict) -> nn.Module:
    """Build a model from a config dictionary."""
    model_config = config["model"]
    name = model_config.pop("name")
    return ModelFactory.create(name, **model_config)
```

---

## 5.4 Observer Pattern — Callbacks & Logging

The Observer pattern lets objects subscribe to events without tight coupling. In ML, this manifests as **callback systems** for logging, checkpointing, and early stopping.

```python
# callbacks.py — Observer pattern for training events
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class TrainingCallback(ABC):
    """Base class for training callbacks (observers)."""

    def on_train_begin(self, state: dict[str, Any]) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...
    def on_epoch_begin(self, epoch: int, state: dict[str, Any]) -> None: ...
    def on_epoch_end(self, epoch: int, state: dict[str, Any]) -> None: ...
    def on_step_end(self, step: int, state: dict[str, Any]) -> None: ...


class MetricsLogger(TrainingCallback):
    """Log metrics to W&B on every step."""

    def __init__(self, project: str, run_name: str):
        import wandb
        self.run = wandb.init(project=project, name=run_name)

    def on_step_end(self, step, state):
        self.run.log({"step": step, **state.get("metrics", {})})

    def on_train_end(self, state):
        self.run.finish()


class CheckpointSaver(TrainingCallback):
    """Save model checkpoints periodically."""

    def __init__(self, save_dir: Path, every_n_steps: int = 1000):
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, step, state):
        if step % self.every_n_steps == 0 and step > 0:
            path = self.save_dir / f"checkpoint-{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": state["model"].state_dict(),
                "optimizer_state_dict": state["optimizer"].state_dict(),
                "metrics": state.get("metrics", {}),
            }, path)


class EarlyStopping(TrainingCallback):
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float = float("inf")
        self.counter: int = 0

    def on_epoch_end(self, epoch, state):
        current = state.get("val_loss", float("inf"))
        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                state["should_stop"] = True


class CallbackHandler:
    """Manages and dispatches events to registered callbacks."""

    def __init__(self, callbacks: list[TrainingCallback]):
        self.callbacks = callbacks

    def fire(self, event: str, **kwargs) -> None:
        for cb in self.callbacks:
            if hasattr(cb, event):
                getattr(cb, event)(**kwargs)


# --- Usage ---
handler = CallbackHandler([
    MetricsLogger(project="my-llm", run_name="run-42"),
    CheckpointSaver(save_dir=Path("checkpoints"), every_n_steps=500),
    EarlyStopping(patience=3),
])

# In training loop:
# handler.fire("on_step_end", step=step, state=state)
# handler.fire("on_epoch_end", epoch=epoch, state=state)
```

---

## 5.5 Registry Pattern — Model & Experiment Registry

The Registry pattern provides a centralized catalog for looking up objects by name. It combines Factory with a global lookup table — extremely common in ML frameworks.

```python
# registry.py — A generic registry with decorator-based registration
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """Generic registry for named objects (models, datasets, metrics, etc.)."""

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable:
        """Decorator to register a class or function."""
        def decorator(obj: T) -> T:
            key = name or obj.__name__  # type: ignore[union-attr]
            if key in self._registry:
                raise ValueError(
                    f"'{key}' already registered in {self.name} registry"
                )
            self._registry[key] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def list_available(self) -> list[str]:
        return sorted(self._registry.keys())


# --- Create registries for different component types ---
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
METRIC_REGISTRY = Registry("metric")
LOSS_REGISTRY = Registry("loss")


# --- Register components with decorators ---
@LOSS_REGISTRY.register("cross_entropy")
def cross_entropy_loss(logits, targets, **kwargs):
    import torch.nn.functional as F
    return F.cross_entropy(logits, targets, **kwargs)


@LOSS_REGISTRY.register("focal")
def focal_loss(logits, targets, gamma=2.0, **kwargs):
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


@METRIC_REGISTRY.register("accuracy")
def accuracy(preds, targets):
    return (preds.argmax(dim=-1) == targets).float().mean()


# --- Usage from config ---
# config.yaml:
#   loss: focal
#   metrics: [accuracy]

def build_from_config(config: dict):
    loss_fn = LOSS_REGISTRY.get(config["loss"])
    metric_fns = [METRIC_REGISTRY.get(m) for m in config["metrics"]]
    return loss_fn, metric_fns
```

---

## 5.6 Pipeline Pattern — Data Processing

The Pipeline pattern chains processing steps into a composable sequence. This is the backbone of every ML data pipeline.

<div class="diagram">
<div class="diagram-title">Data Processing Pipeline</div>
<div class="flow-h">
<div class="flow-node green">Raw Data</div>
<div class="flow-arrow">→</div>
<div class="flow-node blue">Clean &amp; Validate</div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">Tokenize</div>
<div class="flow-arrow">→</div>
<div class="flow-node orange">Augment</div>
<div class="flow-arrow">→</div>
<div class="flow-node accent">Batch &amp; Pad</div>
<div class="flow-arrow">→</div>
<div class="flow-node teal">Model-Ready Tensor</div>
</div>
</div>

```python
# pipeline.py — Composable data processing pipeline
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


class PipelineStep(ABC):
    """Abstract base for a single pipeline step."""

    @abstractmethod
    def process(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single data record. Return None to filter it out."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class Pipeline:
    """Composable data processing pipeline."""

    def __init__(self, steps: list[PipelineStep] | None = None):
        self.steps = steps or []

    def add(self, step: PipelineStep) -> Pipeline:
        """Add a step and return self for chaining."""
        self.steps.append(step)
        return self

    def run(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Run a single record through all steps."""
        for step in self.steps:
            if data is None:
                return None
            data = step.process(data)
        return data

    def run_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run multiple records, filtering out None results."""
        results = []
        for record in records:
            result = self.run(record)
            if result is not None:
                results.append(result)
        return results


# --- Concrete pipeline steps ---
class TextCleaner(PipelineStep):
    """Normalize and clean raw text."""

    def process(self, data):
        text = data.get("text", "")
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        if len(text) < 10:
            return None  # Filter too-short texts
        data["text"] = text
        return data


class LanguageFilter(PipelineStep):
    """Filter to keep only specified languages."""

    def __init__(self, allowed_languages: set[str]):
        self.allowed = allowed_languages

    def process(self, data):
        if data.get("language", "en") not in self.allowed:
            return None
        return data


class Tokenizer(PipelineStep):
    """Tokenize text into input IDs."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process(self, data):
        encoded = self.tokenizer(
            data["text"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        data["input_ids"] = encoded["input_ids"]
        data["attention_mask"] = encoded["attention_mask"]
        return data


class QualityScorer(PipelineStep):
    """Score text quality and filter low-quality samples."""

    def __init__(self, min_score: float = 0.3):
        self.min_score = min_score

    def process(self, data):
        text = data.get("text", "")
        # Simple heuristic quality scoring
        score = min(1.0, len(set(text.split())) / max(len(text.split()), 1))
        data["quality_score"] = score
        if score < self.min_score:
            return None
        return data


# --- Build pipeline from config ---
def build_pipeline(config: dict) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add(TextCleaner())

    if "languages" in config:
        pipeline.add(LanguageFilter(set(config["languages"])))
    if config.get("quality_filter", True):
        pipeline.add(QualityScorer(min_score=config.get("min_quality", 0.3)))

    return pipeline
```

---

## 5.7 Singleton Pattern — Configuration Management

The Singleton pattern ensures a class has exactly one instance. In ML, this is used for global configuration, device management, and resource pools.

```python
# config.py — Singleton configuration manager
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Singleton configuration manager for ML projects.

    Ensures all modules read from the same configuration state.
    """

    _instance: Config | None = None
    _config: dict[str, Any] = {}

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        instance = cls()
        with open(path) as f:
            instance._config = yaml.safe_load(f)
        return instance

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation: 'model.d_model'."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — useful for testing."""
        cls._instance = None
        cls._config = {}


# --- Usage ---
# config = Config.load("configs/train.yaml")
# lr = config.get("training.lr", 3e-4)
# d_model = config.get("model.d_model", 512)
```

> ⚠️ **Use Singletons sparingly.** They introduce hidden global state that makes testing harder. Prefer dependency injection when possible; reserve Singletons for truly global concerns like app-level configuration.

---

## 5.8 Template Method Pattern — Training Loops

The Template Method defines the skeleton of an algorithm, letting subclasses override specific steps without changing the overall structure.

```python
# trainer_template.py — Template Method for training loops
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Template for training loops — override hooks, not the loop."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def fit(self, epochs: int) -> dict[str, list[float]]:
        """Main training loop — the template method."""
        self.optimizer, self.scheduler = self.configure_optimizers()
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        self.on_training_start()

        for epoch in range(epochs):
            # --- Training phase ---
            self.model.train()
            train_loss = 0.0
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.on_before_optimizer_step()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            history["train_loss"].append(train_loss)

            # --- Validation phase ---
            val_loss = self.validate()
            history["val_loss"].append(val_loss)

            self.scheduler.step()
            self.on_epoch_end(epoch, train_loss, val_loss)

        self.on_training_end()
        return history

    @abstractmethod
    def configure_optimizers(self) -> tuple:
        """Subclasses define their optimizer and scheduler."""
        ...

    @abstractmethod
    def training_step(self, batch: dict) -> torch.Tensor:
        """Subclasses define the forward pass and loss computation."""
        ...

    def validate(self) -> float:
        """Default validation — can be overridden."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.training_step(batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    # --- Hooks for subclasses to override ---
    def on_training_start(self) -> None: ...
    def on_training_end(self) -> None: ...
    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None: ...
    def on_before_optimizer_step(self) -> None: ...


class LanguageModelTrainer(BaseTrainer):
    """Concrete trainer for causal language models."""

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000
        )
        return optimizer, scheduler

    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            batch["labels"].view(-1),
        )

    def on_before_optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def on_epoch_end(self, epoch, train_loss, val_loss):
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

---

## 5.9 Adapter Pattern — Dataset Adapters

The Adapter pattern converts one interface to another. In ML, different datasets come in wildly different formats — Adapters provide a uniform interface.

```python
# adapters.py — Uniform interface for diverse dataset formats
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import json
import csv


class DatasetAdapter(ABC):
    """Uniform interface for reading different dataset formats."""

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield records as dictionaries."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class JSONLAdapter(DatasetAdapter):
    """Adapter for JSON Lines format (one JSON object per line)."""

    def __init__(self, path: Path, text_field: str = "text"):
        self.path = path
        self.text_field = text_field
        self._lines = path.read_text().strip().split("\n")

    def __iter__(self):
        for line in self._lines:
            record = json.loads(line)
            yield {"text": record[self.text_field], "metadata": record}

    def __len__(self):
        return len(self._lines)


class CSVAdapter(DatasetAdapter):
    """Adapter for CSV datasets."""

    def __init__(self, path: Path, text_column: str = "text",
                 label_column: str | None = "label"):
        self.path = path
        self.text_column = text_column
        self.label_column = label_column
        with open(path) as f:
            self._rows = list(csv.DictReader(f))

    def __iter__(self):
        for row in self._rows:
            record = {"text": row[self.text_column]}
            if self.label_column and self.label_column in row:
                record["label"] = row[self.label_column]
            yield record

    def __len__(self):
        return len(self._rows)


class HuggingFaceAdapter(DatasetAdapter):
    """Adapter for Hugging Face datasets."""

    def __init__(self, name: str, split: str = "train",
                 text_field: str = "text"):
        from datasets import load_dataset
        self.dataset = load_dataset(name, split=split)
        self.text_field = text_field

    def __iter__(self):
        for record in self.dataset:
            yield {"text": record[self.text_field], "metadata": dict(record)}

    def __len__(self):
        return len(self.dataset)


# --- Unified loading function ---
def load_dataset(path: str, format: str = "auto", **kwargs) -> DatasetAdapter:
    """Load any dataset through a uniform adapter interface."""
    adapters = {
        "jsonl": JSONLAdapter,
        "csv": CSVAdapter,
        "huggingface": HuggingFaceAdapter,
    }

    if format == "auto":
        if path.endswith(".jsonl"):
            format = "jsonl"
        elif path.endswith(".csv"):
            format = "csv"
        else:
            format = "huggingface"

    adapter_cls = adapters.get(format)
    if adapter_cls is None:
        raise ValueError(f"Unknown format: {format}")

    if format == "huggingface":
        return adapter_cls(name=path, **kwargs)
    return adapter_cls(path=Path(path), **kwargs)
```

---

## 5.10 Choosing the Right Pattern

<div class="diagram">
<div class="diagram-title">Pattern Selection Guide for ML</div>
<div class="diagram-grid">
<div class="diagram-card green">
<strong>Need to swap algorithms?</strong><br/>
→ Strategy Pattern<br/>
<small>Optimizers, losses, schedulers</small>
</div>
<div class="diagram-card blue">
<strong>Need to create from config?</strong><br/>
→ Factory + Registry<br/>
<small>Models, datasets, tokenizers</small>
</div>
<div class="diagram-card purple">
<strong>Need event-driven hooks?</strong><br/>
→ Observer (Callbacks)<br/>
<small>Logging, checkpointing, early stop</small>
</div>
<div class="diagram-card orange">
<strong>Need sequential processing?</strong><br/>
→ Pipeline Pattern<br/>
<small>Data cleaning, feature engineering</small>
</div>
<div class="diagram-card accent">
<strong>Need a fixed loop skeleton?</strong><br/>
→ Template Method<br/>
<small>Training loops, eval loops</small>
</div>
<div class="diagram-card teal">
<strong>Need format compatibility?</strong><br/>
→ Adapter Pattern<br/>
<small>Dataset formats, API wrappers</small>
</div>
</div>
</div>

### Anti-Patterns to Avoid

| Anti-Pattern | Problem | Better Approach |
|---|---|---|
| **God class** | One class does everything (data, model, training, eval) | Separate concerns with Strategy + Observer |
| **Hardcoded config** | Values buried in code | Singleton Config or dependency injection |
| **Copy-paste models** | Duplicated code for similar models | Factory + Registry |
| **Spaghetti pipeline** | Monolithic data processing function | Pipeline pattern with composable steps |
| **Callback hell** | Deeply nested, unstructured hooks | Observer pattern with clean callback interface |

> *"Patterns are tools, not rules. Use them when they clarify your design, not to impress code reviewers. The best code uses the simplest pattern that solves the problem."*

---

*Last updated: April 2026*
