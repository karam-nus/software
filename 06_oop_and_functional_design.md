[← Back to Table of Contents](./README.md)

# Chapter 6 — OOP & Functional Design

> "The goal of software architecture is to minimize the human resources required to build and maintain the required system." — Robert C. Martin

Modern ML systems blend object-oriented and functional paradigms. Training pipelines compose objects; data transforms chain pure functions. Knowing when to reach for each style — and how to combine them — separates brittle notebooks from production-grade ML code.

---

## 6.1 SOLID Principles in ML

The SOLID principles were coined for enterprise Java, but they apply directly to ML codebases. Violating them is the fastest way to create training scripts that nobody can modify six months later.

### Single Responsibility Principle (SRP)

Each class should have **one reason to change**.

```python
# ❌ BAD — This class does too much
class ModelPipeline:
    def load_data(self, path: str) -> pd.DataFrame: ...
    def preprocess(self, df: pd.DataFrame) -> np.ndarray: ...
    def train(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict: ...
    def save_to_s3(self, bucket: str) -> None: ...
    def send_slack_notification(self, message: str) -> None: ...

# ✅ GOOD — Separate responsibilities
class DataLoader:
    def load(self, path: str) -> pd.DataFrame: ...

class Preprocessor:
    def transform(self, df: pd.DataFrame) -> np.ndarray: ...

class Trainer:
    def train(self, X: np.ndarray, y: np.ndarray) -> Model: ...

class Evaluator:
    def evaluate(self, model: Model, X: np.ndarray, y: np.ndarray) -> dict: ...
```

### Open/Closed Principle (OCP)

Classes should be **open for extension, closed for modification**.

```python
from abc import ABC, abstractmethod

class AugmentationStrategy(ABC):
    @abstractmethod
    def augment(self, image: np.ndarray) -> np.ndarray: ...

class RandomFlip(AugmentationStrategy):
    def augment(self, image: np.ndarray) -> np.ndarray:
        return np.fliplr(image) if random.random() > 0.5 else image

class GaussianNoise(AugmentationStrategy):
    def __init__(self, std: float = 0.01):
        self.std = std

    def augment(self, image: np.ndarray) -> np.ndarray:
        return image + np.random.normal(0, self.std, image.shape)

# Adding new augmentations never requires modifying existing code
class CutoutAugmentation(AugmentationStrategy):
    def augment(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = random.randint(0, w), random.randint(0, h)
        image[cy-8:cy+8, cx-8:cx+8] = 0
        return image
```

### Dependency Inversion Principle (DIP)

High-level modules should not depend on low-level modules; both should depend on **abstractions**.

```python
from abc import ABC, abstractmethod

# Abstraction
class ExperimentTracker(ABC):
    @abstractmethod
    def log_metric(self, key: str, value: float, step: int) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str) -> None: ...

# Low-level implementations
class WandbTracker(ExperimentTracker):
    def log_metric(self, key, value, step):
        wandb.log({key: value}, step=step)

    def log_artifact(self, path):
        wandb.save(path)

class MLflowTracker(ExperimentTracker):
    def log_metric(self, key, value, step):
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path):
        mlflow.log_artifact(path)

# High-level module depends only on the abstraction
class TrainingLoop:
    def __init__(self, model, tracker: ExperimentTracker):
        self.model = model
        self.tracker = tracker

    def train(self, dataloader, epochs: int):
        for epoch in range(epochs):
            loss = self._train_epoch(dataloader)
            self.tracker.log_metric("loss", loss, step=epoch)
```

<div class="diagram">
<div class="diagram-title">SOLID Principles — ML Quick Reference</div>
<div class="diagram-grid">
<div class="diagram-card accent">
<strong>S — Single Responsibility</strong><br>
DataLoader loads data.<br>
Trainer trains models.<br>
Evaluator evaluates.
</div>
<div class="diagram-card green">
<strong>O — Open/Closed</strong><br>
Add augmentations via<br>
new classes, not by editing<br>
existing transform code.
</div>
<div class="diagram-card blue">
<strong>L — Liskov Substitution</strong><br>
Any DataLoader subclass<br>
must work wherever the<br>
base class is expected.
</div>
<div class="diagram-card purple">
<strong>I — Interface Segregation</strong><br>
Don't force a batch predictor<br>
to implement a streaming<br>
interface it doesn't need.
</div>
<div class="diagram-card orange">
<strong>D — Dependency Inversion</strong><br>
TrainingLoop depends on<br>
ExperimentTracker ABC,<br>
not on wandb directly.
</div>
</div>
</div>

---

## 6.2 Composition over Inheritance

Deep inheritance hierarchies are the bane of ML codebases. When `ResNetWithAttentionAndDropoutV2` inherits from `ResNetWithAttention` which inherits from `ResNet` which inherits from `BaseModel`, debugging is a nightmare.

**Prefer composition**: assemble behavior from independent components.

```python
# ❌ BAD — Deep inheritance
class BaseModel(nn.Module): ...
class ResNet(BaseModel): ...
class ResNetWithAttention(ResNet): ...
class ResNetWithAttentionAndDropout(ResNetWithAttention): ...

# ✅ GOOD — Composition
class ModelBuilder:
    def __init__(self):
        self.backbone: nn.Module | None = None
        self.attention: nn.Module | None = None
        self.head: nn.Module | None = None
        self.dropout: float = 0.0

    def with_backbone(self, backbone: nn.Module) -> "ModelBuilder":
        self.backbone = backbone
        return self

    def with_attention(self, attention: nn.Module) -> "ModelBuilder":
        self.attention = attention
        return self

    def with_head(self, head: nn.Module) -> "ModelBuilder":
        self.head = head
        return self

    def build(self) -> nn.Module:
        layers = [self.backbone]
        if self.attention:
            layers.append(self.attention)
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(self.head)
        return nn.Sequential(*layers)

# Usage
model = (
    ModelBuilder()
    .with_backbone(ResNet50Backbone())
    .with_attention(SelfAttention(dim=2048))
    .with_head(ClassificationHead(2048, num_classes=1000))
    .build()
)
```

<div class="diagram">
<div class="diagram-title">Inheritance vs Composition</div>
<div class="compare">
<div class="compare-side red">
<strong>❌ Inheritance Hierarchy</strong><br><br>
BaseModel<br>
└── ResNet<br>
&nbsp;&nbsp;&nbsp;&nbsp;└── ResNetAttention<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── ResNetAttDropout<br><br>
• Rigid, hard to mix features<br>
• Fragile base class problem<br>
• Tight coupling
</div>
<div class="compare-side green">
<strong>✅ Composition</strong><br><br>
Model = Backbone + Attention + Head<br><br>
• Swap any component<br>
• Mix features freely<br>
• Test components in isolation<br>
• Easy to add new pieces
</div>
</div>
</div>

---

## 6.3 Dataclasses and Pydantic for Configuration

Hyperparameters stored in raw dicts are a leading cause of silent training bugs. **Typed configuration objects** catch errors before GPU hours are wasted.

### Dataclasses for Internal Config

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    tags: list[str] = field(default_factory=list)

config = TrainingConfig(learning_rate=1e-4, batch_size=64)
```

### Pydantic for Validated External Config

```python
from pydantic import BaseModel, Field, field_validator

class DataConfig(BaseModel):
    train_path: str
    val_path: str
    max_seq_length: int = Field(ge=1, le=8192, default=512)
    num_workers: int = Field(ge=0, le=32, default=4)
    tokenizer: str = "tiktoken"

    @field_validator("train_path", "val_path")
    @classmethod
    def path_must_exist(cls, v: str) -> str:
        from pathlib import Path
        if not Path(v).exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

class ExperimentConfig(BaseModel):
    training: TrainingConfig
    data: DataConfig
    experiment_name: str
    run_id: str | None = None

# Load from YAML or JSON with full validation
import yaml

with open("config.yaml") as f:
    raw = yaml.safe_load(f)

config = ExperimentConfig(**raw)  # Validates all fields
```

| Feature | `dataclass` | `Pydantic BaseModel` |
|---|---|---|
| Validation | Manual | Built-in validators |
| Serialization | Manual / `asdict()` | `.model_dump()` / `.model_dump_json()` |
| Immutability | `frozen=True` | `model_config = ConfigDict(frozen=True)` |
| Performance | Faster (no validation) | Slightly slower |
| Best for | Internal config, simple structs | API boundaries, user-facing config |

---

## 6.4 Abstract Base Classes

ABCs define **contracts** that subclasses must fulfill. They're the backbone of plugin architectures in ML frameworks.

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseDataset(ABC):
    """Contract for all datasets in our training framework."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @abstractmethod
    def collate_fn(self, batch: list[dict]) -> dict[str, Any]: ...

    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        from torch.utils.data import DataLoader
        return DataLoader(
            self, batch_size=batch_size,
            shuffle=shuffle, collate_fn=self.collate_fn,
        )

class TextClassificationDataset(BaseDataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tokens = self.tokenizer.encode(self.texts[idx])
        return {"input_ids": tokens, "label": self.labels[idx]}

    def collate_fn(self, batch):
        # Pad sequences to max length in batch
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = [item["input_ids"] + [0] * (max_len - len(item["input_ids"])) for item in batch]
        labels = [item["label"] for item in batch]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}
```

---

## 6.5 Functional Patterns for Data Processing

Functional programming shines in **data pipelines**: transforms are pure, composable, and easy to test.

### Pure Functions for Transforms

```python
# Pure functions — no side effects, same input → same output
def normalize(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / std

def tokenize(text: str, max_length: int = 512) -> list[int]:
    tokens = encoder.encode(text)
    return tokens[:max_length]

def compose(*fns):
    """Compose functions left to right: compose(f, g, h)(x) = h(g(f(x)))"""
    def composed(x):
        for fn in fns:
            x = fn(x)
        return x
    return composed

# Build pipeline from pure functions
preprocess = compose(
    str.lower,
    str.strip,
    lambda text: re.sub(r"[^\w\s]", "", text),
    lambda text: tokenize(text, max_length=256),
)

tokens = preprocess("  Hello, World!  ")
```

### Map / Filter / Reduce for Data Processing

```python
from functools import reduce

# Filter bad samples
clean_samples = list(filter(
    lambda s: len(s["text"]) > 10 and s["label"] is not None,
    raw_dataset,
))

# Map transform across dataset
processed = list(map(
    lambda s: {**s, "tokens": tokenize(s["text"])},
    clean_samples,
))

# Reduce to compute corpus statistics
total_tokens = reduce(
    lambda acc, s: acc + len(s["tokens"]),
    processed,
    0,
)
```

<div class="diagram">
<div class="diagram-title">Functional Data Pipeline</div>
<div class="flow-h">
<div class="flow-node blue">Raw Data</div>
<div class="flow-arrow">→</div>
<div class="flow-node green">filter(quality_check)</div>
<div class="flow-arrow">→</div>
<div class="flow-node green">map(tokenize)</div>
<div class="flow-arrow">→</div>
<div class="flow-node green">map(normalize)</div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">batch(collate)</div>
<div class="flow-arrow">→</div>
<div class="flow-node accent">DataLoader</div>
</div>
</div>

---

## 6.6 Closures and Decorators

Decorators are the Swiss Army knife of Python ML code: timing, retries, caching, and logging — all without modifying the decorated function.

### Timing Decorator

```python
import time
import functools
import logging

logger = logging.getLogger(__name__)

def timed(fn):
    """Log execution time of a function."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{fn.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

@timed
def train_epoch(model, dataloader, optimizer):
    ...
```

### Retry with Exponential Backoff

```python
import random

def retry(max_retries: int = 3, base_delay: float = 1.0, jitter: bool = True):
    """Retry decorator with exponential backoff — essential for API calls."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)
                    logger.warning(f"{fn.__name__} failed (attempt {attempt+1}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_retries=5, base_delay=2.0)
def download_model_weights(url: str, dest: str):
    """Download from a flaky model hub."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    Path(dest).write_bytes(response.content)
```

### LRU Cache for Expensive Computations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_tokenizer(model_name: str):
    """Cache tokenizer loading — only load once per model name."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)

# First call loads from disk/network, subsequent calls are instant
tok = load_tokenizer("meta-llama/Llama-3-8b")  # slow
tok = load_tokenizer("meta-llama/Llama-3-8b")  # instant (cached)
```

---

## 6.7 When to Use OOP vs Functional

There is no universal answer — the best ML code **blends both styles**.

| Aspect | OOP | Functional |
|---|---|---|
| **State management** | ✅ Models, optimizers, trackers | ❌ Avoid mutable state |
| **Data transforms** | ❌ Overkill | ✅ Pure, composable functions |
| **Plugin systems** | ✅ ABCs + subclasses | ❌ Awkward |
| **Config management** | ✅ Pydantic / dataclasses | ❌ Dicts are fragile |
| **Testing** | Needs mocking for deps | ✅ Pure functions are trivial to test |
| **Parallelism** | Harder (shared state) | ✅ No side effects = safe parallelism |

<div class="diagram">
<div class="diagram-title">OOP vs Functional — Where Each Shines</div>
<div class="compare">
<div class="compare-side blue">
<strong>OOP — Stateful Components</strong><br><br>
• nn.Module subclasses<br>
• Experiment trackers<br>
• Data loaders with state<br>
• Configuration objects<br>
• Plugin / strategy patterns<br>
• Anything with lifecycle
</div>
<div class="compare-side green">
<strong>Functional — Stateless Transforms</strong><br><br>
• Data preprocessing<br>
• Feature engineering<br>
• Tokenization pipelines<br>
• Metric computation<br>
• Augmentation chains<br>
• Anything that's pure
</div>
</div>
</div>

---

## 6.8 Putting It All Together

A well-structured ML project uses both paradigms:

```python
# config.py — OOP (Pydantic for validation)
class TrainConfig(BaseModel):
    lr: float = Field(ge=1e-6, le=1.0, default=3e-4)
    batch_size: int = 32

# transforms.py — Functional (pure composable functions)
def normalize(x, mean, std):
    return (x - mean) / std

pipeline = compose(tokenize, pad_sequence, normalize)

# model.py — OOP (stateful nn.Module)
class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

# train.py — Mix of both
@timed
@retry(max_retries=3)
def train(config: TrainConfig):                    # Functional decorator
    model = TransformerLM(config.model)             # OOP
    dataset = load_and_preprocess(config.data)      # Functional pipeline
    tracker = WandbTracker(config.experiment_name)  # OOP (DIP)
    loop = TrainingLoop(model, tracker)             # OOP (composition)
    loop.run(dataset, config.max_epochs)
```

> **Rule of thumb**: Use OOP for things that *have state and identity* (models, trackers, loaders). Use functional for things that *transform data* (preprocessing, augmentation, metrics).

---

*Last updated: April 2026*
