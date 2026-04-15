[← Back to Table of Contents](./README.md)

# Chapter 3 — Code Quality & Style

> *"Any fool can write code that a computer can understand. Good programmers write code that humans can understand."* — Martin Fowler

Code quality is not vanity — it directly impacts the velocity, correctness, and safety of ML systems. Research code that "just works on my machine" becomes a liability when it needs to be reproduced, debugged by a teammate, or deployed to production. This chapter covers the tools and practices that keep ML codebases clean, consistent, and maintainable.

---

## 3.1 Why Style Matters in ML

ML codebases are especially vulnerable to quality issues:

- **Notebook-to-production gap** — quick experiments become production code without cleanup
- **Complex numerics** — subtle bugs in tensor operations hide behind "it still trains"
- **Long feedback loops** — a bug discovered after 8 hours of training is expensive
- **Team diversity** — data scientists, ML engineers, and platform engineers all contribute

| Cost of Bug Found At | Relative Cost |
|---|---|
| Code review | 1× |
| Unit test | 3× |
| Integration test | 10× |
| Training run (8h on 8×A100) | 100× |
| Production (serving wrong predictions) | 1000× |

> *"The best time to catch a bug is before you write it. The second best time is at the linter."*

---

## 3.2 PEP 8 Essentials

PEP 8 is Python's official style guide. Key rules for ML code:

```python
# ✅ Good — clear, PEP 8 compliant ML code
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LEARNING_RATE = 3e-4          # Constants in UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH = 2048
DEFAULT_BATCH_SIZE = 32


class TransformerEncoder(nn.Module):    # Classes in PascalCase
    """Multi-layer transformer encoder with rotary embeddings."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input through transformer layers.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            attention_mask: Optional mask of shape (batch, seq_len).

        Returns:
            Encoded tensor of shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:                      # Functions in snake_case
    """Compute cross-entropy loss, ignoring padding tokens."""
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
```

```python
# ❌ Bad — common anti-patterns in ML code
import torch; import torch.nn as nn  # Multiple imports on one line
from torch.utils.data import *       # Wildcard imports

lr = 3e-4                            # Ambiguous abbreviation
MSL = 2048                           # Cryptic constant name

class transformer_encoder(nn.Module): # Wrong casing
    def __init__(self, d, h, l, p):   # Single-letter params without context
        super().__init__()
        self.l = nn.ModuleList([TransformerBlock(d,h,p) for _ in range(l)])

    def forward(self,x,m=None):       # Missing spaces, no type hints
        for l in self.l: x=l(x,attention_mask=m)  # One-liner loop
        return x
```

---

## 3.3 The Modern Python Toolchain

<div class="diagram">
<div class="diagram-title">Python Code Quality Toolchain</div>
<div class="flow">
<div class="flow-node accent">Ruff<br/><small>Linting + import sorting</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node green">Black<br/><small>Code formatting</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node blue">mypy<br/><small>Type checking</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">pre-commit<br/><small>Git hook automation</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node orange">CI Pipeline<br/><small>Enforce on every PR</small></div>
</div>
</div>

### Black — The Uncompromising Formatter

Black enforces a single, deterministic style. No debates about formatting — ever.

```bash
# Install and run
pip install black
black src/ tests/               # Format all Python files
black --check src/              # Check without modifying (for CI)
black --diff src/model.py       # Preview changes
```

**Before Black:**
```python
def create_optimizer(model,lr=3e-4,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8,use_8bit=False):
    if use_8bit:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(),lr=lr,weight_decay=weight_decay,betas=betas,eps=eps)
    else:
        return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay,betas=betas,eps=eps)
```

**After Black:**
```python
def create_optimizer(
    model,
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    use_8bit=False,
):
    if use_8bit:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    else:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
```

### Ruff — The Fast Linter

Ruff replaces Flake8, isort, pyflakes, and dozens of other tools — and runs 10–100× faster.

```bash
# Install and run
pip install ruff
ruff check src/                 # Lint
ruff check --fix src/           # Lint and auto-fix
ruff format src/                # Format (Black-compatible)
```

### mypy — Static Type Checking

Type hints catch bugs before runtime — especially valuable for complex tensor operations.

```python
# Type hints for ML code
import torch
from torch import Tensor
from typing import TypeAlias

# Type aliases for clarity
BatchedTensor: TypeAlias = Tensor   # shape: (batch, ...)
LogitsTensor: TypeAlias = Tensor    # shape: (batch, seq_len, vocab_size)
LossDict: TypeAlias = dict[str, Tensor]


def training_step(
    model: torch.nn.Module,
    batch: dict[str, BatchedTensor],
    device: torch.device,
) -> LossDict:
    """Execute a single training step."""
    input_ids: BatchedTensor = batch["input_ids"].to(device)
    labels: BatchedTensor = batch["labels"].to(device)
    attention_mask: BatchedTensor = batch["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    return {
        "loss": outputs.loss,
        "logits": outputs.logits,
    }
```

```bash
mypy src/ --strict               # Run with strict mode
mypy src/ --ignore-missing-imports  # Ignore untyped third-party libs
```

---

## 3.4 Docstring Conventions

<div class="diagram">
<div class="diagram-title">Docstring Styles Compared</div>
<div class="compare">
<div class="compare-side">
<h4>Google Style (Recommended)</h4>
<pre>
def train(
    model: nn.Module,
    epochs: int,
    lr: float = 3e-4,
) -> dict[str, float]:
    """Train model for N epochs.

    Args:
        model: The neural network.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Dictionary with final metrics.

    Raises:
        ValueError: If epochs < 1.
    """
</pre>
</div>
<div class="compare-side">
<h4>NumPy Style</h4>
<pre>
def train(
    model: nn.Module,
    epochs: int,
    lr: float = 3e-4,
) -> dict[str, float]:
    """Train model for N epochs.

    Parameters
    ----------
    model : nn.Module
        The neural network.
    epochs : int
        Number of training epochs.
    lr : float, optional
        Learning rate (default 3e-4).

    Returns
    -------
    dict[str, float]
        Dictionary with final metrics.
    """
</pre>
</div>
</div>
</div>

> **Team recommendation:** Use Google style for ML projects. It's more compact, widely used in the PyTorch/JAX ecosystem, and supported by all major documentation tools.

---

## 3.5 pyproject.toml — Unified Configuration

Modern Python projects configure all tools in a single `pyproject.toml`:

```toml
# pyproject.toml — Complete ML project configuration

[project]
name = "ml-training-pipeline"
version = "0.4.0"
description = "Production training pipeline for transformer models"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3.0",
    "transformers>=4.40.0",
    "wandb>=0.17.0",
    "safetensors>=0.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4.0",
    "mypy>=1.10",
    "pre-commit>=3.7",
]

# --- Black ---
[tool.black]
line-length = 99
target-version = ["py311"]

# --- Ruff ---
[tool.ruff]
line-length = 99
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "W",     # pycodestyle warnings
    "F",     # pyflakes
    "I",     # isort
    "N",     # pep8-naming
    "UP",    # pyupgrade
    "B",     # flake8-bugbear
    "SIM",   # flake8-simplify
    "TCH",   # flake8-type-checking
    "RUF",   # Ruff-specific rules
    "NPY",   # NumPy-specific rules
]
ignore = [
    "E501",  # Line too long (handled by Black)
    "B008",  # Function call in default argument (common in FastAPI)
]

[tool.ruff.lint.isort]
known-first-party = ["ml_pipeline"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]   # Allow assert in tests

# --- mypy ---
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = [
    "transformers.*",
    "wandb.*",
    "safetensors.*",
]
ignore_missing_imports = true

# --- pytest ---
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "integration: marks integration tests",
]
```

---

## 3.6 Pre-commit Hooks

Pre-commit hooks run quality checks before every commit, catching issues at the earliest possible moment.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=5000']    # Catch accidental large file commits
      - id: no-commit-to-branch
        args: ['--branch', 'main']  # Prevent direct commits to main
      - id: detect-private-key     # Never commit secrets

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies:
          - torch-stubs
          - types-PyYAML

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks    # Detect secrets and credentials
```

```bash
# Setup
pip install pre-commit
pre-commit install                   # Install hooks
pre-commit run --all-files           # Run on all files (first time)
pre-commit autoupdate                # Update hook versions
```

---

## 3.7 Code Complexity Metrics

Track complexity to identify code that's hard to maintain:

| Metric | What It Measures | Target |
|---|---|---|
| **Cyclomatic complexity** | Number of independent paths | ≤ 10 per function |
| **Cognitive complexity** | How hard code is to understand | ≤ 15 per function |
| **Lines per function** | Function length | ≤ 50 lines |
| **Function parameters** | Number of arguments | ≤ 5 (use dataclasses for more) |
| **Module coupling** | Dependencies between modules | Low (use dependency injection) |

```python
# ❌ High complexity — hard to test and maintain
def process_training_data(data, config, augment=True, balance=True,
                          normalize=True, tokenize=True, cache=True,
                          max_length=512, pad=True, truncate=True):
    if augment:
        if config.augment_type == "random_crop":
            data = random_crop(data, config.crop_size)
        elif config.augment_type == "mixup":
            data = mixup(data, config.alpha)
        elif config.augment_type == "cutout":
            data = cutout(data, config.cutout_size)
    if balance:
        if config.balance_strategy == "oversample":
            data = oversample(data)
        elif config.balance_strategy == "undersample":
            data = undersample(data)
    # ... 80 more lines of nested conditionals
    return data


# ✅ Low complexity — compose small, testable functions
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data processing pipeline."""
    augmentation: str | None = "random_crop"
    balance_strategy: str | None = "oversample"
    normalize: bool = True
    max_length: int = 512


def build_processing_pipeline(config: DataConfig) -> list[Callable]:
    """Build a composable data processing pipeline from config."""
    steps: list[Callable] = []

    if config.augmentation:
        steps.append(get_augmentation(config.augmentation))
    if config.balance_strategy:
        steps.append(get_balancer(config.balance_strategy))
    if config.normalize:
        steps.append(normalize_features)

    return steps


def apply_pipeline(data: Dataset, steps: list[Callable]) -> Dataset:
    """Apply a sequence of processing steps to a dataset."""
    for step in steps:
        data = step(data)
    return data
```

---

## 3.8 Notebook Hygiene

Jupyter notebooks are indispensable for exploration but dangerous for production. Follow these rules:

<div class="diagram">
<div class="diagram-title">Notebook → Production Pipeline</div>
<div class="flow">
<div class="flow-node green">Explore in Notebook<br/><small>Prototype, visualize</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node blue">Extract to .py Modules<br/><small>Functions, classes</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">Add Tests &amp; Types<br/><small>pytest, mypy</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node orange">Review &amp; Merge<br/><small>PR, CI/CD</small></div>
</div>
</div>

**Rules for notebooks in version control:**

1. **Clear outputs before committing** — use `nbstripout` as a pre-commit hook
2. **Number cells logically** — each cell should be runnable top-to-bottom
3. **Keep notebooks short** — < 50 cells, focused on one analysis
4. **Extract reusable code** — if you copy-paste between notebooks, make a module
5. **Use `nbqa`** — run Ruff and Black on notebook code cells

```yaml
# Add to .pre-commit-config.yaml
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout           # Strip notebook outputs before commit
```

---

## 3.9 Summary Checklist

| Practice | Tool | Automated? |
|---|---|---|
| Code formatting | Black / Ruff format | ✅ Pre-commit + CI |
| Linting | Ruff | ✅ Pre-commit + CI |
| Import sorting | Ruff (isort rules) | ✅ Pre-commit + CI |
| Type checking | mypy | ✅ Pre-commit + CI |
| Docstrings | Google style convention | 🔶 Manual review |
| Complexity | Ruff (McCabe rules) | ✅ CI |
| Secrets detection | gitleaks | ✅ Pre-commit |
| Large file prevention | check-added-large-files | ✅ Pre-commit |
| Notebook cleanup | nbstripout | ✅ Pre-commit |

> *"Automate everything you can. The goal is that by the time a human reviewer sees your code, all the mechanical issues are already resolved — they can focus on logic, design, and correctness."*

---

*Last updated: April 2026*
