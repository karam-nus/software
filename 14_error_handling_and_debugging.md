[← Back to Table of Contents](./README.md)

# Chapter 14 — Error Handling & Debugging

> "Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it." — Brian Kernighan

ML code is uniquely challenging to debug. Bugs often manifest as degraded model performance rather than crashes — a subtle data leak, a transposed dimension, or an incorrect loss function can silently produce a model that trains but performs terribly. This chapter covers systematic approaches to error handling and debugging that make ML code more robust and diagnosable.

---

## 14.1 Exception Handling in Python

### The Basics: try/except/finally

```python
import torch
import json

def load_checkpoint(path: str) -> dict:
    """Load a model checkpoint with proper error handling."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        logger.info("Loaded checkpoint from %s (step %d)", path, checkpoint["step"])
        return checkpoint
    except FileNotFoundError:
        logger.error("Checkpoint not found: %s", path)
        raise
    except (torch.serialization.StorageError, RuntimeError) as e:
        logger.error("Corrupt checkpoint at %s: %s", path, e)
        raise CorruptCheckpointError(path) from e
    finally:
        # Always runs — good for cleanup
        logger.debug("Checkpoint load attempt finished for %s", path)


def load_config(path: str) -> dict:
    """Load YAML config with specific error handling."""
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigNotFoundError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML in {path}: {e}")

    if not isinstance(config, dict):
        raise ConfigParseError(f"Expected dict, got {type(config).__name__}")

    return config
```

### Exception Handling Anti-Patterns

```python
# BAD: Bare except catches everything including KeyboardInterrupt
try:
    train(model)
except:
    pass

# BAD: Catching too broadly
try:
    result = model(input_tensor)
except Exception:
    result = None  # Silences real bugs!

# BAD: Using exceptions for control flow
try:
    value = config["learning_rate"]
except KeyError:
    value = 3e-4

# GOOD: Use .get() for dictionaries
value = config.get("learning_rate", 3e-4)

# GOOD: Catch specific exceptions, let unexpected ones propagate
try:
    result = model(input_tensor)
except torch.cuda.OutOfMemoryError:
    logger.warning("OOM — reducing batch size and retrying")
    torch.cuda.empty_cache()
    result = model(input_tensor[:len(input_tensor) // 2])
```

---

## 14.2 Custom Exception Hierarchies for ML

```python
class MLPipelineError(Exception):
    """Base exception for all ML pipeline errors."""
    pass


class DataError(MLPipelineError):
    """Errors related to data loading and processing."""
    pass

class DataNotFoundError(DataError):
    """Dataset file or directory not found."""
    pass

class DataCorruptionError(DataError):
    """Data file is corrupted or has unexpected format."""
    pass

class DataValidationError(DataError):
    """Data fails validation checks."""
    def __init__(self, message: str, invalid_samples: list[int] | None = None):
        super().__init__(message)
        self.invalid_samples = invalid_samples or []


class ModelError(MLPipelineError):
    """Errors related to model building and inference."""
    pass

class ShapeMismatchError(ModelError):
    """Tensor shape doesn't match expected dimensions."""
    def __init__(self, expected: tuple, actual: tuple, tensor_name: str = ""):
        self.expected = expected
        self.actual = actual
        self.tensor_name = tensor_name
        msg = f"Shape mismatch for '{tensor_name}': expected {expected}, got {actual}"
        super().__init__(msg)

class CorruptCheckpointError(ModelError):
    """Checkpoint file is corrupted or incompatible."""
    pass


class TrainingError(MLPipelineError):
    """Errors during training."""
    pass

class NaNLossError(TrainingError):
    """Loss became NaN during training."""
    def __init__(self, step: int, last_valid_loss: float | None = None):
        self.step = step
        self.last_valid_loss = last_valid_loss
        msg = f"NaN loss at step {step}"
        if last_valid_loss is not None:
            msg += f" (last valid: {last_valid_loss:.4f})"
        super().__init__(msg)

class GradientExplosionError(TrainingError):
    """Gradient norm exceeded threshold."""
    pass
```

<div class="diagram">
  <div class="diagram-title">ML Exception Hierarchy</div>
  <div class="layer-stack">
    <div class="layer purple">MLPipelineError (base)</div>
    <div class="layer blue">├── DataError</div>
    <div class="layer cyan">│   ├── DataNotFoundError</div>
    <div class="layer cyan">│   ├── DataCorruptionError</div>
    <div class="layer cyan">│   └── DataValidationError</div>
    <div class="layer green">├── ModelError</div>
    <div class="layer teal">│   ├── ShapeMismatchError</div>
    <div class="layer teal">│   └── CorruptCheckpointError</div>
    <div class="layer orange">└── TrainingError</div>
    <div class="layer yellow">    ├── NaNLossError</div>
    <div class="layer yellow">    └── GradientExplosionError</div>
  </div>
</div>

---

## 14.3 Defensive Programming

### Assertions and Preconditions

```python
import torch
import numpy as np

def train_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Single training step with defensive checks."""
    # Preconditions
    assert model.training, "Model must be in training mode"
    assert "input_ids" in batch, f"Missing 'input_ids' in batch. Keys: {batch.keys()}"
    assert "labels" in batch, f"Missing 'labels' in batch. Keys: {batch.keys()}"

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Shape checks
    assert input_ids.dim() == 2, f"Expected 2D input_ids, got {input_ids.dim()}D"
    assert input_ids.shape == labels.shape, (
        f"Shape mismatch: input_ids {input_ids.shape} vs labels {labels.shape}"
    )

    # Value checks
    assert input_ids.dtype == torch.long, f"Expected long dtype, got {input_ids.dtype}"
    assert not torch.isnan(input_ids.float()).any(), "NaN in input_ids"

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    # Post-condition: loss should be valid
    if torch.isnan(loss):
        raise NaNLossError(step=global_step, last_valid_loss=last_loss)
    if torch.isinf(loss):
        raise TrainingError(f"Infinite loss at step {global_step}: {loss.item()}")

    loss.backward()

    # Check gradient norms before clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if grad_norm > 100:
        logger.warning("Large gradient norm: %.2f at step %d", grad_norm, global_step)

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

### LBYL vs EAFP

<div class="diagram">
  <div class="diagram-title">LBYL vs EAFP in Python</div>
  <div class="compare">
    <div class="compare-side blue">
      <strong>LBYL: Look Before You Leap</strong><br/>
      Check conditions before acting<br/>
      ✓ Explicit intent<br/>
      ✓ Avoids exception overhead<br/>
      ✗ Race conditions possible<br/>
      ✗ Verbose
    </div>
    <div class="compare-side green">
      <strong>EAFP: Easier to Ask Forgiveness</strong><br/>
      Try it, handle failure<br/>
      ✓ Pythonic<br/>
      ✓ Atomically correct<br/>
      ✓ Concise<br/>
      ✗ Can mask bugs
    </div>
  </div>
</div>

```python
# LBYL style
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# EAFP style
try:
    model.gradient_checkpointing_enable()
except AttributeError:
    logger.warning("Model doesn't support gradient checkpointing")

# For ML: prefer LBYL for shape/type checks, EAFP for I/O and optional features
```

---

## 14.4 Debugging Tools

### pdb and ipdb

```python
# Insert a breakpoint anywhere
def forward(self, x):
    hidden = self.encoder(x)
    breakpoint()  # Python 3.7+ built-in — drops into pdb
    output = self.decoder(hidden)
    return output

# Or use ipdb for a better experience
import ipdb; ipdb.set_trace()
```

```
# Common pdb commands
(Pdb) n          # Next line (step over)
(Pdb) s          # Step into function
(Pdb) c          # Continue execution
(Pdb) p x.shape  # Print expression
(Pdb) pp vars()  # Pretty-print variables
(Pdb) l          # List source code
(Pdb) w          # Show call stack
(Pdb) u          # Go up one frame
(Pdb) d          # Go down one frame
(Pdb) b 42       # Set breakpoint at line 42
(Pdb) condition 1 loss > 10  # Conditional breakpoint
```

### Post-Mortem Debugging

```python
# Automatically drop into debugger on unhandled exception
import sys

def excepthook(type, value, tb):
    if type != KeyboardInterrupt:
        import ipdb
        ipdb.post_mortem(tb)

sys.excepthook = excepthook

# Or run your script with:
# python -m pdb -c continue train.py
# This drops into pdb when an exception occurs
```

### Debugging in VSCode

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Model",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/src/train.py",
            "args": ["--config", "configs/debug.yaml"],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Attach to Running Process",
            "type": "debugpy",
            "request": "attach",
            "connect": { "host": "localhost", "port": 5678 }
        }
    ]
}
```

### Remote Debugging

```python
# Add to the start of your training script on a remote machine
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached!")
```

### Debugging Tool Comparison

| Tool | Best For | Interactive | Remote |
|---|---|---|---|
| `pdb` | Quick inspection, always available | Yes | No |
| `ipdb` | Better UX (tab completion, colors) | Yes | No |
| `breakpoint()` | IDE-agnostic breakpoints | Yes | No |
| VSCode debugger | Visual debugging, watches, call stack | Yes | Yes |
| `debugpy` | Remote debugging (SSH, containers) | Yes | Yes |
| `torch.autograd.set_detect_anomaly` | NaN/Inf in gradients | No | No |

---

## 14.5 Profiling

### cProfile for Function-Level Profiling

```python
import cProfile
import pstats

# Profile a training function
profiler = cProfile.Profile()
profiler.enable()

train_one_epoch(model, dataloader, optimizer)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)  # Top 20 functions by cumulative time
```

```bash
# Profile from command line
python -m cProfile -o profile.prof src/train.py
python -m pstats profile.prof
# In pstats: sort cumulative, stats 20
```

### line_profiler for Line-Level Profiling

```python
# Install: pip install line_profiler

# Decorate the function to profile
@profile  # Special decorator recognized by kernprof
def train_step(model, batch, optimizer):
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

```bash
# Run with kernprof
kernprof -l -v src/train.py
```

### memory_profiler for Memory Tracking

```python
from memory_profiler import profile

@profile
def load_dataset(path: str):
    """Profile memory usage of data loading."""
    import pandas as pd
    df = pd.read_parquet(path)           # Watch memory spike here
    features = df.select_dtypes("float")
    normalized = (features - features.mean()) / features.std()
    return normalized.values
```

### PyTorch Profiler

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        train_step(model, batch, optimizer)
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 14.6 Common ML Bugs

<div class="diagram">
  <div class="diagram-title">Common ML Bug Categories</div>
  <div class="diagram-grid">
    <div class="diagram-card red">
      <strong>Shape Mismatches</strong><br/>
      Transposed dimensions, wrong batch dim, broadcasting surprises
    </div>
    <div class="diagram-card orange">
      <strong>NaN / Inf Gradients</strong><br/>
      Learning rate too high, log(0), division by zero
    </div>
    <div class="diagram-card yellow">
      <strong>Data Leakage</strong><br/>
      Test data in training, future data in features
    </div>
    <div class="diagram-card green">
      <strong>Off-by-One</strong><br/>
      Wrong sequence length, batch indexing, padding
    </div>
    <div class="diagram-card blue">
      <strong>Device Mismatches</strong><br/>
      Some tensors on CPU, others on GPU
    </div>
    <div class="diagram-card purple">
      <strong>Silent Failures</strong><br/>
      Model in eval mode during training, frozen layers, wrong loss
    </div>
  </div>
</div>

### Shape Mismatches

```python
# BUG: Broadcasting hides shape mismatch
logits = model(x)       # Shape: [batch, seq_len, vocab_size]
labels = batch["labels"] # Shape: [batch, seq_len]

# This silently broadcasts — wrong!
# loss = (logits - labels).mean()

# CORRECT: Reshape for cross-entropy
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),  # [batch * seq_len, vocab_size]
    labels.view(-1),                     # [batch * seq_len]
)

# DEFENSIVE: Always check shapes explicitly
def safe_cross_entropy(logits, labels):
    assert logits.dim() == 3, f"Expected 3D logits, got {logits.dim()}D: {logits.shape}"
    assert labels.dim() == 2, f"Expected 2D labels, got {labels.dim()}D: {labels.shape}"
    assert logits.shape[:2] == labels.shape, (
        f"Batch/seq mismatch: logits {logits.shape[:2]} vs labels {labels.shape}"
    )
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
```

### NaN Gradient Detection

```python
# Enable anomaly detection during debugging
torch.autograd.set_detect_anomaly(True)

# Custom NaN checker
def check_for_nan(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                raise NaNLossError(
                    step=step,
                    last_valid_loss=None,
                )
            if torch.isinf(param.grad).any():
                raise GradientExplosionError(
                    f"Inf gradient in {name} at step {step}"
                )
```

### Data Leakage Detection

```python
def check_no_data_leakage(train_df, val_df, test_df, id_column="sample_id"):
    """Verify no overlap between data splits."""
    train_ids = set(train_df[id_column])
    val_ids = set(val_df[id_column])
    test_ids = set(test_df[id_column])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap:
        raise DataValidationError(
            f"{len(train_val_overlap)} samples in both train and val",
            invalid_samples=list(train_val_overlap)[:10],
        )
    if train_test_overlap:
        raise DataValidationError(
            f"{len(train_test_overlap)} samples in both train and test",
            invalid_samples=list(train_test_overlap)[:10],
        )

    logger.info(
        "No data leakage detected. Train: %d, Val: %d, Test: %d",
        len(train_ids), len(val_ids), len(test_ids),
    )
```

---

## 14.7 Error Handling Strategy

<div class="diagram">
  <div class="diagram-title">Error Handling Decision Flow</div>
  <div class="flow">
    <div class="flow-node blue">Error Occurs</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Can we recover? (retry, fallback, skip)</div>
    <div class="flow-arrow">→ Yes →</div>
    <div class="flow-node teal">Log warning, apply recovery strategy</div>
    <div class="flow-arrow">→ No →</div>
    <div class="flow-node orange">Is it a known failure mode?</div>
    <div class="flow-arrow">→ Yes →</div>
    <div class="flow-node accent">Raise specific custom exception</div>
    <div class="flow-arrow">→ No →</div>
    <div class="flow-node red">Let it propagate, log full context</div>
  </div>
</div>

### Retry with Exponential Backoff

```python
import time
import random

def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """Retry a function with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except exceptions as e:
            if attempt == max_retries:
                logger.error("All %d retries failed: %s", max_retries, e)
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            logger.warning(
                "Attempt %d/%d failed (%s), retrying in %.1fs",
                attempt + 1, max_retries, e, delay + jitter,
            )
            time.sleep(delay + jitter)


# Usage: retry flaky operations
checkpoint = retry_with_backoff(
    lambda: torch.load(checkpoint_url),
    max_retries=3,
    exceptions=(ConnectionError, TimeoutError),
)
```

---

## 14.8 Key Takeaways

1. **Catch specific exceptions** — never use bare `except:` or `except Exception`.
2. **Build a custom exception hierarchy** that maps to your ML pipeline stages.
3. **Use assertions liberally** for shape checks, dtype checks, and value range checks.
4. **`breakpoint()` is your friend** — use it to inspect tensors, gradients, and intermediate values.
5. **Profile before optimizing** — use cProfile for bottlenecks, line_profiler for hot loops, PyTorch profiler for GPU.
6. **Know the common ML bugs** — shape mismatches, NaN gradients, data leakage, and device mismatches.
7. **Enable `torch.autograd.set_detect_anomaly(True)`** during debugging to find NaN sources.
8. **Log full context on errors** — the step number, batch contents, and config are all essential for diagnosis.

---

*Last updated: April 2026*
