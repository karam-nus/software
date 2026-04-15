[← Back to Table of Contents](./README.md)

# Chapter 17 — Performance & Profiling

> "Premature optimization is the root of all evil — but so is premature pessimism about performance." — Adapted from Donald Knuth

In ML engineering, performance bottlenecks lurk everywhere: slow data loading starves GPUs, unoptimized feature computation delays training, and naive serving code blows latency budgets. The key discipline is **measure first, optimize second**.

---

## 17.1 The Performance Mindset

Before optimizing anything, establish:

1. **A measurable goal** — "Inference latency < 50ms at p99"
2. **A baseline measurement** — "Current p99 is 320ms"
3. **A profile** — "85% of time is in feature preprocessing"

<div class="diagram">
  <div class="diagram-title">The Profiling Workflow</div>
  <div class="flow">
    <div class="flow-node accent">Define Performance Goal</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Measure Baseline</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node blue">Profile to Find Bottleneck</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">Optimize the Bottleneck</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Measure Again</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node teal">Goal Met?</div>
  </div>
</div>

> **Critical rule:** Never guess where the bottleneck is. Profile first. Engineers are notoriously bad at predicting performance bottlenecks — the actual hotspot is almost never where you think.

---

## 17.2 Python Profiling Tools

### cProfile — Built-in Function-Level Profiler

```python
import cProfile
import pstats

def train_epoch(model, dataloader):
    for batch in dataloader:
        loss = model.training_step(batch)
        loss.backward()
        model.optimizer.step()

# Profile a training epoch
profiler = cProfile.Profile()
profiler.enable()
train_epoch(model, train_loader)
profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)  # Top 20 functions by cumulative time
```

```bash
# Command-line profiling
python -m cProfile -s cumulative train.py | head -30
```

### py-spy — Sampling Profiler (No Code Changes)

```bash
# Profile a running training job without modifying code
pip install py-spy
py-spy top --pid $(pgrep -f "python train.py")

# Generate a flame graph
py-spy record -o profile.svg -- python train.py --epochs 5

# Profile only Python code (hide C extensions)
py-spy record --native -o profile.svg -- python train.py
```

### line_profiler — Line-by-Line Profiling

```python
# Install: pip install line_profiler

# Decorate functions to profile
@profile  # This decorator is recognized by kernprof
def preprocess_batch(raw_data):
    normalized = normalize_features(raw_data)       # Line 1
    encoded = one_hot_encode(normalized)             # Line 2
    tensor = torch.tensor(encoded, dtype=torch.float32)  # Line 3
    return tensor
```

```bash
# Run with kernprof
kernprof -l -v preprocess.py

# Output shows time per line:
# Line #  Hits   Time    Per Hit  % Time  Line Contents
#   3     1000   45.2ms  45.2us   12.1%  normalized = normalize_features(...)
#   4     1000  298.1ms  298.1us  79.8%  encoded = one_hot_encode(...)
#   5     1000   30.1ms  30.1us    8.1%  tensor = torch.tensor(...)
```

### memory_profiler — Track Memory Usage

```python
# Install: pip install memory_profiler
from memory_profiler import profile

@profile
def load_training_data(path: str):
    import pandas as pd
    df = pd.read_parquet(path)              # Memory spike here
    features = df.select_dtypes("float64")
    labels = df["target"].values
    return features.values, labels
```

```bash
# Run with memory tracking
python -m memory_profiler load_data.py

# Output:
# Line #    Mem usage    Increment   Line Contents
#   5       100.2 MiB    0.0 MiB    df = pd.read_parquet(path)
#   5       845.7 MiB  745.5 MiB    (after load)
#   6       845.7 MiB    0.0 MiB    features = df.select_dtypes(...)
#   7       923.1 MiB   77.4 MiB    labels = df["target"].values
```

| Tool | Type | Overhead | Best For |
|---|---|---|---|
| cProfile | Deterministic | Medium | Function-level hotspots |
| py-spy | Sampling | Very low | Production profiling, flame graphs |
| line_profiler | Deterministic | High | Line-level optimization |
| memory_profiler | Memory tracking | High | Finding memory leaks |
| `torch.profiler` | GPU + CPU | Medium | PyTorch training loops |
| Scalene | CPU + memory + GPU | Low | All-in-one Python profiling |

---

## 17.3 Understanding Python's GIL

The **Global Interpreter Lock (GIL)** prevents multiple threads from executing Python bytecode simultaneously. This is critical to understand for ML workloads:

```python
import threading
import time

def cpu_bound_task(n):
    """Simulates CPU-bound work — GIL will serialize this."""
    total = 0
    for i in range(n):
        total += i * i
    return total

# This will NOT be faster than sequential due to GIL
threads = [threading.Thread(target=cpu_bound_task, args=(10_000_000,))
           for _ in range(4)]
start = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"Threaded: {time.time() - start:.2f}s")  # ~Same as sequential!
```

**When threading DOES help:** I/O-bound operations (network calls, disk reads) release the GIL while waiting. NumPy and PyTorch also release the GIL during C/CUDA operations.

---

## 17.4 Concurrency for ML Workloads

<div class="diagram">
  <div class="diagram-title">Concurrency Approaches Compared</div>
  <div class="compare">
    <div class="compare-side">
      <h4>Threading</h4>
      <div class="flow-node green">Best for I/O-bound</div>
      <p>Data loading, API calls, file I/O. Shared memory. GIL limits CPU parallelism.</p>
    </div>
    <div class="compare-side">
      <h4>Multiprocessing</h4>
      <div class="flow-node blue">Best for CPU-bound</div>
      <p>Feature engineering, data transforms. Separate memory. True parallelism.</p>
    </div>
    <div class="compare-side">
      <h4>Asyncio</h4>
      <div class="flow-node purple">Best for many I/O ops</div>
      <p>Model serving, concurrent API calls. Single thread, cooperative multitasking.</p>
    </div>
  </div>
</div>

### Multiprocessing for Data Preprocessing

```python
from multiprocessing import Pool
import numpy as np

def compute_features(chunk: np.ndarray) -> np.ndarray:
    """CPU-intensive feature computation."""
    rolling_mean = np.convolve(chunk, np.ones(10)/10, mode="valid")
    rolling_std = np.array([chunk[i:i+10].std() for i in range(len(chunk)-9)])
    return np.column_stack([rolling_mean, rolling_std])

# Split data into chunks and process in parallel
data = np.random.randn(1_000_000)
chunks = np.array_split(data, 8)

with Pool(processes=8) as pool:
    results = pool.map(compute_features, chunks)

features = np.concatenate(results)
```

### Async Model Serving

```python
import asyncio
import aiohttp
from fastapi import FastAPI
import numpy as np

app = FastAPI()

async def fetch_features(session, user_id: str) -> dict:
    """Fetch features from feature store (I/O-bound)."""
    async with session.get(f"http://feature-store/features/{user_id}") as resp:
        return await resp.json()

@app.post("/predict/batch")
async def predict_batch(user_ids: list[str]):
    async with aiohttp.ClientSession() as session:
        # Fetch all features concurrently — 100x faster than sequential
        tasks = [fetch_features(session, uid) for uid in user_ids]
        all_features = await asyncio.gather(*tasks)

    features_array = np.array([f["vector"] for f in all_features])
    predictions = model.predict(features_array)
    return {"predictions": predictions.tolist()}
```

### PyTorch DataLoader Parallelism

```python
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        return self.transform(image)

# num_workers > 0 uses multiprocessing for data loading
# pin_memory=True speeds up CPU→GPU transfer
# prefetch_factor controls how many batches are pre-loaded
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,          # 8 worker processes for data loading
    pin_memory=True,        # Pin to page-locked memory for fast GPU transfer
    prefetch_factor=2,      # Each worker pre-fetches 2 batches
    persistent_workers=True # Don't respawn workers each epoch
)
```

---

## 17.5 Caching Strategies

```python
from functools import lru_cache
import hashlib
import json

# In-memory caching for repeated feature lookups
@lru_cache(maxsize=10_000)
def get_user_embedding(user_id: str) -> tuple:
    """Cache frequently accessed embeddings in memory."""
    embedding = expensive_embedding_lookup(user_id)
    return tuple(embedding)  # Must be hashable for lru_cache

# Redis caching for distributed serving
import redis
import numpy as np

r = redis.Redis(host="redis-cache", port=6379)

def get_features_cached(entity_id: str) -> np.ndarray:
    cache_key = f"features:{entity_id}"
    cached = r.get(cache_key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)

    features = compute_features_expensive(entity_id)
    r.setex(cache_key, 3600, features.tobytes())  # TTL: 1 hour
    return features
```

---

## 17.6 Memory Optimization

### Generators for Large Datasets

```python
# BAD: Loads everything into memory
def load_all_samples(file_paths):
    samples = []
    for path in file_paths:
        samples.extend(json.load(open(path)))
    return samples  # Could be 50GB in memory!

# GOOD: Generator yields one sample at a time
def stream_samples(file_paths):
    for path in file_paths:
        with open(path) as f:
            for line in f:
                yield json.loads(line)
```

### __slots__ for Memory-Efficient Objects

```python
# Without __slots__: each instance has a __dict__ (~200 bytes overhead)
class Sample:
    def __init__(self, features, label, weight):
        self.features = features
        self.label = label
        self.weight = weight

# With __slots__: fixed attributes, ~64 bytes per instance
class SampleSlots:
    __slots__ = ["features", "label", "weight"]

    def __init__(self, features, label, weight):
        self.features = features
        self.label = label
        self.weight = weight

# For 10M samples: ~2GB savings with __slots__
```

### Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in float16 — 2x less memory, faster on modern GPUs
    with autocast():
        outputs = model(batch["input_ids"])
        loss = criterion(outputs, batch["labels"])

    # Backward pass with gradient scaling to prevent underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 17.7 NumPy & PyTorch Performance Tips

| Tip | Slow | Fast |
|---|---|---|
| Vectorize | `for x in arr: x * 2` | `arr * 2` |
| Pre-allocate | `np.append` in loop | `np.empty` then fill |
| Avoid copies | `arr[mask].copy()` | Use views when possible |
| Batch GPU ops | Transfer per sample | Transfer per batch |
| Contiguous memory | Random access patterns | `np.ascontiguousarray` |
| Disable gradients | Default in eval | `torch.no_grad()` context |

```python
# Batching strategies for model inference
import torch
import numpy as np

def naive_inference(model, samples):
    """BAD: One sample at a time — massive overhead."""
    results = []
    for sample in samples:
        tensor = torch.tensor(sample).unsqueeze(0).cuda()
        with torch.no_grad():
            results.append(model(tensor).cpu().numpy())
    return np.concatenate(results)

def batched_inference(model, samples, batch_size=256):
    """GOOD: Process in batches — amortize transfer overhead."""
    results = []
    for i in range(0, len(samples), batch_size):
        batch = torch.tensor(samples[i:i+batch_size]).cuda()
        with torch.no_grad():
            results.append(model(batch).cpu().numpy())
    return np.concatenate(results)

# batched_inference is typically 10-100x faster
```

---

## 17.8 Profiling PyTorch Training

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= 5:
            break
        with record_function("forward"):
            outputs = model(batch["input_ids"].cuda())
            loss = criterion(outputs, batch["labels"].cuda())
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        prof.step()

# Print summary sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 17.9 Summary Checklist

| Step | Tool | Action |
|---|---|---|
| 1. Set goal | — | Define measurable target (latency, throughput, memory) |
| 2. Baseline | `time`, `timeit` | Measure current performance |
| 3. Profile CPU | `py-spy`, `cProfile` | Find hotspot functions |
| 4. Profile memory | `memory_profiler`, `tracemalloc` | Find memory leaks |
| 5. Profile GPU | `torch.profiler` | Find CUDA bottlenecks |
| 6. Optimize | See tips above | Fix the #1 bottleneck only |
| 7. Re-measure | Same as step 2 | Verify improvement |
| 8. Repeat | — | Until goal is met |

> **Advice from production ML teams:** At OpenAI and Anthropic, data loading is often the #1 training bottleneck. Before optimizing model code, ensure your data pipeline can saturate your GPUs. Use `nvidia-smi` to check GPU utilization — if it's below 80%, your bottleneck is likely data loading or preprocessing.

---

*Last updated: April 2026*
