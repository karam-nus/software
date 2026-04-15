[← Back to Table of Contents](./README.md)

# Chapter 9 — Dependency & Package Management

> "It works on my machine." — Every developer, moments before production breaks.

ML projects have notoriously complex dependency graphs: PyTorch depends on CUDA, which depends on specific GPU drivers, which depend on the kernel version. This chapter covers how to tame that complexity with modern Python tooling.

---

## 9.1 Why Dependency Management Matters

A machine learning project typically depends on **hundreds** of packages. Without proper management:

- **Reproducibility breaks**: Training run from January can't be repeated in March because `transformers` released a breaking change.
- **Conflicts multiply**: `package-A` needs `numpy>=1.24` but `package-B` needs `numpy<1.24`.
- **Security rots**: Untracked transitive dependencies accumulate known CVEs.
- **Onboarding slows**: New team members spend days getting the environment right.

<div class="diagram">
<div class="diagram-title">The Dependency Problem in ML</div>
<div class="flow">
<div class="flow-node red">Your Code</div>
<div class="flow-arrow">↓ depends on</div>
<div class="flow-node orange">transformers 4.38</div>
<div class="flow-arrow">↓ depends on</div>
<div class="flow-node yellow">torch 2.2</div>
<div class="flow-arrow">↓ depends on</div>
<div class="flow-node blue">CUDA 12.1</div>
<div class="flow-arrow">↓ depends on</div>
<div class="flow-node purple">GPU Driver ≥ 530</div>
<div class="flow-arrow">↓ depends on</div>
<div class="flow-node accent">Linux Kernel ≥ 5.15</div>
</div>
</div>

---

## 9.2 Virtual Environments

**Never install ML packages into the system Python.** Always use an isolated environment.

### venv (Built-in)

```bash
# Create
python -m venv .venv

# Activate
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate       # Windows

# Verify isolation
which python   # Should point to .venv/bin/python

# Deactivate
deactivate
```

### Conda Environments

```bash
# Create from scratch
conda create -n myproject python=3.12 -y

# Activate
conda activate myproject

# Create from file
conda env create -f environment.yml

# Export current environment
conda env export --from-history > environment.yml
```

| Feature | `venv` | `conda` |
|---|---|---|
| Installs Python itself | ❌ | ✅ |
| Non-Python packages (CUDA, MKL) | ❌ | ✅ |
| Speed | Fast | Slow (solving) |
| Disk usage | Small | Large |
| Best for | Pure Python projects | ML with CUDA deps |

---

## 9.3 pip and requirements.txt

The original Python package manager — simple but powerful.

### Basic Workflow

```bash
# Install packages
pip install torch transformers wandb

# Freeze current state
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

### Structured Requirements Files

```
# requirements.txt — direct dependencies only
torch>=2.2.0,<2.4.0
transformers>=4.38.0,<5.0.0
datasets>=2.16.0
wandb>=0.16.0
pydantic>=2.0.0

# requirements-dev.txt
-r requirements.txt
pytest>=8.0.0
ruff>=0.3.0
mypy>=1.8.0
pre-commit>=3.6.0

# requirements-gpu.txt
-r requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.2.0+cu121
```

> ⚠️ **`pip freeze` is dangerous**: It captures *everything*, including transitive dependencies. If you freeze `numpy==1.26.3` and later a security patch releases `1.26.4`, your pinned version blocks the fix. Use **lock files** instead (see §9.5).

---

## 9.4 Conda Environments and environment.yml

Conda manages both Python packages **and** system libraries like CUDA, cuDNN, and MKL.

```yaml
# environment.yml
name: ml-project
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.12
  - pytorch=2.2
  - pytorch-cuda=12.1
  - torchvision
  - numpy>=1.26
  - pandas>=2.1
  - scipy
  - pip:
    # Packages not available on conda
    - transformers>=4.38
    - wandb>=0.16
    - pydantic>=2.0
```

```bash
# Create environment
conda env create -f environment.yml

# Update after changing environment.yml
conda env update -f environment.yml --prune

# Remove environment
conda env remove -n ml-project
```

---

## 9.5 uv — The Fast Python Package Manager

[uv](https://github.com/astral-sh/uv) is a Rust-based replacement for pip, pip-compile, virtualenv, and more. It's **10-100x faster** than pip.

### Basic Usage

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project
uv init ml-project
cd ml-project

# Add dependencies
uv add torch transformers wandb
uv add --dev pytest ruff mypy

# Install from lock file (deterministic)
uv sync

# Run commands in the environment
uv run python train.py
uv run pytest
```

### pyproject.toml with uv

```toml
[project]
name = "ml-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.38.0",
    "wandb>=0.16.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

[tool.uv]
# Extra index for CUDA wheels
extra-index-url = ["https://download.pytorch.org/whl/cu121"]
```

<div class="diagram">
<div class="diagram-title">Package Manager Speed Comparison</div>
<div class="compare">
<div class="compare-side red">
<strong>pip install (cold)</strong><br><br>
████████████████████████████<br>
~45 seconds<br><br>
• Pure Python resolver<br>
• No parallel downloads<br>
• No caching by default
</div>
<div class="compare-side green">
<strong>uv sync (cold)</strong><br><br>
███<br>
~3 seconds<br><br>
• Rust resolver<br>
• Parallel downloads<br>
• Global cache
</div>
</div>
</div>

---

## 9.6 Lock Files

Lock files pin **every** dependency (direct + transitive) to exact versions, ensuring byte-for-byte reproducible installs.

### pip-compile (pip-tools)

```bash
pip install pip-tools

# Compile requirements.in → requirements.txt (with locked versions)
pip-compile requirements.in -o requirements.txt --generate-hashes

# Upgrade all
pip-compile requirements.in --upgrade

# Install locked deps
pip-sync requirements.txt
```

```
# requirements.in — what you maintain (loose constraints)
torch>=2.2
transformers>=4.38
wandb

# requirements.txt — auto-generated (locked, DO NOT EDIT)
# This file is autogenerated by pip-compile with Python 3.12
torch==2.2.1 \
    --hash=sha256:abc123...
transformers==4.38.2 \
    --hash=sha256:def456...
tokenizers==0.15.2 \
    --hash=sha256:ghi789...
wandb==0.16.4 \
    --hash=sha256:jkl012...
```

### conda-lock

```bash
pip install conda-lock

# Generate lock file from environment.yml
conda-lock -f environment.yml -p linux-64 -p osx-arm64

# Install from lock file (fast, no solving)
conda-lock install conda-lock.yml
```

### uv.lock

uv generates lock files automatically:

```bash
# Lock file is created/updated automatically by uv add / uv lock
uv lock

# Install exactly what's in the lock file
uv sync --frozen  # Fail if lock file is out of date
```

| Tool | Lock File | Speed | Cross-Platform | Hash Verification |
|---|---|---|---|---|
| pip-compile | `requirements.txt` | Medium | ❌ | ✅ |
| conda-lock | `conda-lock.yml` | Slow | ✅ | ✅ |
| uv | `uv.lock` | Fast | ✅ | ✅ |
| poetry | `poetry.lock` | Medium | ✅ | ✅ |

---

## 9.7 Dependency Resolution

When two packages demand conflicting versions of a shared dependency, you have a **resolution conflict**.

<div class="diagram">
<div class="diagram-title">Dependency Resolution Flow</div>
<div class="flow">
<div class="flow-node blue">Read constraints<br>(pyproject.toml)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node green">Fetch package metadata<br>(PyPI index)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node purple">Build dependency graph<br>(all transitive deps)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node orange">Solve constraints<br>(SAT solver / backtracking)</div>
<div class="flow-arrow">↓ success? </div>
<div class="flow-node green">Write lock file</div>
<div class="flow-arrow">↓ conflict?</div>
<div class="flow-node red">Error: No solution<br>"package-a needs X&lt;2 but package-b needs X≥2"</div>
</div>
</div>

### Debugging Conflicts

```bash
# pip — verbose resolver output
pip install torch transformers --verbose 2>&1 | grep "conflict"

# uv — much better error messages
uv add conflicting-package
# error: No solution found when resolving dependencies:
#   Because pack-a==1.0.0 depends on numpy<1.24
#   and pack-b==2.0.0 depends on numpy>=1.24,
#   pack-a==1.0.0 and pack-b==2.0.0 are incompatible.

# pipdeptree — visualize dependency tree
pip install pipdeptree
pipdeptree --warn fail
```

---

## 9.8 Publishing Packages

When your ML code is mature enough to share, publish it to PyPI.

### Modern pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "mlplatform"
dynamic = ["version"]
description = "Production ML training and serving framework"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.11"
authors = [{name = "ML Team", email = "ml@example.com"}]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "torch>=2.2.0",
    "pydantic>=2.0.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.3"]
docs = ["mkdocs-material", "mkdocstrings[python]"]
gpu = ["torch>=2.2.0+cu121"]

[project.scripts]
mlplatform = "mlplatform.cli:main"

[project.urls]
Homepage = "https://github.com/org/mlplatform"
Documentation = "https://mlplatform.readthedocs.io"
Repository = "https://github.com/org/mlplatform"

[tool.setuptools-scm]
# Version from git tags

[tool.setuptools.packages.find]
where = ["src"]
```

### Build and Publish

```bash
# Build
pip install build
python -m build   # Creates dist/mlplatform-0.1.0.tar.gz and .whl

# Upload to PyPI
pip install twine
twine upload dist/*

# Or use uv
uv build
uv publish
```

---

## 9.9 Managing CUDA Dependencies

CUDA is the most painful dependency in ML. Here's how to handle it.

```bash
# Option 1: conda (manages CUDA toolkit)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Option 2: pip with CUDA-specific index
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Option 3: Docker (most reproducible)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Option 4: uv with extra index
# In pyproject.toml:
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu121"]
```

### CUDA Compatibility Matrix

| PyTorch | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 |
|---|---|---|---|
| 2.0.x | ✅ | ✅ | ❌ |
| 2.1.x | ✅ | ✅ | ❌ |
| 2.2.x | ✅ | ✅ | ✅ |
| 2.3.x | ❌ | ✅ | ✅ |

> **Pro tip**: In CI, use Docker images with a pre-installed CUDA toolkit. Locally, use conda for CUDA. In production, use `nvidia/cuda` base images.

### Verifying GPU Setup

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"CUDA version:    {torch.version.cuda}")
print(f"cuDNN version:   {torch.backends.cudnn.version()}")
print(f"GPU count:       {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

<div class="diagram">
<div class="diagram-title">Recommended Tool Stack by Project Stage</div>
<div class="diagram-grid">
<div class="diagram-card green">
<strong>🧪 Research / Notebook</strong><br><br>
• conda for environments<br>
• pip install as needed<br>
• environment.yml checked in<br>
• Low overhead
</div>
<div class="diagram-card blue">
<strong>🏗️ Team Project</strong><br><br>
• uv for speed + lock files<br>
• pyproject.toml for config<br>
• uv.lock for reproducibility<br>
• pre-commit for consistency
</div>
<div class="diagram-card purple">
<strong>📦 Published Package</strong><br><br>
• pyproject.toml (PEP 621)<br>
• Build with setuptools or hatch<br>
• Publish to PyPI with twine/uv<br>
• Semantic versioning
</div>
<div class="diagram-card accent">
<strong>🚀 Production Deploy</strong><br><br>
• Docker with pinned base<br>
• Lock file for installs<br>
• Multi-stage builds<br>
• Vulnerability scanning
</div>
</div>
</div>

---

*Last updated: April 2026*
