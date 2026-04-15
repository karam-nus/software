[← Back to Table of Contents](./README.md)

# Chapter 10 — Build Systems & Automation

> "Automate anything you do more than twice." — Every senior engineer, eventually.

ML projects have a unique build challenge: you're not just compiling code — you're downloading data, training models, evaluating checkpoints, building Docker images, and deploying endpoints. This chapter covers the tools that keep all of that from becoming a manual nightmare.

---

## 10.1 Why Automation Matters

Without automation, the "deploy a model" process looks like this:

1. SSH into a GPU machine
2. `git pull` (hope it's the right branch)
3. `pip install -r requirements.txt` (hope nothing broke)
4. `python train.py --config ...` (hope you remembered the right flags)
5. Copy checkpoint to S3 (hope you named it right)
6. SSH into the serving machine
7. Download the checkpoint, restart the server
8. Manually test the endpoint

**With automation**, it's: `git push` → CI handles everything → Slack notification with evaluation results.

<div class="diagram">
<div class="diagram-title">Manual vs Automated ML Workflow</div>
<div class="compare">
<div class="compare-side red">
<strong>❌ Manual Process</strong><br><br>
• 12 steps, 45 minutes<br>
• Error-prone (wrong branch, wrong config)<br>
• Not reproducible<br>
• Knowledge in one person's head<br>
• "It worked when I ran it"
</div>
<div class="compare-side green">
<strong>✅ Automated Pipeline</strong><br><br>
• git push → done<br>
• Reproducible every time<br>
• Documented in code (Makefile, CI)<br>
• Anyone can trigger it<br>
• Audit trail in CI logs
</div>
</div>
</div>

---

## 10.2 Makefiles for ML Projects

Make is 47 years old and still the best way to define project tasks. It's pre-installed on every Unix system and requires zero dependencies.

### Complete ML Project Makefile

```makefile
# Makefile for ML project
.PHONY: help install lint format type-check test test-fast train eval
.PHONY: serve docker-build docker-run clean

PYTHON := python
UV := uv
DOCKER_IMAGE := ml-project
DOCKER_TAG := $(shell git rev-parse --short HEAD)

# Default target
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────

install: ## Install all dependencies
	$(UV) sync

install-dev: ## Install with dev dependencies
	$(UV) sync --all-extras

# ── Code Quality ─────────────────────────────────────────

lint: ## Run linter (ruff)
	$(UV) run ruff check src/ tests/

format: ## Format code (ruff)
	$(UV) run ruff format src/ tests/
	$(UV) run ruff check --fix src/ tests/

type-check: ## Run type checker (mypy)
	$(UV) run mypy src/ --ignore-missing-imports

quality: lint type-check ## Run all code quality checks

# ── Testing ──────────────────────────────────────────────

test: ## Run full test suite
	$(UV) run pytest tests/ -v --tb=short

test-fast: ## Run tests excluding slow/GPU tests
	$(UV) run pytest tests/ -v --tb=short -m "not slow and not gpu"

test-cov: ## Run tests with coverage
	$(UV) run pytest tests/ --cov=src --cov-report=html --cov-report=term

# ── Training & Evaluation ────────────────────────────────

train: ## Train model with default config
	$(UV) run $(PYTHON) -m mlproject.train --config configs/base.yaml

train-debug: ## Train with small dataset for debugging
	$(UV) run $(PYTHON) -m mlproject.train \
		--config configs/base.yaml \
		--override max_steps=100 batch_size=4

eval: ## Evaluate latest checkpoint
	$(UV) run $(PYTHON) -m mlproject.eval \
		--checkpoint runs/latest/model.pt \
		--output results/eval.json

# ── Serving ──────────────────────────────────────────────

serve: ## Start model serving API locally
	$(UV) run uvicorn mlproject.serve:app --reload --port 8000

# ── Docker ───────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):latest

docker-run: ## Run Docker container
	docker run --gpus all -p 8000:8000 $(DOCKER_IMAGE):latest

docker-push: ## Push to container registry
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):latest

# ── Cleanup ──────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

### Usage

```bash
make help          # Show all targets
make install-dev   # Set up environment
make quality       # Lint + type-check
make test-fast     # Quick test run
make train         # Train model
make docker-build  # Build container
```

---

## 10.3 CI/CD with GitHub Actions

GitHub Actions is the standard CI/CD platform for open-source ML projects. Here's a production-grade workflow.

### Complete CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: "3.12"
  UV_CACHE_DIR: .uv-cache

jobs:
  # ── Code Quality ─────────────────────────────────────
  quality:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Lint with ruff
        run: uv run ruff check src/ tests/

      - name: Check formatting
        run: uv run ruff format --check src/ tests/

      - name: Type check with mypy
        run: uv run mypy src/ --ignore-missing-imports

  # ── Tests ────────────────────────────────────────────
  test:
    name: Tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-extras --python ${{ matrix.python-version }}

      - name: Run tests
        run: uv run pytest tests/ -v --tb=short -m "not gpu"

      - name: Upload coverage
        if: matrix.python-version == '3.12'
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: htmlcov/

  # ── Docker Build ─────────────────────────────────────
  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: [quality, test]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.sha }}
            ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Model Evaluation (on main only) ─────────────────
  evaluate:
    name: Model Evaluation
    runs-on: ubuntu-latest
    needs: [quality, test]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Run evaluation
        run: uv run python -m mlproject.eval --config configs/eval.yaml

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: results/
```

<div class="diagram">
<div class="diagram-title">CI/CD Pipeline Flow</div>
<div class="flow">
<div class="flow-node blue">git push / PR</div>
<div class="flow-arrow">↓ triggers</div>
<div class="flow-node green">Lint & Format Check<br>(ruff, mypy)</div>
<div class="flow-arrow">↓ pass?</div>
<div class="flow-node green">Unit Tests<br>(pytest, Python 3.11 + 3.12)</div>
<div class="flow-arrow">↓ pass?</div>
<div class="flow-node purple">Docker Build<br>(multi-stage, cached)</div>
<div class="flow-arrow">↓ main branch only</div>
<div class="flow-node orange">Model Evaluation<br>(benchmark suite)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node accent">Deploy to Staging<br>(auto) → Production (manual)</div>
</div>
</div>

---

## 10.4 Pre-commit Hooks

Pre-commit hooks catch issues **before** they enter the repository — no more "fix lint" commits.

### Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ["--maxkb=1000"]   # Catch accidental model uploads
      - id: check-merge-conflict
      - id: debug-statements      # No breakpoint() in commits

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, torch-stubs]
        args: [--ignore-missing-imports]

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks   # Prevent secrets from being committed
```

### Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks into .git/hooks/
pre-commit install

# Run against all files (first time)
pre-commit run --all-files

# Skip hooks when needed (use sparingly!)
git commit --no-verify -m "WIP: debug commit"
```

---

## 10.5 Task Runners — just & invoke

When Makefiles feel too ancient, modern task runners offer better ergonomics.

### just (Recommended)

```just
# justfile
set dotenv-load

default:
    @just --list

# Install dependencies
install:
    uv sync --all-extras

# Run linter
lint:
    uv run ruff check src/ tests/

# Format code
format:
    uv run ruff format src/ tests/

# Run tests
test *args='':
    uv run pytest tests/ {{args}}

# Train model with optional config override
train config='configs/base.yaml':
    uv run python -m mlproject.train --config {{config}}

# Start serving locally
serve port='8000':
    uv run uvicorn mlproject.serve:app --reload --port {{port}}

# Build and tag Docker image
docker-build:
    docker build -t ml-project:$(git rev-parse --short HEAD) .
```

### invoke (Python-native)

```python
# tasks.py
from invoke import task

@task
def install(c):
    """Install all dependencies."""
    c.run("uv sync --all-extras")

@task
def lint(c):
    """Run linter."""
    c.run("uv run ruff check src/ tests/")

@task
def test(c, fast=False):
    """Run tests. Use --fast to skip slow tests."""
    marks = '-m "not slow and not gpu"' if fast else ""
    c.run(f"uv run pytest tests/ -v {marks}")

@task
def train(c, config="configs/base.yaml"):
    """Train a model."""
    c.run(f"uv run python -m mlproject.train --config {config}")
```

| Feature | Make | just | invoke |
|---|---|---|---|
| Language | Make DSL | Custom DSL | Python |
| Pre-installed | ✅ Linux/Mac | ❌ | ❌ |
| Arguments | Awkward | `{{arg}}` | Python args |
| Tab sensitivity | ✅ (notorious) | ❌ | ❌ |
| Dependency graph | ✅ Built-in | ❌ | Manual |
| Best for | Universal default | Modern projects | Python-only teams |

---

## 10.6 Automated Testing Pipelines

ML projects need **multiple levels** of automated tests.

```yaml
# .github/workflows/test-matrix.yml
name: Test Matrix

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-extras
      - run: uv run pytest tests/unit/ -v --tb=short

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      redis:
        image: redis:7
        ports: [6379:6379]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-extras
      - run: uv run pytest tests/integration/ -v
        env:
          REDIS_URL: redis://localhost:6379

  gpu-tests:
    runs-on: [self-hosted, gpu]
    needs: unit-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: make install
      - run: make test
        env:
          RUN_GPU_TESTS: "1"
```

<div class="diagram">
<div class="diagram-title">Test Pyramid for ML Projects</div>
<div class="layer-stack">
<div class="layer red">E2E / Model Eval (slow, expensive, main-only)</div>
<div class="layer orange">Integration Tests (API, DB, Redis)</div>
<div class="layer yellow">Unit Tests — Transforms, Utils (fast, every PR)</div>
<div class="layer green">Static Analysis — ruff, mypy, pre-commit (instant)</div>
</div>
</div>

---

## 10.7 Artifact Management

CI pipelines produce **artifacts**: trained models, evaluation reports, Docker images. Managing them is critical.

```yaml
# Upload model checkpoint as CI artifact
- name: Upload model checkpoint
  uses: actions/upload-artifact@v4
  with:
    name: model-checkpoint-${{ github.sha }}
    path: |
      runs/best/model.pt
      runs/best/config.yaml
      results/eval.json
    retention-days: 90

# Download in a later job
- name: Download checkpoint
  uses: actions/download-artifact@v4
  with:
    name: model-checkpoint-${{ github.sha }}
```

### Artifact Storage Strategy

| Artifact | Storage | Retention |
|---|---|---|
| Model checkpoints | S3 / GCS / HuggingFace Hub | Permanent (versioned) |
| Evaluation reports | GitHub Actions artifacts | 90 days |
| Docker images | ghcr.io / ECR | Last 10 versions |
| Training logs | Weights & Biases / MLflow | Permanent |
| Coverage reports | GitHub Actions artifacts | 30 days |

---

## 10.8 Release Automation

Automate version bumps and releases with GitHub Actions.

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write   # For PyPI trusted publishing

    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4

      - name: Build package
        run: uv build

      - name: Publish to PyPI
        run: uv publish
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
          files: dist/*
```

### Release Process

```bash
# 1. Update version in pyproject.toml (or use setuptools-scm)
# 2. Create and push a tag
git tag v1.2.0
git push origin v1.2.0

# 3. CI automatically:
#    - Builds the package
#    - Publishes to PyPI
#    - Creates GitHub Release with auto-generated notes
```

---

## 10.9 Putting It All Together

A mature ML project's automation stack:

<div class="diagram">
<div class="diagram-title">Complete Automation Stack</div>
<div class="diagram-grid">
<div class="diagram-card green">
<strong>Local Development</strong><br><br>
• Makefile / justfile<br>
• pre-commit hooks<br>
• uv for fast installs<br>
• Hot-reload dev server
</div>
<div class="diagram-card blue">
<strong>Pull Request</strong><br><br>
• Lint + format check<br>
• Unit + integration tests<br>
• Type checking<br>
• Docker build (no push)
</div>
<div class="diagram-card purple">
<strong>Merge to Main</strong><br><br>
• All PR checks<br>
• GPU model evaluation<br>
• Docker push to registry<br>
• Deploy to staging
</div>
<div class="diagram-card accent">
<strong>Release (tag)</strong><br><br>
• Build Python package<br>
• Publish to PyPI<br>
• GitHub Release<br>
• Deploy to production
</div>
</div>
</div>

> **Key takeaway**: Automation is an investment that pays compound interest. Every manual step you automate is a step that can never be forgotten, done wrong, or skipped under pressure. Start with a `Makefile` and pre-commit hooks, then add CI/CD as the project matures.

---

*Last updated: April 2026*
