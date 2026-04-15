[← Back to Table of Contents](./README.md)

# Chapter 11 — Containerization & Reproducibility

> "Works on my machine" is not a deployment strategy. Containers make your ML pipelines portable, reproducible, and production-ready.

Machine learning code is notoriously difficult to reproduce. Between CUDA versions, Python dependencies, system libraries, and hardware-specific optimizations, getting a training script to run identically on two different machines is a real challenge. **Containerization** solves this by packaging your code, runtime, and dependencies into a single, portable artifact.

This chapter covers Docker fundamentals through the lens of ML engineering — from writing efficient Dockerfiles for GPU workloads to orchestrating multi-service ML applications.

---

## 11.1 Why Containers for ML?

ML workflows have unique dependency challenges that make containers essential:

| Challenge | Without Containers | With Containers |
|---|---|---|
| CUDA/cuDNN versions | Manual install, conflicts with other projects | Pinned in base image |
| Python dependencies | Virtual envs can leak, conda conflicts | Isolated filesystem |
| System libraries (OpenCV, FFmpeg) | `apt-get` on host, version drift | Declarative in Dockerfile |
| Reproducibility | "It worked last month" | Exact same image every time |
| Team onboarding | Multi-page setup docs | `docker run` |
| Production parity | Dev ≠ Staging ≠ Prod | Same image everywhere |

> "At Google, we learned that the most expensive bugs are environment bugs — the ones that only appear in production because your dev setup was subtly different." — *Adapted from Software Engineering at Google*

---

## 11.2 Docker Fundamentals

### Images, Containers, and Layers

<div class="diagram">
  <div class="diagram-title">Docker Image Layer Stack</div>
  <div class="layer-stack">
    <div class="layer accent">Your ML Code & Scripts</div>
    <div class="layer orange">Python Packages (torch, transformers, numpy)</div>
    <div class="layer yellow">Python 3.11 Runtime</div>
    <div class="layer green">System Libraries (libcudnn, libcublas)</div>
    <div class="layer blue">NVIDIA CUDA Toolkit 12.1</div>
    <div class="layer purple">Ubuntu 22.04 Base</div>
  </div>
</div>

Key concepts:

- **Image**: A read-only template with layered filesystem. Each instruction in a Dockerfile creates a layer.
- **Container**: A running instance of an image with a writable layer on top.
- **Registry**: A repository for images (Docker Hub, GHCR, ECR, GCR).
- **Layer caching**: Docker reuses unchanged layers, making rebuilds fast.

```bash
# Pull an NVIDIA base image
docker pull nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Build an image from a Dockerfile
docker build -t my-training:v1.0 .

# Run a container with GPU access
docker run --gpus all -v ./data:/data my-training:v1.0

# List running containers
docker ps

# View image layers and sizes
docker history my-training:v1.0
```

---

## 11.3 Writing Dockerfiles for ML

### Basic ML Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (changes frequently — last layer)
COPY src/ ./src/
COPY configs/ ./configs/

# Default command
CMD ["python", "src/train.py", "--config", "configs/default.yaml"]
```

### Multi-Stage Build for Lean Production Images

Multi-stage builds let you separate the build environment from the runtime, producing smaller final images:

```dockerfile
# syntax=docker/dockerfile:1

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["python3", "src/serve.py"]
```

### pip vs conda in Docker

| Approach | Pros | Cons |
|---|---|---|
| `pip` + `requirements.txt` | Fast installs, small layers, simple | No non-Python deps |
| `pip` + `pip-tools` | Deterministic lock files | Extra tool in pipeline |
| `conda` + `environment.yml` | Handles C libs (CUDA, MKL) | Slow, large images (2-5 GB+) |
| `conda-lock` | Fully deterministic conda | Still large images |
| `uv` + `pyproject.toml` | Extremely fast, deterministic | Newer tool, less ecosystem support |

**Recommendation**: Use `pip` with pinned versions for production Docker images. Use `conda` only when you need system-level packages that are hard to install otherwise.

---

## 11.4 Docker Compose for Multi-Service ML Apps

Real ML applications are rarely a single container. You typically need an API server, model workers, a database, and possibly a message queue.

<div class="diagram">
  <div class="diagram-title">Multi-Service ML Application</div>
  <div class="flow">
    <div class="flow-node blue">Client Request</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">NGINX Reverse Proxy</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">FastAPI Gateway</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">Model Worker (GPU)</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node orange">Redis Cache</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node teal">PostgreSQL (Predictions Log)</div>
  </div>
</div>

```yaml
# docker-compose.yml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - MODEL_ENDPOINT=http://model-worker:8080
      - REDIS_URL=redis://cache:6379
      - DATABASE_URL=postgresql://ml:secret@db:5432/predictions
    depends_on:
      - model-worker
      - cache
      - db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  model-worker:
    build:
      context: .
      dockerfile: Dockerfile.model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/models/v2.1
      - BATCH_SIZE=32
    volumes:
      - ./models:/models:ro

  cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ml
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: predictions
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
# Start all services
docker compose up -d

# View logs from the model worker
docker compose logs -f model-worker

# Scale model workers
docker compose up -d --scale model-worker=3

# Tear down everything
docker compose down -v
```

---

## 11.5 NVIDIA Container Toolkit for GPU Access

The NVIDIA Container Toolkit allows Docker containers to access host GPUs:

```bash
# Install NVIDIA Container Toolkit (Ubuntu)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access inside container
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Choosing NVIDIA Base Images

| Image Tag | Use Case | Size |
|---|---|---|
| `nvidia/cuda:12.1.1-base-ubuntu22.04` | Just CUDA runtime | ~120 MB |
| `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | + cuBLAS, cuFFT | ~800 MB |
| `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` | + cuDNN (inference) | ~1.5 GB |
| `nvidia/cuda:12.1.1-devel-ubuntu22.04` | + headers, nvcc (compiling) | ~3.5 GB |

**Rule of thumb**: Use `-runtime` for inference, `-devel` only when you need to compile CUDA kernels (e.g., Flash Attention, custom ops).

---

## 11.6 Reproducible Builds

### Pinning Everything

```txt
# requirements.txt — pin EVERY version
torch==2.2.1+cu121
transformers==4.38.2
datasets==2.18.0
numpy==1.26.4
scipy==1.12.0
safetensors==0.4.2
accelerate==0.27.2
```

```dockerfile
# Pin the base image by digest, not just tag
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:a1b2c3d4e5f6...

# Pin apt packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11=3.11.8-1+jammy1 \
    && rm -rf /var/lib/apt/lists/*
```

### Deterministic Installs with pip-compile

```bash
# Generate locked requirements from abstract deps
pip-compile requirements.in --generate-hashes --output-file requirements.txt

# Install with hash checking (fails if any hash doesn't match)
pip install --require-hashes -r requirements.txt
```

### Build Arguments for Flexibility

```dockerfile
ARG CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.11
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION
RUN apt-get update && apt-get install -y python${PYTHON_VERSION}
```

```bash
# Build with different CUDA versions
docker build --build-arg CUDA_VERSION=11.8.0 -t model:cu118 .
docker build --build-arg CUDA_VERSION=12.1.1 -t model:cu121 .
```

---

## 11.7 Docker Best Practices for ML

### Layer Caching Strategy

<div class="diagram">
  <div class="diagram-title">Optimal Layer Ordering (Least → Most Frequently Changed)</div>
  <div class="layer-stack">
    <div class="layer red">COPY src/ (changes every commit)</div>
    <div class="layer orange">COPY configs/ (changes occasionally)</div>
    <div class="layer yellow">RUN pip install -r requirements.txt (changes weekly)</div>
    <div class="layer green">COPY requirements.txt (changes weekly)</div>
    <div class="layer blue">RUN apt-get install ... (changes rarely)</div>
    <div class="layer purple">FROM nvidia/cuda:... (changes per quarter)</div>
  </div>
</div>

### .dockerignore

```gitignore
# .dockerignore
__pycache__/
*.pyc
.git/
.github/
*.egg-info/
dist/
build/
.venv/
.env
wandb/
outputs/
checkpoints/
data/raw/
*.pt
*.ckpt
*.safetensors
notebooks/
tests/
docs/
README.md
```

### Security Best Practices

```dockerfile
# Run as non-root
RUN groupadd -r mluser && useradd -r -g mluser mluser
USER mluser

# Don't store secrets in images — use runtime env vars or mounted secrets
# BAD: ENV HF_TOKEN=hf_abc123...
# GOOD: pass at runtime with docker run -e HF_TOKEN=$HF_TOKEN

# Use read-only filesystem where possible
# docker run --read-only --tmpfs /tmp my-model:v1
```

### Image Size Reduction

```bash
# Check image size
docker images my-training

# Analyze layers
docker history my-training:v1.0 --no-trunc

# Use dive for interactive analysis
dive my-training:v1.0
```

| Technique | Savings |
|---|---|
| Multi-stage builds | 50-70% |
| `--no-install-recommends` | 10-30% |
| `rm -rf /var/lib/apt/lists/*` | 50-200 MB |
| `pip --no-cache-dir` | 100-500 MB |
| `.dockerignore` for data/checkpoints | GBs |
| Use `-slim` or `-alpine` base | 200-500 MB |

---

## 11.8 CI/CD Integration

```yaml
# .github/workflows/docker-build.yml
name: Build & Push ML Image

on:
  push:
    branches: [main]
    paths:
      - "src/**"
      - "requirements.txt"
      - "Dockerfile"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
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
            ghcr.io/${{ github.repository }}/model:${{ github.sha }}
            ghcr.io/${{ github.repository }}/model:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 11.9 Key Takeaways

1. **Containers are essential** for ML reproducibility — they pin everything from CUDA to pip packages.
2. **Order Dockerfile layers** from least to most frequently changed for optimal caching.
3. **Use multi-stage builds** to separate build-time and runtime dependencies.
4. **Pin versions aggressively** — use image digests, exact package versions, and hash checking.
5. **Docker Compose** orchestrates multi-service ML apps (API + model + cache + DB).
6. **NVIDIA Container Toolkit** bridges host GPUs into containers — use the right base image for your workload.
7. **Never bake secrets into images** — pass them at runtime via environment variables or mounted secrets.

---

*Last updated: April 2026*
