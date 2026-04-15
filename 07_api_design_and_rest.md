[← Back to Table of Contents](./README.md)

# Chapter 7 — API Design & REST

> "A good API is not just easy to use but also hard to misuse." — Joshua Bloch

Deploying an ML model means wrapping it in an API that the rest of the world can call. This chapter covers REST principles, FastAPI for model serving, streaming responses for LLMs, gRPC for high-throughput inference, and everything you need to take a model from a notebook to a production endpoint.

---

## 7.1 REST Fundamentals

REST (Representational State Transfer) maps operations onto **resources** using standard HTTP methods.

| HTTP Method | Purpose | ML Example | Idempotent? |
|---|---|---|---|
| `GET` | Read a resource | Fetch model metadata | ✅ Yes |
| `POST` | Create / trigger action | Submit inference request | ❌ No |
| `PUT` | Replace a resource | Upload a new model version | ✅ Yes |
| `PATCH` | Partial update | Update model config | ✅ Yes |
| `DELETE` | Remove a resource | Delete experiment run | ✅ Yes |

### Status Codes That Matter

| Code | Meaning | When to Use |
|---|---|---|
| `200 OK` | Success | Inference completed |
| `201 Created` | Resource created | New model registered |
| `202 Accepted` | Async job started | Long-running training job queued |
| `400 Bad Request` | Client error | Invalid input shape |
| `404 Not Found` | Resource missing | Model version not found |
| `422 Unprocessable Entity` | Validation error | Text exceeds max tokens |
| `429 Too Many Requests` | Rate limited | Exceeded API quota |
| `500 Internal Server Error` | Server error | Model inference crashed |
| `503 Service Unavailable` | Server overloaded | Model loading, GPU OOM |

### URL Design

```
# Good — nouns, plural, hierarchical
GET    /v1/models
GET    /v1/models/gpt-4o
POST   /v1/models/gpt-4o/completions
GET    /v1/experiments/42/metrics
DELETE /v1/models/gpt-4o/versions/3

# Bad — verbs in URLs, flat structure
POST   /getModelPrediction
GET    /run-inference?model=gpt4
POST   /api/doTraining
```

<div class="diagram">
<div class="diagram-title">Model Serving API — Request Flow</div>
<div class="flow">
<div class="flow-node blue">Client</div>
<div class="flow-arrow">↓ POST /v1/completions</div>
<div class="flow-node orange">API Gateway<br>(rate limit, auth)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node green">Load Balancer</div>
<div class="flow-arrow">↓</div>
<div class="flow-node purple">FastAPI Server</div>
<div class="flow-arrow">↓ validate input</div>
<div class="flow-node accent">Model Worker<br>(GPU inference)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node cyan">Response<br>{ "text": "..." }</div>
</div>
</div>

---

## 7.2 FastAPI for ML Model Serving

FastAPI is the de facto framework for Python ML APIs: async-native, Pydantic integration, automatic OpenAPI docs.

### Complete Model Serving Application

```python
"""ML Model Serving API with FastAPI."""
import asyncio
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Schemas ──────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    model: str = "our-model-v1"

class CompletionResponse(BaseModel):
    id: str
    text: str
    model: str
    usage: dict[str, int]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    uptime_seconds: float

# ── Application Lifecycle ────────────────────────────────

models: dict = {}
start_time: float = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global start_time
    start_time = time.time()
    models["default"] = load_model("our-model-v1")
    yield
    models.clear()
    torch.cuda.empty_cache()

app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded="default" in models,
        gpu_available=torch.cuda.is_available(),
        uptime_seconds=time.time() - start_time,
    )

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    if request.model not in models and request.model != "our-model-v1":
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    start = time.perf_counter()
    model = models["default"]

    # Run inference in thread pool to avoid blocking the event loop
    result = await asyncio.to_thread(
        model.generate,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        text=result.text,
        model=request.model,
        usage={
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        latency_ms=round(elapsed_ms, 2),
    )

@app.get("/v1/models")
async def list_models():
    return {
        "models": [
            {"id": "our-model-v1", "ready": True, "backend": "pytorch"},
        ]
    }
```

### Running the Server

```bash
# Development
uvicorn serve:app --reload --host 0.0.0.0 --port 8000

# Production with multiple workers
gunicorn serve:app -w 4 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 --timeout 120
```

---

## 7.3 Streaming Responses for LLMs

Modern LLM APIs stream tokens via **Server-Sent Events (SSE)**, so users see text appear word-by-word instead of waiting for the full response.

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/v1/completions/stream")
async def stream_completion(request: CompletionRequest):
    async def generate_stream():
        model = models["default"]
        async for token in model.stream_generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            chunk = {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion.chunk",
                "choices": [{"text": token, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### Client-Side Consumption

```python
import httpx

async def consume_stream(prompt: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/completions/stream",
            json={"prompt": prompt, "max_tokens": 100},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    print(chunk["choices"][0]["text"], end="", flush=True)
```

<div class="diagram">
<div class="diagram-title">Streaming vs Non-Streaming Inference</div>
<div class="compare">
<div class="compare-side orange">
<strong>Non-Streaming</strong><br><br>
Client ──POST──→ Server<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(waits 3s...)<br>
Client ←──200───  Server<br>
&nbsp;&nbsp;{"text": "full response"}<br><br>
• Simple implementation<br>
• High time-to-first-token<br>
• Good for batch jobs
</div>
<div class="compare-side green">
<strong>Streaming (SSE)</strong><br><br>
Client ──POST──→ Server<br>
Client ←─ "The"<br>
Client ←─ " quick"<br>
Client ←─ " brown"<br>
Client ←─ [DONE]<br><br>
• Low time-to-first-token<br>
• Better UX for chat<br>
• Industry standard for LLMs
</div>
</div>
</div>

---

## 7.4 gRPC for High-Performance Inference

For internal microservice-to-microservice calls, gRPC offers **2-10x lower latency** than REST thanks to HTTP/2 multiplexing and Protocol Buffer serialization.

### Proto Definition

```protobuf
// inference.proto
syntax = "proto3";

package inference;

service InferenceService {
    rpc Predict(PredictRequest) returns (PredictResponse);
    rpc StreamPredict(PredictRequest) returns (stream PredictChunk);
    rpc HealthCheck(Empty) returns (HealthStatus);
}

message PredictRequest {
    string model_id = 1;
    repeated float input_tensor = 2;
    repeated int32 input_shape = 3;
    map<string, string> metadata = 4;
}

message PredictResponse {
    repeated float output_tensor = 1;
    repeated int32 output_shape = 2;
    float latency_ms = 3;
}

message PredictChunk {
    repeated float partial_output = 1;
    bool is_final = 2;
}

message Empty {}

message HealthStatus {
    bool serving = 1;
    string model_id = 2;
}
```

### gRPC Server Implementation

```python
import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, model):
        self.model = model

    def Predict(self, request, context):
        import numpy as np
        input_array = np.array(request.input_tensor).reshape(request.input_shape)
        output = self.model.predict(input_array)
        return inference_pb2.PredictResponse(
            output_tensor=output.flatten().tolist(),
            output_shape=list(output.shape),
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(model), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
```

| Feature | REST (FastAPI) | gRPC |
|---|---|---|
| Serialization | JSON | Protocol Buffers (binary) |
| Transport | HTTP/1.1 or HTTP/2 | HTTP/2 only |
| Streaming | SSE / WebSocket | Native bidirectional |
| Browser support | ✅ Native | ⚠️ Requires grpc-web proxy |
| Latency | Higher | Lower |
| Best for | External APIs, web clients | Internal microservices |

---

## 7.5 API Versioning

ML models evolve. Your API must support **multiple versions** without breaking existing clients.

```python
from fastapi import APIRouter

# Version 1 — original response format
v1_router = APIRouter(prefix="/v1")

@v1_router.post("/completions")
async def v1_completions(request: CompletionRequestV1):
    return {"text": result.text, "tokens_used": result.total_tokens}

# Version 2 — richer response format with usage breakdown
v2_router = APIRouter(prefix="/v2")

@v2_router.post("/completions")
async def v2_completions(request: CompletionRequestV2):
    return {
        "choices": [{"text": result.text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
    }

app.include_router(v1_router)
app.include_router(v2_router)
```

---

## 7.6 Rate Limiting & Middleware

Protect your GPU-backed endpoints from abuse.

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()

        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < 60
        ]

        if len(self.requests[client_ip]) >= self.rpm:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )

        self.requests[client_ip].append(now)
        response = await call_next(request)
        return response

app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
```

---

## 7.7 Testing Your API

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_completion_validation():
    # Missing required field
    response = client.post("/v1/completions", json={})
    assert response.status_code == 422

    # Temperature out of range
    response = client.post("/v1/completions", json={
        "prompt": "Hello",
        "temperature": 5.0,  # max is 2.0
    })
    assert response.status_code == 422

def test_completion_success():
    response = client.post("/v1/completions", json={
        "prompt": "Explain gradient descent",
        "max_tokens": 50,
    })
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["usage"]["total_tokens"] > 0
```

<div class="diagram">
<div class="diagram-title">API Design Checklist</div>
<div class="diagram-grid">
<div class="diagram-card green">
<strong>✅ Request Validation</strong><br>
Pydantic schemas with<br>
min/max constraints on<br>
every field.
</div>
<div class="diagram-card blue">
<strong>✅ Health Endpoint</strong><br>
GET /health returns<br>
model status, GPU info,<br>
uptime.
</div>
<div class="diagram-card purple">
<strong>✅ Versioning</strong><br>
/v1/ prefix on all<br>
routes. Never break<br>
existing clients.
</div>
<div class="diagram-card orange">
<strong>✅ Rate Limiting</strong><br>
Protect GPU endpoints<br>
from abuse. Return 429<br>
with Retry-After header.
</div>
<div class="diagram-card accent">
<strong>✅ Error Handling</strong><br>
Structured error responses<br>
with error codes, not<br>
stack traces.
</div>
<div class="diagram-card cyan">
<strong>✅ Documentation</strong><br>
OpenAPI auto-generated<br>
at /docs. Add examples<br>
in Pydantic schemas.
</div>
</div>
</div>

---

## 7.8 OpenAPI & Auto-Generated Docs

FastAPI generates interactive docs automatically at `/docs` (Swagger UI) and `/redoc` (ReDoc). Enhance them with examples:

```python
class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(default=256, ge=1, le=4096)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Explain backpropagation in simple terms.",
                    "max_tokens": 150,
                }
            ]
        }
    }
```

> **Key takeaway**: A well-designed API is the difference between a model that sits in a notebook and a model that powers a product. Invest in schemas, versioning, streaming, and tests — your future self (and your users) will thank you.

---

*Last updated: April 2026*
