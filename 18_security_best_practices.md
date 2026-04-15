[← Back to Table of Contents](./README.md)

# Chapter 18 — Security Best Practices

> "Security is not a feature — it's a property of the entire system." — Ross Anderson

ML engineers routinely handle sensitive data, powerful compute credentials, and models that can be adversarially exploited. Yet security is often an afterthought in ML workflows. This chapter covers practical security for the ML practitioner — from secrets management to model serialization risks.

---

## 18.1 The Security Mindset for ML Engineers

Security in ML systems extends beyond traditional software concerns:

| Traditional Security | ML-Specific Security |
|---|---|
| SQL injection | Adversarial inputs to models |
| Credential leaks | Training data exposure |
| Dependency vulnerabilities | Poisoned training data |
| API authentication | Model theft / extraction |
| Input validation | Prompt injection (LLMs) |

> **Principle of least privilege:** Every component should have the minimum permissions needed. A training job doesn't need production database write access. A serving endpoint doesn't need S3 full access.

---

## 18.2 Secrets Management

<div class="diagram">
  <div class="diagram-title">Secrets Management — From Worst to Best</div>
  <div class="layer-stack">
    <div class="layer red">❌ Hardcoded in source code</div>
    <div class="layer orange">⚠️ .env files (local dev only)</div>
    <div class="layer yellow">⚠️ Environment variables (CI/CD)</div>
    <div class="layer green">✅ GitHub Secrets / CI provider secrets</div>
    <div class="layer blue">✅ Vault systems (HashiCorp Vault, AWS Secrets Manager)</div>
    <div class="layer purple">✅ Workload identity (no secrets at all)</div>
  </div>
</div>

### Never Hardcode Secrets

```python
# ❌ NEVER DO THIS
AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLE"
DB_PASSWORD = "super_secret_password_123"
WANDB_API_KEY = "abc123def456"

# ✅ Use environment variables
import os

AWS_SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
```

### .env Files for Local Development

```bash
# .env — NEVER commit this file
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
WANDB_API_KEY=abc123def456ghi789
DATABASE_URL=postgresql://user:pass@localhost:5432/mldb
```

```python
# Load .env in development
from dotenv import load_dotenv
import os

load_dotenv()  # Reads .env file into os.environ

# Works the same way regardless of environment
db_url = os.environ["DATABASE_URL"]
```

```gitignore
# .gitignore — ALWAYS include these
.env
.env.local
.env.*.local
*.pem
*.key
secrets/
credentials.json
wandb/
```

### Vault Systems for Production

```python
# HashiCorp Vault client
import hvac

def get_secret(path: str) -> dict:
    client = hvac.Client(url=os.environ["VAULT_ADDR"])
    client.token = os.environ["VAULT_TOKEN"]
    secret = client.secrets.kv.v2.read_secret_version(path=path)
    return secret["data"]["data"]

# Usage
db_creds = get_secret("ml-platform/database")
connection_string = (
    f"postgresql://{db_creds['username']}:{db_creds['password']}"
    f"@{db_creds['host']}:5432/{db_creds['database']}"
)
```

```python
# AWS Secrets Manager
import boto3
import json

def get_aws_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

# Usage in training scripts
secrets = get_aws_secret("ml/training/credentials")
wandb.login(key=secrets["wandb_api_key"])
```

### Comparison of Approaches

| Approach | Security | Convenience | Best For |
|---|---|---|---|
| Hardcoded | ❌ Terrible | High | Never |
| `.env` files | ⚠️ Okay | High | Local development |
| Env vars (CI) | ✅ Good | Medium | CI/CD pipelines |
| GitHub Secrets | ✅ Good | Medium | GitHub Actions workflows |
| AWS Secrets Manager | ✅ Very good | Low | Production AWS workloads |
| HashiCorp Vault | ✅ Excellent | Low | Multi-cloud production |
| Workload Identity | ✅ Best | Medium | Cloud-native services |

---

## 18.3 Preventing Secret Commits

```bash
# Install git-secrets (pre-commit hook)
git secrets --install
git secrets --register-aws

# Scans for AWS key patterns before every commit
git secrets --scan

# Install detect-secrets (more comprehensive)
pip install detect-secrets
detect-secrets scan > .secrets.baseline
detect-secrets audit .secrets.baseline
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
        exclude: "tests/.*|.*\\.lock"
```

> **Recovery if you commit a secret:** Removing it in a new commit is NOT enough — it's still in git history. You must: (1) rotate the credential immediately, (2) use `git filter-repo` or BFG Repo Cleaner to purge history, and (3) force-push. Assume the secret is already compromised.

---

## 18.4 Input Validation and Sanitization

```python
from pydantic import BaseModel, Field, validator
import numpy as np

class PredictionRequest(BaseModel):
    """Validated prediction request — rejects malformed inputs."""
    features: list[float] = Field(..., min_length=1, max_length=1000)
    model_version: str = Field(..., pattern=r"^v\d+\.\d+\.\d+$")
    user_id: str = Field(..., min_length=1, max_length=128)

    @validator("features")
    def features_are_finite(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Features must be finite numbers (no NaN/Inf)")
        return v

    @validator("features")
    def features_in_range(cls, v):
        if any(abs(x) > 1e6 for x in v):
            raise ValueError("Feature values out of expected range")
        return v

# FastAPI automatically validates using this model
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict(request: PredictionRequest):
    features = np.array(request.features, dtype=np.float32)
    prediction = model.predict(features.reshape(1, -1))
    return {"prediction": float(prediction[0])}
```

---

## 18.5 Dependency Vulnerability Scanning

```bash
# pip-audit — scan Python dependencies for known vulnerabilities
pip install pip-audit
pip-audit

# Safety — another Python vulnerability scanner
pip install safety
safety check --full-report

# Scan a requirements file
pip-audit -r requirements.txt
safety check -r requirements.txt
```

```yaml
# GitHub Actions: automated dependency scanning
name: Security Scan
on:
  push:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"  # Weekly Monday scan

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pip-audit
      - run: pip-audit --strict --desc
```

```yaml
# Dependabot configuration — .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "security"
```

---

## 18.6 Pickle Security Risks

> **Critical warning:** Python's `pickle` module can execute arbitrary code during deserialization. Never unpickle data from untrusted sources.

```python
import pickle

# ❌ This is a remote code execution vulnerability
class MaliciousModel:
    def __reduce__(self):
        import os
        return (os.system, ("rm -rf / --no-preserve-root",))

# If someone gives you a "model file" that's actually this:
# pickle.loads(malicious_bytes)  # Executes the command!
```

<div class="diagram">
  <div class="diagram-title">Model Serialization Security</div>
  <div class="compare">
    <div class="compare-side">
      <h4>❌ Unsafe Formats</h4>
      <div class="flow-node red">pickle / joblib</div>
      <div class="flow-node red">torch.save (uses pickle)</div>
      <div class="flow-node orange">numpy.load (allow_pickle)</div>
      <p>Can execute arbitrary code</p>
    </div>
    <div class="compare-side">
      <h4>✅ Safer Formats</h4>
      <div class="flow-node green">ONNX</div>
      <div class="flow-node green">TorchScript (torch.jit)</div>
      <div class="flow-node green">SafeTensors</div>
      <div class="flow-node green">TensorFlow SavedModel</div>
      <p>Store only weights and graph</p>
    </div>
  </div>
</div>

```python
# ✅ Safe model serialization with SafeTensors
from safetensors.torch import save_file, load_file

# Save — only tensors, no arbitrary code
tensors = {k: v for k, v in model.state_dict().items()}
save_file(tensors, "model.safetensors")

# Load — guaranteed safe, no code execution
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
```

```python
# ✅ Safe alternative: ONNX export
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["prediction"],
    dynamic_axes={"image": {0: "batch_size"}},
)
```

---

## 18.7 API Authentication

```python
# JWT-based authentication for model serving
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os

app = FastAPI()
security = HTTPBearer()
JWT_SECRET = os.environ["JWT_SECRET"]

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict(request: PredictionRequest, user=Depends(verify_token)):
    # user is the decoded JWT payload — use for auditing
    prediction = model.predict(request.features)
    return {"prediction": prediction, "user": user["sub"]}
```

```python
# API key authentication — simpler but less flexible
from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")
VALID_API_KEYS = set(os.environ["API_KEYS"].split(","))

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

---

## 18.8 Supply Chain Security

```bash
# Pin exact dependency versions with hashes
pip install pip-tools
pip-compile --generate-hashes requirements.in -o requirements.txt

# requirements.txt now includes hashes:
# numpy==1.26.4 \
#     --hash=sha256:ab123...
#     --hash=sha256:cd456...

# Install with hash verification
pip install --require-hashes -r requirements.txt
```

```yaml
# Pin GitHub Actions to commit SHAs, not tags
# ❌ Tag can be moved to point to malicious code
- uses: actions/checkout@v4

# ✅ SHA is immutable
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11
```

---

## 18.9 Security Checklist for ML Projects

| Category | Check | Tool |
|---|---|---|
| Secrets | No secrets in code or git history | `detect-secrets`, `git-secrets` |
| Secrets | Credentials rotated regularly | Vault, AWS Secrets Manager |
| Dependencies | No known vulnerabilities | `pip-audit`, Dependabot |
| Dependencies | Pinned with hashes | `pip-compile --generate-hashes` |
| Serialization | No unpickling untrusted data | SafeTensors, ONNX |
| API | Authentication on all endpoints | JWT, API keys |
| API | Input validation | Pydantic, JSON Schema |
| Data | Training data access controlled | IAM, row-level security |
| Data | PII handled per regulations | Anonymization, encryption |
| Infra | Least-privilege IAM roles | AWS IAM, GCP IAM |
| Infra | Network policies restrict access | VPC, security groups |
| Models | Model integrity verified | Checksums, signing |

> **Practice at top labs:** Anthropic and OpenAI use workload identity federation wherever possible — eliminating long-lived secrets entirely. Training jobs authenticate via short-lived tokens tied to their execution environment.

---

*Last updated: April 2026*
