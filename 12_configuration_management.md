[← Back to Table of Contents](./README.md)

# Chapter 12 — Configuration Management

> "Hardcoding is the root of all evil in ML experimentation. Every magic number, file path, and hyperparameter should live in configuration."

Machine learning code without proper configuration management quickly becomes unmaintainable. When hyperparameters are scattered across Python files, data paths are hardcoded, and experiment settings live in Jupyter notebook cells, reproducing results or collaborating with teammates becomes nearly impossible.

This chapter covers configuration management from simple YAML files to sophisticated frameworks like Hydra, with a focus on the unique challenges of ML experimentation.

---

## 12.1 Why Configuration Matters for ML

In traditional software, configuration usually means database URLs and feature flags. In ML, it encompasses a much larger surface:

| Config Category | Examples |
|---|---|
| **Model architecture** | `hidden_size=768`, `num_layers=12`, `dropout=0.1` |
| **Training** | `learning_rate=3e-4`, `batch_size=32`, `num_epochs=100` |
| **Data** | `dataset_path`, `train_split=0.8`, `max_seq_length=512` |
| **Infrastructure** | `num_gpus=4`, `precision=bf16`, `num_workers=8` |
| **Experiment tracking** | `wandb_project`, `run_name`, `tags` |
| **Paths** | `checkpoint_dir`, `output_dir`, `cache_dir` |

> "At Anthropic, every training run is fully defined by its configuration. If you can't reproduce a run from its config file alone, something is wrong." — *Adapted from ML engineering best practices*

---

## 12.2 Configuration File Formats

### YAML — The ML Standard

YAML is the dominant format in ML because of its readability and support for nested structures:

```yaml
# configs/train.yaml
model:
  name: transformer
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  dropout: 0.1
  vocab_size: 50257

training:
  learning_rate: 3.0e-4
  weight_decay: 0.01
  batch_size: 32
  max_steps: 100000
  warmup_steps: 1000
  gradient_clip: 1.0
  precision: bf16

data:
  train_path: data/train.jsonl
  val_path: data/val.jsonl
  max_seq_length: 512
  num_workers: 8

logging:
  wandb_project: my-llm
  log_every: 100
  eval_every: 1000
  save_every: 5000
```

### TOML — For Application Config

```toml
# pyproject.toml or config.toml
[model]
name = "transformer"
hidden_size = 768
num_layers = 12

[training]
learning_rate = 3.0e-4
batch_size = 32
max_steps = 100_000

[data]
train_path = "data/train.jsonl"
max_seq_length = 512
```

### JSON — For APIs and Schemas

```json
{
  "model": {
    "name": "transformer",
    "hidden_size": 768,
    "num_layers": 12
  },
  "training": {
    "learning_rate": 3e-4,
    "batch_size": 32
  }
}
```

### Format Comparison

<div class="diagram">
  <div class="diagram-title">Choosing the Right Config Format</div>
  <div class="compare">
    <div class="compare-side accent">
      <strong>YAML</strong><br/>
      ✓ Human-readable<br/>
      ✓ Comments supported<br/>
      ✓ Complex nesting<br/>
      ✓ Anchors & aliases<br/>
      ✗ Indentation-sensitive<br/>
      ✗ YAML "Norway problem"
    </div>
    <div class="compare-side blue">
      <strong>TOML</strong><br/>
      ✓ Explicit typing<br/>
      ✓ No indentation issues<br/>
      ✓ Great for flat config<br/>
      ✓ Python standard (pyproject)<br/>
      ✗ Verbose for deep nesting<br/>
      ✗ Less common in ML
    </div>
    <div class="compare-side green">
      <strong>JSON</strong><br/>
      ✓ Universal format<br/>
      ✓ Strict parsing<br/>
      ✓ API-friendly<br/>
      ✓ Schema validation (JSON Schema)<br/>
      ✗ No comments<br/>
      ✗ Verbose
    </div>
  </div>
</div>

**Recommendation**: Use YAML for ML experiment configs, TOML for project-level settings (`pyproject.toml`), and JSON for API contracts and schemas.

---

## 12.3 Environment Variables and .env Files

### The 12-Factor App Approach

The [12-Factor App](https://12factor.net) methodology states that configuration should be stored in environment variables, not in code:

```python
import os

# Read from environment with sensible defaults
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///local.db")
MODEL_PATH = os.environ["MODEL_PATH"]  # Required — fail loudly if missing
HF_TOKEN = os.environ.get("HF_TOKEN")  # Optional
NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
```

### python-dotenv for Local Development

```bash
# .env (NEVER commit this file!)
DATABASE_URL=postgresql://ml:secret@localhost:5432/experiments
MODEL_PATH=/models/gpt2-large
HF_TOKEN=hf_abc123...
WANDB_API_KEY=wandb_xyz789...
NUM_GPUS=2
```

```python
# settings.py
from dotenv import load_dotenv
import os

load_dotenv()  # Reads .env into os.environ

class Settings:
    database_url: str = os.environ["DATABASE_URL"]
    model_path: str = os.environ["MODEL_PATH"]
    hf_token: str | None = os.environ.get("HF_TOKEN")
    num_gpus: int = int(os.environ.get("NUM_GPUS", "1"))
```

```gitignore
# .gitignore — always exclude secrets
.env
.env.local
.env.production
```

---

## 12.4 Hydra for ML Configuration

[Hydra](https://hydra.cc/) by Meta is the most powerful configuration framework for ML experiments. It supports composition, overrides, sweeps, and multirun.

### Basic Hydra Setup

```
project/
├── configs/
│   ├── config.yaml          # Main config
│   ├── model/
│   │   ├── small.yaml
│   │   ├── base.yaml
│   │   └── large.yaml
│   ├── data/
│   │   ├── wikitext.yaml
│   │   └── openwebtext.yaml
│   └── training/
│       ├── default.yaml
│       └── aggressive.yaml
├── src/
│   └── train.py
└── outputs/                  # Hydra auto-creates this
```

```yaml
# configs/config.yaml
defaults:
  - model: base
  - data: wikitext
  - training: default
  - _self_

experiment_name: ${model.name}_${data.name}
seed: 42
```

```yaml
# configs/model/base.yaml
name: transformer-base
hidden_size: 768
num_layers: 12
num_heads: 12
dropout: 0.1
```

```yaml
# configs/model/large.yaml
name: transformer-large
hidden_size: 1024
num_layers: 24
num_heads: 16
dropout: 0.1
```

```yaml
# configs/training/default.yaml
learning_rate: 3e-4
batch_size: 32
max_steps: 100000
warmup_steps: 1000
weight_decay: 0.01
gradient_clip: 1.0
```

### Using Hydra in Python

```python
# src/train.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> float:
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))

    # Access config values with dot notation
    model = build_model(
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Hydra auto-saves config to outputs/<date>/<time>/.hydra/
    # Full reproducibility: just rerun with the saved config
    return final_loss

if __name__ == "__main__":
    train()
```

### Command-Line Overrides

```bash
# Override individual values
python src/train.py model.hidden_size=1024 training.learning_rate=1e-4

# Swap entire config groups
python src/train.py model=large data=openwebtext

# Multirun: sweep over hyperparameters
python src/train.py --multirun \
    training.learning_rate=1e-4,3e-4,1e-3 \
    training.batch_size=16,32,64

# Override with nested values
python src/train.py +new_param=42  # Add new parameter
python src/train.py ~model.dropout  # Remove parameter
```

<div class="diagram">
  <div class="diagram-title">Hydra Config Composition Flow</div>
  <div class="flow">
    <div class="flow-node blue">CLI Overrides</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Defaults List</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">Config Composition</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">OmegaConf Resolution</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Final DictConfig</div>
  </div>
</div>

---

## 12.5 OmegaConf Deep Dive

OmegaConf powers Hydra's config resolution and provides powerful features:

```python
from omegaconf import OmegaConf, DictConfig

# Create from dict
cfg = OmegaConf.create({
    "model": {"name": "gpt2", "size": "base"},
    "output_dir": "outputs/${model.name}-${model.size}",
})

# Variable interpolation
print(cfg.output_dir)  # "outputs/gpt2-base"

# Merge configs (later values override earlier)
base = OmegaConf.load("configs/base.yaml")
override = OmegaConf.load("configs/experiment.yaml")
merged = OmegaConf.merge(base, override)

# Read-only config (prevent accidental mutation)
OmegaConf.set_readonly(cfg, True)

# Missing values with ??? (must be provided before access)
schema = OmegaConf.create({
    "model_path": "???",      # Required
    "batch_size": 32,          # Has default
})

# Convert to plain dict/yaml
plain_dict = OmegaConf.to_container(cfg, resolve=True)
yaml_str = OmegaConf.to_yaml(cfg)
```

---

## 12.6 Config Validation with Pydantic

Use Pydantic to validate configuration with type checking and constraints:

```python
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class ModelConfig(BaseModel):
    name: str
    hidden_size: int = Field(ge=64, le=16384)
    num_layers: int = Field(ge=1, le=128)
    num_heads: int = Field(ge=1, le=128)
    dropout: float = Field(ge=0.0, le=1.0)

    @field_validator("hidden_size")
    @classmethod
    def hidden_size_divisible_by_heads(cls, v, info):
        num_heads = info.data.get("num_heads")
        if num_heads and v % num_heads != 0:
            raise ValueError(
                f"hidden_size ({v}) must be divisible by num_heads ({num_heads})"
            )
        return v


class TrainingConfig(BaseModel):
    learning_rate: float = Field(gt=0, le=1.0)
    batch_size: int = Field(ge=1, le=4096)
    max_steps: int = Field(ge=1)
    warmup_steps: int = Field(ge=0)
    precision: Precision = Precision.BF16
    gradient_clip: float = Field(ge=0.0)

    @field_validator("warmup_steps")
    @classmethod
    def warmup_less_than_max(cls, v, info):
        max_steps = info.data.get("max_steps")
        if max_steps and v >= max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        return v


class ExperimentConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    experiment_name: str


# Usage
import yaml

with open("configs/train.yaml") as f:
    raw_config = yaml.safe_load(f)

config = ExperimentConfig(**raw_config)  # Validates on construction
print(f"Training {config.model.name} with lr={config.training.learning_rate}")
```

---

## 12.7 Feature Flags

Feature flags let you toggle functionality without redeploying:

```python
from dataclasses import dataclass, field


@dataclass
class FeatureFlags:
    use_flash_attention: bool = True
    enable_gradient_checkpointing: bool = False
    use_fused_optimizer: bool = True
    enable_speculative_decoding: bool = False
    new_tokenizer_v2: bool = False

    @classmethod
    def from_env(cls):
        import os
        return cls(
            use_flash_attention=os.getenv("FF_FLASH_ATTN", "true").lower() == "true",
            enable_gradient_checkpointing=os.getenv("FF_GRAD_CKPT", "false").lower() == "true",
            use_fused_optimizer=os.getenv("FF_FUSED_OPT", "true").lower() == "true",
        )


# Usage in training code
flags = FeatureFlags.from_env()

if flags.use_flash_attention:
    from flash_attn import flash_attn_func
    attn_output = flash_attn_func(q, k, v)
else:
    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

---

## 12.8 Experiment Config Management

### Linking Configs to Runs

<div class="diagram">
  <div class="diagram-title">Experiment Configuration Lifecycle</div>
  <div class="cycle">
    <div class="cycle-step accent">Define Config (YAML)</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step green">Validate (Pydantic)</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step blue">Log to W&amp;B / MLflow</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step purple">Save with Checkpoint</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step orange">Reproduce from Config</div>
    <div class="cycle-arrow">↩</div>
  </div>
</div>

```python
import wandb
import json
from omegaconf import OmegaConf

def run_experiment(cfg: DictConfig):
    # Resolve all interpolations
    resolved = OmegaConf.to_container(cfg, resolve=True)

    # Log config to Weights & Biases
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.experiment_name,
        config=resolved,
    )

    # Save config alongside checkpoint
    def save_checkpoint(model, optimizer, step, cfg):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": OmegaConf.to_yaml(cfg),
        }
        path = f"checkpoints/step_{step}.pt"
        torch.save(checkpoint, path)

        # Also save human-readable config
        with open(f"checkpoints/step_{step}_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # Resume from checkpoint with config verification
    def load_checkpoint(path):
        ckpt = torch.load(path)
        saved_cfg = OmegaConf.create(ckpt["config"])
        # Warn if current config differs from saved
        if OmegaConf.to_yaml(cfg) != OmegaConf.to_yaml(saved_cfg):
            logger.warning("Config has changed since checkpoint was saved!")
            logger.warning(f"Diff: {OmegaConf.to_yaml(OmegaConf.merge(saved_cfg, cfg))}")
        return ckpt
```

---

## 12.9 Anti-Patterns to Avoid

| Anti-Pattern | Problem | Solution |
|---|---|---|
| Hardcoded paths | Breaks on other machines | Use config files + env vars |
| Magic numbers in code | Can't reproduce experiments | Put all hyperparameters in config |
| Mutable global config | Race conditions, hard to test | Use frozen dataclasses or `set_readonly` |
| Config in Jupyter cells | Lost when kernel restarts | External YAML files |
| Secrets in config files | Security vulnerability | Use `.env` files, never commit |
| Copy-pasting configs | Drift between experiments | Use Hydra composition |
| No config validation | Silent bugs from typos | Use Pydantic or structured configs |

---

## 12.10 Key Takeaways

1. **Externalize all configuration** — no hardcoded hyperparameters, paths, or magic numbers.
2. **Use YAML for ML experiments**, TOML for project config, JSON for API schemas.
3. **Hydra is the gold standard** for ML config — it provides composition, overrides, and multirun.
4. **Validate configs with Pydantic** to catch errors before training starts.
5. **Follow 12-Factor App** — separate config from code, use environment variables for secrets.
6. **Link configs to experiment runs** — save the full config alongside every checkpoint and log it to your experiment tracker.
7. **Feature flags** let you toggle behavior at runtime without code changes.

---

*Last updated: April 2026*
