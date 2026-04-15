[← Back to Table of Contents](./README.md)

# Chapter 8 — Documentation

> "Documentation is a love letter that you write to your future self." — Damian Conway

The best ML code in the world is useless if nobody — including future-you — can understand how to use it. This chapter covers the science of documentation: what types exist, how to write each one, and how to automate the boring parts.

---

## 8.1 The Diátaxis Framework

The [Diátaxis framework](https://diataxis.fr) classifies documentation into four types based on **what the reader needs**.

<div class="diagram">
<div class="diagram-title">Diátaxis — Four Types of Documentation</div>
<div class="diagram-grid">
<div class="diagram-card green">
<strong>📖 Tutorial</strong><br>
<em>Learning-oriented</em><br><br>
"Follow these steps to fine-tune a model."<br><br>
Holds the reader's hand. Always works end-to-end.
</div>
<div class="diagram-card blue">
<strong>🔧 How-To Guide</strong><br>
<em>Task-oriented</em><br><br>
"How to deploy a model to Kubernetes."<br><br>
Assumes competence. Goal-directed, not explanatory.
</div>
<div class="diagram-card purple">
<strong>💡 Explanation</strong><br>
<em>Understanding-oriented</em><br><br>
"Why we use mixed precision training."<br><br>
Discusses concepts, tradeoffs, architecture decisions.
</div>
<div class="diagram-card orange">
<strong>📋 Reference</strong><br>
<em>Information-oriented</em><br><br>
"TrainingConfig accepts these 12 parameters."<br><br>
Complete, accurate, terse. Auto-generated when possible.
</div>
</div>
</div>

| Aspect | Tutorial | How-To | Explanation | Reference |
|---|---|---|---|---|
| **Reader needs** | To learn | To accomplish | To understand | To look up |
| **Author focus** | Guided steps | Problem → solution | Concepts & context | Completeness |
| **Examples** | "Your first training run" | "Add a custom metric" | "How attention works" | "API parameter list" |
| **Maintenance cost** | High (must stay runnable) | Medium | Low | Auto-generate! |

---

## 8.2 Docstring Conventions

Python has two major docstring styles. **Pick one and enforce it project-wide.**

### Google Style (Recommended for ML)

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> dict[str, float]:
    """Train the model for one epoch.

    Iterates over the full dataloader, computes loss, and updates
    model parameters. Returns training metrics for the epoch.

    Args:
        model: The neural network to train. Must be in train mode.
        dataloader: Iterable of (input, target) batches.
        optimizer: Optimizer with parameters already registered.
        device: Device to move batches to. Defaults to "cuda".

    Returns:
        Dictionary with keys "loss", "accuracy", and "learning_rate".

    Raises:
        RuntimeError: If model is in eval mode.
        torch.cuda.OutOfMemoryError: If batch size exceeds GPU memory.

    Example:
        >>> model = MyModel().to("cuda").train()
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        >>> metrics = train_epoch(model, train_loader, optimizer)
        >>> print(f"Loss: {metrics['loss']:.4f}")
    """
```

### NumPy Style

```python
def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute classification metrics for model evaluation.

    Parameters
    ----------
    predictions : np.ndarray
        Model output probabilities, shape (n_samples,).
    targets : np.ndarray
        Ground truth binary labels, shape (n_samples,).
    threshold : float, optional
        Classification threshold, by default 0.5.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'accuracy', 'precision', 'recall', 'f1'.

    Notes
    -----
    Uses micro-averaging for multi-class extension.

    Examples
    --------
    >>> preds = np.array([0.9, 0.2, 0.8, 0.3])
    >>> targets = np.array([1, 0, 1, 0])
    >>> metrics = compute_metrics(preds, targets)
    >>> metrics['accuracy']
    1.0
    """
```

<div class="diagram">
<div class="diagram-title">Docstring Style Comparison</div>
<div class="compare">
<div class="compare-side green">
<strong>Google Style</strong><br><br>
Args:<br>
&nbsp;&nbsp;param: Description here.<br>
Returns:<br>
&nbsp;&nbsp;Description of return value.<br>
Raises:<br>
&nbsp;&nbsp;ErrorType: When it happens.<br><br>
✅ Compact, readable<br>
✅ Used by TensorFlow, JAX<br>
✅ Works well with Sphinx
</div>
<div class="compare-side blue">
<strong>NumPy Style</strong><br><br>
Parameters<br>
----------<br>
param : type<br>
&nbsp;&nbsp;Description here.<br>
Returns<br>
-------<br>
type<br>
&nbsp;&nbsp;Description of return value.<br><br>
✅ More structured<br>
✅ Used by NumPy, SciPy, scikit-learn<br>
✅ Easier to scan for long param lists
</div>
</div>
</div>

---

## 8.3 Inline Comments — Best Practices

Comments should explain **why**, not **what**.

```python
# ❌ BAD — States the obvious
# Increment the counter by 1
counter += 1

# ❌ BAD — Paraphrases the code
# Set learning rate to 3e-4
lr = 3e-4

# ✅ GOOD — Explains non-obvious reasoning
# Use 3e-4 as per Karpathy's "the most common LR" — works well for
# AdamW with transformers when combined with cosine decay.
lr = 3e-4

# ✅ GOOD — Explains a workaround
# Clamp logits to [-100, 100] to prevent NaN in cross-entropy loss
# when using mixed precision. See: github.com/pytorch/pytorch/issues/12345
logits = logits.clamp(-100, 100)

# ✅ GOOD — Documents a known issue
# TODO(alex): This loads the full dataset into memory. Switch to
# streaming for datasets > 50GB. Tracked in JIRA-1234.
data = load_dataset(path)
```

---

## 8.4 Sphinx — The Python Standard

Sphinx generates beautiful HTML documentation from reStructuredText or Markdown.

### Quick Setup

```bash
# Install Sphinx and extensions
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser

# Initialize in your project
cd docs/
sphinx-quickstart --sep --project "MLPlatform" --author "ML Team"
```

### Sphinx Configuration

```python
# docs/conf.py
project = "MLPlatform"
author = "ML Team"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",         # Google/NumPy docstring support
    "sphinx.ext.viewcode",         # Links to source code
    "sphinx_autodoc_typehints",    # Type hints in docs
    "myst_parser",                 # Markdown support
    "sphinx.ext.doctest",          # Run docstring examples
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# Support both .rst and .md files
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
```

### Auto-Generate API Docs

```bash
# Generate .rst stubs from source code
sphinx-apidoc -o docs/api/ src/mlplatform/ --force --module-first

# Build HTML
cd docs/ && make html
```

---

## 8.5 MkDocs with Material Theme

MkDocs is the **modern alternative** to Sphinx — Markdown-native, faster to set up, and the Material theme is gorgeous.

### Installation

```bash
pip install mkdocs-material mkdocstrings[python] mkdocs-gen-files
```

### MkDocs Configuration

```yaml
# mkdocs.yml
site_name: ML Platform Documentation
site_url: https://mlplatform.example.com/docs
repo_url: https://github.com/org/mlplatform

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - content.code.copy
    - content.code.annotate
    - search.suggest
    - search.highlight

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details

nav:
  - Home: index.md
  - Tutorials:
    - Quick Start: tutorials/quickstart.md
    - Fine-Tuning: tutorials/fine-tuning.md
  - How-To Guides:
    - Deploy to K8s: guides/deploy-k8s.md
    - Custom Metrics: guides/custom-metrics.md
  - API Reference: api/
  - Architecture:
    - ADRs: architecture/decisions/
```

### Building and Serving

```bash
# Local development with hot reload
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

<div class="diagram">
<div class="diagram-title">Documentation Tool Comparison</div>
<div class="compare">
<div class="compare-side purple">
<strong>Sphinx</strong><br><br>
• reStructuredText native<br>
• Mature ecosystem<br>
• Rich cross-references<br>
• Best for large API docs<br>
• Used by Python, PyTorch<br>
• Steeper learning curve
</div>
<div class="compare-side green">
<strong>MkDocs + Material</strong><br><br>
• Markdown native<br>
• Beautiful default theme<br>
• Fast setup & build<br>
• Best for project docs<br>
• Used by FastAPI, Pydantic<br>
• Easy to contribute to
</div>
</div>
</div>

---

## 8.6 Architecture Decision Records (ADRs)

ADRs capture **why** architectural decisions were made. Six months later, when someone asks "why do we use Ray instead of Dask?", the ADR has the answer.

### ADR Template

```markdown
# ADR-003: Use Ray for Distributed Training

## Status
Accepted (2025-03-15)

## Context
We need to scale training beyond a single GPU node. Our options are:
1. PyTorch DDP (built-in)
2. Ray Train (orchestration layer)
3. DeepSpeed (ZeRO optimization)
4. Horovod (legacy)

## Decision
We will use **Ray Train** as our distributed training orchestrator,
with DeepSpeed as the optimization backend.

## Rationale
- Ray Train provides fault tolerance and elastic scaling
- Integrates with our existing Ray Serve deployment
- DeepSpeed ZeRO-3 enables training models larger than GPU memory
- PyTorch DDP alone lacks fault tolerance for multi-day runs
- Horovod is in maintenance mode

## Consequences
- **Positive**: Unified platform for training + serving
- **Positive**: Automatic recovery from node failures
- **Negative**: Additional operational complexity (Ray cluster)
- **Negative**: Team needs Ray training (estimated 1 week)

## Alternatives Considered
See context above. Full evaluation matrix in doc/eval/distributed-training.md
```

### ADR File Naming

```
docs/architecture/decisions/
├── 001-use-pytorch-over-tensorflow.md
├── 002-adopt-pydantic-for-configs.md
├── 003-use-ray-for-distributed-training.md
├── 004-model-versioning-with-dvc.md
└── README.md   # Index of all ADRs
```

---

## 8.7 README-Driven Development

Write the README **before** writing the code. This forces you to think about the user experience first.

### ML Project README Template

```markdown
# 🚀 ProjectName

One-sentence description of what this does and why it matters.

## Quick Start

​```bash
pip install projectname
​```

​```python
from projectname import Model

model = Model.from_pretrained("our-model-v1")
result = model.predict("Hello, world!")
print(result)
​```

## Features

- ⚡ Fast inference (< 50ms on CPU)
- 🎯 State-of-the-art accuracy on benchmark X
- 📦 Simple pip install, no CUDA required for inference

## Installation

​```bash
# CPU only
pip install projectname

# With GPU support
pip install projectname[gpu]
​```

## Usage

### Training
​```bash
python -m projectname.train --config configs/base.yaml
​```

### Evaluation
​```bash
python -m projectname.eval --checkpoint runs/best/model.pt
​```

## Model Zoo

| Model | Params | Accuracy | Download |
|-------|--------|----------|----------|
| base  | 110M   | 92.3%    | [link]() |
| large | 340M   | 95.1%    | [link]() |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

## License

Apache 2.0
```

---

## 8.8 Documentation Testing with doctest

Docstrings with examples can be **automatically tested** to ensure they stay accurate.

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].

    Example:
        >>> import numpy as np
        >>> cosine_similarity(np.array([1, 0]), np.array([1, 0]))
        1.0
        >>> cosine_similarity(np.array([1, 0]), np.array([0, 1]))
        0.0
        >>> round(cosine_similarity(np.array([1, 1]), np.array([1, 0])), 4)
        0.7071
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

```bash
# Run doctests
python -m doctest src/mlplatform/utils.py -v

# Or with pytest
pytest --doctest-modules src/
```

### Integration in CI

```yaml
# .github/workflows/docs.yml
name: Documentation
on: [push, pull_request]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[docs]"
      - run: pytest --doctest-modules src/
      - run: mkdocs build --strict  # Fail on warnings
```

---

## 8.9 The Documentation Workflow

<div class="diagram">
<div class="diagram-title">Documentation as Part of Development</div>
<div class="cycle">
<div class="cycle-step green">Write README<br>(before code)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step blue">Write docstrings<br>(with code)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step purple">Generate API docs<br>(automated)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step orange">Write guides<br>(after release)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step accent">Review & update<br>(ongoing)</div>
<div class="cycle-arrow">↺</div>
</div>
</div>

> **The cardinal rule**: If you change the code, update the docs in the same PR. Documentation that's out of sync is worse than no documentation — it actively misleads.

### Documentation Checklist

| ✅ | Item | Who |
|---|---|---|
| ☐ | README updated | PR author |
| ☐ | Docstrings on all public functions | PR author |
| ☐ | ADR for architectural changes | Tech lead |
| ☐ | Changelog entry | PR author |
| ☐ | doctests pass | CI |
| ☐ | Docs build without warnings | CI |

---

*Last updated: April 2026*
