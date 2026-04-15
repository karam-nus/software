[← Back to Table of Contents](./README.md)

# Chapter 4 — Testing for ML Software

> *"If it's not tested, it's broken — you just don't know it yet."*

Testing ML software is uniquely challenging. Traditional software has deterministic outputs — given the same input, you expect the same output. ML systems involve stochastic training, floating-point arithmetic, and models that produce probabilistic outputs. This chapter covers testing strategies adapted for the realities of ML development.

---

## 4.1 The Testing Pyramid

The testing pyramid provides a framework for balancing test types. For ML, we add a layer for **model-specific tests** that validate statistical properties.

<div class="diagram">
<div class="diagram-title">ML Testing Pyramid</div>
<div class="layer-stack">
<div class="layer red">E2E / System Tests<br/><small>Full pipeline: data → train → evaluate → serve</small></div>
<div class="layer orange">Integration Tests<br/><small>Component interactions: data pipeline + model, API + model</small></div>
<div class="layer blue">Model Tests<br/><small>Shapes, gradients, determinism, overfitting, invariances</small></div>
<div class="layer green">Unit Tests<br/><small>Individual functions: loss, metrics, preprocessing, augmentation</small></div>
</div>
</div>

| Test Type | Quantity | Speed | Confidence | Example |
|---|---|---|---|---|
| **Unit** | Many (hundreds) | Fast (ms) | Low-level correctness | Loss function returns expected value |
| **Model** | Moderate (dozens) | Medium (seconds) | Model behavior | Gradients flow through all layers |
| **Integration** | Some (tens) | Slow (minutes) | Components work together | DataLoader feeds correct shapes to model |
| **E2E** | Few (< 10) | Very slow (hours) | System works | Train for 100 steps, eval loss decreases |

> *"Write many unit tests, fewer integration tests, and a handful of end-to-end tests. Invert this and your CI takes hours and tells you nothing useful."* — Adapted from MIT 6.031

---

## 4.2 pytest Fundamentals

pytest is the standard testing framework for Python ML projects. It's powerful, extensible, and integrates with every major ML library.

### Basic Test Structure

```python
# tests/test_loss.py
import pytest
import torch
from ml_pipeline.losses import focal_loss, label_smoothed_cross_entropy


class TestFocalLoss:
    """Tests for focal loss implementation."""

    def test_perfect_prediction_gives_zero_loss(self):
        """Focal loss should be ~0 when prediction matches target."""
        logits = torch.tensor([[10.0, -10.0, -10.0]])  # Confident class 0
        targets = torch.tensor([0])
        loss = focal_loss(logits, targets, gamma=2.0)
        assert loss.item() < 1e-4

    def test_uniform_prediction_gives_high_loss(self):
        """Focal loss should be high for uncertain predictions."""
        logits = torch.zeros(1, 3)  # Uniform distribution
        targets = torch.tensor([0])
        loss = focal_loss(logits, targets, gamma=2.0)
        assert loss.item() > 0.5

    def test_gamma_zero_equals_cross_entropy(self):
        """Focal loss with gamma=0 should equal standard CE."""
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        focal = focal_loss(logits, targets, gamma=0.0)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(focal, ce, atol=1e-5, rtol=1e-5)

    def test_output_shape_is_scalar(self):
        """Loss should return a scalar tensor."""
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = focal_loss(logits, targets)
        assert loss.dim() == 0  # Scalar


class TestLabelSmoothing:
    """Tests for label-smoothed cross entropy."""

    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.2, 0.5])
    def test_smoothing_range(self, smoothing):
        """Loss should be finite for valid smoothing values."""
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = label_smoothed_cross_entropy(logits, targets, smoothing=smoothing)
        assert torch.isfinite(loss)

    def test_zero_smoothing_equals_cross_entropy(self):
        """No smoothing should equal standard cross entropy."""
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        smoothed = label_smoothed_cross_entropy(logits, targets, smoothing=0.0)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(smoothed, ce, atol=1e-5, rtol=1e-5)
```

### Fixtures

```python
# tests/conftest.py
import pytest
import torch
from pathlib import Path
from ml_pipeline.models import TransformerEncoder
from ml_pipeline.data import TextDataset


@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set deterministic seeds for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    return 42


@pytest.fixture
def small_model(device):
    """Create a small model for fast testing."""
    model = TransformerEncoder(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=1000,
        max_seq_len=128,
    ).to(device)
    return model


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing."""
    batch_size, seq_len = 4, 32
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
        "labels": torch.randint(0, 1000, (batch_size, seq_len), device=device),
    }


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small dataset for testing."""
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        '{"text": "The cat sat on the mat."}\n'
        '{"text": "Machine learning is great."}\n'
        '{"text": "Testing ensures quality."}\n'
    )
    return TextDataset(data_file, max_length=64)
```

### Parametrize for Thorough Coverage

```python
# tests/test_activations.py
import pytest
import torch
from ml_pipeline.activations import get_activation


@pytest.mark.parametrize("activation_name,expected_range", [
    ("relu", (0, float("inf"))),
    ("sigmoid", (0, 1)),
    ("tanh", (-1, 1)),
    ("gelu", (-0.17, float("inf"))),    # GELU minimum ≈ -0.17
    ("swish", (-0.28, float("inf"))),
])
def test_activation_output_range(activation_name, expected_range):
    """Each activation should produce values within its expected range."""
    activation = get_activation(activation_name)
    x = torch.randn(1000)
    y = activation(x)
    lo, hi = expected_range
    assert y.min().item() >= lo - 0.01  # Small tolerance for numerics
    assert y.max().item() <= hi + 0.01
```

---

## 4.3 Testing ML Code

### Shape Tests

The most common ML bug is a shape mismatch. Shape tests are fast and catch many issues.

```python
# tests/test_model_shapes.py
import pytest
import torch


class TestModelShapes:
    """Verify tensor shapes throughout the model."""

    def test_forward_output_shape(self, small_model, sample_batch, device):
        """Model output should have shape (batch, seq_len, vocab_size)."""
        output = small_model(**sample_batch)
        batch_size = sample_batch["input_ids"].shape[0]
        seq_len = sample_batch["input_ids"].shape[1]
        assert output.logits.shape == (batch_size, seq_len, 1000)

    def test_embedding_shape(self, small_model, device):
        """Embedding layer should produce (batch, seq_len, d_model)."""
        x = torch.randint(0, 1000, (2, 16), device=device)
        embedded = small_model.embedding(x)
        assert embedded.shape == (2, 16, 64)  # d_model=64

    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 1),      # Minimum
        (1, 128),    # Single sample, max length
        (32, 64),    # Typical batch
        (64, 128),   # Large batch, max length
    ])
    def test_variable_input_shapes(self, small_model, device, batch_size, seq_len):
        """Model should handle various input shapes."""
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, device=device)
        output = small_model(input_ids=x, attention_mask=mask)
        assert output.logits.shape == (batch_size, seq_len, 1000)
```

### Gradient Tests

Ensure gradients flow properly through your model — a model with dead gradients trains but learns nothing.

```python
# tests/test_gradients.py
import torch


class TestGradients:
    """Verify gradient flow through the model."""

    def test_all_parameters_receive_gradients(self, small_model, sample_batch):
        """Every trainable parameter should have a non-zero gradient."""
        output = small_model(**sample_batch)
        loss = output.logits.sum()
        loss.backward()

        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_magnitudes_reasonable(self, small_model, sample_batch):
        """Gradients should not be excessively large or small."""
        output = small_model(**sample_batch)
        loss = output.logits.mean()
        loss.backward()

        for name, param in small_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1e6, f"Exploding gradient in {name}: {grad_norm}"
                assert grad_norm > 1e-12, f"Vanishing gradient in {name}: {grad_norm}"

    def test_loss_decreases_on_overfit(self, small_model, sample_batch, device):
        """Model should be able to overfit a single batch."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        initial_loss = None

        for step in range(50):
            optimizer.zero_grad()
            output = small_model(**sample_batch)
            loss = torch.nn.functional.cross_entropy(
                output.logits.view(-1, 1000),
                sample_batch["labels"].view(-1),
            )
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss * 0.5, (
            f"Loss didn't decrease enough: {initial_loss:.4f} → {loss.item():.4f}"
        )
```

### Determinism Tests

```python
# tests/test_determinism.py
import torch


class TestDeterminism:
    """Verify reproducibility of model operations."""

    def test_forward_pass_is_deterministic(self, small_model, sample_batch):
        """Same input should produce same output."""
        small_model.eval()
        with torch.no_grad():
            output1 = small_model(**sample_batch).logits.clone()
            output2 = small_model(**sample_batch).logits.clone()
        torch.testing.assert_close(output1, output2)

    def test_training_is_reproducible_with_seed(self, device):
        """Training with same seed should produce identical results."""
        def train_with_seed(seed):
            torch.manual_seed(seed)
            model = torch.nn.Linear(10, 2).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(4, 10, device=device)
            y = torch.tensor([0, 1, 0, 1], device=device)

            for _ in range(10):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
            return loss.item()

        loss1 = train_with_seed(42)
        loss2 = train_with_seed(42)
        assert abs(loss1 - loss2) < 1e-6
```

### Data Pipeline Tests

```python
# tests/test_data_pipeline.py
import pytest
import torch
from ml_pipeline.data import TextDataset, create_dataloader


class TestDataPipeline:
    """Test the data loading and preprocessing pipeline."""

    def test_dataset_length(self, sample_dataset):
        """Dataset should report correct length."""
        assert len(sample_dataset) == 3

    def test_dataset_item_keys(self, sample_dataset):
        """Each item should have required keys."""
        item = sample_dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_no_data_leakage(self, sample_dataset):
        """Labels should be shifted input_ids (causal LM)."""
        item = sample_dataset[0]
        # For causal LM, labels[t] = input_ids[t+1]
        assert torch.equal(
            item["labels"][:-1],
            item["input_ids"][1:],
        )

    def test_tokenization_is_reversible(self, sample_dataset):
        """Decoding encoded text should recover original (approximately)."""
        item = sample_dataset[0]
        decoded = sample_dataset.tokenizer.decode(
            item["input_ids"],
            skip_special_tokens=True,
        )
        assert "cat" in decoded.lower()

    def test_dataloader_batching(self, sample_dataset):
        """DataLoader should produce correctly batched tensors."""
        loader = create_dataloader(sample_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].dtype == torch.long
```

---

## 4.4 Property-Based Testing with Hypothesis

Property-based testing generates random inputs to find edge cases you wouldn't think to test manually.

```python
# tests/test_properties.py
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from ml_pipeline.preprocessing import normalize_features, pad_sequence


@given(
    batch_size=st.integers(min_value=1, max_value=64),
    features=st.integers(min_value=1, max_value=256),
)
@settings(max_examples=100)
def test_normalize_preserves_shape(batch_size, features):
    """Normalization should not change tensor shape."""
    x = torch.randn(batch_size, features)
    normalized = normalize_features(x)
    assert normalized.shape == x.shape


@given(
    batch_size=st.integers(min_value=1, max_value=32),
    features=st.integers(min_value=1, max_value=128),
)
@settings(max_examples=50)
def test_normalize_output_is_finite(batch_size, features):
    """Normalized output should never contain NaN or Inf."""
    x = torch.randn(batch_size, features)
    normalized = normalize_features(x)
    assert torch.isfinite(normalized).all()


@given(
    seq_len=st.integers(min_value=1, max_value=512),
    target_len=st.integers(min_value=1, max_value=1024),
)
def test_pad_sequence_reaches_target_length(seq_len, target_len):
    """Padded sequence should be exactly target_len or original length."""
    assume(target_len >= seq_len)
    x = torch.randint(0, 100, (seq_len,))
    padded = pad_sequence(x, target_len, pad_value=0)
    assert padded.shape[0] == target_len
    # Original content should be preserved
    assert torch.equal(padded[:seq_len], x)
```

---

## 4.5 Mocking External Services

ML code often depends on external services (GPU clusters, model registries, feature stores). Mock these for fast, reliable tests.

```python
# tests/test_model_registry.py
from unittest.mock import MagicMock, patch
import pytest
from ml_pipeline.registry import register_model, fetch_best_model


class TestModelRegistry:
    """Test model registry interactions without a live server."""

    @patch("ml_pipeline.registry.mlflow")
    def test_register_model_logs_metrics(self, mock_mlflow):
        """Model registration should log all required metrics."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = lambda s: mock_run
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        register_model(
            model_path="models/v2.1",
            metrics={"accuracy": 0.95, "f1": 0.93},
            params={"lr": 3e-4, "epochs": 10},
        )

        mock_mlflow.log_metrics.assert_called_once_with(
            {"accuracy": 0.95, "f1": 0.93}
        )
        mock_mlflow.log_params.assert_called_once_with(
            {"lr": 3e-4, "epochs": 10}
        )

    @patch("ml_pipeline.registry.mlflow")
    def test_fetch_best_model_by_metric(self, mock_mlflow):
        """Should fetch the model with the best target metric."""
        mock_mlflow.search_runs.return_value = [
            MagicMock(data=MagicMock(metrics={"f1": 0.90}), info=MagicMock(run_id="run1")),
            MagicMock(data=MagicMock(metrics={"f1": 0.95}), info=MagicMock(run_id="run2")),
        ]

        best = fetch_best_model(metric="f1", direction="max")
        assert best.info.run_id == "run2"
```

---

## 4.6 Test Coverage & CI Integration

<div class="diagram">
<div class="diagram-title">ML Testing in CI Pipeline</div>
<div class="flow">
<div class="flow-node green">Push / PR</div>
<div class="flow-arrow">→</div>
<div class="flow-node blue">Lint &amp; Type Check<br/><small>Ruff + mypy (30s)</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">Unit Tests<br/><small>CPU only (2 min)</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node orange">Model Tests<br/><small>Small models (5 min)</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node accent">Integration Tests<br/><small>GPU runner (15 min)</small></div>
<div class="flow-arrow">→</div>
<div class="flow-node teal">E2E (nightly)<br/><small>Full training run</small></div>
</div>
</div>

### pytest Configuration

```ini
# pytest.ini (alternative to pyproject.toml section)
[pytest]
testpaths = tests
addopts =
    -ra                          # Show summary of all non-passing tests
    -q                           # Quiet output
    --strict-markers             # Error on unknown markers
    --tb=short                   # Shorter tracebacks
    --cov=ml_pipeline            # Measure coverage
    --cov-report=term-missing    # Show missing lines
    --cov-fail-under=80          # Fail if coverage < 80%
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require a GPU
    integration: marks integration tests
    nightly: marks tests that only run nightly
```

### GitHub Actions CI

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: mypy src/
      - run: pytest tests/ -m "not slow and not gpu" --cov

  gpu-tests:
    runs-on: [self-hosted, gpu, a100]
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -m "gpu" --timeout=300
```

### Running Tests Locally

```bash
# Run all fast tests
pytest tests/ -m "not slow and not gpu"

# Run with coverage
pytest tests/ --cov=ml_pipeline --cov-report=html
open htmlcov/index.html

# Run a specific test class
pytest tests/test_model_shapes.py::TestModelShapes -v

# Run tests matching a name pattern
pytest -k "gradient" -v

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

---

## 4.7 Test Patterns Summary

| Pattern | What It Catches | Example |
|---|---|---|
| **Shape test** | Dimension mismatches | Output is `(B, T, V)` |
| **Gradient test** | Dead layers, vanishing gradients | All params receive nonzero grads |
| **Overfit test** | Model can't learn at all | Loss decreases on single batch |
| **Determinism test** | Non-reproducible results | Same seed → same output |
| **Data leakage test** | Train/test contamination | Labels don't appear in inputs |
| **Numerical stability** | NaN/Inf values | Output is always finite |
| **Invariance test** | Incorrect equivariance | Translation-invariant model is actually invariant |
| **Boundary test** | Edge cases | Batch size 1, sequence length 1 |

> *"Good ML tests are not about achieving 100% code coverage. They're about covering the failure modes that are unique to ML: shape errors, gradient pathologies, data leakage, and non-determinism."*

---

*Last updated: April 2026*
