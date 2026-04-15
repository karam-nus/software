[← Back to Table of Contents](./README.md)

# Chapter 15 — Code Review & Collaboration

> "Code review is not about finding bugs — automated tools do that. Code review is about sharing knowledge, maintaining standards, and building a team that can move fast together." — Adapted from *Software Engineering at Google*

In ML teams, code review takes on additional importance. A missed off-by-one error in a data pipeline can corrupt months of training. A poorly chosen loss function can waste thousands of GPU hours. Code review is your last line of defense before code reaches production — and in ML, the cost of defects is measured in compute dollars, not just engineering time.

This chapter covers how to do code review well, how to structure collaborative workflows, and how to build a culture of constructive feedback.

---

## 15.1 Why Code Review Matters

| Benefit | Description |
|---|---|
| **Bug prevention** | Catch logic errors, off-by-ones, wrong hyperparameters before they waste GPU time |
| **Knowledge sharing** | Spread understanding of the codebase across the team |
| **Code quality** | Maintain consistent style, patterns, and best practices |
| **Onboarding** | New team members learn by reviewing and being reviewed |
| **Documentation** | PR descriptions become a record of why changes were made |
| **Security** | Catch leaked credentials, unsafe deserialization, injection vulnerabilities |

> "At Google, no code enters the repository without being reviewed. This single practice has done more for code quality than any tool, process, or methodology." — *Software Engineering at Google, Chapter 9*

Studies show that code review catches 60-90% of defects before they reach production. For ML code, where bugs are often silent (the code runs but produces wrong results), this is even more critical.

---

## 15.2 Pull Request Best Practices

### Small, Focused PRs

<div class="diagram">
  <div class="diagram-title">PR Size and Review Quality</div>
  <div class="compare">
    <div class="compare-side green">
      <strong>Small PR (< 400 lines)</strong><br/>
      ✓ Reviewed in < 30 minutes<br/>
      ✓ Thorough feedback<br/>
      ✓ Quick turnaround<br/>
      ✓ Easy to revert if needed<br/>
      ✓ Lower merge conflict risk
    </div>
    <div class="compare-side red">
      <strong>Large PR (> 1000 lines)</strong><br/>
      ✗ "LGTM" rubber-stamp reviews<br/>
      ✗ Days to review<br/>
      ✗ Reviewer fatigue<br/>
      ✗ Risky to merge or revert<br/>
      ✗ Merge conflict nightmare
    </div>
  </div>
</div>

**Guidelines for PR size:**
- Aim for **200-400 lines** of meaningful changes
- One logical change per PR (don't mix refactoring with features)
- If a feature requires 2000+ lines, break it into a stack of PRs
- Infrastructure changes (new service, new pipeline) may be larger — that's OK

### Writing a Good PR Description

```markdown
## Summary
Add learning rate warmup scheduler to training pipeline.

## Motivation
Training runs with large batch sizes diverge in the first ~1000 steps
when using a constant learning rate. Linear warmup stabilizes early
training and improves final loss by ~3% (see experiment W&B link below).

## Changes
- Added `WarmupCosineScheduler` in `src/schedulers.py`
- Integrated scheduler into `train.py` training loop
- Added config options: `warmup_steps`, `min_lr_ratio`
- Added unit tests for scheduler step values

## Testing
- Unit tests: `pytest tests/test_schedulers.py` ✅
- Integration: Ran 5K steps on debug dataset, verified LR curve
- W&B run: https://wandb.ai/team/project/runs/abc123

## Screenshots
[LR schedule plot showing warmup + cosine decay]

## Checklist
- [x] Tests pass locally
- [x] No new linting warnings
- [x] Config changes documented
- [x] Backward compatible (existing configs work unchanged)
```

### PR Lifecycle

<div class="diagram">
  <div class="diagram-title">Pull Request Lifecycle</div>
  <div class="flow">
    <div class="flow-node blue">Create Branch</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Write Code + Tests</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node accent">Open PR with Description</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node purple">CI Runs (lint, test, build)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node orange">Review &amp; Feedback</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node teal">Address Comments</div>
    <div class="flow-arrow">→</div>
    <div class="flow-node green">Approve &amp; Merge</div>
  </div>
</div>

---

## 15.3 Code Review Checklist

Use this checklist when reviewing ML code:

| Category | What to Check |
|---|---|
| **Correctness** | Does the logic do what the PR claims? Are edge cases handled? |
| **Data handling** | Any data leakage? Are train/val/test splits respected? |
| **Shapes & types** | Are tensor shapes documented or asserted? Correct dtypes? |
| **Numerics** | Division by zero? Log of zero? NaN propagation? |
| **Readability** | Clear variable names? Comments on non-obvious logic? |
| **Performance** | Unnecessary copies? Data on wrong device? Missing `torch.no_grad()`? |
| **Config** | Are new parameters in config files? Defaults sensible? |
| **Tests** | Unit tests for new functions? Edge cases covered? |
| **Reproducibility** | Random seeds set? Non-determinism documented? |
| **Security** | No hardcoded credentials? Safe deserialization? |
| **Dependencies** | New deps justified? Pinned versions? License compatible? |
| **Backward compat** | Does this break existing configs, checkpoints, or APIs? |

### ML-Specific Review Items

```python
# REVIEW CHECK: Is model in correct mode?
model.eval()  # Must be set for inference
with torch.no_grad():  # Must be set to save memory during eval
    predictions = model(test_batch)

# REVIEW CHECK: Data leakage in preprocessing
# BAD — fits scaler on full dataset before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)  # LEAKS test statistics into train
X_train, X_test = split(X_scaled)

# GOOD — fit only on training data
X_train, X_test = split(X_all)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Transform only, don't fit

# REVIEW CHECK: Random seed for reproducibility
import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

---

## 15.4 Giving and Receiving Feedback

### Giving Feedback

**Do:**
- Be specific: "This tensor reshape might fail when batch_size=1" not "This looks wrong"
- Ask questions: "What happens if the dataset is empty?" instead of "Handle empty datasets"
- Praise good code: "Nice use of `torch.no_grad()` here — easy to forget"
- Suggest alternatives: "Consider using `einops.rearrange` — makes the shape transform clearer"
- Distinguish blocking vs non-blocking: Use prefixes like `nit:`, `question:`, `suggestion:`, `blocking:`

**Don't:**
- Make it personal: "You always forget..." → "This case isn't handled..."
- Be vague: "Clean this up" → "Extract this into a named function for readability"
- Nitpick style if there's a linter for it
- Request changes that are out of scope for the PR

### Receiving Feedback

**Do:**
- Assume good intent — reviewers want the code to be better
- Explain your reasoning if you disagree, with evidence
- Thank reviewers for catching issues
- Update the code and respond to each comment

**Don't:**
- Take it personally — the review is about the code, not you
- Dismiss feedback without explanation
- Resolve comments without addressing them

### Comment Conventions

```
# Prefix conventions for review comments:

nit: Consider renaming `x` to `input_tensor` for clarity
# → Non-blocking style suggestion

question: Why do we need to clone() here? Is the tensor shared?
# → Seeking understanding, not necessarily requesting a change

suggestion: Could use torch.nn.functional.cross_entropy instead
#   of manually computing softmax + nll_loss — it's numerically
#   more stable.
# → Optional improvement

blocking: This computes loss on padded tokens. Need to mask
#   padding positions before computing the mean, otherwise the
#   loss is diluted.
# → Must fix before merging
```

---

## 15.5 Branching Strategies

<div class="diagram">
  <div class="diagram-title">Branching Strategy Comparison</div>
  <div class="diagram-grid">
    <div class="diagram-card green">
      <strong>Trunk-Based Development</strong><br/>
      Short-lived branches (< 1 day)<br/>
      Merge to main frequently<br/>
      Feature flags for incomplete work<br/>
      ✓ Fastest integration<br/>
      ✓ Minimal merge conflicts<br/>
      ✓ Used by Google, Meta
    </div>
    <div class="diagram-card blue">
      <strong>GitHub Flow</strong><br/>
      Feature branches from main<br/>
      PR → Review → Merge<br/>
      Deploy from main<br/>
      ✓ Simple and effective<br/>
      ✓ Good for most teams<br/>
      ✓ Used by GitHub, Shopify
    </div>
    <div class="diagram-card orange">
      <strong>Git Flow</strong><br/>
      develop, feature, release, hotfix<br/>
      Long-lived branches<br/>
      Scheduled releases<br/>
      ✗ Complex and slow<br/>
      ✗ Merge conflicts<br/>
      ✗ Rarely appropriate for ML
    </div>
  </div>
</div>

**Recommendation for ML teams**: Use **GitHub Flow** or **trunk-based development**. ML experimentation benefits from fast iteration — long-lived branches mean your experiments diverge from the team's latest code.

### Trunk-Based Development in Practice

```bash
# Start from latest main
git checkout main && git pull

# Create a short-lived branch
git checkout -b add-warmup-scheduler

# Make changes, commit frequently
git add -A && git commit -m "Add WarmupCosineScheduler"

# Push and create PR
git push -u origin add-warmup-scheduler
gh pr create --title "Add LR warmup scheduler" --body "..."

# After approval, squash merge
gh pr merge --squash

# Delete branch
git branch -d add-warmup-scheduler
```

---

## 15.6 Feature Flags for Long-Running Work

When a feature takes more than a day, use feature flags instead of long-lived branches:

```python
# src/config.py
FEATURE_FLAGS = {
    "use_flash_attention_v2": True,
    "new_data_pipeline": False,       # In development
    "speculative_decoding": False,     # Experimental
    "streaming_inference": True,
}

# src/model.py
from src.config import FEATURE_FLAGS

def build_attention(config):
    if FEATURE_FLAGS["use_flash_attention_v2"]:
        return FlashAttentionV2(config)
    return StandardAttention(config)
```

```bash
# Toggle at runtime without code changes
USE_FLASH_ATTENTION_V2=true python src/serve.py
```

This lets you merge incomplete work to main daily, hidden behind a flag, avoiding merge conflicts and keeping CI green.

---

## 15.7 RFC Process for Major Changes

For significant changes (new training framework, architecture redesign, infrastructure migration), write an RFC before coding:

```markdown
# RFC: Migrate Training Pipeline to PyTorch Lightning

## Status: Proposed
## Author: @alice
## Date: 2024-03-15
## Reviewers: @bob, @carol, @dave

## Summary
Propose migrating our custom training loop to PyTorch Lightning to reduce
boilerplate and get built-in support for multi-GPU, mixed precision, and
checkpoint management.

## Motivation
Our custom training loop has grown to 800 lines with subtle bugs around
gradient accumulation and checkpoint resumption. Lightning handles these
correctly by default.

## Proposed Design
[Architecture diagram, key interfaces, migration plan]

## Alternatives Considered
1. **Hugging Face Trainer** — Too opinionated for our custom architectures
2. **Keep custom loop** — Maintenance burden is unsustainable
3. **JAX/Flax** — Too large a migration for the team

## Migration Plan
- Phase 1: Wrap existing model in LightningModule (1 week)
- Phase 2: Port data loading to LightningDataModule (1 week)
- Phase 3: Replace custom training loop (2 weeks)
- Phase 4: Validate on full training run (1 week)

## Risks
- Performance regression from Lightning overhead (~2% based on benchmarks)
- Team needs to learn Lightning API
- Custom callbacks for our experiment tracking

## Decision
[To be filled after review meeting]
```

---

## 15.8 Pair Programming and Mob Programming

### Pair Programming

Two developers work together on one task:

| Role | Responsibility |
|---|---|
| **Driver** | Writes the code, focuses on syntax and implementation |
| **Navigator** | Thinks about strategy, catches errors, considers edge cases |

**When to pair:**
- Onboarding a new team member
- Debugging a tricky issue
- Designing a new component
- Working on critical or high-risk code

### Mob Programming

The entire team works on one task together:

```
Mob Programming Session: Debug Training Data Pipeline
Duration: 2 hours
Driver rotates every 15 minutes

Team: Alice (Driver) → Bob → Carol → Dave → Alice...
Navigator: Whoever isn't driving
```

**When to mob:**
- Critical production incident
- Complex system design decisions
- Knowledge transfer for a system only one person understands

---

## 15.9 Collaborative Tools

| Tool | Purpose | ML Team Use Case |
|---|---|---|
| **GitHub Projects** | Project management, Kanban boards | Track experiment backlog, model releases |
| **Linear** | Fast issue tracking | Sprint planning, bug tracking |
| **Notion** | Documentation, wikis | Experiment logs, architecture decisions |
| **Slack/Discord** | Real-time communication | Alerts, experiment results, quick questions |
| **W&B Reports** | Experiment documentation | Share experiment comparisons with team |
| **Google Docs** | Collaborative writing | RFCs, design docs, postmortems |
| **Figma/Excalidraw** | Diagramming | System architecture, data flow diagrams |

### Automating Collaboration

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on:
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: |
          ruff check src/ tests/
          ruff format --check src/ tests/

      - name: Type check
        run: mypy src/ --ignore-missing-imports

      - name: Test
        run: pytest tests/ -v --tb=short

      - name: Check for large files
        run: |
          find . -name "*.pt" -o -name "*.ckpt" -o -name "*.safetensors" | \
            while read f; do
              size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
              if [ "$size" -gt 10485760 ]; then
                echo "ERROR: Large model file committed: $f ($size bytes)"
                exit 1
              fi
            done
```

### CODEOWNERS for ML Repos

```
# .github/CODEOWNERS

# Default reviewers
* @ml-team

# Model architecture changes need senior review
src/models/ @senior-ml-eng @ml-lead

# Data pipeline changes need data team review
src/data/ @data-team

# Infrastructure and deployment
Dockerfile* @infra-team
docker-compose* @infra-team
.github/workflows/ @infra-team

# Config changes — anyone can review
configs/ @ml-team

# Training scripts — need experiment review
src/train*.py @ml-lead @senior-ml-eng
```

---

## 15.10 Code Review Workflow Summary

<div class="diagram">
  <div class="diagram-title">Effective Code Review Workflow</div>
  <div class="cycle">
    <div class="cycle-step blue">Author: Write small, focused PR with good description</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step green">CI: Automated checks (lint, test, type check)</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step accent">Reviewer: Use checklist, give specific feedback</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step purple">Author: Address comments, push updates</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step orange">Reviewer: Approve when satisfied</div>
    <div class="cycle-arrow">→</div>
    <div class="cycle-step teal">Merge: Squash merge to main</div>
    <div class="cycle-arrow">↩</div>
  </div>
</div>

---

## 15.11 Key Takeaways

1. **Every change gets reviewed** — no exceptions, even for "small fixes."
2. **Keep PRs small** (< 400 lines) — large PRs get rubber-stamped, not reviewed.
3. **Write descriptive PR descriptions** — explain the why, not just the what.
4. **Use a checklist** — especially for ML code where bugs are silent.
5. **Be kind in reviews** — critique the code, not the person. Use prefixes (`nit:`, `blocking:`).
6. **Prefer trunk-based development** or GitHub Flow — avoid long-lived branches.
7. **Use feature flags** to merge incomplete work safely.
8. **Write RFCs for major changes** — get alignment before writing 5000 lines of code.
9. **Automate what you can** — lint, test, type check, and CODEOWNERS in CI.
10. **Code review is a skill** — practice it deliberately, just like writing code.

---

*Last updated: April 2026*
