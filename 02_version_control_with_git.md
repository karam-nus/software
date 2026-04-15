[← Back to Table of Contents](./README.md)

# Chapter 2 — Version Control with Git

> *"If you can't reproduce it, you can't trust it. Version control is the foundation of reproducibility."*

Version control is the single most important practice in professional software engineering. For ML, it extends beyond code to encompass data versions, model artifacts, experiment configurations, and training environments. This chapter covers Git fundamentals and the workflows that make ML teams productive.

---

## 2.1 Why Version Control Matters for ML

In traditional software, version control tracks code changes. In ML, we must track a richer set of artifacts:

| Artifact | Tool | Why |
|---|---|---|
| **Source code** | Git | Track training scripts, pipelines, serving code |
| **Configuration** | Git | Hyperparameters, model architecture configs |
| **Data** | DVC, Git LFS, lakeFS | Dataset versions, preprocessing transformations |
| **Models** | MLflow, W&B, DVC | Trained weights, evaluation metrics |
| **Environments** | Docker, conda-lock | Exact dependency versions for reproducibility |
| **Experiments** | W&B, MLflow, Neptune | Hyperparameter sweeps, metric comparisons |

> *"A model is not just code — it is code + data + config + environment. Version all four."* — Adapted from Google's ML Best Practices

---

## 2.2 Git Fundamentals

### Core Concepts

<div class="diagram">
<div class="diagram-title">Git's Three Trees</div>
<div class="flow">
<div class="flow-node accent">Working Directory<br/><small>Your files on disk</small></div>
<div class="flow-arrow">git add →</div>
<div class="flow-node green">Staging Area (Index)<br/><small>Next commit snapshot</small></div>
<div class="flow-arrow">git commit →</div>
<div class="flow-node blue">Repository (.git)<br/><small>Permanent history</small></div>
<div class="flow-arrow">git push →</div>
<div class="flow-node purple">Remote (GitHub)<br/><small>Shared history</small></div>
</div>
</div>

### Essential Commands

```bash
# --- Initialize & Configure ---
git init                              # Create a new repository
git clone <url>                       # Clone an existing repository
git config user.name "Ada Lovelace"   # Set author name
git config user.email "ada@ml.org"    # Set author email

# --- Daily Workflow ---
git status                            # See what's changed
git diff                              # See unstaged changes
git diff --staged                     # See staged changes
git add model/trainer.py              # Stage a specific file
git add -p                            # Stage interactively (hunk by hunk)
git commit -m "feat: add learning rate warmup scheduler"
git push origin feature/lr-warmup     # Push to remote branch

# --- Inspect History ---
git log --oneline --graph -20         # Visual commit history
git log --follow -- src/model.py      # Track file through renames
git blame src/trainer.py              # Who changed each line
git show abc1234                      # Show a specific commit

# --- Undo Mistakes ---
git restore src/model.py              # Discard unstaged changes
git restore --staged src/model.py     # Unstage a file
git commit --amend                    # Fix the last commit message/content
git revert abc1234                    # Create a new commit that undoes abc1234
git reset --soft HEAD~1               # Undo last commit, keep changes staged
```

---

## 2.3 Branching Strategies

Choosing the right branching strategy is critical for team productivity. Here are the three most common approaches:

<div class="diagram">
<div class="diagram-title">Branching Strategies Compared</div>
<div class="compare">
<div class="compare-side">
<h4>Git Flow</h4>
<div class="flow">
<div class="flow-node accent">main</div>
<div class="flow-arrow">↓</div>
<div class="flow-node blue">develop</div>
<div class="flow-arrow">↓</div>
<div class="flow-node green">feature/*</div>
<div class="flow-arrow">↓</div>
<div class="flow-node orange">release/*</div>
<div class="flow-arrow">↓</div>
<div class="flow-node red">hotfix/*</div>
</div>
</div>
<div class="compare-side">
<h4>GitHub Flow</h4>
<div class="flow">
<div class="flow-node accent">main (always deployable)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node green">feature branch</div>
<div class="flow-arrow">↓</div>
<div class="flow-node blue">Pull Request + Review</div>
<div class="flow-arrow">↓</div>
<div class="flow-node purple">Merge &amp; Deploy</div>
</div>
</div>
</div>
</div>

### Strategy Comparison

| Strategy | Complexity | Best For | ML Suitability |
|---|---|---|---|
| **Git Flow** | High | Large teams, versioned releases | ⚠️ Overkill for most ML projects |
| **GitHub Flow** | Low | Continuous deployment, small teams | ✅ Great for ML services |
| **Trunk-Based** | Medium | High-velocity teams, feature flags | ✅ Great for MLOps with feature flags |

### Recommended: GitHub Flow for ML Teams

```bash
# 1. Create a feature branch from main
git checkout main
git pull origin main
git checkout -b feat/add-attention-pooling

# 2. Make changes, commit frequently
git add src/models/attention_pool.py tests/test_attention_pool.py
git commit -m "feat: add multi-head attention pooling layer"

git add src/models/encoder.py
git commit -m "refactor: integrate attention pooling into encoder"

# 3. Push and open a pull request
git push origin feat/add-attention-pooling
gh pr create --title "feat: add attention pooling layer" \
             --body "## Summary\nAdds multi-head attention pooling..."

# 4. After review approval, merge
gh pr merge --squash  # Clean history with squash merge
```

### Branch Naming Conventions

```
feat/add-transformer-encoder     # New feature
fix/data-loader-memory-leak      # Bug fix
exp/lora-rank-ablation           # Experiment (may not merge)
refactor/training-loop-cleanup   # Code improvement
docs/update-model-card           # Documentation
ci/add-gpu-test-runner           # CI/CD changes
```

---

## 2.4 Merge vs Rebase

<div class="diagram">
<div class="diagram-title">Merge vs Rebase</div>
<div class="compare">
<div class="compare-side">
<h4>Merge (preserves history)</h4>
<div class="flow">
<div class="flow-node blue">A — B — C (main)</div>
<div class="flow-arrow">↘ ↗</div>
<div class="flow-node green">D — E (feature)</div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">M (merge commit)</div>
</div>
</div>
<div class="compare-side">
<h4>Rebase (linear history)</h4>
<div class="flow">
<div class="flow-node blue">A — B — C (main)</div>
<div class="flow-arrow">→</div>
<div class="flow-node green">D' — E' (rebased)</div>
</div>
</div>
</div>
</div>

```bash
# Merge approach — preserves branch history
git checkout main
git merge feat/new-tokenizer

# Rebase approach — linear history
git checkout feat/new-tokenizer
git rebase main
git checkout main
git merge --ff-only feat/new-tokenizer

# Interactive rebase — clean up before merging
git rebase -i HEAD~5   # Squash/reorder last 5 commits
```

**Team guideline:** Rebase your feature branch onto main before merging. Use squash merges for PRs. Never rebase shared branches.

---

## 2.5 Conventional Commits

Conventional Commits provide a structured format that enables automated changelog generation and semantic versioning.

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types for ML Projects

| Type | Description | Example |
|---|---|---|
| `feat` | New feature | `feat(model): add rotary position embeddings` |
| `fix` | Bug fix | `fix(dataloader): resolve memory leak in prefetch` |
| `exp` | Experiment | `exp(training): test cosine annealing with warm restarts` |
| `data` | Data changes | `data(corpus): add CommonCrawl 2025-Q4 snapshot` |
| `perf` | Performance | `perf(inference): enable Flash Attention 2 for 40% speedup` |
| `refactor` | Code restructuring | `refactor(pipeline): extract feature engineering module` |
| `test` | Testing | `test(model): add gradient flow assertions` |
| `ci` | CI/CD changes | `ci: add A100 GPU runner for integration tests` |
| `docs` | Documentation | `docs: update model card with bias analysis` |
| `chore` | Maintenance | `chore: upgrade PyTorch to 2.3.0` |

```bash
# Good commit messages
git commit -m "feat(serving): add batched inference endpoint

Implements dynamic batching with configurable max_batch_size
and max_wait_time_ms parameters. Throughput improves 3.2x on
A100 for batch sizes >= 16.

Closes #142"

git commit -m "fix(training): prevent NaN loss with gradient clipping

Loss was diverging after ~10k steps when learning rate exceeded 3e-4.
Added gradient norm clipping at 1.0, matching the approach in
Llama 2 (Touvron et al., 2023).

Fixes #287"
```

---

## 2.6 .gitignore for ML Projects

ML projects have unique files that should never be committed — large datasets, model checkpoints, experiment logs, and secrets.

```gitignore
# ============================================
# .gitignore for ML Projects
# ============================================

# --- Python ---
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg
.venv/
venv/

# --- Jupyter Notebooks ---
.ipynb_checkpoints/
*.ipynb_metadata/

# --- ML Data & Artifacts ---
data/raw/
data/processed/
data/interim/
*.csv
*.parquet
*.arrow
*.tfrecord
*.h5
*.hdf5
!data/README.md          # Keep the data README

# --- Model Checkpoints ---
checkpoints/
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.onnx
models/weights/

# --- Experiment Tracking ---
wandb/
mlruns/
outputs/                  # Hydra outputs
multirun/                 # Hydra multi-run
neptune/
lightning_logs/

# --- Environment & Secrets ---
.env
.env.*
*.pem
secrets/

# --- IDE ---
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# --- Large Files (use Git LFS instead) ---
*.zip
*.tar.gz
*.tar.bz2
```

---

## 2.7 Git LFS for Large Files

Git LFS (Large File Storage) replaces large files with lightweight pointers in the repository while storing file contents on a separate server.

```bash
# Install and set up Git LFS
git lfs install

# Track large file types
git lfs track "*.safetensors"
git lfs track "*.parquet"
git lfs track "*.onnx"
git lfs track "data/embeddings/*.npy"

# This creates/updates .gitattributes — commit it!
git add .gitattributes
git commit -m "chore: configure Git LFS for model and data files"

# Now use Git normally — LFS handles large files transparently
git add models/encoder.safetensors
git commit -m "data: add trained encoder weights v2.1"
git push origin main
```

### When to Use Git LFS vs DVC

| Criteria | Git LFS | DVC |
|---|---|---|
| **File size** | < 2 GB per file | Any size |
| **Storage backend** | GitHub / GitLab LFS | S3, GCS, Azure, SSH |
| **Data pipelines** | ❌ No | ✅ Yes |
| **Experiment tracking** | ❌ No | ✅ Yes |
| **Team size** | Small–medium | Any |
| **Complexity** | Low | Medium |

```yaml
# .gitattributes — LFS tracking rules
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
data/embeddings/**/*.npy filter=lfs diff=lfs merge=lfs -text
```

---

## 2.8 Monorepos for ML

Large ML organizations (Google, Meta) often use monorepos — a single repository containing multiple projects, shared libraries, and configurations.

<div class="diagram">
<div class="diagram-title">ML Monorepo Structure</div>
<div class="diagram-grid">
<div class="diagram-card accent">
<strong>models/</strong><br/>
• transformer/<br/>
• diffusion/<br/>
• retrieval/
</div>
<div class="diagram-card green">
<strong>data/</strong><br/>
• pipelines/<br/>
• schemas/<br/>
• validation/
</div>
<div class="diagram-card blue">
<strong>training/</strong><br/>
• configs/<br/>
• distributed/<br/>
• callbacks/
</div>
<div class="diagram-card purple">
<strong>serving/</strong><br/>
• api/<br/>
• batch/<br/>
• monitoring/
</div>
<div class="diagram-card orange">
<strong>libs/</strong><br/>
• tokenizers/<br/>
• metrics/<br/>
• utils/
</div>
<div class="diagram-card teal">
<strong>infra/</strong><br/>
• docker/<br/>
• terraform/<br/>
• ci/
</div>
</div>
</div>

### Monorepo Advantages for ML

- **Shared libraries** — common utilities, metrics, data loaders used across projects
- **Atomic changes** — update a shared tokenizer and all dependent models in one commit
- **Consistent tooling** — one linter config, one CI pipeline, one dependency set
- **Code discoverability** — search across all projects to find patterns and examples

### Monorepo Tools

```bash
# Pants — popular for Python ML monorepos
pants test src/models/transformer/::   # Test everything under transformer/
pants package src/serving/api:docker   # Build Docker image for serving

# Bazel — used at Google scale
bazel build //models/transformer:train
bazel test //data/pipelines/...:all
```

---

## 2.9 Practical Git Workflows for ML

### Starting a New Experiment

```bash
# Create an experiment branch
git checkout -b exp/rope-scaling-ablation

# Make your changes
vim configs/rope_scaling.yaml
vim src/models/position_encoding.py

# Commit with experiment metadata
git commit -m "exp(positional): test RoPE scaling factors [1.0, 2.0, 4.0]

Hypothesis: RoPE scaling factor of 2.0 extends context window from 4k
to 8k tokens without significant quality degradation.

Tracking: wandb.ai/team/project/sweeps/rope-ablation
Configs: configs/rope_scaling.yaml"

# Push and track
git push origin exp/rope-scaling-ablation
```

### Recovering from Common Mistakes

```bash
# Accidentally committed a large model file
git rm --cached models/large_model.bin
echo "models/large_model.bin" >> .gitignore
git commit -m "fix: remove accidentally committed model binary"

# Committed to wrong branch
git stash                          # Save current changes
git checkout correct-branch
git stash pop                      # Apply changes here

# Need to split a commit into two
git reset HEAD~1                   # Undo the commit, keep changes
git add src/model.py
git commit -m "feat: add model architecture"
git add src/trainer.py
git commit -m "feat: add training loop"

# Find which commit introduced a bug
git bisect start
git bisect bad HEAD                # Current commit is broken
git bisect good v1.2.0             # This tag was working
# Git will checkout commits for you to test — run your test each time
git bisect good                    # or: git bisect bad
```

---

## 2.10 Summary

| Practice | Recommendation |
|---|---|
| **Branching** | Use GitHub Flow; prefix branches with `feat/`, `fix/`, `exp/` |
| **Commits** | Conventional Commits format; small, atomic commits |
| **Merging** | Squash merge PRs; rebase feature branches on main |
| **Large files** | Git LFS for < 2GB; DVC for larger datasets |
| **Secrets** | Never commit; use `.env` + `.gitignore` |
| **History** | Never rewrite shared history; use `git revert` for public fixes |

---

*Last updated: April 2026*
