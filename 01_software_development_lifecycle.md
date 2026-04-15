[← Back to Table of Contents](./README.md)

# Chapter 1 — Software Development Lifecycle

> *"Plans are worthless, but planning is everything."* — Dwight D. Eisenhower

The Software Development Lifecycle (SDLC) provides a structured framework for building software systems. In Machine Learning, the lifecycle extends beyond traditional software to encompass data collection, experimentation, model training, evaluation, and deployment — each introducing unique challenges that demand adapted processes.

---

## 1.1 SDLC Overview

The SDLC defines the phases a software project moves through from inception to retirement. For ML systems, these phases interleave with the experimental, iterative nature of research.

<div class="diagram">
<div class="diagram-title">SDLC Phases for ML Systems</div>
<div class="flow">
<div class="flow-node accent">Requirements &amp; Problem Framing</div>
<div class="flow-arrow">→</div>
<div class="flow-node green">Data Collection &amp; Analysis</div>
<div class="flow-arrow">→</div>
<div class="flow-node blue">System &amp; Model Design</div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">Implementation &amp; Training</div>
<div class="flow-arrow">→</div>
<div class="flow-node orange">Testing &amp; Evaluation</div>
<div class="flow-arrow">→</div>
<div class="flow-node teal">Deployment &amp; Monitoring</div>
<div class="flow-arrow">→</div>
<div class="flow-node pink">Maintenance &amp; Retraining</div>
</div>
</div>

Each phase feeds back into earlier stages — a model that underperforms in evaluation may require new data, revised features, or an entirely different architecture.

### Key Differences from Traditional Software

| Aspect | Traditional Software | ML Software |
|---|---|---|
| **Requirements** | Deterministic specs | Probabilistic performance targets |
| **Design** | Architecture diagrams | Architecture + experiment hypotheses |
| **Implementation** | Write code | Write code + train models |
| **Testing** | Pass/fail assertions | Statistical evaluation metrics |
| **Deployment** | Ship binary | Ship model + inference code + data pipeline |
| **Maintenance** | Bug fixes, features | Retraining, data drift monitoring |

---

## 1.2 Waterfall vs Agile

### Waterfall

The Waterfall model is a sequential, phase-gate process: each phase must be completed before the next begins. It works well when requirements are fully known upfront — which is rarely the case in ML.

**When Waterfall may apply in ML:**
- Regulatory or compliance-driven projects (medical ML, autonomous vehicles) where documentation gates are mandatory
- Well-understood problems with proven architectures (e.g., deploying a known fraud detection model to a new market)

### Agile

Agile methodologies embrace iterative development, frequent feedback, and adaptive planning — a natural fit for the experimental nature of ML work.

<div class="diagram">
<div class="diagram-title">Waterfall vs Agile for ML Projects</div>
<div class="compare">
<div class="compare-side">
<h4>Waterfall</h4>
<div class="flow">
<div class="flow-node accent">Requirements (fixed)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node blue">Design (complete)</div>
<div class="flow-arrow">↓</div>
<div class="flow-node green">Implementation</div>
<div class="flow-arrow">↓</div>
<div class="flow-node orange">Testing</div>
<div class="flow-arrow">↓</div>
<div class="flow-node purple">Deployment</div>
</div>
</div>
<div class="compare-side">
<h4>Agile / Iterative</h4>
<div class="cycle">
<div class="cycle-step green">Plan Sprint</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step blue">Experiment &amp; Build</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step orange">Evaluate &amp; Review</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step purple">Retrospect &amp; Adapt</div>
<div class="cycle-arrow">↩</div>
</div>
</div>
</div>
</div>

> *"In ML, you don't know if your approach will work until you try it. Agile's inspect-and-adapt loop is not optional — it's survival."* — Adapted from CMU 17-445

---

## 1.3 Scrum for ML Teams

Scrum provides a lightweight framework of roles, events, and artifacts that ML teams can adapt for their dual nature: **research exploration** and **production engineering**.

### Roles in an ML Scrum Team

| Role | Traditional Scrum | ML Adaptation |
|---|---|---|
| **Product Owner** | Defines features | Defines ML success metrics, prioritizes experiments |
| **Scrum Master** | Removes blockers | Manages GPU allocation, data access, infra blockers |
| **Dev Team** | Engineers | ML engineers, data scientists, data engineers, MLOps |

### Sprints: Research vs Production

A key tension in ML teams is balancing open-ended research with production delivery. One effective pattern is the **dual-track sprint**:

- **Research track (exploration):** Time-boxed experiments with clear hypotheses. Success is measured by learning, not just deliverables.
- **Production track (exploitation):** Traditional sprint work — shipping features, fixing bugs, improving infrastructure.

```yaml
# sprint_config.yaml — Dual-track sprint planning
sprint:
  number: 14
  duration_weeks: 2
  start_date: "2026-03-16"
  end_date: "2026-03-30"

  tracks:
    research:
      capacity_percent: 40
      goals:
        - hypothesis: "LoRA fine-tuning reduces training cost by 60% with <2% quality loss"
          experiment_id: "exp-2026-03-lora-finetune"
          success_criteria:
            - metric: "eval_loss"
              threshold: 0.35
              direction: "lower_is_better"
            - metric: "gpu_hours"
              threshold: 120
              direction: "lower_is_better"
          tracking_url: "https://wandb.ai/team/project/runs/exp-lora"

    production:
      capacity_percent: 60
      goals:
        - title: "Deploy v2.3 model to staging"
          story_points: 8
          acceptance_criteria:
            - "Model serves at p99 < 200ms"
            - "A/B test framework configured"
            - "Rollback runbook documented"
        - title: "Add data validation to ingestion pipeline"
          story_points: 5
          acceptance_criteria:
            - "Great Expectations suite passes on new data"
            - "Alert fires on schema drift"

  ceremonies:
    standup: "daily 9:30 AM, 15 min"
    planning: "sprint day 1, 2 hours"
    review: "sprint day 10, 1 hour"
    retrospective: "sprint day 10, 1 hour"
    experiment_review: "weekly Wednesday, 30 min"
```

### User Stories for ML Features

Traditional user stories follow the format: *"As a [role], I want [feature], so that [benefit]."* For ML, we extend this with **acceptance criteria that include model performance**:

```markdown
## User Story Template for ML Features

**Title:** Improve product search relevance with semantic embeddings

**As a** e-commerce customer,
**I want** search results to understand the meaning of my query,
**So that** I find relevant products even when I don't use exact keywords.

### Acceptance Criteria
- [ ] Semantic search returns relevant results for 85%+ of test queries
- [ ] MRR@10 >= 0.45 on the held-out evaluation set
- [ ] p95 latency < 150ms for embedding lookup + retrieval
- [ ] Graceful fallback to keyword search if embedding service is unavailable
- [ ] A/B test shows >= 3% improvement in click-through rate

### Technical Notes
- Model: fine-tuned `all-MiniLM-L6-v2` on product catalog
- Infrastructure: FAISS index served via FastAPI
- Data dependency: nightly product catalog export from warehouse

### Definition of Done
- [ ] Code reviewed and merged to main
- [ ] Unit and integration tests passing
- [ ] Model registered in MLflow with evaluation metrics
- [ ] Deployed to staging, load tested
- [ ] Monitoring dashboard configured (latency, error rate, drift)
- [ ] Runbook updated with rollback procedure
```

---

## 1.4 Kanban for MLOps

While Scrum works well for feature development, **Kanban** suits continuous operational work like MLOps — where work items flow continuously rather than in fixed sprints.

<div class="diagram">
<div class="diagram-title">MLOps Kanban Board</div>
<div class="diagram-grid">
<div class="diagram-card accent">
<strong>Backlog</strong><br/>
• Set up model monitoring<br/>
• Migrate to new GPU cluster<br/>
• Add data lineage tracking
</div>
<div class="diagram-card blue">
<strong>Ready</strong><br/>
• Automate retraining pipeline<br/>
• Fix feature store latency
</div>
<div class="diagram-card green">
<strong>In Progress (WIP: 3)</strong><br/>
• Deploy canary for v2.4<br/>
• Debug data drift alert<br/>
• Update Docker base image
</div>
<div class="diagram-card orange">
<strong>Review / Validate</strong><br/>
• Load test new endpoint<br/>
• Verify rollback procedure
</div>
<div class="diagram-card purple">
<strong>Done</strong><br/>
• Upgrade CUDA to 12.3<br/>
• Add GPU utilization alerts
</div>
</div>
</div>

### Kanban Principles for MLOps

1. **Visualize the workflow** — make all in-flight work visible
2. **Limit work in progress (WIP)** — prevent context-switching; a WIP limit of 2-3 per person is typical
3. **Manage flow** — measure cycle time from "Ready" to "Done"
4. **Make policies explicit** — define when a model is "production-ready"
5. **Improve collaboratively** — use metrics (lead time, deployment frequency) to drive improvements

---

## 1.5 Sprint Planning with Experiment Tracking

Effective ML sprint planning integrates experiment tracking tools (W&B, MLflow, Neptune) directly into the planning process.

```python
"""
sprint_planning.py — Automated sprint planning helper that pulls
experiment results from W&B to inform sprint priorities.
"""
import wandb
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentResult:
    """Summary of an ML experiment for sprint review."""
    run_id: str
    hypothesis: str
    metric_name: str
    metric_value: float
    target_value: float
    gpu_hours: float
    status: str  # "success", "failure", "inconclusive"

    @property
    def met_target(self) -> bool:
        return self.metric_value <= self.target_value


def fetch_sprint_experiments(
    project: str,
    sprint_tag: str,
) -> list[ExperimentResult]:
    """Fetch all experiments tagged with the current sprint."""
    api = wandb.Api()
    runs = api.runs(
        project,
        filters={"tags": {"$in": [sprint_tag]}},
    )

    results = []
    for run in runs:
        summary = run.summary
        config = run.config
        results.append(ExperimentResult(
            run_id=run.id,
            hypothesis=config.get("hypothesis", "N/A"),
            metric_name=config.get("primary_metric", "eval_loss"),
            metric_value=summary.get(config.get("primary_metric", "eval_loss"), float("inf")),
            target_value=config.get("target_value", 0.0),
            gpu_hours=summary.get("_wandb", {}).get("runtime", 0) / 3600,
            status=run.state,
        ))

    return results


def generate_sprint_report(results: list[ExperimentResult]) -> str:
    """Generate a markdown report for sprint review."""
    lines = ["# Sprint Experiment Report\n"]

    succeeded = [r for r in results if r.met_target]
    failed = [r for r in results if not r.met_target]

    lines.append(f"**Total experiments:** {len(results)}")
    lines.append(f"**Met target:** {len(succeeded)}")
    lines.append(f"**Below target:** {len(failed)}")
    lines.append(f"**Total GPU hours:** {sum(r.gpu_hours for r in results):.1f}\n")

    lines.append("## Results\n")
    lines.append("| Hypothesis | Metric | Value | Target | Status |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        status = "✅" if r.met_target else "❌"
        lines.append(
            f"| {r.hypothesis} | {r.metric_name} | {r.metric_value:.4f} "
            f"| {r.target_value:.4f} | {status} |"
        )

    return "\n".join(lines)
```

---

## 1.6 Definition of Done for ML

A robust "Definition of Done" (DoD) for ML work extends traditional software DoD with ML-specific criteria:

| Category | Criteria |
|---|---|
| **Code** | Code reviewed, merged, linted, type-checked |
| **Tests** | Unit tests pass, integration tests pass, model tests pass |
| **Model** | Evaluation metrics meet acceptance criteria on held-out test set |
| **Data** | Data pipeline tested, schema validated, no data leakage confirmed |
| **Documentation** | Model card written, API docs updated, runbook reviewed |
| **Deployment** | Deployed to staging, load tested, canary passed |
| **Monitoring** | Dashboards configured for latency, errors, data drift, model quality |
| **Reproducibility** | Random seeds fixed, environment pinned, training reproducible within tolerance |

> *"A model that can't be reproduced, monitored, and rolled back is not done — it's a liability."*

---

## 1.7 Putting It All Together

The best ML teams combine elements from multiple methodologies:

1. **Scrum** for feature sprints with dual-track research/production planning
2. **Kanban** for continuous MLOps and incident response
3. **Experiment tracking** integrated into sprint ceremonies
4. **Extended DoD** that covers model quality, data integrity, and operational readiness

<div class="diagram">
<div class="diagram-title">Integrated ML Development Process</div>
<div class="cycle">
<div class="cycle-step accent">Sprint Planning<br/><small>Prioritize experiments + features</small></div>
<div class="cycle-arrow">→</div>
<div class="cycle-step green">Development<br/><small>Code, train, experiment</small></div>
<div class="cycle-arrow">→</div>
<div class="cycle-step blue">Evaluation<br/><small>Metrics, A/B tests, review</small></div>
<div class="cycle-arrow">→</div>
<div class="cycle-step orange">Deployment<br/><small>Canary, monitor, validate</small></div>
<div class="cycle-arrow">→</div>
<div class="cycle-step purple">Retrospective<br/><small>Learn, adapt, improve</small></div>
<div class="cycle-arrow">↩</div>
</div>
</div>

### Further Reading

- Sculley et al., *"Hidden Technical Debt in Machine Learning Systems"* (NeurIPS 2015)
- Amershi et al., *"Software Engineering for Machine Learning: A Case Study"* (ICSE 2019)
- CMU 17-445: *Software Engineering for AI-Enabled Systems*
- Google, *"Rules of Machine Learning"*

---

*Last updated: April 2026*
