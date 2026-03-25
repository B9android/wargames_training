# Wargames Training

[![CI](https://github.com/B9android/wargames_training/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/B9android/wargames_training/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reinforcement learning research project for training AI agents to control
Napoleonic-era military battalions in a continuous 2D simulation.

## Overview

The core atomic unit is a battalion (a 1D line segment on a 2D plane). The
training stack supports PPO-first workflows and extends into self-play,
multi-agent, league, and hierarchical RL pipelines.

### Tech Stack

- **Python 3.11** with **PyTorch** and **Stable-Baselines3**
- **Gymnasium** and **PettingZoo** for environment APIs
- **W&B** for experiment tracking
- **JAX/NumPy** for simulation performance

## Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/B9android/wargames_training.git
cd wargames_training
```

### 2 — Create a virtual environment

**Linux / macOS**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Set up W&B experiment tracking

```bash
# Create a free account at https://wandb.ai if you don't have one
wandb login
```

The default W&B project is configured in `configs/default.yaml`:

```yaml
wandb:
  project: "wargames_training"
  entity: null   # set to your W&B team/org name if using a shared workspace
```

### 5 — Run a quick smoke test

```bash
pytest tests/ -q
```

### 6 — Launch a baseline training run

```bash
python training/train.py
```

> **No W&B account?**  Run offline with `WANDB_MODE=offline python training/train.py` —
> logs are saved locally and can be synced later with `wandb sync`.

### 7 — Evaluate a trained checkpoint

```bash
python training/evaluate.py \
    --checkpoint checkpoints/best/best_model.zip \
    --opponent scripted_l5 \
    --n-episodes 100
```

Supported opponents: `scripted_l1` ... `scripted_l5`, `random`, or a path to
any `.zip` checkpoint.

## Common Commands

```bash
# Run full unit tests
pytest tests/ -q

# Run offline training (no W&B upload)
WANDB_MODE=offline python training/train.py

# Evaluate a custom checkpoint
python training/evaluate.py --checkpoint <path_to_model.zip> --opponent scripted_l3 --n-episodes 50
```

## Repository Layout

```
wargames_training/
├── configs/             # YAML config files (training, curricula, orchestration)
├── envs/                # Gymnasium and PettingZoo environments + simulation core
├── models/              # Policy/value network architectures
├── training/            # Train/eval entry points and callbacks
├── analysis/            # Explainability and course-of-action analysis tools
├── api/                 # Inference and COA-serving endpoints
├── frontend/            # Vite-based front-end client
├── docs/                # Project docs, guides, and architecture notes
├── scripts/             # Automation scripts and project agents
└── tests/               # Unit and integration tests
```

## GitHub Labels & Milestones

All required labels (`type:`, `priority:`, `status:`, `domain:`) and milestones
(M0–M4) are bootstrapped automatically.

**To create labels and milestones** in a fresh fork, dispatch the workflow:

1. Go to **Actions → 🏷️ Bootstrap: Labels & Milestones**
2. Set `dry_run = false` and click **Run workflow**

Or run locally from the repository root:

```bash
GITHUB_TOKEN=<your-pat> REPO_NAME=B9android/wargames_training \
    python scripts/project_agent/setup_labels_and_milestones.py
```

## Triaging Issues

The triage agent runs automatically when a new issue is opened (`.github/workflows/agent-triage.yml`). It reads title markers and applies labels + milestone:

| Title marker | Labels applied |
|---|---|
| `[BUG]` | `type: bug`, `priority: high` |
| `[EXP]` | `type: experiment`, `priority: medium` |
| `[EPIC]` | `type: epic`, `priority: high` |
| `[FEAT]` / `[FEATURE]` | `type: feature`, `priority: medium` |
| `[RESEARCH]` | `type: research`, `priority: low` |

To retriage an issue manually, dispatch **Actions → 🤖 Agent: Triage** with the issue number.

## Development Workflow

This project follows a **vertical-slice + iterative-deepening** approach:
build the thinnest end-to-end pipeline first, then deepen layer-by-layer
while keeping `main` always runnable.

| Resource | Description |
|---|---|
| **[Development Playbook](docs/development_playbook.md)** | Walking skeleton, iteration cycle, spike rules, Definition of Done, and per-feature checklists |
| **[Contributing Guide](CONTRIBUTING.md)** | Branch naming, commit style, conventions, and PR process |
| **[Project Roadmap](docs/ROADMAP.md)** | Version overview (v1–v5) and sprint schedule |
| **[Training Guide](docs/TRAINING_GUIDE.md)** | Hyperparameters reference, curriculum, self-play, W&B tips |
| **[Environment Spec](docs/ENVIRONMENT_SPEC.md)** | Observation/action space, reward function, scripted opponents |

## Documentation Index

- [Training Guide](docs/TRAINING_GUIDE.md)
- [Environment Spec](docs/ENVIRONMENT_SPEC.md)
- [Multi-Agent Guide](docs/multi_agent_guide.md)
- [League Training Guide](docs/league_training_guide.md)
- [HRL Architecture](docs/hrl_architecture.md)
- [Metrics Reference](docs/metrics_reference.md)
- [Project Report](docs/PROJECT_REPORT.md)

## For Operators

The project uses GitHub Actions for automated orchestration of experiments, milestones, and releases.

**See [Orchestration Runbook](docs/ORCHESTRATION_RUNBOOK.md)** for:

- Dispatch procedures for running agents manually
- Per-action checklists (triage, experiment approval, milestone closure)
- Release sync verification
- Local test commands

**Required secrets / environment variables:**

| Name | Purpose |
|---|---|
| `GITHUB_TOKEN` | GitHub API access (provided automatically in Actions) |
| `OPENAI_API_KEY` | Optional — enables AI triage fallback |
| `GH_PAT_PROJECT` | PAT with `project` scope — required for Project board automation |
| `DRY_RUN=true` | Set to run any agent in read-only / audit mode |

## Key Conventions

- All environments inherit from `gymnasium.Env` or `pettingzoo.ParallelEnv`
- Observations are always normalized to reasonable ranges
- Angles are represented as (cos θ, sin θ) pairs — never raw radians
- Positions are normalized by map dimensions
- Every training run **must** be logged to W&B with a config dict
- Every significant training run should have a corresponding GitHub issue using the `[EXP]` template

## Project Status

The repository includes single-battalion, multi-agent, league, and hierarchical
training pipelines, plus analysis and deployment tooling.

For the authoritative status, known gaps, and roadmap direction, see
[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) and
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

This project is licensed under the [MIT License](LICENSE).
