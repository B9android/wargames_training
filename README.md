# Wargames Training

Reinforcement learning research project training AI agents to control Napoleonic-era military battalions in a continuous 2D simulation.

## Overview

Agents are trained using PPO, self-play, and eventually league training and hierarchical RL. The atomic unit is a battalion (1D line segment on a 2D plane).

### Tech Stack

- **Python 3.11** with **PyTorch** and **Stable-Baselines3**
- **Gymnasium** and **PettingZoo** for environment APIs
- **W&B** for experiment tracking
- **JAX/NumPy** for simulation performance

## Getting Started

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

### 6 — Launch a training run

```bash
python training/train.py
```

## Project Structure

```
wargames_training/
├── configs/             # YAML config files (default.yaml, experiment_*.yaml)
├── envs/                # Gymnasium environments and simulation engine
│   └── sim/             # Core sim: battalion, combat, terrain modules
├── models/              # Neural network architectures
├── notebooks/           # Jupyter notebooks for analysis and debugging
├── scripts/
│   └── project_agent/   # GitHub automation agents
├── tests/               # Unit tests
└── training/            # Training scripts and callbacks
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

## Current Version: v1

Focus is on 1v1 battalion training against scripted opponents, then self-play.
Multi-agent complexity will be added after v1 milestones are met.

## License

[Add license here]
