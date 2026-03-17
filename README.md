# 🪖 Wargames Training

Reinforcement learning research project training AI agents to control Napoleonic-era military battalions in a continuous 2D simulation.

---

## 📋 Overview

The atomic unit is a **battalion** — a 1D line segment on a 2D plane. Agents are trained using PPO, self-play, and (eventually) league training and hierarchical RL. Experiments are tracked with Weights & Biases.

**Current version: v1** — focus is 1v1 battalion training against scripted opponents, then self-play.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- `git`
- A [Weights & Biases](https://wandb.ai) account (free tier is fine)

### 1. Clone the repository

```bash
git clone https://github.com/B9android/wargames_training.git
cd wargames_training
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Weights & Biases

```bash
wandb login          # paste your API key when prompted
```

Then set your W&B entity in `configs/default.yaml`:

```yaml
wandb:
  project: wargames_training
  entity: YOUR_WANDB_USERNAME
```

### 5. Verify the environment

```bash
python -c "from envs.battalion_env import BattalionEnv; print('✅ Environment OK')"
```

### 6. Run a training session

```bash
python training/train.py --config configs/default.yaml
```

---

## 🗂️ Project Structure

```
wargames_training/
├── envs/               # Gymnasium environments & simulation engine
│   ├── battalion_env.py
│   ├── rendering/
│   └── sim/
├── models/             # Neural network architectures
│   ├── mlp_policy.py
│   └── transformer_policy.py
├── training/           # Training & evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── configs/            # YAML configuration files
│   ├── default.yaml    # Default hyperparameters & W&B config
│   └── experiment_1.yaml
├── scripts/
│   └── project_agent/  # GitHub automation agents
│       ├── triage_agent.py
│       ├── issue_writer.py
│       ├── milestone_checker.py
│       ├── progress_reporter.py
│       ├── training_monitor.py
│       ├── setup_labels.py      # Creates all repo labels
│       └── setup_milestones.py  # Creates M0–M4 milestones
├── notebooks/          # Jupyter analysis notebooks
├── docs/               # Design documents
└── requirements.txt
```

---

## ⚙️ Configuration

All training behaviour is controlled by YAML files in `configs/`.

Key settings in `configs/default.yaml`:

| Key | Description |
|-----|-------------|
| `wandb.project` | W&B project name |
| `wandb.entity` | Your W&B username / org |
| `env.map_width` | Simulation map width (metres) |
| `env.map_height` | Simulation map height (metres) |
| `training.total_timesteps` | Total environment steps |
| `training.n_envs` | Number of parallel environments |
| `ppo.*` | PPO hyperparameters |

Override any value on the command line:

```bash
python training/train.py training.total_timesteps=1000000 ppo.learning_rate=0.0001
```

---

## 🏷️ GitHub Workflow

### Labels

The repo uses a structured label taxonomy:

- **`type:`** — `bug`, `feature`, `experiment`, `epic`, `research`, `infrastructure`, `documentation`, `chore`
- **`priority:`** — `critical`, `high`, `medium`, `low`
- **`status:`** — `in-progress`, `blocked`, `needs-review`, `ready`, `on-hold`, `stale`, `agent-created`

Run the setup script once to create all labels and milestones:

```bash
GITHUB_TOKEN=<token> REPO_NAME=B9android/wargames_training \
  python scripts/project_agent/setup_labels.py

GITHUB_TOKEN=<token> REPO_NAME=B9android/wargames_training \
  python scripts/project_agent/setup_milestones.py
```

### Milestones

| Milestone | Description |
|-----------|-------------|
| M0: Project Bootstrap | Repo, CI, tooling, dev environment |
| M1: 1v1 Competence | Beat scripted opponent reliably |
| M2: Self-Play | Stable self-play loop |
| M3: League Training | Multi-agent league |
| M4: HRL | Hierarchical reinforcement learning |

### Automation Agents

GitHub Actions run AI agents for routine project management:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `agent-triage` | New issue opened | Label, milestone, and comment |
| `agent-milestone-check` | Daily | Flag at-risk milestones & stale issues |
| `agent-progress-report` | Weekly | Summary of milestone progress |
| `agent-issue-writer` | Manual dispatch | Generate follow-up issues from context |
| `training-monitor` | Manual dispatch | Parse training results & open issues |

---

## 🧪 Testing

```bash
pytest
```

---

## 🤝 Contributing

1. Open an issue (or find an existing one) using the appropriate template.
2. Create a branch from `main`.
3. Open a PR using the PR template — link the issue.
4. Ensure `pytest` passes and (if env changed) `check_env` passes.

See `.github/PULL_REQUEST_TEMPLATE.md` for the full checklist.

---

## 📊 Experiment Tracking

All training runs are logged to [Weights & Biases](https://wandb.ai/B9android/wargames_training).

Each significant experiment should have:
- A GitHub issue using the `[EXP]` template
- A W&B run linked in the issue body
- A config file committed to `configs/`
