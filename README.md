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

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Project Structure

- **envs/** — Gymnasium environments and simulation engine
- **models/** — Neural network architectures  
- **training/** — Training scripts and callbacks
- **configs/** — YAML configuration files
- **scripts/project_agent/** — GitHub automation agents
- **notebooks/** — Jupyter notebooks for analysis and debugging

## For Operators

The project uses GitHub Actions for automated orchestration of experiments, milestones, and releases.

**See [Orchestration Runbook](docs/ORCHESTRATION_RUNBOOK.md)** for:

- Dispatch procedures for running agents manually
- Per-action checklists (triage, experiment approval, milestone closure)
- Release sync verification
- Local test commands

**Environment Setup:**

- `GITHUB_TOKEN`: Personal access token for GitHub API (required for all agents)
- `DRY_RUN=true`: Run orchestration agents in dry-run mode (no mutations)

## Key Conventions

- All environments inherit from `gymnasium.Env` or `pettingzoo.ParallelEnv`
- Observations are always normalized to reasonable ranges
- Angles are represented as (cos θ, sin θ) pairs — never raw radians
- Positions are normalized by map dimensions
- Every training run **must** be logged to W&B with a config dict
- Every significant training run should have a corresponding GitHub issue using the [EXP] template

## Current Version: v1

Focus is on 1v1 battalion training against scripted opponents, then self-play.
Multi-agent complexity will be added after v1 milestones are met.

## License

[Add license here]
