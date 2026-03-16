# Wargames Training — AI Agent Instructions

## Project Overview
This is a reinforcement learning research project training AI agents to control
Napoleonic-era military battalions in a continuous 2D simulation. The atomic unit
is a battalion (a 1D line segment on a 2D plane). Agents are trained using PPO,
self-play, and eventually league training and HRL.

## Tech Stack
- Python 3.11, PyTorch, Stable-Baselines3, Gymnasium, PettingZoo
- W&B for experiment tracking
- JAX/NumPy for simulation speed
- GitHub Actions for CI and agent automation

## Key Conventions
- All environments inherit from `gymnasium.Env` or `pettingzoo.ParallelEnv`
- Observations are always normalized to reasonable ranges
- Angles are always represented as (cos θ, sin θ) pairs — never raw radians
- Positions are normalized by map dimensions
- Every training run must be logged to W&B with a config dict
- Every significant training run should have a corresponding GitHub issue
  using the [EXP] template

## Project Structure
- `envs/` — Gymnasium environments and simulation engine
- `models/` — Neural network architectures
- `training/` — Training scripts and callbacks
- `configs/` — YAML configuration files
- `scripts/project_agent/` — GitHub automation agents

## Current Version: v1
Focus is on 1v1 battalion training against scripted opponents,
then self-play. Do not add multi-agent complexity until v1 milestones are met.

## Agent Behavior Guidelines
- When creating issues, always use the appropriate issue template format
- Label all agent-created issues with `status: agent-created`
- Never close issues — only humans close issues
- When in doubt about scope, create a smaller, more focused issue
- Prefer opening new issues over editing existing ones
