# Changelog

All notable changes to wargames_training will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] — 2026-03-19

### Added
- **Simulation engine** (`envs/sim/`) — battalion, combat (damage accumulation,
  morale mechanics, routing threshold), and procedural terrain with elevation
  and cover.
- **`BattalionEnv`** — Gymnasium 1v1 environment with 12-dim observation space,
  3-dim continuous action space, scripted Red opponent (curriculum levels 1–5),
  randomized terrain, and configurable reward shaping.
- **`BattalionMlpPolicy`** — Stable-Baselines3 `ActorCriticPolicy` with a
  dedicated `BattalionFeaturesExtractor` (obs → 128 → 64, ReLU + LayerNorm).
- **PPO training pipeline** (`training/train.py`) — Hydra config loading, W&B
  experiment tracking, `CheckpointCallback`, `EvalCallback`,
  `WandbCallback`, `RewardBreakdownCallback`.
- **Elo tracking** (`training/elo.py`) — `EloRegistry` with JSON persistence;
  `EloEvalCallback` for per-opponent Elo logging during training.
- **Evaluation script** (`training/evaluate.py`) — CLI evaluation against
  scripted opponents, random baseline, or any `.zip` checkpoint.
- **Self-play** (`training/self_play.py`) — `OpponentPool`, `SelfPlayCallback`,
  `WinRateVsPoolCallback`; wired into `train.py` via `self_play.enabled`.
- **Configuration system** — Hydra-based YAML configs (`default.yaml`,
  `self_play.yaml`, `experiment_1.yaml`, `orchestration.yaml`).
- **GitHub automation** — triage agent, label/milestone bootstrap, project
  board sync, orchestration workflow, governance policy.
- **Documentation** — `README.md`, `CONTRIBUTING.md`, `docs/TRAINING_GUIDE.md`,
  `docs/ENVIRONMENT_SPEC.md`, `docs/ROADMAP.md`, `docs/development_playbook.md`,
  `docs/ORCHESTRATION_RUNBOOK.md`.

### Changed
- `MORALE_CASUALTY_WEIGHT` raised from `0.4` to `1.5` to make routing
  reachable at the default `MORALE_ROUT_THRESHOLD` of `0.25`.

---

