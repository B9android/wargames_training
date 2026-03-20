# Changelog

All notable changes to wargames_training will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [2.0.0] — 2026-03-20

### Added
- **`MultiBattalionEnv`** (`envs/multi_battalion_env.py`) — PettingZoo
  `ParallelEnv` supporting NvN battalion combat.  Per-agent local observations
  with fog-of-war via `visibility_radius`; global state tensor exposed for the
  centralized critic.  Passes `pettingzoo.test.parallel_api_test`.
- **MAPPO** (`models/mappo_policy.py`) — Multi-Agent Proximal Policy
  Optimization with centralized training and decentralized execution (CTDE).
  `MAPPOActor` (local obs → Gaussian), `MAPPOCritic` (global state → value),
  `MAPPOPolicy` wrapping both with optional `share_parameters`.
- **MAPPO training pipeline** (`training/train_mappo.py`) — `MAPPORolloutBuffer`
  with per-agent GAE, `MAPPOTrainer`, Hydra config entry-point, W&B logging of
  per-agent and aggregate rewards.
- **3-stage curriculum** (`training/curriculum_scheduler.py`) —
  `CurriculumScheduler` with rolling win-rate promotion across
  `STAGE_1V1 → STAGE_2V1 → STAGE_2V2`; `load_v1_weights_into_mappo` for
  warm-starting from v1 PPO checkpoints.
- **Coordination metrics** (`envs/metrics/coordination.py`) — `flanking_ratio`,
  `fire_concentration`, `mutual_support_score` logged per episode to W&B.
- **NvN scaling** — `MultiBattalionEnv` parameterized by `n_blue` / `n_red`;
  scenario configs added for `2v2`, `3v3`, `4v4`, and `6v6`
  (`configs/scenarios/`); scaling notes documented in `docs/scaling_notes.md`.
- **Multi-agent self-play** (`training/self_play.py`) — `TeamOpponentPool`
  saving MAPPO policy snapshots; `TeamEloRegistry` extending `EloRegistry` with
  team Elo baselines; `nash_exploitability_proxy` estimator.
- **Experiment config** (`configs/experiment_mappo_2v2.yaml`) — reference
  MAPPO 2v2 training config (200 k timesteps, shared actor, 128→64 MLP).
- **Curriculum config** (`configs/curriculum_2v2.yaml`) — three-stage
  curriculum schedule with win-rate thresholds.
- **Documentation** — `docs/multi_agent_guide.md`, `docs/v2_architecture.md`,
  `docs/scaling_notes.md`.

### Changed
- `training/self_play.py` extended with `TeamOpponentPool` and
  `evaluate_team_vs_pool` alongside the existing v1 `OpponentPool`.
- `training/elo.py` extended with `TeamEloRegistry` and team-specific
  `TEAM_BASELINE_RATINGS`.
- `envs/__init__.py` updated to export `MultiBattalionEnv`.

---

## [1.0.0] — 2026-03-19

### Added
- **Simulation engine** (`envs/sim/`) — battalion, combat (damage accumulation,
  morale mechanics, routing threshold), and procedural terrain with elevation
  and cover.
- **`BattalionEnv`** — Gymnasium 1v1 environment with 12-dim observation space,
  3-dim continuous action space, scripted Red opponent (curriculum levels 1–5),
  randomized terrain, and configurable reward shaping.
- **`BattalionMlpPolicy`** — Stable-Baselines3 `ActorCriticPolicy` subclass
  with a two-hidden-layer MLP (obs(12) → 128 → Tanh → 128 → Tanh) shared
  by the actor and critic heads; registered as `PPO.policy_aliases["BattalionMlpPolicy"]`.
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

