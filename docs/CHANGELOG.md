# Changelog

All notable changes to wargames_training will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [4.0.0] — 2026-03-21

### Added
- **League infrastructure** (`training/league/agent_pool.py`,
  `training/league/match_database.py`) — `AgentPool` (JSON manifest, atomic
  writes) and `MatchDatabase` (JSONL append-log) as the persistence backbone
  for the v4 league.  `AgentType` enum: `MAIN_AGENT`, `MAIN_EXPLOITER`,
  `LEAGUE_EXPLOITER`.
- **League matchmaker** (`training/league/matchmaker.py`) —
  `LeagueMatchmaker` implementing Prioritized Fictitious Self-Play (PFSP)
  with a hard-first weight function `f(w) = 1 − w`; `set_weight_function()`
  for custom weighting; `set_nash_weights()` to switch to Nash distribution
  sampling; matchup rules: main agents face all roles, main exploiters face
  main agents only, league exploiters face all roles.
- **Main agent training loop** (`training/league/train_main_agent.py`) —
  `MainAgentTrainer` combining PFSP matchmaking, MAPPO policy updates, Elo
  rating, and periodic pool snapshots; `make_pfsp_weight_fn(T)` temperature
  factory; config `configs/league/main_agent.yaml`.
- **Main exploiter** (`training/league/train_exploiter.py`) —
  `MainExploiterTrainer` targeting the latest main agent snapshot; rolling
  win-rate reset via `_orthogonal_reinit`; `MAIN_EXPLOITER` pool snapshots;
  W&B `exploiter/*` metrics; config `configs/league/main_exploiter.yaml`.
- **League exploiter** (`training/league/train_league_exploiter.py`) —
  `LeagueExploiterTrainer` using PFSP against the full historical pool;
  Nash exploitability computation via `compute_league_exploitability()`;
  `LEAGUE_EXPLOITER` pool snapshots; W&B `league_exploiter/*` metrics;
  config `configs/league/league_exploiter.yaml`.
- **Nash distribution sampling** (`training/league/nash.py`) —
  `build_payoff_matrix` (win-rate callable → NumPy array);
  `compute_nash_distribution` (LP + regret matching → `{agent_id: prob}`);
  `nash_entropy` (Shannon entropy in nats); W&B key `league/nash_entropy`.
- **Strategy diversity metrics** (`training/league/diversity.py`) —
  `TrajectoryBatch`; `embed_trajectory` (action histogram + position heatmap
  + movement stats, L2-normalised); `pairwise_cosine_distances`;
  `diversity_score` (mean/min/median); `DiversityTracker`; W&B key
  `league/diversity_score`.
- **Distributed training** (`training/league/distributed_runner.py`,
  `envs/remote_multi_battalion_env.py`) — `RemoteMultiBattalionEnv`
  (`@ray.remote` actor); `make_remote_envs(n)`; `DistributedRolloutRunner`;
  `RolloutResult`; `benchmark()` throughput utility; Ray cluster config
  `configs/distributed/ray_cluster.yaml`; smoke-test CI workflow
  `.github/workflows/ray_smoke_test.yml`.
- **Documentation** — `docs/league_training_guide.md` (agent types,
  matchmaking, Nash sampling, diversity, distributed execution),
  `docs/v4_architecture.md` (ASCII component and data-flow diagrams).

---

## [3.0.0] — 2026-03-20

### Added
- **SMDP / Options framework** (`envs/smdp_wrapper.py`, `envs/options.py`) —
  Semi-Markov Decision Process wrapper enabling temporal abstraction; option
  primitives (advance, hold, retreat, flank-left, flank-right) with configurable
  `max_steps`; `make_default_options()` factory.
- **Brigade Commander** (`envs/brigade_env.py`, `training/train_brigade.py`) —
  `BrigadeEnv` wrapping `MultiBattalionEnv`; obs_dim = 3 + 7 × n_blue + 1
  (sector control, battalion strength/morale, threat vectors, step); PPO-based
  brigade training with frozen MAPPO battalion sub-policies; config
  `configs/experiment_brigade.yaml`.
- **Division Commander** (`envs/division_env.py`, `training/train_division.py`) —
  `DivisionEnv` wrapping `BrigadeEnv`; obs_dim = 5 + 8 × n_brigades + 1
  (theatre sectors, brigade status, threat, step); `_forced_red_options` for
  injecting Red brigade commands; config `configs/experiment_division.yaml`.
- **Hierarchical curriculum** (`training/hrl_curriculum.py`) —
  `HRLCurriculumScheduler` with `HRLPhase` enum
  (PHASE_1_BATTALION → PHASE_2_BRIGADE → PHASE_3_DIVISION); dual promotion
  criteria: rolling win-rate ≥ threshold **and** cached Elo ≥ elo_threshold.
- **Adaptive temporal abstraction** (`training/adaptive_temporal.py`) —
  `AdaptiveTemporalScheduler` varying temporal ratio from `base_ratio` to
  `min_ratio` across episode progress; `SWEEP_RATIOS` constant for grid-search.
- **Policy registry** (`training/policy_registry.py`) — `PolicyRegistry` backed
  by a JSON manifest; `Echelon` enum (battalion/brigade/division); versioned
  register/get/remove/list/load/save; CLI: `python -m training.policy_registry`.
- **Freeze utilities** (`training/utils/freeze_policy.py`) —
  `freeze_mappo_policy()`, `freeze_sb3_policy()`, `assert_frozen()`,
  `load_and_freeze_mappo()`, `load_and_freeze_sb3()` for bottom-up curriculum.
- **HRL evaluation harness** (`training/evaluate_hrl.py`) — end-to-end
  HRL vs. flat MARL tournament (`run_tournament()`); bootstrapped 95 % CIs
  (`bootstrap_ci()`); JSON output; CLI entry-point.
- **HRL analysis notebook** (`notebooks/v3_hrl_analysis.ipynb`) — tournament
  result loading, win-rate bar charts, CI error bars, echelon latency plots.
- **HRL configs** (`configs/hrl/phase1_battalion.yaml`,
  `phase2_brigade.yaml`, `phase3_division.yaml`) — per-phase training configs
  wiring curriculum promotions, temporal ratios, and checkpoint paths.
- **Documentation** — `docs/hrl_architecture.md` (three-echelon command
  hierarchy, observation/action spaces, reward flow),
  `docs/hrl_training_protocol.md` (bottom-up training protocol, phase
  descriptions, promotion criteria, evaluation methodology).

### Changed
- `envs/__init__.py` updated to export `BrigadeEnv` and `DivisionEnv`.
- `training/self_play.py` — `TeamOpponentPool` now supports brigade-level
  MAPPO snapshots alongside battalion-level policies.

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

