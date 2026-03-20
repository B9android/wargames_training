# Multi-Agent Guide — v2 MAPPO & Curriculum

This guide covers the v2 multi-agent components: the `MultiBattalionEnv`
PettingZoo environment, the MAPPO algorithm, the three-stage curriculum,
multi-agent self-play, and coordination metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [Environment: MultiBattalionEnv](#environment-multibattalionenv)
3. [Observation Design](#observation-design)
4. [MAPPO Setup](#mappo-setup)
5. [Three-Stage Curriculum](#three-stage-curriculum)
6. [Multi-Agent Self-Play](#multi-agent-self-play)
7. [Coordination Metrics](#coordination-metrics)
8. [Running an Experiment](#running-an-experiment)
9. [Configuration Reference](#configuration-reference)

---

## Overview

v2 extends the v1 1v1 framework to **NvN multi-agent combat** using
Multi-Agent Proximal Policy Optimization (MAPPO).  The design follows the
Centralized Training with Decentralized Execution (CTDE) paradigm:

- **Decentralized execution** — each battalion agent acts using only its
  local observation (fog-of-war, partial visibility).
- **Centralized training** — the critic receives the full global state
  tensor during training to reduce variance.

---

## Environment: MultiBattalionEnv

`MultiBattalionEnv` is a [PettingZoo](https://pettingzoo.farama.org/)
`ParallelEnv` located at `envs/multi_battalion_env.py`.

### Instantiation

```python
from envs.multi_battalion_env import MultiBattalionEnv

env = MultiBattalionEnv(
    n_blue=2,                 # Blue (learning) agents
    n_red=2,                  # Red (scripted/opponent) agents
    map_width=1000.0,         # metres
    map_height=1000.0,
    max_steps=500,
    visibility_radius=600.0,  # fog-of-war radius (metres)
    randomize_terrain=True,
    seed=42,
)
env.reset()
```

### Agent IDs

Agents are identified by strings of the form `"blue_0"`, `"blue_1"`, …,
`"red_0"`, `"red_1"`, ….  Only Blue agents are controlled by the learned
policy; Red agents are driven by a scripted or opponent-pool policy.

### Global State

The centralized critic accesses the full global state via:

```python
state = env.state()  # np.ndarray, shape (state_dim,)
```

See [Observation Design](#observation-design) for dimensionality.

### PettingZoo Compliance

The environment passes `pettingzoo.test.parallel_api_test`:

```python
from pettingzoo.test import parallel_api_test
parallel_api_test(MultiBattalionEnv(), num_cycles=10)
```

---

## Observation Design

### Per-Agent Local Observation

Each agent receives a local observation vector of dimension:

```
obs_dim = 6 + 5 * (n_total - 1) + 1
```

where `n_total = n_blue + n_red`.

| Component | Dimensions | Description |
|---|---|---|
| Self state | 6 | `(x/W, y/H, cos θ, sin θ, hp, morale)` |
| Other units | 5 × (n_total−1) | Per other unit: `(Δx/W, Δy/H, cos θ, sin θ, hp)`; set to zero if outside `visibility_radius` |
| Terrain | 1 | Terrain cover at agent position `[0, 1]` |

**Key conventions:**
- All positions normalized by map dimensions (`W`, `H`) → `[-1, 1]`.
- Angles encoded as `(cos θ, sin θ)` — never raw radians.
- HP and morale normalized to `[0, 1]`.
- Units outside `visibility_radius` have their directional features zeroed
  but distance is still encoded (one-sided partial observability).

### Dimensionality by Scenario

| Scenario | n_total | obs_dim | state_dim |
|---|---|---|---|
| 1v1 | 2 | 12 | 13 |
| 2v2 | 4 | 22 | 25 |
| 3v3 | 6 | 32 | 37 |
| 4v4 | 8 | 42 | 49 |
| 6v6 | 12 | 62 | 73 |

### Global State (Centralized Critic)

The global state has dimension `state_dim = 6 * n_total + 1` and contains
the full, unobscured state of every unit plus the normalized step count.
The critic receives this vector during training regardless of
`visibility_radius`.

---

## MAPPO Setup

### Architecture

```
MAPPOActor
  Input : obs_dim
  Hidden: actor_hidden_sizes  (default [128, 64], ReLU)
  Output: action_dim (3) × 2  (mean, log_std for Gaussian)

MAPPOCritic
  Input : state_dim
  Hidden: critic_hidden_sizes (default [128, 64], ReLU)
  Output: 1 (scalar value estimate)
```

With `share_parameters=True` (default) all Blue agents share a single
`MAPPOActor`.  Setting `share_parameters=False` gives each agent its own
actor but they still share the centralized critic.

### Training Hyperparameters (2v2 defaults)

| Parameter | Default | Notes |
|---|---|---|
| `total_timesteps` | 200 000 | Increase for larger scenarios |
| `n_steps` | 256 | Rollout length per update |
| `n_epochs` | 10 | PPO update epochs per rollout |
| `batch_size` | 64 | Minibatch size |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE-λ |
| `clip_range` | 0.2 | PPO ε clip |
| `vf_coef` | 0.5 | Value-function loss weight |
| `ent_coef` | 0.01 | Entropy bonus weight |
| `max_grad_norm` | 0.5 | Gradient clip norm |

### Launching Training

```bash
# 2v2 reference experiment (Hydra config)
python training/train_mappo.py --config-name=experiment_mappo_2v2

# Override via CLI
python training/train_mappo.py \
    training.total_timesteps=500000 \
    env.n_blue=3 env.n_red=3

# Disable W&B for local testing
WANDB_MODE=disabled python training/train_mappo.py
```

Checkpoints are saved to `checkpoints/mappo_2v2/` (configurable via
`eval.checkpoint_dir`).

### Swapping the Red Policy at Runtime

`MAPPOTrainer` exposes `set_red_policy(policy)` for swapping in a snapshot
opponent without restarting training:

```python
from models.mappo_policy import MAPPOPolicy
opponent = MAPPOPolicy.load("checkpoints/snapshot_500k.pt")
trainer.set_red_policy(opponent)
```

---

## Three-Stage Curriculum

The curriculum (`training/curriculum_scheduler.py`) progressively increases
difficulty to stabilize early learning:

```
Stage 1 — 1v1  (BattalionEnv, scripted Red)
        ↓  win-rate ≥ threshold over rolling window
Stage 2 — 2v1  (MultiBattalionEnv, 2 Blue vs 1 stationary Red)
        ↓  win-rate ≥ threshold over rolling window
Stage 3 — 2v2  (MultiBattalionEnv, 2 Blue vs 2 scripted Red)
```

### Enabling the Curriculum

```bash
python training/train_mappo.py --config-name=curriculum_2v2
```

### Weight Transfer from v1

When transitioning from Stage 1 (v1 PPO) to Stage 2 (MAPPO), the scheduler
calls `load_v1_weights_into_mappo()` to warm-start the MAPPO actor from the
v1 checkpoint.  This matches SB3 `mlp_extractor.policy_net` Linear layers to
MAPPO `actor.trunk` Linear layers by position.

```python
from training.curriculum_scheduler import load_v1_weights_into_mappo

load_v1_weights_into_mappo(
    v1_model_path="checkpoints/ppo_battalion_final.zip",
    mappo_policy=policy,
)
```

### Win-Rate Promotion Logic

`CurriculumScheduler.record_episode_result(won)` maintains a rolling win-rate
window.  When the window is full and win-rate exceeds `win_rate_threshold`,
`promote()` advances to the next stage and resets the window.

```python
scheduler = CurriculumScheduler(
    win_rate_threshold=0.70,
    window_size=100,
)
```

---

## Multi-Agent Self-Play

Multi-agent self-play uses `TeamOpponentPool` in `training/self_play.py`.

### How It Works

1. Every `snapshot_freq` timesteps the current MAPPO policy is saved to the
   pool directory as a `.pt` file.
2. At each rollout collection step, a snapshot is sampled from the pool
   (uniform or Elo-weighted) and used as the Red opponent.
3. `TeamEloRegistry` (`training/elo.py`) tracks relative team strength and
   provides K-factor-weighted Elo updates.

### Enabling Self-Play in MAPPO Training

```yaml
# configs/experiment_mappo_2v2.yaml (or any override)
self_play:
  enabled: true
  pool_dir: "checkpoints/mappo_pool/"
  pool_max_size: 20
  snapshot_freq: 50000
  eval_freq: 25000
  n_eval_episodes: 20
```

### Evaluating Against the Pool

```python
from training.self_play import evaluate_team_vs_pool

results = evaluate_team_vs_pool(
    policy=current_policy,
    pool=opponent_pool,
    env_fn=lambda: MultiBattalionEnv(n_blue=2, n_red=2),
    n_episodes=20,
)
print(f"Win rate vs pool: {results.win_rate:.2%}")
print(f"Nash exploitability proxy: {results.nash_proxy:.4f}")
```

---

## Coordination Metrics

`envs/metrics/coordination.py` provides three episode-level metrics logged
to W&B under the `metrics/` namespace:

| Metric | Range | Description |
|---|---|---|
| `flanking_ratio` | `[0, 1]` | Fraction of attacks coming from a flank angle (> 45° off the target's front). Values > 0.3 indicate meaningful flanking. |
| `fire_concentration` | `[0, 1]` | Degree to which Blue agents focus fire on a single Red target. 1.0 = perfect concentration. |
| `mutual_support_score` | `[0, 1]` | Average fraction of Blue agents within support range of each other. Higher = tighter coordination. |

These are computed by `MAPPOTrainer` at the end of each rollout and written
to W&B via the standard `wandb.log()` call.

---

## Running an Experiment

### Quick-Start (2v2, no curriculum)

```bash
# Install dependencies
pip install -r requirements.txt

# Run 2v2 MAPPO training
python training/train_mappo.py --config-name=experiment_mappo_2v2

# Monitor in W&B
# → project: wargames_training, tags: v2, mappo, 2v2
```

### Curriculum Training (1v1 → 2v1 → 2v2)

```bash
# Requires a v1 checkpoint for weight transfer
python training/train_mappo.py --config-name=curriculum_2v2 \
    curriculum.v1_checkpoint=checkpoints/ppo_battalion_final.zip
```

### NvN Scaling (3v3, 4v4, 6v6)

```bash
# 3v3 — use the scenario config
python training/train_mappo.py \
    --config-name=experiment_mappo_2v2 \
    env.n_blue=3 env.n_red=3 \
    training.total_timesteps=500000

# 6v6 — adjust hidden sizes for larger obs/state dims
python training/train_mappo.py \
    --config-name=experiment_mappo_2v2 \
    env.n_blue=6 env.n_red=6 \
    training.actor_hidden_sizes=[256,128] \
    training.critic_hidden_sizes=[256,128] \
    training.total_timesteps=2000000
```

---

## Configuration Reference

| Config File | Purpose |
|---|---|
| `configs/experiment_mappo_2v2.yaml` | Reference 2v2 MAPPO experiment |
| `configs/curriculum_2v2.yaml` | Three-stage curriculum schedule |
| `configs/scenarios/2v2.yaml` | 2v2 scenario parameters |
| `configs/scenarios/3v3.yaml` | 3v3 scenario parameters |
| `configs/scenarios/4v4.yaml` | 4v4 scenario parameters |
| `configs/scenarios/6v6.yaml` | 6v6 scenario parameters |

Key configuration keys:

```yaml
env:
  n_blue: 2                    # Blue team size
  n_red: 2                     # Red team size
  visibility_radius: 600.0     # Fog-of-war radius (metres)
  red_random: false            # false=stationary Red, true=random Red

training:
  share_parameters: true       # Shared actor across Blue agents
  actor_hidden_sizes: [128, 64]
  critic_hidden_sizes: [128, 64]
  total_timesteps: 200000

self_play:
  enabled: false               # Enable multi-agent self-play
  pool_dir: "checkpoints/mappo_pool/"
  snapshot_freq: 50000
```

---

## See Also

- `docs/ENVIRONMENT_SPEC.md` — Full v1 environment specification
- `docs/TRAINING_GUIDE.md` — v1 PPO training guide
- `docs/scaling_notes.md` — NvN scaling analysis and benchmarks
- `docs/v2_architecture.md` — v2 system architecture diagram
