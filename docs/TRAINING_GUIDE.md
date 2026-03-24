# Wargames Training — Training Guide

This guide covers everything you need to run, configure, and monitor training
experiments for the `wargames_training` project.

---

## Prerequisites

Complete the [Getting Started](https://github.com/B9android/wargames_training#getting-started) steps in the README first:

```bash
git clone https://github.com/B9android/wargames_training.git
cd wargames_training
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login          # optional — see Offline Mode below
pytest tests/ -q     # all tests must pass before training
```

---

## Quickstart

Run the default training configuration (1 M steps, 8 parallel envs, W&B logging):

```bash
python training/train.py
```

Run without W&B internet access (offline mode — W&B writes a local run directory
under `./wandb/` and the eval logs go to `logs/`; sync to the cloud later):

```bash
WANDB_MODE=offline python training/train.py
```

Override individual hyperparameters at the command line using Hydra syntax:

```bash
python training/train.py training.learning_rate=1e-4 training.total_timesteps=500000
```

Use an experiment config file (e.g. `configs/experiment_1.yaml`):

```bash
python training/train.py --config-name experiment_1
```

---

## Configuration

All settings live in `configs/default.yaml`. The sections below cover every
config key and its effect.

### W&B experiment tracking

| Key | Default | Description |
|---|---|---|
| `wandb.project` | `"wargames_training"` | W&B project name |
| `wandb.entity` | `null` | W&B team/org — `null` uses your personal account |
| `wandb.tags` | `["v1", "ppo"]` | Default tags for each run (overridable via Hydra) |
| `wandb.log_freq` | `1000` | Log rollout metrics every N environment steps |

### Environment

| Key | Default | Description |
|---|---|---|
| `env.map_width` | `1000.0` | Map width in metres |
| `env.map_height` | `1000.0` | Map height in metres |
| `env.max_steps` | `500` | Maximum steps per episode |
| `env.num_envs` | `8` | Parallel training environments (increase for faster wall-clock) |
| `env.randomize_terrain` | `true` | Generate new terrain each episode for generalization |
| `env.hill_speed_factor` | `0.5` | Speed multiplier on max-elevation hills (range `(0, 1]`) |
| `env.curriculum_level` | `5` | Red opponent difficulty: 1 = stationary … 5 = full combat |

### Reward shaping

| Key | Default | Description |
|---|---|---|
| `reward.delta_enemy_strength` | `5.0` | Reward per unit of enemy strength destroyed |
| `reward.delta_own_strength` | `5.0` | Penalty per unit of own strength lost |
| `reward.survival_bonus` | `0.0` | Per-step bonus scaled by Blue's remaining strength |
| `reward.win_bonus` | `10.0` | Terminal bonus when Blue wins |
| `reward.loss_penalty` | `-10.0` | Terminal penalty when Blue loses |
| `reward.time_penalty` | `-0.01` | Per-step time penalty (discourages stalling) |

### PPO training

| Key | Default | Description |
|---|---|---|
| `training.algorithm` | `"PPO"` | RL algorithm (only PPO is currently supported) |
| `training.total_timesteps` | `1000000` | Total environment steps to train for |
| `training.learning_rate` | `3e-4` | Adam learning rate |
| `training.n_steps` | `2048` | Rollout steps per environment before a PPO update |
| `training.batch_size` | `64` | Minibatch size |
| `training.n_epochs` | `10` | Number of epochs per PPO update |
| `training.gamma` | `0.99` | Discount factor |
| `training.gae_lambda` | `0.95` | GAE λ for advantage estimation |
| `training.clip_range` | `0.2` | PPO clip ratio ε |
| `training.ent_coef` | `0.01` | Entropy regularization coefficient |
| `training.vf_coef` | `0.5` | Value function loss coefficient |
| `training.max_grad_norm` | `0.5` | Gradient clip norm |
| `training.seed` | `42` | Random seed for reproducibility |

### Evaluation

| Key | Default | Description |
|---|---|---|
| `eval.n_eval_episodes` | `20` | Episodes per evaluation interval |
| `eval.eval_freq` | `50000` | Evaluate every N timesteps |
| `eval.checkpoint_freq` | `100000` | Save checkpoint every N timesteps |
| `eval.checkpoint_dir` | `"checkpoints/"` | Directory for checkpoint `.zip` files |
| `eval.elo_registry` | `"checkpoints/elo_registry.json"` | Path to Elo registry JSON |
| `eval.elo_opponents` | `[]` | List of scripted opponents for Elo tracking |
| `eval.elo_eval_freq` | `50000` | Elo evaluation interval (timesteps) |
| `eval.elo_n_eval_episodes` | `20` | Episodes per Elo evaluation |

### Artifact naming and manifest

| Key | Default | Description |
|---|---|---|
| `artifacts.enable_naming_v2` | `true` | Include seed and curriculum level in checkpoint filenames |
| `artifacts.keep_legacy_aliases` | `true` | Also write `ppo_battalion_final.zip` alongside the v2-named file |
| `artifacts.write_manifest` | `true` | Append every checkpoint event to a local JSONL index |
| `artifacts.manifest_path` | `"checkpoints/manifest.jsonl"` | Path to the checkpoint manifest |

When `enable_naming_v2` is `true`, checkpoint filenames include the run seed and curriculum
level, for example `ppo_battalion_s42_c5_200000_steps.zip`.  This eliminates cross-run
overwrite collisions when multiple runs share the same `checkpoints/` directory.

The manifest (`checkpoints/manifest.jsonl`) is an append-only JSONL file — one JSON object
per line — that records every periodic, final, and best checkpoint with its step count, seed,
curriculum level, W&B run ID, and a stable config hash.  It is used by the resume resolution
logic (see [Resuming a Run](#resuming-a-run) below) and by the UI artifact browser.

Periodic checkpoints are registered in the manifest when the checkpoint callback writes them,
and best-model entries are appended when evaluation produces a new best checkpoint.  The final
checkpoint and any legacy aliases are registered during end-of-run finalization.

### Resume from checkpoint

| Key | Default | Description |
|---|---|---|
| `resume.auto` | `false` | Automatically resume from the latest checkpoint in the manifest |
| `resume.checkpoint` | `null` | Explicit `.zip` path to resume from (overrides `auto`) |

### Self-play (disabled by default)

| Key | Default | Description |
|---|---|---|
| `self_play.enabled` | `false` | Enable self-play loop |
| `self_play.pool_dir` | `"checkpoints/pool"` | Directory for snapshot `.zip` files |
| `self_play.pool_max_size` | `10` | Keep the N most recent policy snapshots |
| `self_play.snapshot_freq` | `50000` | Save snapshot every N environment steps |
| `self_play.eval_freq` | `50000` | Evaluate win-rate vs pool every N environment steps |
| `self_play.n_eval_episodes` | `20` | Episodes per win-rate evaluation |
| `self_play.use_latest_for_eval` | `false` | `false` = sample uniformly; `true` = always use latest |

### Logging

| Key | Default | Description |
|---|---|---|
| `logging.level` | `"INFO"` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `logging.log_dir` | `"logs/"` | Directory for eval logs written by `EvalCallback` |

---

## Curriculum Training

Training against progressively harder scripted opponents accelerates early
learning.  A typical progression:

1. Start at `env.curriculum_level=1` (stationary Red) until win rate > 90 %.
2. Advance to level 2–3.  Adjust `reward.time_penalty` if the agent stalls.
3. Advance to level 4–5 for full combat capability.

To run at level 3:

```bash
python training/train.py env.curriculum_level=3
```

---

## Self-Play

Once the agent reliably beats the scripted level-5 opponent, switch to
self-play using the pre-made config:

```bash
python training/train.py --config-name self_play
```

The self-play config runs for 2 M steps, periodically snapshots the current
policy into `checkpoints/pool/`, and evaluates win-rate against the pool.

---

## Elo Tracking

Enable Elo tracking by listing opponents in the config:

```bash
python training/train.py \
    "eval.elo_opponents=[scripted_l1,scripted_l3,scripted_l5]"
```

Elo ratings are persisted to `checkpoints/elo_registry.json` and logged to
W&B under the `elo/` key prefix.

Baseline ratings:

| Opponent | Baseline Elo |
|---|---|
| `random` | 500 |
| `scripted_l1` | 600 |
| `scripted_l2` | 700 |
| `scripted_l3` | 800 |
| `scripted_l4` | 900 |
| `scripted_l5` | 1000 |

---

## W&B Integration

Every training run calls `wandb.init()` automatically.  The following metrics
are logged:

| W&B key | Description |
|---|---|
| `rollout/ep_rew_mean` | Mean episodic reward (rolling buffer) |
| `rollout/ep_len_mean` | Mean episode length |
| `train/policy_gradient_loss` | PPO policy gradient loss |
| `train/value_loss` | Value function loss |
| `train/entropy_loss` | Entropy regularization loss |
| `reward_breakdown/*` | Per-component mean reward (per episode) |
| `elo/rating_vs_<opponent>` | Elo rating after evaluating vs the named opponent |
| `elo/win_rate_vs_<opponent>` | Win rate vs the named opponent at each Elo checkpoint |
| `elo/delta_vs_<opponent>` | Elo rating change from the last evaluation vs the named opponent |
| `self_play/win_rate_vs_pool` | Win rate vs self-play pool (when self-play is enabled) |

**Tips:**

- Set `wandb.entity` to your team name in `configs/default.yaml` to share
  runs with collaborators.
- Use `WANDB_MODE=offline` to train without internet access; sync later with
  `wandb sync`.
- Post your W&B run URL in the tracking issue when opening a PR for any
  experiment.

---

## Checkpoints

Checkpoints are saved as Stable-Baselines3 `.zip` files.  When `artifacts.enable_naming_v2`
is `true` (the default), filenames embed the run seed and curriculum level:

| Path | Contents |
|---|---|
| `checkpoints/ppo_battalion_s<seed>_c<level>_<N>_steps.zip` | Periodic checkpoint every `checkpoint_freq` steps |
| `checkpoints/best/ppo_battalion_s<seed>_c<level>_best.zip` | Best model by mean eval reward |
| `checkpoints/ppo_battalion_s<seed>_c<level>_final.zip` | Final model at end of training |
| `checkpoints/ppo_battalion_final.zip` | Legacy alias (written when `keep_legacy_aliases: true`) |
| `checkpoints/best/best_model.zip` | Legacy alias for best (written by SB3 `EvalCallback`) |
| `checkpoints/manifest.jsonl` | Append-only JSONL index of all checkpoint events |

Load a checkpoint:

```python
from stable_baselines3 import PPO
model = PPO.load("checkpoints/best/ppo_battalion_s42_c5_best.zip")
# or using the legacy alias:
model = PPO.load("checkpoints/best/best_model.zip")
```

---

## Resuming a Run

To automatically resume from the latest checkpoint found in the manifest:

```bash
python training/train.py resume.auto=true
```

To resume from a specific checkpoint:

```bash
python training/train.py resume.checkpoint=checkpoints/ppo_battalion_s42_c5_200000_steps.zip
```

**Resume resolution order:**
1. Explicit `resume.checkpoint` path (error if file not found).
2. Manifest `latest_periodic()` scan (fastest, uses the JSONL index).
3. Filesystem glob scan for `{prefix}_*_steps.zip` (fallback when no manifest).

A warning is logged if the recorded config hash for the checkpoint differs from the current
run's config hash — meaning hyperparameters may have changed since the checkpoint was saved.
Warnings are also emitted when the manifest metadata shows a different training seed or
curriculum level than the current run configuration.

---

## Evaluation

After training, evaluate a checkpoint against scripted opponents:

```bash
python training/evaluate.py \
    --checkpoint checkpoints/best/best_model.zip \
    --opponent scripted_l5 \
    --n-episodes 100
```

See `python training/evaluate.py --help` for all options, and
[`docs/ENVIRONMENT_SPEC.md`](ENVIRONMENT_SPEC.md) for opponent identifiers.

---

## Hyperparameter Tips

- **Slow learning / no improvement:** increase `training.learning_rate` to
  `1e-3` or raise `training.n_epochs` to 15.
- **Unstable training (loss spikes):** lower `training.clip_range` to `0.1`
  and reduce `training.learning_rate`.
- **Agent stalls without fighting:** increase `|reward.time_penalty|` to
  `-0.05` or add a small `reward.survival_bonus`.
- **Out of memory:** reduce `env.num_envs` to 4 or lower `training.n_steps`
  to 1024.
- **Slow wall-clock speed:** increase `env.num_envs` to 16–32 if your CPU
  has enough cores.

---

## Creating a Custom Experiment Config

Copy `configs/default.yaml` to `configs/experiment_myname.yaml`, change only
the keys you want to override, and run:

```bash
python training/train.py --config-name experiment_myname
```

Add a `[EXP]` GitHub issue before starting any significant experiment — see
[CONTRIBUTING.md](CONTRIBUTING.md#opening-issues).
