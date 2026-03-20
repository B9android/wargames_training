# HRL Architecture — Brigade & Division Commander Layers (E3.2 / E3.3)

This document describes the Hierarchical RL (HRL) architecture for the
Brigade Commander (v3, E3.2) and Division Commander (v3, E3.3), which form
a three-echelon command hierarchy above the battalion-level policies trained
in v2.

---

## Overview

The full HRL stack has **three echelons**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Division Commander (PPO)                                               │
│  Observes: theatre sectors, brigade status summaries, objective ctrl    │
│  Acts:     operational command per brigade (MultiDiscrete)              │
│  Reward:   brigade reward (passed through from BrigadeEnv)              │
├─────────────────────────────────────────────────────────────────────────┤
│  Command Translator (envs/division_env.py)                              │
│  Expands per-brigade operational command → per-battalion option index   │
├─────────────────────────────────────────────────────────────────────────┤
│  Brigade Commander (PPO / option dispatcher)                            │
│  Observes: sector control, battalion states, enemy threats              │
│  Acts:     option selection per battalion (MultiDiscrete)               │
│  Reward:   mean of battalion rewards over the macro-step                │
├─────────────────────────────────────────────────────────────────────────┤
│  Option Dispatcher (envs/brigade_env.py)                                │
│  Translates brigade macro-command → Option object                       │
│  Executes option for K primitive steps until termination                │
├─────────────────────────────────────────────────────────────────────────┤
│  Battalion Options (envs/options.py)                                    │
│  6 hardcoded macro-actions: advance, defend, flank L/R,                 │
│  withdraw, concentrate fire                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  MultiBattalionEnv (envs/multi_battalion_env.py)                        │
│  Primitive continuous actions: [move, rotate, fire]                     │
│  PettingZoo ParallelEnv, fog-of-war, full combat physics                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Division Commander Layer (E3.3)

### Division Observation Space

`Box(shape=(obs_dim,), dtype=float32)` where
`obs_dim = N_THEATRE_SECTORS + 8 * n_brigades + 1`

For the default 2-brigade scenario (`n_brigades=2`): **`obs_dim = 22`**

| Slice                        | Feature                                        | Range      |
|------------------------------|------------------------------------------------|------------|
| `[0 : N_THEATRE_SECTORS]`    | Theatre sector control (5 vertical strips)     | `[0, 1]`   |
|                              | `sector_control[s]` = blue strength /          |            |
|                              | (blue + red) in sector *s*.                    |            |
|                              | 0.5 when no units occupy the sector.           |            |
| `[5 : 5+3*n_brigades]`       | Per-brigade status (3 per brigade)             | `[0, 1]`   |
|                              | `[avg_strength, avg_morale, alive_ratio]`      |            |
|                              | Zeros for fully destroyed brigades.            |            |
| `[5+3*nb : 5+8*nb]`          | Per-brigade threat vector (5 per brigade)      | mixed      |
|                              | `[dist/diag, cos_bearing, sin_bearing,`        |            |
|                              | `enemy_avg_strength, enemy_avg_morale]`        |            |
|                              | — nearest alive Red brigade centroid.          |            |
|                              | Sentinel `[1, 0, 0, 0, 0]` if no enemy alive. |            |
| `[-1]`                       | Step progress: `step / max_steps`              | `[0, 1]`   |

### Division Action Space

`MultiDiscrete([n_div_options] * n_brigades)` — one operational command per
Blue brigade.  For `n_brigades=2`, `n_div_options=6`: shape `(2,)` with each
element in `[0, 5]`.

#### Operational command vocabulary

| Index | Division command          | Translated brigade option |
|-------|---------------------------|---------------------------|
| 0     | `advance_theatre`         | `advance_sector`   (0)    |
| 1     | `hold_position`           | `defend_position`  (1)    |
| 2     | `envelop_left`            | `flank_left`       (2)    |
| 3     | `envelop_right`           | `flank_right`      (3)    |
| 4     | `strategic_withdrawal`    | `withdraw`         (4)    |
| 5     | `mass_fires`              | `concentrate_fire` (5)    |

### Brigade Grouping

Blue battalions are assigned to brigades in consecutive index order:
**brigade *i*** = battalions `[i * n_blue_per_brigade … (i+1) * n_blue_per_brigade)`.

The division command for brigade *i* is broadcast to **all** battalions in
that brigade:
```
brigade_action[i*n_per + j] = div_command[i]  for j in range(n_per)
```

### Frozen Brigade Policy for Red

An SB3 PPO brigade checkpoint can be passed to `DivisionEnv` as
`brigade_policy`.  At each division step:

1. `DivisionEnv._get_red_brigade_obs()` builds a symmetric brigade-style
   observation for the Red side.
2. The frozen policy's `predict()` returns Red brigade actions (option indices
   per Red brigade).
3. These are fanned out to Red battalions and injected via
   `BrigadeEnv._forced_red_options`, which overrides
   `BrigadeEnv._get_red_action()` to execute the option's primitive policy.

---

## Brigade Commander Layer (E3.2)

### Brigade Observation Space

`Box(shape=(obs_dim,), dtype=float32)` where `obs_dim = 3 + 7 * n_blue + 1`

For the default 2v2 scenario (`n_blue=2`): **`obs_dim = 18`**

### Layout

| Slice                   | Feature                                    | Range      |
|-------------------------|--------------------------------------------|------------|
| `[0:3]`                 | Sector control (3 vertical strips)         | `[0, 1]`   |
|                         | `sector_control[s]` = blue strength in     |            |
|                         | sector *s* / (blue + red strength in *s*). |            |
|                         | 0.5 when no units occupy the sector.       |            |
| `[3 : 3+2*n_blue]`      | Per-blue battalion `[strength, morale]`    | `[0, 1]`   |
|                         | Zeros for dead battalions.                 |            |
| `[3+2*nb : 3+7*nb]`     | Per-blue enemy threat vector (5 per bat.)  | mixed      |
|                         | `[dist/diag, cos_bearing, sin_bearing,`    |            |
|                         | `enemy_strength, enemy_morale]`            |            |
|                         | — nearest alive red battalion.             |            |
|                         | Sentinel `[1, 0, 0, 0, 0]` if no enemy.   |            |
| `[-1]`                  | Step progress: `step / max_steps`          | `[0, 1]`   |

### Threat vector per battalion

Each blue battalion `i` contributes a 5-float block to the threat section:

| Index | Feature                            | Range      |
|-------|------------------------------------|------------|
| 0     | Distance to nearest red / diagonal | `[0, 1]`   |
| 1     | `cos(bearing_to_nearest_red)`      | `[-1, 1]`  |
| 2     | `sin(bearing_to_nearest_red)`      | `[-1, 1]`  |
| 3     | Nearest red battalion strength     | `[0, 1]`   |
| 4     | Nearest red battalion morale       | `[0, 1]`   |

---

## Brigade Action Space

`MultiDiscrete([n_options] * n_blue)` — one option index per Blue battalion.

For `n_blue=2`, `n_options=6`: shape `(2,)` with each element in `[0, 5]`.

### Option vocabulary

| Index | Name               | Behaviour                                      |
|-------|--------------------|------------------------------------------------|
| 0     | `advance_sector`   | Forward at 0.8 speed + suppression fire (0.2) |
| 1     | `defend_position`  | Stationary, full sustained fire (1.0)          |
| 2     | `flank_left`       | Move at 0.6 speed + full CCW rotation          |
| 3     | `flank_right`      | Move at 0.6 speed + full CW rotation           |
| 4     | `withdraw`         | Retreat at full speed (-1.0), no fire          |
| 5     | `concentrate_fire` | Stationary + tracking rotation + full fire     |

Options execute for up to 30 primitive steps (default `max_steps`).
Flanking options cap at 15 steps.

---

## Option Dispatcher (Brigade Layer)

Option selection and dispatch are implemented inside `BrigadeEnv.step()`.
At each brigade-level macro-step, `step(...)`:

1. Maps the brigade action for each alive Blue battalion (an option index) to an `Option` object from the vocabulary.
2. Runs the inner `MultiBattalionEnv` step-by-step until **all** selected
   options have terminated (condition fired, hard cap, or env episode ends).
3. Aggregates rewards across primitive steps.
4. Returns the mean reward over Blue battalions as the brigade scalar reward.

---

## Frozen Battalion Policy (Brigade Layer)

During brigade training, a v2 MAPPO checkpoint can be loaded to drive Red
agents as a challenging opponent.  The checkpoint is loaded via
`training.train_brigade.load_frozen_battalion_policy()`:

```python
from training.train_brigade import load_frozen_battalion_policy

policy = load_frozen_battalion_policy(
    checkpoint_path=Path("checkpoints/mappo_2v2/mappo_policy_final.pt"),
    obs_dim=22,    # MultiBattalionEnv obs_dim for 2v2
    action_dim=3,
    state_dim=25,  # MultiBattalionEnv state_dim for 2v2
    n_agents=2,
)
# All parameters are frozen:
assert all(not p.requires_grad for p in policy.parameters())
```

---

## Training Scripts

### Brigade training (E3.2)

`training/train_brigade.py` uses Stable-Baselines3 PPO:

```bash
# Default config (500k macro-steps, stationary Red)
python training/train_brigade.py

# With frozen v2 Red policy
python training/train_brigade.py \
    env.battalion_checkpoint=checkpoints/mappo_2v2/mappo_policy_final.pt
```

### Division training (E3.3)

`training/train_division.py` uses Stable-Baselines3 PPO:

```bash
# Default config (1M division macro-steps, stationary Red)
python training/train_division.py

# With frozen brigade Red policy
python training/train_division.py \
    env.brigade_checkpoint=checkpoints/brigade/ppo_brigade_final.zip

# Override timesteps
python training/train_division.py training.total_timesteps=2000000
```

Key hyperparameters (`configs/experiment_division.yaml`):

| Parameter             | Default   | Description                            |
|-----------------------|-----------|----------------------------------------|
| `n_brigades`          | 2         | Blue brigades                          |
| `n_blue_per_brigade`  | 2         | Battalions per brigade                 |
| `total_timesteps`     | 1 000 000 | Division macro-steps                   |
| `n_steps`             | 512       | PPO rollout buffer size                |
| `eval_freq`           | 20 000    | Win-rate evaluation frequency          |
| `checkpoint_freq`     | 100 000   | Checkpoint save frequency              |

---

## Acceptance Criteria Mapping

### E3.2 — Brigade Commander

| Criterion                                               | Implementation                                                          |
|---------------------------------------------------------|-------------------------------------------------------------------------|
| Brigade PPO converges within 500k steps                 | `train_brigade.py` + `BrigadeWinRateCallback` for W&B tracking         |
| Battalion policies are frozen (no gradient updates)     | `load_frozen_battalion_policy()` + `requires_grad_(False)` on all params |
| Brigade obs / action shapes documented                  | This document (see tables above)                                        |
| `tests/test_brigade_env.py` passes                      | Full test suite in `tests/test_brigade_env.py`                          |

### E3.3 — Division Commander

| Criterion                                                | Implementation                                                               |
|----------------------------------------------------------|------------------------------------------------------------------------------|
| Three-echelon hierarchy end-to-end                       | `DivisionEnv` → `BrigadeEnv` → `MultiBattalionEnv` → simulation             |
| Division policy converges within 1M steps                | `train_division.py` + `DivisionWinRateCallback` for W&B tracking            |
| Brigade policies frozen during division training         | `load_frozen_brigade_policy()` + `requires_grad_(False)` on policy params    |
| Division obs / action shapes documented                  | This document (see Division Commander section above)                         |
| `tests/test_division_env.py` passes                      | Full test suite in `tests/test_division_env.py`                              |

---

## File Index

| File                                   | Description                                                |
|----------------------------------------|------------------------------------------------------------|
| `envs/division_env.py`                 | Division Gymnasium env + command translator                |
| `envs/brigade_env.py`                  | Brigade Gymnasium env + option dispatcher                  |
| `envs/options.py`                      | Option vocabulary (6 macro-actions)                        |
| `envs/multi_battalion_env.py`          | Underlying PettingZoo env (primitive actions)              |
| `training/train_division.py`           | SB3 PPO division training loop + frozen brigade loader     |
| `training/train_brigade.py`            | SB3 PPO brigade training loop + frozen policy loader       |
| `configs/experiment_division.yaml`     | Default hyperparameters for division training              |
| `configs/experiment_brigade.yaml`      | Default hyperparameters for brigade training               |
| `tests/test_division_env.py`           | Unit and integration tests for `DivisionEnv`               |
| `tests/test_brigade_env.py`            | Unit and integration tests for `BrigadeEnv`                |
| `models/mappo_policy.py`               | MAPPOPolicy (v2 checkpoint format)                         |


This document describes the Hierarchical RL (HRL) architecture for the
Brigade Commander (v3), which sits above the battalion-level policies
trained in v2.

---

## Overview

The HRL stack has two layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  Brigade Commander (PPO)                                        │
│  Observes: sector control, battalion states, enemy threats      │
│  Acts:     option selection per battalion (MultiDiscrete)       │
│  Reward:   mean of battalion rewards over the macro-step        │
├─────────────────────────────────────────────────────────────────┤
│  Option Dispatcher (envs/brigade_env.py)                        │
│  Translates brigade macro-command → Option object               │
│  Executes option for K primitive steps until termination        │
├─────────────────────────────────────────────────────────────────┤
│  Battalion Options (envs/options.py)                            │
│  6 hardcoded macro-actions: advance, defend, flank L/R,         │
│  withdraw, concentrate fire                                     │
├─────────────────────────────────────────────────────────────────┤
│  MultiBattalionEnv (envs/multi_battalion_env.py)                │
│  Primitive continuous actions: [move, rotate, fire]             │
│  PettingZoo ParallelEnv, fog-of-war, full combat physics        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Brigade Observation Space

`Box(shape=(obs_dim,), dtype=float32)` where `obs_dim = 3 + 7 * n_blue + 1`

For the default 2v2 scenario (`n_blue=2`): **`obs_dim = 18`**

### Layout

| Slice                   | Feature                                    | Range      |
|-------------------------|--------------------------------------------|------------|
| `[0:3]`                 | Sector control (3 vertical strips)         | `[0, 1]`   |
|                         | `sector_control[s]` = blue strength in     |            |
|                         | sector *s* / (blue + red strength in *s*). |            |
|                         | 0.5 when no units occupy the sector.       |            |
| `[3 : 3+2*n_blue]`      | Per-blue battalion `[strength, morale]`    | `[0, 1]`   |
|                         | Zeros for dead battalions.                 |            |
| `[3+2*nb : 3+7*nb]`     | Per-blue enemy threat vector (5 per bat.)  | mixed      |
|                         | `[dist/diag, cos_bearing, sin_bearing,`    |            |
|                         | `enemy_strength, enemy_morale]`            |            |
|                         | — nearest alive red battalion.             |            |
|                         | Sentinel `[1, 0, 0, 0, 0]` if no enemy.   |            |
| `[-1]`                  | Step progress: `step / max_steps`          | `[0, 1]`   |

### Threat vector per battalion

Each blue battalion `i` contributes a 5-float block to the threat section:

| Index | Feature                            | Range      |
|-------|------------------------------------|------------|
| 0     | Distance to nearest red / diagonal | `[0, 1]`   |
| 1     | `cos(bearing_to_nearest_red)`      | `[-1, 1]`  |
| 2     | `sin(bearing_to_nearest_red)`      | `[-1, 1]`  |
| 3     | Nearest red battalion strength     | `[0, 1]`   |
| 4     | Nearest red battalion morale       | `[0, 1]`   |

---

## Brigade Action Space

`MultiDiscrete([n_options] * n_blue)` — one option index per Blue battalion.

For `n_blue=2`, `n_options=6`: shape `(2,)` with each element in `[0, 5]`.

### Option vocabulary

| Index | Name               | Behaviour                                      |
|-------|--------------------|------------------------------------------------|
| 0     | `advance_sector`   | Forward at 0.8 speed + suppression fire (0.2) |
| 1     | `defend_position`  | Stationary, full sustained fire (1.0)          |
| 2     | `flank_left`       | Move at 0.6 speed + full CCW rotation          |
| 3     | `flank_right`      | Move at 0.6 speed + full CW rotation           |
| 4     | `withdraw`         | Retreat at full speed (-1.0), no fire          |
| 5     | `concentrate_fire` | Stationary + tracking rotation + full fire     |

Options execute for up to 30 primitive steps (default `max_steps`).
Flanking options cap at 15 steps.

---

## Option Dispatcher

Option selection and dispatch are implemented inline inside `BrigadeEnv.step()`.
At each brigade-level macro-step, `step(...)`:

1. Maps the brigade action for each alive Blue battalion (an option index) to an `Option` object from the vocabulary.
2. Runs the inner `MultiBattalionEnv` step-by-step until **all** selected
   options have terminated (condition fired, hard cap, or env episode ends).
3. Aggregates rewards across primitive steps.
4. Returns the mean reward over Blue battalions as the brigade scalar reward.

---

## Frozen Battalion Policy

During brigade training, a v2 MAPPO checkpoint can be loaded to drive Red
agents as a challenging opponent.  The checkpoint is loaded via
`training.train_brigade.load_frozen_battalion_policy()`:

```python
from training.train_brigade import load_frozen_battalion_policy

policy = load_frozen_battalion_policy(
    checkpoint_path=Path("checkpoints/mappo_2v2/mappo_policy_final.pt"),
    obs_dim=22,    # MultiBattalionEnv obs_dim for 2v2
    action_dim=3,
    state_dim=25,  # MultiBattalionEnv state_dim for 2v2
    n_agents=2,
)
# All parameters are frozen:
assert all(not p.requires_grad for p in policy.parameters())
```

The `BrigadeEnv` also accepts a `battalion_policy` constructor argument:

```python
env = BrigadeEnv(n_blue=2, n_red=2, battalion_policy=policy)
```

`BrigadeEnv.set_battalion_policy(policy)` freezes the policy in-place by
calling `param.requires_grad_(False)` on every parameter and setting the
module to `eval()` mode.

---

## Training Script

`training/train_brigade.py` uses Stable-Baselines3 PPO:

```bash
# Default config (500k macro-steps, stationary Red)
python training/train_brigade.py

# With frozen v2 Red policy
python training/train_brigade.py \
    env.battalion_checkpoint=checkpoints/mappo_2v2/mappo_policy_final.pt

# Override timesteps
python training/train_brigade.py training.total_timesteps=1000000
```

Key hyperparameters (`configs/experiment_brigade.yaml`):

| Parameter             | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `total_timesteps`     | 500 000 | Macro-steps for training           |
| `n_steps`             | 512     | PPO rollout buffer size            |
| `n_epochs`            | 10      | PPO update epochs per rollout      |
| `batch_size`          | 64      | Minibatch size                     |
| `lr`                  | 3e-4    | Adam learning rate                 |
| `eval_freq`           | 10 000  | Win-rate evaluation frequency      |
| `checkpoint_freq`     | 50 000  | Checkpoint save frequency          |

---

## Acceptance Criteria Mapping

| Criterion                                               | Implementation                                                          |
|---------------------------------------------------------|-------------------------------------------------------------------------|
| Brigade PPO converges within 500k steps                 | `train_brigade.py` + `BrigadeWinRateCallback` for W&B tracking         |
| Battalion policies are frozen (no gradient updates)     | `load_frozen_battalion_policy()` + `requires_grad_(False)` on all params |
| Brigade obs / action shapes documented                  | This document (see tables above)                                        |
| `tests/test_brigade_env.py` passes                      | Full test suite in `tests/test_brigade_env.py`                          |

---

## File Index

| File                                 | Description                                             |
|--------------------------------------|---------------------------------------------------------|
| `envs/brigade_env.py`                | Brigade Gymnasium env + option dispatcher               |
| `envs/options.py`                    | Option vocabulary (6 macro-actions)                     |
| `envs/multi_battalion_env.py`        | Underlying PettingZoo env (primitive actions)           |
| `training/train_brigade.py`          | SB3 PPO training loop + frozen policy loader            |
| `configs/experiment_brigade.yaml`    | Default hyperparameters for brigade training            |
| `tests/test_brigade_env.py`          | Unit and integration tests for `BrigadeEnv`             |
| `models/mappo_policy.py`             | MAPPOPolicy (v2 checkpoint format)                      |
