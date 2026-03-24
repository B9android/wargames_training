# HRL Training Protocol — Hierarchical Curriculum (E3.4)

This document describes the **three-phase bottom-up training protocol** for
the Hierarchical RL (HRL) stack implemented in E3.4.

---

## Overview

The HRL hierarchy has three command echelons:

```
Division Commander  (Phase 3)
      │  frozen brigade policy
Brigade Commander   (Phase 2)
      │  frozen battalion policy
Battalion Agents    (Phase 1)
      │  primitive actions
MultiBattalionEnv
```

**Bottom-up training** means each echelon is trained from scratch *after* the
echelon below it has converged and been frozen.  This prevents the higher-level
agent from exploiting an unstable lower-level policy — a common failure mode in
naive HRL.

The three phases are managed by
[`training/hrl_curriculum.py`](https://github.com/B9android/wargames_training/blob/main/training/hrl_curriculum.py) and the
corresponding config files under
[`configs/hrl/`](https://github.com/B9android/wargames_training/tree/main/configs/hrl).

---

## Phase 1 — Battalion Training

| Item               | Value                                              |
|--------------------|----------------------------------------------------|
| **Env**            | `MultiBattalionEnv` (2v2, via `train_mappo.py`)    |
| **Algorithm**      | MAPPO                                              |
| **Config**         | `configs/hrl/phase1_battalion.yaml`                |
| **Checkpoint**     | `checkpoints/hrl/phase1_battalion/mappo_policy_final.pt` |
| **Win-rate crit.** | ≥ 0.70 over 50 episodes                           |
| **Elo criterion**  | ≥ 1100 (above default Elo of 1000)                |

### What is trained

Multi-agent MAPPO battalion agents learn to cooperate in a 2v2 scenario.
Observations and actions follow the standard `MultiBattalionEnv` spec
(see [`docs/ENVIRONMENT_SPEC.md`](ENVIRONMENT_SPEC.md)).

### Promotion

Once **both** criteria are met simultaneously:

1. The MAPPO checkpoint is saved to
   `checkpoints/hrl/phase1_battalion/mappo_policy_final.pt`.
2. All parameters are frozen using
   [`training/utils/freeze_policy.py`](https://github.com/B9android/wargames_training/blob/main/training/utils/freeze_policy.py):
   ```python
   from training.utils.freeze_policy import freeze_mappo_policy, assert_frozen
   freeze_mappo_policy(policy)
   assert_frozen(policy)   # raises RuntimeError if any param has requires_grad=True
   ```
3. The frozen policy is passed to `BrigadeEnv` as `battalion_policy`.

### Verify frozen weights

```python
from training.utils.freeze_policy import assert_frozen
assert_frozen(frozen_battalion_policy)  # must not raise
```

---

## Phase 2 — Brigade Training

| Item               | Value                                              |
|--------------------|----------------------------------------------------|
| **Env**            | `BrigadeEnv` (2v2, frozen battalion policy)        |
| **Algorithm**      | Stable-Baselines3 PPO                              |
| **Config**         | `configs/hrl/phase2_brigade.yaml`                  |
| **Checkpoint**     | `checkpoints/hrl/phase2_brigade/ppo_brigade_final.zip` |
| **Win-rate crit.** | ≥ 0.65 over 30 episodes                           |
| **Elo criterion**  | ≥ 1100 (above default Elo of 1000)                |

### What is trained

A brigade PPO commander issues macro-commands (options) to frozen battalion
agents.  The brigade observes aggregated sector control and battalion states;
the frozen battalion policies execute the chosen options at the primitive level.

The battalion policy is frozen before `BrigadeEnv` is constructed:
```python
env = BrigadeEnv(
    n_blue=2, n_red=2,
    battalion_policy=frozen_battalion_policy,  # all params have requires_grad=False
)
```

### Promotion

Once **both** criteria are met simultaneously:

1. The brigade checkpoint is saved to
   `checkpoints/hrl/phase2_brigade/ppo_brigade_final.zip`.
2. The SB3 PPO model is frozen:
   ```python
   from training.utils.freeze_policy import freeze_sb3_policy, assert_frozen
   freeze_sb3_policy(brigade_ppo)
   assert_frozen(brigade_ppo.policy)
   ```
3. The frozen policy is passed to `DivisionEnv` as `brigade_policy`.

---

## Phase 3 — Division Training

| Item               | Value                                                |
|--------------------|------------------------------------------------------|
| **Env**            | `DivisionEnv` (2×2 brigades, frozen brigade policy)  |
| **Algorithm**      | Stable-Baselines3 PPO                                |
| **Config**         | `configs/hrl/phase3_division.yaml`                   |
| **Checkpoint**     | `checkpoints/hrl/phase3_division/ppo_division_final.zip` |
| **Win-rate crit.** | ≥ 0.60 over 20 episodes                             |
| **Elo criterion**  | ≥ 1100 (above default Elo of 1000)                  |

### What is trained

A division PPO commander issues operational commands to frozen brigade
commanders.  The division observes theatre-sector control and brigade status
summaries; the frozen brigade policies issue macro-commands to their frozen
battalion sub-policies.

### Curriculum complete

Once Phase 3 criteria are met, the full three-echelon hierarchy is trained.
The final checkpoint is saved to
`checkpoints/hrl/phase3_division/ppo_division_final.zip`.

---

## Promotion Criteria — Programmatic Enforcement

All promotion checks are implemented in
[`training/hrl_curriculum.py`](https://github.com/B9android/wargames_training/blob/main/training/hrl_curriculum.py) via the
`HRLCurriculumScheduler` class:

```python
from training.hrl_curriculum import HRLPhase, HRLCurriculumScheduler
from training.elo import EloRegistry

scheduler = HRLCurriculumScheduler(
    win_rate_threshold=0.70,   # Phase 1 default
    win_rate_window=50,
    elo_threshold=1100.0,
)
elo_registry = EloRegistry(path=None)  # in-memory for testing

# After each evaluation:
scheduler.record_episode(win=True)
scheduler.update_elo(elo_registry, agent_name="phase1_run", opponent="scripted_l3",
                     win_rate=0.72, n_episodes=20)

if scheduler.should_promote():
    new_phase = scheduler.promote()
```

The `should_promote()` method enforces:
1. `len(outcome_window) >= win_rate_window` — minimum episodes observed.
2. `win_rate() >= win_rate_threshold`.
3. `elo >= elo_threshold` (if configured).

---

## Freezing Utilities

[`training/utils/freeze_policy.py`](https://github.com/B9android/wargames_training/blob/main/training/utils/freeze_policy.py)
exposes five public functions:

| Function                  | Purpose                                                        |
|---------------------------|----------------------------------------------------------------|
| `freeze_mappo_policy(m)`  | Set `requires_grad=False` + `eval()` on a `torch.nn.Module`   |
| `freeze_sb3_policy(ppo)`  | Freeze `.policy` of an SB3 PPO model                          |
| `assert_frozen(m)`        | Raise `RuntimeError` if any parameter has `requires_grad=True` |
| `load_and_freeze_mappo(…)`| Load a MAPPO `.pt` checkpoint and return a frozen policy       |
| `load_and_freeze_sb3(…)`  | Load an SB3 `.zip` checkpoint and return a frozen PPO model    |

Example — load and verify a frozen battalion policy:

```python
from training.utils.freeze_policy import load_and_freeze_mappo, assert_frozen

policy = load_and_freeze_mappo(
    checkpoint_path="checkpoints/hrl/phase1_battalion/mappo_policy_final.pt",
    obs_dim=22,
    action_dim=3,
    state_dim=25,
    n_agents=2,
)
assert_frozen(policy)   # guarantees no gradient leaks into battalion weights
```

---

## Running the Full Curriculum

### Phase 1 (Battalion)

```bash
python training/train_mappo.py \
    training.total_timesteps=500000
```

### Phase 2 (Brigade) — after Phase 1 promotes

```bash
python training/train_brigade.py \
    env.battalion_checkpoint=checkpoints/hrl/phase1_battalion/mappo_policy_final.pt \
    training.total_timesteps=500000
```

### Phase 3 (Division) — after Phase 2 promotes

```bash
python training/train_division.py \
    env.brigade_checkpoint=checkpoints/hrl/phase2_brigade/ppo_brigade_final.zip \
    training.total_timesteps=1000000
```

Each script writes a `ppo_*_final.zip` / `mappo_policy_final.pt` to the
corresponding `checkpoints/hrl/phaseN_*/` directory and emits W&B metrics
tagged with `hrl`, `curriculum`, and `phaseN`.

---

## CI Integration Test

A small-scenario integration test is provided in
[`tests/test_hrl_curriculum.py`](https://github.com/B9android/wargames_training/blob/main/tests/test_hrl_curriculum.py).

It trains each phase for approximately **512 environment steps** (well within
the CI time budget) on a small 300 × 300 m map, verifies that frozen weights
are not modified during subsequent training, checks that `should_promote()`
responds correctly to win-rate and Elo signals, and asserts that at least one
step of training completes without error.

The config files under `configs/hrl/` set `training.total_timesteps: 200000`
for full training runs.  For CI or quick smoke tests, override this value:

```bash
python training/train_mappo.py training.total_timesteps=10000
```

Run the integration test with:

```bash
python -m pytest tests/test_hrl_curriculum.py -v
```

---

## File Index

| Path                                          | Purpose                                   |
|-----------------------------------------------|-------------------------------------------|
| `training/hrl_curriculum.py`                  | `HRLPhase` enum + `HRLCurriculumScheduler`|
| `training/utils/freeze_policy.py`             | `freeze_mappo_policy`, `freeze_sb3_policy`, `assert_frozen` |
| `configs/hrl/phase1_battalion.yaml`           | Phase 1 hyperparameters + promotion config|
| `configs/hrl/phase2_brigade.yaml`             | Phase 2 hyperparameters + promotion config|
| `configs/hrl/phase3_division.yaml`            | Phase 3 hyperparameters + promotion config|
| `tests/test_hrl_curriculum.py`                | Integration test (3-phase, ~512 steps/phase)|
| `docs/hrl_training_protocol.md`               | This document                             |
| `docs/hrl_architecture.md`                    | Brigade/Division architecture details      |
