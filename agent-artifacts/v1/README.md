# v1 Baseline Checkpoint — `best_model.zip`

## Overview

This is the first committed PPO checkpoint for the v1 training pipeline,
trained on the **M1: 1v1** configuration against `scripted_l3` opponents.

## Training Details

| Field               | Value                              |
|---------------------|------------------------------------|
| Algorithm           | PPO (Stable-Baselines3)            |
| Policy              | `BattalionMlpPolicy` (128 × 128)   |
| Environment         | `BattalionEnv` (1v1, 1000×1000 m)  |
| Curriculum level    | 3 (scripted Red, movement only — no Red fire) |
| Total timesteps     | ~50 000 (bootstrap baseline)       |
| Parallel envs       | 4                                  |
| Learning rate       | 3 × 10⁻⁴                          |
| n_steps / batch     | 1 024 / 256                        |
| Seed                | 42                                 |
| Trained on          | 2026-03-25                         |

## Evaluation Results

Evaluated over **20 episodes per level** (deterministic policy, seed 0).

| Opponent         | Wins | Draws | Losses | Win Rate |
|------------------|------|-------|--------|----------|
| `scripted_l1`    | 17   | 3     | 0      | 85 %     |
| `scripted_l2`    | 17   | 3     | 0      | 85 %     |
| `scripted_l3`    | 20   | 0     | 0      | 100 % ✅ |
| `scripted_l4`    | 20   | 0     | 0      | 100 %    |
| `scripted_l5`    | 5    | 0     | 15     | 25 %     |

The checkpoint **meets the v1 acceptance criterion** of ≥ 80 % win rate against
`scripted_l3`.

## Usage

```python
from stable_baselines3 import PPO

model = PPO.load("agent-artifacts/v1/best_model.zip")
```

Or via the evaluation CLI:

```bash
python training/evaluate.py \
    --checkpoint agent-artifacts/v1/best_model.zip \
    --opponent scripted_l3 \
    --n-episodes 20 \
    --seed 0
```
