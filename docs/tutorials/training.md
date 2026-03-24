# Training Guide

This tutorial covers PPO training, self-play, and league training on
battalion-level environments.

## Single-Agent PPO

```python
from envs import BattalionEnv
from training.train import train

env = BattalionEnv()
train(env=env, total_timesteps=500_000)
```

Configuration is driven by YAML files in `configs/`:

```bash
python -m training.train --config configs/experiment_1.yaml
```

## Self-Play

Train a policy against a pool of frozen past checkpoints:

```python
from training.self_play import OpponentPool, SelfPlayCallback
```

See `configs/self_play.yaml` for the canonical self-play configuration.

## League Training

For competitive multi-agent training see
[`docs/league_training_guide.md`](../league_training_guide.md).

## Experiment Tracking

All training runs **must** be logged to W&B:

```python
import wandb
wandb.init(project="wargames_training", config=your_config_dict)
```

Post the W&B run URL in your PR or `[EXP]` tracking issue.
