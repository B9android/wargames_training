# Environments API

The `envs` package exposes all simulation environments and supporting
utilities as a clean programmatic Python API.

## Quick-start

```python
from envs import BattalionEnv, RewardWeights, Formation

# Custom reward shaping
rw = RewardWeights(win_bonus=20.0, loss_penalty=-20.0, time_penalty=-0.005)
env = BattalionEnv(reward_weights=rw, curriculum_level=3)
obs, info = env.reset(seed=42)
```

## Environments

::: envs.battalion_env.BattalionEnv
::: envs.brigade_env.BrigadeEnv
::: envs.division_env.DivisionEnv
::: envs.corps_env.CorpsEnv
::: envs.cavalry_corps_env.CavalryCorpsEnv
::: envs.artillery_corps_env.ArtilleryCorpsEnv
::: envs.multi_battalion_env.MultiBattalionEnv

## Reward shaping

::: envs.reward.RewardWeights
::: envs.reward.RewardComponents
::: envs.reward.compute_reward

## Configuration types

::: envs.battalion_env.LogisticsConfig
::: envs.battalion_env.MoraleConfig
::: envs.battalion_env.WeatherConfig
::: envs.battalion_env.Formation

## Simulation primitives

::: envs.sim.engine.SimEngine
::: envs.sim.engine.EpisodeResult

## HRL options framework

::: envs.options.MacroAction
::: envs.options.Option
::: envs.options.make_default_options
::: envs.smdp_wrapper.SMDPWrapper
