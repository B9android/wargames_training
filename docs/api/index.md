# API Reference

Full auto-generated API documentation for all stable public interfaces.
All four packages expose their symbols at the **top-level namespace** so
you only need a single import per package.

```python
from envs     import BattalionEnv, RewardWeights, Formation
from models   import BattalionMlpPolicy, MAPPOPolicy, WFM1Policy
from training import train, TrainingConfig, evaluate, OpponentPool
from analysis import COAGenerator, SaliencyAnalyzer, compute_gradient_saliency
```

## Packages

- [envs](envs.md) — simulation environments, reward shaping, sim primitives, HRL options
- [models](models.md) — MLP, MAPPO, transformer, recurrent, and WFM-1 policies
- [training](training.md) — training runners, callbacks, evaluation, self-play, Elo, benchmarks
- [analysis](analysis.md) — COA generation and policy saliency
- [benchmarks](benchmarks.md) — WargamesBench runner

## Top-level `envs` symbols

::: envs
    options:
      show_source: false
      members:
        - BattalionEnv
        - RewardWeights
        - Formation
        - SimEngine

## Top-level `models` symbols

::: models
    options:
      show_source: false
      members:
        - BattalionMlpPolicy
        - MAPPOPolicy
        - EntityEncoder
        - RecurrentActorCriticPolicy
        - WFM1Policy

## Top-level `training` symbols

::: training
    options:
      show_source: false
      members:
        - TrainingConfig
        - train
        - evaluate
        - OpponentPool
        - EloRegistry
