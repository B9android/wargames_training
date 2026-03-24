# Models API

The `models` package exposes all neural network policy architectures.

## Quick-start

```python
from models import BattalionMlpPolicy, MAPPOPolicy, WFM1Policy, ScenarioCard
from stable_baselines3 import PPO
from envs import BattalionEnv

# Standard MLP policy with SB3
env = BattalionEnv()
model = PPO(BattalionMlpPolicy, env)

# Multi-agent MAPPO
from models import MAPPOPolicy
policy = MAPPOPolicy(obs_dim=12, action_dim=3, n_agents=2)

# WFM-1 foundation model
from models import WFM1Policy, ScenarioCard, ECHELON_BATTALION, TERRAIN_PROCEDURAL
card = ScenarioCard(echelon=ECHELON_BATTALION, terrain_type=TERRAIN_PROCEDURAL)
wfm1 = WFM1Policy()
```

## MLP policy (SB3-compatible)

::: models.mlp_policy.BattalionMlpPolicy

## MAPPO multi-agent policy

::: models.mappo_policy.MAPPOActor
::: models.mappo_policy.MAPPOCritic
::: models.mappo_policy.MAPPOPolicy

## Entity encoder (transformer)

::: models.entity_encoder.EntityEncoder
::: models.entity_encoder.EntityActorCriticPolicy
::: models.entity_encoder.SpatialPositionalEncoding

## Recurrent policy (LSTM)

::: models.recurrent_policy.RecurrentActorCriticPolicy
::: models.recurrent_policy.RecurrentEntityEncoder
::: models.recurrent_policy.LSTMHiddenState
::: models.recurrent_policy.RecurrentRolloutBuffer

## WFM-1 foundation model

::: models.wfm1.WFM1Policy
::: models.wfm1.ScenarioCard
::: models.wfm1.EchelonEncoder
::: models.wfm1.CrossEchelonTransformer
