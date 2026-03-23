"""Neural network models for wargames_training.

Exports
-------
* :mod:`models.mlp_policy` — SB3-compatible MLP actor-critic for BattalionEnv
* :mod:`models.mappo_policy` — MAPPO actor and centralized critic
* :mod:`models.entity_encoder` — entity-based transformer encoder (E8.1)
* :mod:`models.recurrent_policy` — LSTM recurrent actor-critic policy (E8.2)
"""
from models.entity_encoder import (
    ENTITY_TOKEN_DIM,
    UNIT_TYPE_INFANTRY,
    UNIT_TYPE_CAVALRY,
    UNIT_TYPE_ARTILLERY,
    TEAM_BLUE,
    TEAM_RED,
    SpatialPositionalEncoding,
    EntityEncoder,
    EntityActorCriticPolicy,
)
from models.recurrent_policy import (
    LSTMHiddenState,
    RecurrentEntityEncoder,
    RecurrentActorCriticPolicy,
    RecurrentRolloutBuffer,
)

__all__ = [
    "ENTITY_TOKEN_DIM",
    "UNIT_TYPE_INFANTRY",
    "UNIT_TYPE_CAVALRY",
    "UNIT_TYPE_ARTILLERY",
    "TEAM_BLUE",
    "TEAM_RED",
    "SpatialPositionalEncoding",
    "EntityEncoder",
    "EntityActorCriticPolicy",
    "LSTMHiddenState",
    "RecurrentEntityEncoder",
    "RecurrentActorCriticPolicy",
    "RecurrentRolloutBuffer",
]
