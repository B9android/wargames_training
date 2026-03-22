"""Neural network models for wargames_training.

Exports
-------
* :mod:`models.mlp_policy` — SB3-compatible MLP actor-critic for BattalionEnv
* :mod:`models.mappo_policy` — MAPPO actor and centralized critic
* :mod:`models.entity_encoder` — entity-based transformer encoder (E8.1)
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
]
