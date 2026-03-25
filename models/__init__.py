# SPDX-License-Identifier: MIT
"""Neural network models for wargames_training.

Exports
-------
MLP policy (SB3-compatible)
    :class:`~models.mlp_policy.BattalionMlpPolicy` — MLP actor-critic for
    :class:`~envs.battalion_env.BattalionEnv`.

MAPPO multi-agent policy
    :class:`~models.mappo_policy.MAPPOActor` — stochastic Gaussian actor.
    :class:`~models.mappo_policy.MAPPOCritic` — centralised value critic.
    :class:`~models.mappo_policy.MAPPOPolicy` — combined actor-critic policy.

Entity encoder (transformer)
    :data:`~models.entity_encoder.ENTITY_TOKEN_DIM` — token dimensionality.
    :class:`~models.entity_encoder.SpatialPositionalEncoding` — 2D Fourier encoding.
    :class:`~models.entity_encoder.EntityEncoder` — multi-head transformer encoder.
    :class:`~models.entity_encoder.EntityActorCriticPolicy` — entity-based actor-critic.

Recurrent policy (LSTM)
    :class:`~models.recurrent_policy.LSTMHiddenState` — LSTM hidden-state container.
    :class:`~models.recurrent_policy.RecurrentEntityEncoder` — entity encoder + LSTM.
    :class:`~models.recurrent_policy.RecurrentActorCriticPolicy` — LSTM actor-critic.
    :class:`~models.recurrent_policy.RecurrentRolloutBuffer` — BPTT rollout buffer.

WFM-1 foundation model
    :class:`~models.wfm1.WFM1Policy` — multi-echelon foundation model.
    :class:`~models.wfm1.ScenarioCard` — scenario descriptor for WFM-1.
    :class:`~models.wfm1.EchelonEncoder` — per-echelon entity encoder.
    :class:`~models.wfm1.CrossEchelonTransformer` — cross-echelon transformer.
"""

from models.mlp_policy import BattalionMlpPolicy
from models.mappo_policy import MAPPOActor, MAPPOCritic, MAPPOPolicy
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
from models.wfm1 import (
    ECHELON_BATTALION,
    ECHELON_BRIGADE,
    ECHELON_DIVISION,
    ECHELON_CORPS,
    WEATHER_CLEAR,
    WEATHER_RAIN,
    WEATHER_FOG,
    WEATHER_SNOW,
    TERRAIN_PROCEDURAL,
    TERRAIN_GIS_WATERLOO,
    TERRAIN_GIS_AUSTERLITZ,
    TERRAIN_GIS_BORODINO,
    TERRAIN_GIS_SALAMANCA,
    ScenarioCard,
    EchelonEncoder,
    CrossEchelonTransformer,
    WFM1Policy,
)

__all__ = [
    # MLP policy
    "BattalionMlpPolicy",
    # MAPPO multi-agent policy
    "MAPPOActor",
    "MAPPOCritic",
    "MAPPOPolicy",
    # Entity encoder
    "ENTITY_TOKEN_DIM",
    "UNIT_TYPE_INFANTRY",
    "UNIT_TYPE_CAVALRY",
    "UNIT_TYPE_ARTILLERY",
    "TEAM_BLUE",
    "TEAM_RED",
    "SpatialPositionalEncoding",
    "EntityEncoder",
    "EntityActorCriticPolicy",
    # Recurrent policy
    "LSTMHiddenState",
    "RecurrentEntityEncoder",
    "RecurrentActorCriticPolicy",
    "RecurrentRolloutBuffer",
    # WFM-1 foundation model
    "ECHELON_BATTALION",
    "ECHELON_BRIGADE",
    "ECHELON_DIVISION",
    "ECHELON_CORPS",
    "WEATHER_CLEAR",
    "WEATHER_RAIN",
    "WEATHER_FOG",
    "WEATHER_SNOW",
    "TERRAIN_PROCEDURAL",
    "TERRAIN_GIS_WATERLOO",
    "TERRAIN_GIS_AUSTERLITZ",
    "TERRAIN_GIS_BORODINO",
    "TERRAIN_GIS_SALAMANCA",
    "ScenarioCard",
    "EchelonEncoder",
    "CrossEchelonTransformer",
    "WFM1Policy",
]
