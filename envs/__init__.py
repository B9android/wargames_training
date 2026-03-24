"""Wargames Training — environments public API (E12.2).

Stable interfaces for all simulation environments.  Import from this module
rather than from individual submodules to stay insulated from internal
restructuring.

Environments
------------
:class:`~envs.battalion_env.BattalionEnv`
    Single-battalion 1v1 Gymnasium environment.

:class:`~envs.brigade_env.BrigadeEnv`
    Brigade-level multi-battalion environment.

:class:`~envs.division_env.DivisionEnv`
    Division-level multi-brigade environment.

:class:`~envs.corps_env.CorpsEnv`
    Corps-level operational environment with supply and road networks.

:class:`~envs.cavalry_corps_env.CavalryCorpsEnv`
    Corps environment extended with cavalry reconnaissance.

:class:`~envs.artillery_corps_env.ArtilleryCorpsEnv`
    Corps environment extended with artillery and fortification mechanics.

:class:`~envs.multi_battalion_env.MultiBattalionEnv`
    Vectorised multi-battalion PettingZoo parallel environment.
"""

from __future__ import annotations

from envs.battalion_env import (
    BattalionEnv,
    DESTROYED_THRESHOLD,
    MAP_WIDTH,
    MAP_HEIGHT,
    MAX_STEPS,
    LogisticsConfig,
    LogisticsState,
    SupplyWagon,
    MoraleConfig,
    RewardWeights,
    RedPolicy,
    Formation,
    WeatherConfig,
    WeatherState,
)
from envs.brigade_env import BrigadeEnv
from envs.division_env import DivisionEnv
from envs.corps_env import (
    CorpsEnv,
    ObjectiveType,
    OperationalObjective,
    CORPS_OBS_DIM,
    N_CORPS_SECTORS,
    N_OBJECTIVES,
    CORPS_MAP_WIDTH,
    CORPS_MAP_HEIGHT,
    N_ROAD_FEATURES,
)
from envs.cavalry_corps_env import CavalryCorpsEnv
from envs.artillery_corps_env import ArtilleryCorpsEnv
from envs.multi_battalion_env import MultiBattalionEnv

__all__ = [
    # Battalion
    "BattalionEnv",
    "DESTROYED_THRESHOLD",
    "MAP_WIDTH",
    "MAP_HEIGHT",
    "MAX_STEPS",
    "LogisticsConfig",
    "LogisticsState",
    "SupplyWagon",
    "MoraleConfig",
    "RewardWeights",
    "RedPolicy",
    "Formation",
    "WeatherConfig",
    "WeatherState",
    # Brigade / Division
    "BrigadeEnv",
    "DivisionEnv",
    # Corps
    "CorpsEnv",
    "ObjectiveType",
    "OperationalObjective",
    "CORPS_OBS_DIM",
    "N_CORPS_SECTORS",
    "N_OBJECTIVES",
    "CORPS_MAP_WIDTH",
    "CORPS_MAP_HEIGHT",
    "N_ROAD_FEATURES",
    # Specialised corps variants
    "CavalryCorpsEnv",
    "ArtilleryCorpsEnv",
    # Multi-agent
    "MultiBattalionEnv",
]
