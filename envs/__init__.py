"""Wargames Training — environments public API.

Stable interfaces for all simulation environments, reward shaping,
simulation primitives, and scenario utilities.  Import from this module
rather than from individual submodules to stay insulated from internal
restructuring.

Environments
------------
:class:`BattalionEnv`
    Single-battalion 1v1 Gymnasium environment (continuous action space).

:class:`BrigadeEnv`
    Brigade-level multi-battalion environment (MultiDiscrete actions).

:class:`DivisionEnv`
    Division-level multi-brigade environment (MultiDiscrete actions).

:class:`CorpsEnv`
    Corps-level operational environment with supply and road networks.

:class:`CavalryCorpsEnv`
    Corps environment extended with cavalry reconnaissance.

:class:`ArtilleryCorpsEnv`
    Corps environment extended with artillery and fortification mechanics.

:class:`MultiBattalionEnv`
    Vectorised multi-battalion PettingZoo parallel environment.

Reward shaping
--------------
:class:`RewardWeights`
    Dataclass of per-component reward multipliers.

:class:`RewardComponents`
    Named-tuple of per-step reward component values.

:func:`compute_reward`
    Compute a single-step reward given state deltas and weights.

Environment configuration types
--------------------------------
:class:`LogisticsConfig` / :class:`LogisticsState` / :class:`SupplyWagon`
    Logistics and supply management.

:class:`MoraleConfig`
    Morale dynamics parameters.

:class:`Formation`
    Unit formation enum (LINE / COLUMN / SQUARE / SKIRMISH).

:class:`WeatherConfig` / :class:`WeatherState`
    Weather simulation parameters.

:class:`RedPolicy`
    Protocol for scripted / learned Red opponent policies.

Simulation primitives
---------------------
:class:`SimEngine`
    Headless deterministic 1v1 simulation engine.

:class:`EpisodeResult`
    Structured result from a :class:`SimEngine` episode.

HRL options framework
---------------------
:class:`MacroAction`
    Discrete high-level action enum.

:class:`Option`
    Temporal abstraction option (initiation set, policy, termination).

:func:`make_default_options`
    Build the default option set for the battalion environment.

:class:`SMDPWrapper`
    Semi-MDP wrapper that executes options as macro-actions.

Corps constants
---------------
:data:`CORPS_OBS_DIM`, :data:`N_CORPS_SECTORS`, :data:`N_OBJECTIVES`,
:data:`CORPS_MAP_WIDTH`, :data:`CORPS_MAP_HEIGHT`, :data:`N_ROAD_FEATURES`
"""

from __future__ import annotations

# ── Environments ──────────────────────────────────────────────────────────
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

# ── Reward shaping ────────────────────────────────────────────────────────
from envs.reward import RewardComponents, compute_reward

# ── Simulation primitives ─────────────────────────────────────────────────
from envs.sim.engine import EpisodeResult, SimEngine

# ── HRL options framework ─────────────────────────────────────────────────
from envs.options import (
    MacroAction,
    Option,
    make_default_options,
)
from envs.smdp_wrapper import SMDPWrapper

__all__ = [
    # ── Environments ──────────────────────────────────────────────────────
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
    "BrigadeEnv",
    "DivisionEnv",
    "CorpsEnv",
    "ObjectiveType",
    "OperationalObjective",
    "CORPS_OBS_DIM",
    "N_CORPS_SECTORS",
    "N_OBJECTIVES",
    "CORPS_MAP_WIDTH",
    "CORPS_MAP_HEIGHT",
    "N_ROAD_FEATURES",
    "CavalryCorpsEnv",
    "ArtilleryCorpsEnv",
    "MultiBattalionEnv",
    # ── Reward shaping ────────────────────────────────────────────────────
    "RewardComponents",
    "compute_reward",
    # ── Simulation primitives ─────────────────────────────────────────────
    "EpisodeResult",
    "SimEngine",
    # ── HRL options ───────────────────────────────────────────────────────
    "MacroAction",
    "Option",
    "make_default_options",
    "SMDPWrapper",
]
