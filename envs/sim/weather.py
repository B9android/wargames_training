# envs/sim/weather.py
"""Weather and time-of-day model for Napoleonic battalion simulation.

This module parameterizes simulation episodes with weather conditions
(CLEAR, OVERCAST, RAIN, FOG, SNOW) and time of day (DAWN, DAY, DUSK, NIGHT).
Weather and daylight affect visibility range, musket/cannon accuracy, movement
speed, and per-step morale drain.

Historical motivation
~~~~~~~~~~~~~~~~~~~~~
Rain-soaked flintlocks misfired at rates of 30 % or higher (Nafziger, 1988).
Thick fog reduced effective fighting range to well under 100 m. Night marches
and dawn attacks were staples of Napoleonic warfare precisely because reduced
visibility neutered long-range firepower. Snowfall both slowed manoeuvre and
sapped troop morale.

Key public API
~~~~~~~~~~~~~~
* :class:`WeatherCondition` — CLEAR, OVERCAST, RAIN, FOG, SNOW.
* :class:`TimeOfDay` — DAWN, DAY, DUSK, NIGHT.
* :class:`ConditionEffects` — per-condition visibility/accuracy/speed/morale table entry.
* :class:`TODEffects` — per-time-of-day visibility/accuracy/morale table entry.
* :data:`CONDITION_EFFECTS` / :data:`TIME_OF_DAY_EFFECTS` — lookup tables.
* :class:`WeatherConfig` — all tunable parameters (fixed/random condition,
  progression schedule, base visibility range).
* :class:`WeatherState` — mutable per-episode state (condition, time of day,
  internal step counter).
* :func:`sample_weather` — draw a random :class:`WeatherState` for a new episode.
* :func:`step_weather` — advance time-of-day by one simulation step (mutates).
* :func:`get_visibility_fraction` — combined [0, 1] visibility from weather + tod.
* :func:`get_effective_visibility_range` — visibility fraction × base range (metres).
* :func:`get_accuracy_modifier` — combined fire-damage multiplier [0, 1].
* :func:`get_speed_modifier` — movement speed multiplier (0, 1].
* :func:`get_morale_stressor` — per-step morale drain (non-negative).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WeatherCondition(IntEnum):
    """Enumeration of weather conditions."""

    CLEAR = 0
    OVERCAST = 1
    RAIN = 2
    FOG = 3
    SNOW = 4


class TimeOfDay(IntEnum):
    """Enumeration of time-of-day periods."""

    DAWN = 0
    DAY = 1
    DUSK = 2
    NIGHT = 3


#: Total number of distinct weather conditions.
NUM_CONDITIONS: int = len(WeatherCondition)
#: Total number of distinct time-of-day periods.
NUM_TIME_OF_DAY: int = len(TimeOfDay)

# ---------------------------------------------------------------------------
# Effect tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionEffects:
    """Effect multipliers for a single weather condition.

    Parameters
    ----------
    visibility_fraction:
        Fraction of the base visibility range that remains clear.
        ``1.0`` = full daylight with no weather; ``0.15`` = dense fog.
    accuracy_modifier:
        Multiplicative penalty applied to fire damage.  A value of ``0.65``
        models the ≥ 30 % musket misfire rate in heavy rain (Nafziger, 1988).
    speed_modifier:
        Multiplicative penalty applied to movement speed.  ``0.5`` for snow
        captures the severe slowdown on frozen, muddy terrain.
    morale_stressor:
        Flat morale drain subtracted from both sides each step.  Non-negative.
    """

    visibility_fraction: float
    accuracy_modifier: float
    speed_modifier: float
    morale_stressor: float


@dataclass(frozen=True)
class TODEffects:
    """Effect multipliers for a single time-of-day period.

    Parameters
    ----------
    visibility_fraction:
        Daylight-based fraction of the base visibility range.
        Combined multiplicatively with :attr:`ConditionEffects.visibility_fraction`.
    accuracy_modifier:
        Multiplicative penalty applied to fire damage due to reduced light.
    morale_stressor:
        Flat morale drain per step due to low-light conditions.
    """

    visibility_fraction: float
    accuracy_modifier: float
    morale_stressor: float


#: Effect lookup table — one entry per :class:`WeatherCondition`.
#:
#: Historical notes:
#: * RAIN: ≥ 30 % accuracy drop from flintlock misfire rates.
#: * FOG:  heavy visibility loss pushes effective range well below 100 m
#:         when combined with a 500 m base visibility range.
#: * SNOW: severe movement penalty, elevated morale drain.
CONDITION_EFFECTS: dict[WeatherCondition, ConditionEffects] = {
    WeatherCondition.CLEAR:    ConditionEffects(
        visibility_fraction=1.00,
        accuracy_modifier=1.00,
        speed_modifier=1.00,
        morale_stressor=0.000,
    ),
    WeatherCondition.OVERCAST: ConditionEffects(
        visibility_fraction=0.90,
        accuracy_modifier=0.95,
        speed_modifier=1.00,
        morale_stressor=0.001,
    ),
    WeatherCondition.RAIN:     ConditionEffects(
        visibility_fraction=0.70,
        accuracy_modifier=0.65,  # 35 % accuracy drop — exceeds historical ≥ 30 %
        speed_modifier=0.85,
        morale_stressor=0.005,
    ),
    WeatherCondition.FOG:      ConditionEffects(
        visibility_fraction=0.15,  # ×500 m base → 75 m effective range (< 100 m)
        accuracy_modifier=0.85,
        speed_modifier=0.90,
        morale_stressor=0.003,
    ),
    WeatherCondition.SNOW:     ConditionEffects(
        visibility_fraction=0.60,
        accuracy_modifier=0.80,
        speed_modifier=0.50,
        morale_stressor=0.008,
    ),
}

#: Effect lookup table — one entry per :class:`TimeOfDay`.
TIME_OF_DAY_EFFECTS: dict[TimeOfDay, TODEffects] = {
    TimeOfDay.DAWN:  TODEffects(
        visibility_fraction=0.50,
        accuracy_modifier=0.85,
        morale_stressor=0.002,
    ),
    TimeOfDay.DAY:   TODEffects(
        visibility_fraction=1.00,
        accuracy_modifier=1.00,
        morale_stressor=0.000,
    ),
    TimeOfDay.DUSK:  TODEffects(
        visibility_fraction=0.60,
        accuracy_modifier=0.85,
        morale_stressor=0.002,
    ),
    TimeOfDay.NIGHT: TODEffects(
        visibility_fraction=0.30,
        accuracy_modifier=0.70,
        morale_stressor=0.005,
    ),
}

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

#: Default base visibility range in metres.  Weather and time-of-day
#: ``visibility_fraction`` values are multiplied by this to obtain the
#: effective sight radius.  A FOG episode at DAY yields 0.15 × 500 = 75 m,
#: satisfying the < 100 m acceptance criterion.
DEFAULT_BASE_VISIBILITY_RANGE: float = 500.0

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class WeatherConfig:
    """All tunable parameters for the weather and time-of-day system.

    Pass an instance to :class:`~envs.battalion_env.BattalionEnv` via the
    *weather_config* argument to enable the full weather model.

    Parameters
    ----------
    fixed_condition:
        Force a specific :class:`WeatherCondition` for every episode.
        ``None`` (the default) selects a condition randomly using
        *condition_weights* at each ``reset()``.
    fixed_time_of_day:
        Force a specific :class:`TimeOfDay` for every episode.
        ``None`` (the default) draws a random time of day at each ``reset()``.
    steps_per_time_of_day:
        Number of simulation steps between time-of-day transitions.
        ``0`` (the default) disables progression — the time of day stays
        fixed for the entire episode.  A positive value causes DAWN → DAY →
        DUSK → NIGHT → DAWN cycling.
    condition_weights:
        Sampling weights for :class:`WeatherCondition` values
        ``[CLEAR, OVERCAST, RAIN, FOG, SNOW]``.  Values need not sum to 1;
        they are normalised internally.  Ignored when *fixed_condition* is set.
    base_visibility_range:
        Base visibility range in metres before weather modification (> 0).
        The combined ``visibility_fraction`` is multiplied by this value to
        produce the effective sight radius in the simulation.
    """

    fixed_condition: Optional[WeatherCondition] = None
    fixed_time_of_day: Optional[TimeOfDay] = None
    steps_per_time_of_day: int = 0
    condition_weights: List[float] = field(
        default_factory=lambda: [0.40, 0.25, 0.15, 0.10, 0.10]
    )
    base_visibility_range: float = DEFAULT_BASE_VISIBILITY_RANGE

    def __post_init__(self) -> None:
        if len(self.condition_weights) != NUM_CONDITIONS:
            raise ValueError(
                f"condition_weights must have {NUM_CONDITIONS} entries "
                f"(one per WeatherCondition), got {len(self.condition_weights)}"
            )
        if any(w < 0.0 for w in self.condition_weights):
            raise ValueError(
                "All condition_weights must be non-negative, "
                f"got {self.condition_weights}"
            )
        if sum(self.condition_weights) <= 0.0:
            raise ValueError(
                "condition_weights must have a positive sum, "
                f"got {self.condition_weights}"
            )
        if self.steps_per_time_of_day < 0:
            raise ValueError(
                f"steps_per_time_of_day must be >= 0, got {self.steps_per_time_of_day}"
            )
        if self.base_visibility_range <= 0.0:
            raise ValueError(
                f"base_visibility_range must be > 0, got {self.base_visibility_range}"
            )


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class WeatherState:
    """Mutable per-episode weather and time-of-day state.

    Parameters
    ----------
    condition:
        Current :class:`WeatherCondition`.
    time_of_day:
        Current :class:`TimeOfDay`.
    _tod_step_counter:
        Internal counter tracking steps elapsed within the current
        time-of-day period.  Not part of the public API.
    """

    condition: WeatherCondition
    time_of_day: TimeOfDay
    _tod_step_counter: int = field(default=0, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Episode initialisation
# ---------------------------------------------------------------------------


def sample_weather(
    rng: np.random.Generator,
    config: WeatherConfig,
) -> WeatherState:
    """Sample a random :class:`WeatherState` for the start of a new episode.

    If :attr:`WeatherConfig.fixed_condition` is set, that condition is always
    used.  Otherwise a condition is drawn according to the normalised
    :attr:`WeatherConfig.condition_weights`.

    If :attr:`WeatherConfig.fixed_time_of_day` is set, that time of day is
    always used.  Otherwise a uniform random time of day is chosen.

    Parameters
    ----------
    rng:
        A :class:`numpy.random.Generator` seeded by the caller.
    config:
        :class:`WeatherConfig` controlling fixed overrides and weights.

    Returns
    -------
    WeatherState
        A fresh state ready for use in a new episode.
    """
    # --- Weather condition ---
    if config.fixed_condition is not None:
        condition = config.fixed_condition
    else:
        weights = np.array(config.condition_weights, dtype=np.float64)
        weights = weights / weights.sum()
        condition_idx = int(rng.choice(NUM_CONDITIONS, p=weights))
        condition = WeatherCondition(condition_idx)

    # --- Time of day ---
    if config.fixed_time_of_day is not None:
        tod = config.fixed_time_of_day
    else:
        tod = TimeOfDay(int(rng.integers(0, NUM_TIME_OF_DAY)))

    return WeatherState(condition=condition, time_of_day=tod)


# ---------------------------------------------------------------------------
# Step-level update
# ---------------------------------------------------------------------------


def step_weather(state: WeatherState, config: WeatherConfig) -> None:
    """Advance time of day by one simulation step (mutates *state* in place).

    When :attr:`WeatherConfig.steps_per_time_of_day` is ``0``, this function
    is a no-op (time of day stays fixed for the episode).  Otherwise the
    internal step counter is incremented; when it reaches
    ``steps_per_time_of_day`` the time of day advances to the next period in
    the cycle DAWN → DAY → DUSK → NIGHT → DAWN and the counter resets.

    Parameters
    ----------
    state:
        The :class:`WeatherState` to update.
    config:
        :class:`WeatherConfig` with ``steps_per_time_of_day``.
    """
    if config.steps_per_time_of_day <= 0:
        return
    state._tod_step_counter += 1
    if state._tod_step_counter >= config.steps_per_time_of_day:
        state._tod_step_counter = 0
        state.time_of_day = TimeOfDay(
            (int(state.time_of_day) + 1) % NUM_TIME_OF_DAY
        )


# ---------------------------------------------------------------------------
# Effect accessors
# ---------------------------------------------------------------------------


def get_visibility_fraction(state: WeatherState) -> float:
    """Combined visibility fraction ``[0, 1]`` from weather condition + time of day.

    The combined value is the product of the condition's
    :attr:`ConditionEffects.visibility_fraction` and the time of day's
    :attr:`TODEffects.visibility_fraction`, clamped to ``[0, 1]``.

    Parameters
    ----------
    state:
        Current :class:`WeatherState`.

    Returns
    -------
    float
        Fraction of the base visibility range that remains clear.
    """
    cond_vis = CONDITION_EFFECTS[state.condition].visibility_fraction
    tod_vis = TIME_OF_DAY_EFFECTS[state.time_of_day].visibility_fraction
    return float(min(max(cond_vis * tod_vis, 0.0), 1.0))


def get_effective_visibility_range(
    state: WeatherState, config: WeatherConfig
) -> float:
    """Effective visibility range in metres for the current weather + time of day.

    Computed as ``get_visibility_fraction(state) × config.base_visibility_range``.

    Parameters
    ----------
    state:
        Current :class:`WeatherState`.
    config:
        :class:`WeatherConfig` with ``base_visibility_range``.

    Returns
    -------
    float
        Metres within which units can see each other.
    """
    return get_visibility_fraction(state) * config.base_visibility_range


def get_accuracy_modifier(state: WeatherState) -> float:
    """Combined fire-damage multiplier ``[0, 1]`` for the current weather.

    The multiplier is the product of the condition's
    :attr:`ConditionEffects.accuracy_modifier` and the time of day's
    :attr:`TODEffects.accuracy_modifier`, clamped to ``[0, 1]``.

    A value below ``1.0`` reduces musket and cannon effectiveness.  For
    heavy rain (RAIN + DAY) this is ``0.65``, giving a ≥ 30 % reduction
    consistent with Nafziger (1988) historical misfire estimates.

    Parameters
    ----------
    state:
        Current :class:`WeatherState`.

    Returns
    -------
    float
        Multiplicative fire-damage modifier.
    """
    cond_acc = CONDITION_EFFECTS[state.condition].accuracy_modifier
    tod_acc = TIME_OF_DAY_EFFECTS[state.time_of_day].accuracy_modifier
    return float(min(max(cond_acc * tod_acc, 0.0), 1.0))


def get_speed_modifier(state: WeatherState) -> float:
    """Movement speed multiplier ``(0, 1]`` from weather condition.

    Only the weather condition affects speed (time of day does not slow
    marching, though it affects visibility and accuracy).

    Parameters
    ----------
    state:
        Current :class:`WeatherState`.

    Returns
    -------
    float
        Multiplicative movement-speed modifier.
    """
    return float(CONDITION_EFFECTS[state.condition].speed_modifier)


def get_morale_stressor(state: WeatherState) -> float:
    """Combined per-step morale drain from weather condition + time of day.

    Returns the sum of :attr:`ConditionEffects.morale_stressor` and
    :attr:`TODEffects.morale_stressor`.  The caller is responsible for
    subtracting this from each battalion's morale each step.

    Parameters
    ----------
    state:
        Current :class:`WeatherState`.

    Returns
    -------
    float
        Non-negative morale drain per step.
    """
    cond_ms = CONDITION_EFFECTS[state.condition].morale_stressor
    tod_ms = TIME_OF_DAY_EFFECTS[state.time_of_day].morale_stressor
    return float(cond_ms + tod_ms)
