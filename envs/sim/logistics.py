# SPDX-License-Identifier: MIT
"""Supply, ammunition, and fatigue model for Napoleonic battalion simulation.

Tracks per-battalion ammunition, food/water supply, and cumulative fatigue.
Resupply requires proximity to a supply wagon unit.  Sustained operations
without resupply degrade effectiveness.

Historical background
~~~~~~~~~~~~~~~~~~~~~
Supply and attrition shaped every Napoleonic campaign.  Armies that outmarched
their logistics trains could fight brilliantly for a day but were helpless
two days later when ammunition wagons had not caught up.  This module gives
agents the incentive to conserve ammo, protect supply wagons, and time attacks
to exploit an exhausted enemy.

Key public API
~~~~~~~~~~~~~~
* :class:`LogisticsConfig` — all tunable parameters in one dataclass.
* :class:`LogisticsState` — per-battalion mutable supply/fatigue state.
* :class:`SupplyWagon` — slow, fragile high-value logistics unit.
* :func:`consume_ammo` — spend ammunition on a volley; returns effective intensity.
* :func:`consume_food` — per-step food/water depletion.
* :func:`update_fatigue` — accumulate or recover fatigue each step.
* :func:`check_resupply` — replenish ammo/food when near a friendly wagon.
* :func:`get_ammo_modifier` — accuracy/volume modifier from ammo level.
* :func:`get_fatigue_speed_modifier` — movement speed modifier from fatigue.
* :func:`get_fatigue_accuracy_modifier` — fire accuracy modifier from fatigue.

Usage example::

    from envs.sim.logistics import (
        LogisticsConfig, LogisticsState, SupplyWagon,
        consume_ammo, update_fatigue, check_resupply,
    )

    config = LogisticsConfig()
    state  = LogisticsState()
    wagon  = SupplyWagon(x=300.0, y=500.0, team=0)

    # Before firing — spend ammo (returns effective intensity)
    effective = consume_ammo(state, intensity=1.0, config=config)

    # After movement — accumulate fatigue
    update_fatigue(state, is_moving=True, is_firing=False, config=config)

    # Each step — try to resupply from nearby wagon
    check_resupply(state, unit_x=295.0, unit_y=505.0, wagon=wagon, config=config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default initial ammunition level (full load-out).
DEFAULT_INITIAL_AMMO: float = 1.0

#: Default initial food/water level (full rations).
DEFAULT_INITIAL_FOOD: float = 1.0

#: Ammo threshold below which the battalion is considered "critically low".
CRITICAL_AMMO_THRESHOLD: float = 0.1

#: Fatigue threshold above which speed and accuracy penalties apply.
FATIGUE_ONSET_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class LogisticsConfig:
    """Configurable parameters for the supply, ammunition, and fatigue model.

    Pass an instance to :class:`~envs.battalion_env.BattalionEnv` (via the
    *logistics_config* parameter) to enable the full logistics model.  All
    parameters have historically-informed defaults; adjust them to tune
    scenario difficulty.

    Parameters
    ----------
    initial_ammo:
        Starting ammunition level in ``[0, 1]``.  ``1.0`` = full load-out.
    initial_food:
        Starting food/water level in ``[0, 1]``.  ``1.0`` = full rations.
    ammo_per_volley:
        Ammunition consumed per unit of fire intensity per step.  A value
        of ``0.01`` means firing at full intensity for 100 steps exhausts
        the load-out.  Scaled by intensity so partial volleys cost less.
    food_per_step:
        Food/water consumed per simulation step regardless of activity.
        Models the steady drain of rations on campaign.
    fatigue_per_move_step:
        Fatigue accumulated per step when the unit is moving (``> 0``).
    fatigue_per_fire_step:
        Fatigue accumulated per step when the unit is firing (``> 0``).
    fatigue_recovery_per_halt_step:
        Fatigue recovered per step when the unit is stationary and not
        firing.  Should exceed ``fatigue_per_move_step`` so that resting
        eventually restores full readiness.
    resupply_radius:
        Distance in metres within which a battalion can draw on a supply
        wagon.  The unit must be within this radius *and* the wagon must be
        alive.
    ammo_resupply_rate:
        Ammunition recovered per step while within resupply radius.
    food_resupply_rate:
        Food/water recovered per step while within resupply radius.
    low_ammo_accuracy_penalty:
        Multiplicative fire-intensity modifier applied when ammo is below
        :data:`CRITICAL_AMMO_THRESHOLD`.  A value of ``0.5`` halves the
        effective volley when critically low.
    fatigue_speed_penalty:
        Maximum fractional speed reduction at full fatigue (``1.0``).
        At fatigue ``f``, speed is multiplied by
        ``1 − fatigue_speed_penalty × max(0, (f − onset) / (1 − onset))``.
    fatigue_accuracy_penalty:
        Maximum fractional fire-intensity reduction at full fatigue.
        Applies the same capped formula as *fatigue_speed_penalty*.
    enable_resupply:
        When ``False`` the :func:`check_resupply` function is a no-op.
        Useful for short training episodes where supply logistics would
        end episodes too quickly.
    wagon_speed:
        Maximum movement speed of supply wagons (metres per step at dt=1).
        Wagons are slow: ~10 m/step vs 50 m/step for infantry.
    wagon_max_strength:
        Initial strength of a new supply wagon in ``[0, 1]``.
    """

    # Initial levels
    initial_ammo: float = DEFAULT_INITIAL_AMMO
    initial_food: float = DEFAULT_INITIAL_FOOD

    # Consumption rates
    ammo_per_volley: float = 0.01
    food_per_step: float = 0.0005

    # Fatigue rates
    fatigue_per_move_step: float = 0.002
    fatigue_per_fire_step: float = 0.001
    fatigue_recovery_per_halt_step: float = 0.003

    # Resupply
    resupply_radius: float = 100.0
    ammo_resupply_rate: float = 0.02
    food_resupply_rate: float = 0.01
    enable_resupply: bool = True

    # Performance modifiers
    low_ammo_accuracy_penalty: float = 0.5
    fatigue_speed_penalty: float = 0.3
    fatigue_accuracy_penalty: float = 0.2

    # Supply wagon properties
    wagon_speed: float = 10.0
    wagon_max_strength: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.initial_ammo <= 1.0):
            raise ValueError(
                f"initial_ammo must be in [0, 1], got {self.initial_ammo}"
            )
        if not (0.0 <= self.initial_food <= 1.0):
            raise ValueError(
                f"initial_food must be in [0, 1], got {self.initial_food}"
            )
        if self.ammo_per_volley < 0.0:
            raise ValueError(
                f"ammo_per_volley must be >= 0, got {self.ammo_per_volley}"
            )
        if self.food_per_step < 0.0:
            raise ValueError(
                f"food_per_step must be >= 0, got {self.food_per_step}"
            )
        if self.fatigue_per_move_step < 0.0:
            raise ValueError(
                f"fatigue_per_move_step must be >= 0, got {self.fatigue_per_move_step}"
            )
        if self.fatigue_per_fire_step < 0.0:
            raise ValueError(
                f"fatigue_per_fire_step must be >= 0, got {self.fatigue_per_fire_step}"
            )
        if self.fatigue_recovery_per_halt_step < 0.0:
            raise ValueError(
                f"fatigue_recovery_per_halt_step must be >= 0, "
                f"got {self.fatigue_recovery_per_halt_step}"
            )
        if self.resupply_radius <= 0.0:
            raise ValueError(
                f"resupply_radius must be positive, got {self.resupply_radius}"
            )
        if not (0.0 <= self.ammo_resupply_rate <= 1.0):
            raise ValueError(
                f"ammo_resupply_rate must be in [0, 1], got {self.ammo_resupply_rate}"
            )
        if not (0.0 <= self.food_resupply_rate <= 1.0):
            raise ValueError(
                f"food_resupply_rate must be in [0, 1], got {self.food_resupply_rate}"
            )
        if not (0.0 <= self.low_ammo_accuracy_penalty <= 1.0):
            raise ValueError(
                f"low_ammo_accuracy_penalty must be in [0, 1], "
                f"got {self.low_ammo_accuracy_penalty}"
            )
        if not (0.0 <= self.fatigue_speed_penalty <= 1.0):
            raise ValueError(
                f"fatigue_speed_penalty must be in [0, 1], "
                f"got {self.fatigue_speed_penalty}"
            )
        if not (0.0 <= self.fatigue_accuracy_penalty <= 1.0):
            raise ValueError(
                f"fatigue_accuracy_penalty must be in [0, 1], "
                f"got {self.fatigue_accuracy_penalty}"
            )
        if self.wagon_speed <= 0.0:
            raise ValueError(
                f"wagon_speed must be positive, got {self.wagon_speed}"
            )
        if not (0.0 < self.wagon_max_strength <= 1.0):
            raise ValueError(
                f"wagon_max_strength must be in (0, 1], got {self.wagon_max_strength}"
            )


# ---------------------------------------------------------------------------
# Per-battalion logistics state
# ---------------------------------------------------------------------------


@dataclass
class LogisticsState:
    """Mutable supply and fatigue state for a single battalion.

    One :class:`LogisticsState` is paired with each
    :class:`~envs.sim.battalion.Battalion` (and its
    :class:`~envs.sim.combat.CombatState`) when the logistics model is active.

    Parameters
    ----------
    ammo:
        Current ammunition level in ``[0, 1]``.
    food:
        Current food/water level in ``[0, 1]``.
    fatigue:
        Current fatigue level in ``[0, 1]``.  ``0`` = fresh, ``1`` = exhausted.
    """

    ammo: float = field(default=DEFAULT_INITIAL_AMMO)
    food: float = field(default=DEFAULT_INITIAL_FOOD)
    fatigue: float = 0.0

    def __post_init__(self) -> None:
        # Clamp core logistics values to their documented [0, 1] range to
        # prevent invalid states (e.g., negative modifiers downstream).
        self.ammo = min(max(self.ammo, 0.0), 1.0)
        self.food = min(max(self.food, 0.0), 1.0)
        self.fatigue = min(max(self.fatigue, 0.0), 1.0)

    @property
    def is_ammo_exhausted(self) -> bool:
        """``True`` when the battalion has no ammunition left."""
        return self.ammo <= 0.0

    @property
    def is_critically_low_ammo(self) -> bool:
        """``True`` when ammo is below the critical threshold."""
        return self.ammo < CRITICAL_AMMO_THRESHOLD

    @property
    def is_starving(self) -> bool:
        """``True`` when food/water is exhausted."""
        return self.food <= 0.0


# ---------------------------------------------------------------------------
# Supply wagon
# ---------------------------------------------------------------------------


@dataclass
class SupplyWagon:
    """A supply wagon unit — slow, fragile, high-value logistics asset.

    Supply wagons are the focal point of the resupply mechanic: battalions
    within :attr:`~LogisticsConfig.resupply_radius` of a friendly wagon
    automatically replenish ammunition and food.  Destroying an enemy wagon
    cuts off its resupply chain.

    Parameters
    ----------
    x, y:
        Current position in metres on the simulation map.
    team:
        Owning side — ``0`` for Blue, ``1`` for Red.
    strength:
        Current health in ``[0, 1]``.  ``0`` = destroyed; resupply stops.
    theta:
        Facing angle (radians).  Wagons typically follow behind the main
        force but can be given movement commands.
    """

    x: float
    y: float
    team: int
    strength: float = 1.0
    theta: float = 0.0

    @property
    def is_alive(self) -> bool:
        """``True`` while the wagon has not been destroyed."""
        return self.strength > 0.0

    def take_damage(self, damage: float) -> None:
        """Apply *damage* to the wagon; strength is clamped to ``[0, 1]``.

        Parameters
        ----------
        damage:
            Non-negative amount of damage to inflict.

        Raises
        ------
        ValueError
            If *damage* is negative.
        """
        if damage < 0.0:
            raise ValueError(f"damage must be non-negative, got {damage!r}")
        self.strength = float(np.clip(self.strength - damage, 0.0, 1.0))

    def move_toward(self, target_x: float, target_y: float, speed: float, dt: float = 0.1) -> None:
        """Move the wagon one step toward ``(target_x, target_y)`` at *speed*.

        Parameters
        ----------
        target_x, target_y:
            Destination coordinates in metres.
        speed:
            Movement speed in metres per step (before dt scaling).
        dt:
            Simulation time step (seconds).
        """
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return
        step = speed * dt
        if step >= dist:
            self.x = target_x
            self.y = target_y
        else:
            self.x += dx / dist * step
            self.y += dy / dist * step
        self.theta = math.atan2(dy, dx)


# ---------------------------------------------------------------------------
# Core logistics functions
# ---------------------------------------------------------------------------


def consume_ammo(
    state: LogisticsState,
    intensity: float,
    config: LogisticsConfig,
) -> float:
    """Consume ammunition for a volley and return the effective fire intensity.

    When the battalion has ammunition the function deducts
    ``intensity × ammo_per_volley`` from ``state.ammo`` and returns the
    original *intensity* (possibly capped by available ammo).  When ammo
    is exhausted the function returns ``0.0`` — the weapon jams.

    Parameters
    ----------
    state:
        The :class:`LogisticsState` of the firing battalion.
    intensity:
        Requested fire intensity in ``[0, 1]``.
    config:
        :class:`LogisticsConfig` with ``ammo_per_volley``.

    Returns
    -------
    float
        Effective fire intensity after applying the ammo constraint.  This
        value should be passed to
        :func:`~envs.sim.combat.compute_fire_damage` in place of the raw
        agent action.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity == 0.0:
        return 0.0

    if state.is_ammo_exhausted:
        return 0.0

    # Maximum volleys available at full intensity
    cost = intensity * config.ammo_per_volley

    if state.ammo < cost:
        # Scale intensity down to what ammo remains
        intensity = intensity * (state.ammo / cost)
        state.ammo = 0.0
    else:
        state.ammo = max(0.0, state.ammo - cost)

    return float(intensity)


def consume_food(
    state: LogisticsState,
    config: LogisticsConfig,
) -> None:
    """Deduct per-step food/water consumption from *state*.

    Parameters
    ----------
    state:
        The :class:`LogisticsState` to update.
    config:
        :class:`LogisticsConfig` with ``food_per_step``.
    """
    state.food = max(0.0, state.food - config.food_per_step)


def update_fatigue(
    state: LogisticsState,
    is_moving: bool,
    is_firing: bool,
    config: LogisticsConfig,
) -> None:
    """Accumulate or recover fatigue for one simulation step.

    * Moving units accumulate ``fatigue_per_move_step`` per step.
    * Firing units accumulate ``fatigue_per_fire_step`` per step
      (on top of movement fatigue if both apply).
    * Units that are neither moving nor firing recover
      ``fatigue_recovery_per_halt_step`` per step.

    Parameters
    ----------
    state:
        The :class:`LogisticsState` to update.
    is_moving:
        ``True`` if the battalion moved this step (non-zero velocity).
    is_firing:
        ``True`` if the battalion fired this step (intensity > 0).
    config:
        :class:`LogisticsConfig` with fatigue rate parameters.
    """
    if is_moving or is_firing:
        delta = 0.0
        if is_moving:
            delta += config.fatigue_per_move_step
        if is_firing:
            delta += config.fatigue_per_fire_step
        state.fatigue = min(1.0, state.fatigue + delta)
    else:
        state.fatigue = max(0.0, state.fatigue - config.fatigue_recovery_per_halt_step)


def check_resupply(
    state: LogisticsState,
    unit_x: float,
    unit_y: float,
    wagon: SupplyWagon,
    config: LogisticsConfig,
) -> bool:
    """Attempt to resupply from *wagon* if within range.

    A battalion can draw resupply when:

    * ``config.enable_resupply`` is ``True``.
    * The wagon is alive (``wagon.strength > 0``).
    * The battalion is within ``config.resupply_radius`` metres of the wagon.

    When eligible, ``state.ammo`` and ``state.food`` are incremented by
    ``ammo_resupply_rate`` and ``food_resupply_rate`` respectively, clamped
    at ``1.0``.

    Parameters
    ----------
    state:
        The battalion's :class:`LogisticsState` to replenish.
    unit_x, unit_y:
        Current position of the battalion in metres.
    wagon:
        The supply wagon to draw from.
    config:
        :class:`LogisticsConfig` with resupply parameters.

    Returns
    -------
    bool
        ``True`` if resupply occurred this step, ``False`` otherwise.
    """
    if not config.enable_resupply:
        return False
    if not wagon.is_alive:
        return False

    dx = unit_x - wagon.x
    dy = unit_y - wagon.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > config.resupply_radius:
        return False

    state.ammo = min(1.0, state.ammo + config.ammo_resupply_rate)
    state.food = min(1.0, state.food + config.food_resupply_rate)
    return True


def get_ammo_modifier(
    state: LogisticsState,
    config: LogisticsConfig,
) -> float:
    """Return a multiplicative fire-intensity modifier based on ammo level.

    * When ammo is above :data:`CRITICAL_AMMO_THRESHOLD` the modifier is
      ``1.0`` (no penalty).
    * Below the threshold the modifier decreases linearly from ``1.0`` at
      the threshold to ``low_ammo_accuracy_penalty`` at zero ammo.

    Parameters
    ----------
    state:
        The battalion's :class:`LogisticsState`.
    config:
        :class:`LogisticsConfig` with ``low_ammo_accuracy_penalty``.

    Returns
    -------
    float
        Modifier in ``[low_ammo_accuracy_penalty, 1.0]``.
    """
    if state.ammo >= CRITICAL_AMMO_THRESHOLD:
        return 1.0
    # Linear interpolation from penalty at 0 to 1.0 at threshold
    ratio = state.ammo / CRITICAL_AMMO_THRESHOLD
    return float(
        config.low_ammo_accuracy_penalty
        + (1.0 - config.low_ammo_accuracy_penalty) * ratio
    )


def get_fatigue_speed_modifier(
    state: LogisticsState,
    config: LogisticsConfig,
) -> float:
    """Return a multiplicative movement-speed modifier based on fatigue.

    Fatigue has no effect below :data:`FATIGUE_ONSET_THRESHOLD`.  Above the
    threshold the penalty grows linearly, reaching a maximum reduction of
    ``fatigue_speed_penalty`` at full fatigue (``1.0``).

    Parameters
    ----------
    state:
        The battalion's :class:`LogisticsState`.
    config:
        :class:`LogisticsConfig` with ``fatigue_speed_penalty``.

    Returns
    -------
    float
        Speed multiplier in ``[1 − fatigue_speed_penalty, 1.0]``.
    """
    onset = FATIGUE_ONSET_THRESHOLD
    if state.fatigue <= onset:
        return 1.0
    # Fraction of the post-onset range [onset, 1] that fatigue occupies
    fraction = (state.fatigue - onset) / (1.0 - onset)
    return float(1.0 - config.fatigue_speed_penalty * fraction)


def get_fatigue_accuracy_modifier(
    state: LogisticsState,
    config: LogisticsConfig,
) -> float:
    """Return a multiplicative fire-accuracy modifier based on fatigue.

    Uses the same onset / linear degradation formula as
    :func:`get_fatigue_speed_modifier`, but applies
    ``fatigue_accuracy_penalty`` instead.

    Parameters
    ----------
    state:
        The battalion's :class:`LogisticsState`.
    config:
        :class:`LogisticsConfig` with ``fatigue_accuracy_penalty``.

    Returns
    -------
    float
        Accuracy multiplier in ``[1 − fatigue_accuracy_penalty, 1.0]``.
    """
    onset = FATIGUE_ONSET_THRESHOLD
    if state.fatigue <= onset:
        return 1.0
    fraction = (state.fatigue - onset) / (1.0 - onset)
    return float(1.0 - config.fatigue_accuracy_penalty * fraction)
