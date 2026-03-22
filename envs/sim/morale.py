# envs/sim/morale.py
"""Morale state machine with stressors for Napoleonic battalion simulation.

This module models battalion morale as a continuous state variable ``[0, 1]`` that:

* Degrades under fire, flanking/rear attack, and casualty accumulation.
* Below :attr:`MoraleConfig.cohesion_threshold`: battalion loses formation
  cohesion — the :func:`cohesion_modifier` function returns a value below
  ``1.0`` that callers can use to scale combat effectiveness.
* Below :attr:`MoraleConfig.rout_threshold`: a probabilistic routing check is
  triggered; the lower morale is, the more likely the unit routs.
* At ``0.0``: the unit is considered dispersed (equivalent to destroyed).
* Recovers each step based on distance from the enemy, nearby friendly
  support, and commander proximity.

Integration with :mod:`envs.sim.combat`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module is complementary to :func:`~envs.sim.combat.morale_check`.
:func:`update_morale` is a drop-in replacement that adds stressor-aware
flanking penalties and configurable distance/support recovery.  Callers
using :mod:`envs.sim.engine` or :mod:`envs.battalion_env` can opt in to
the richer mechanics by passing a :class:`MoraleConfig` instance.

Key public API
~~~~~~~~~~~~~~
* :class:`MoraleConfig` — all tunable parameters in one dataclass.
* :func:`compute_flank_stressor` — extra morale hit from flanking / rear fire.
* :func:`compute_recovery` — morale recovery from distance and support.
* :func:`cohesion_modifier` — multiplicative combat effectiveness factor.
* :func:`update_morale` — full per-step morale update (replaces basic
  :func:`~envs.sim.combat.morale_check` when stressors are desired).
* :func:`rout_velocity` — ``(vx, vy)`` for a routing unit fleeing the enemy.
* :func:`is_dispersed` — ``True`` when morale has reached zero (unit gone).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.sim.combat import CombatState

# ---------------------------------------------------------------------------
# Public constants (re-exported for convenience)
# ---------------------------------------------------------------------------

#: Default morale threshold below which cohesion begins to degrade.
DEFAULT_COHESION_THRESHOLD: float = 0.5

#: Default morale threshold below which a routing check is made each step.
DEFAULT_ROUT_THRESHOLD: float = 0.25

#: Extra morale penalty multiplier applied per unit of damage for a flanking hit.
FLANK_STRESSOR_MULT: float = 1.0

#: Extra morale penalty multiplier applied per unit of damage for a rear hit.
REAR_STRESSOR_MULT: float = 2.0


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MoraleConfig:
    """Configurable parameters for the morale state machine.

    Pass an instance to :func:`update_morale`, :class:`~envs.sim.engine.SimEngine`,
    or :class:`~envs.battalion_env.BattalionEnv` to enable the full morale
    stressor model.  All parameters have sensible defaults; tweak them to
    adjust scenario difficulty or historical calibration.

    Parameters
    ----------
    cohesion_threshold:
        Morale below this value causes cohesion loss.  The
        :func:`cohesion_modifier` function returns a value below ``1.0``
        when morale is beneath this threshold.
    rout_threshold:
        Morale below this value triggers the probabilistic routing check.
        Must be ``< cohesion_threshold``.
    base_recovery_rate:
        Passive morale recovery per step when the unit is not under fire.
        Increase to make recovery faster (easier scenario).
    distance_recovery_bonus:
        Additional recovery per step, scaled by ``enemy_dist / safe_distance``
        (capped at ``1.0``).  Rewards moving away from the enemy.
    friendly_support_bonus:
        Per-step morale bonus when a friendly unit is within
        ``commander_range`` metres.  Rewards cohesive formations.
    commander_proximity_bonus:
        Per-step morale bonus when the brigade commander is within
        ``commander_range`` metres.  Represents Napoleonic leadership effect.
    commander_range:
        Radius in metres within which the commander or a friendly unit must
        be to grant a proximity bonus.
    rally_threshold_multiplier:
        A routing unit can only attempt to rally once its morale has
        recovered to at least ``rout_threshold × rally_threshold_multiplier``.
    rally_probability:
        Probability per step that a unit attempts to rally (given it has
        reached the rally morale gate).
    rout_speed_multiplier:
        Routing units move at ``max_speed × rout_speed_multiplier``.
    safe_distance:
        Reference distance (metres) used to normalise the distance recovery
        bonus.  Units at or beyond this distance from the enemy receive the
        full ``distance_recovery_bonus``.
    """

    cohesion_threshold: float = DEFAULT_COHESION_THRESHOLD
    rout_threshold: float = DEFAULT_ROUT_THRESHOLD

    # Recovery rates
    base_recovery_rate: float = 0.01
    distance_recovery_bonus: float = 0.01
    friendly_support_bonus: float = 0.015
    commander_proximity_bonus: float = 0.02
    commander_range: float = 300.0

    # Rally mechanics
    rally_threshold_multiplier: float = 2.0
    rally_probability: float = 0.05

    # Rout movement
    rout_speed_multiplier: float = 1.5
    safe_distance: float = 400.0

    def __post_init__(self) -> None:
        if not (0.0 < self.rout_threshold < self.cohesion_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < rout_threshold ({self.rout_threshold}) "
                f"< cohesion_threshold ({self.cohesion_threshold}) <= 1.0"
            )
        if self.base_recovery_rate < 0:
            raise ValueError(f"base_recovery_rate must be >= 0, got {self.base_recovery_rate}")
        if self.rout_speed_multiplier <= 0:
            raise ValueError(
                f"rout_speed_multiplier must be > 0, got {self.rout_speed_multiplier}"
            )
        if self.safe_distance <= 0:
            raise ValueError(f"safe_distance must be > 0, got {self.safe_distance}")
        if self.commander_range <= 0:
            raise ValueError(f"commander_range must be > 0, got {self.commander_range}")


# ---------------------------------------------------------------------------
# Internal geometry helper
# ---------------------------------------------------------------------------


def _angle_diff(a: float, b: float) -> float:
    """Return the smallest signed difference *a − b* wrapped to ``(-π, π]``."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Stressor computation
# ---------------------------------------------------------------------------


def compute_flank_stressor(
    attacker_x: float,
    attacker_y: float,
    target_x: float,
    target_y: float,
    target_theta: float,
    base_damage: float,
) -> float:
    """Compute the additional morale penalty from a flanking or rear attack.

    A frontal hit (within ±45° of the target's facing direction) produces no
    additional penalty beyond the normal casualty-based morale hit.  A flanking
    hit (45°–135°) adds ``FLANK_STRESSOR_MULT × base_damage`` and a rear hit
    (>135°) adds ``REAR_STRESSOR_MULT × base_damage``.

    Parameters
    ----------
    attacker_x, attacker_y:
        Position of the attacking unit.
    target_x, target_y:
        Position of the defending unit.
    target_theta:
        Facing angle of the defending unit (radians).
    base_damage:
        The actual damage dealt this step (after all modifiers).

    Returns
    -------
    float
        Extra morale penalty (non-negative).  Returns ``0.0`` if *base_damage*
        is zero or the attack is frontal.
    """
    if base_damage <= 0.0:
        return 0.0

    dx = target_x - attacker_x
    dy = target_y - attacker_y
    # Arrival angle: direction from which bullets arrive at the target
    arrival_angle = math.atan2(dy, dx) + math.pi
    diff = abs(_angle_diff(arrival_angle, target_theta))

    if diff < math.pi / 4:          # frontal ±45°
        return 0.0
    elif diff < 3 * math.pi / 4:    # flanking 45°–135°
        return FLANK_STRESSOR_MULT * base_damage
    else:                            # rear >135°
        return REAR_STRESSOR_MULT * base_damage


# ---------------------------------------------------------------------------
# Recovery computation
# ---------------------------------------------------------------------------


def compute_recovery(
    enemy_dist: float,
    config: MoraleConfig,
    friendly_dist: Optional[float] = None,
    commander_dist: Optional[float] = None,
) -> float:
    """Compute the morale recovery bonus for a step when the unit is not under fire.

    Parameters
    ----------
    enemy_dist:
        Distance to the nearest enemy unit (metres).  Higher is better.
    config:
        :class:`MoraleConfig` with recovery rate parameters.
    friendly_dist:
        Distance to the nearest friendly unit (metres).  ``None`` if no
        friendly unit exists.  Grants a bonus if within ``commander_range``.
    commander_dist:
        Distance to the brigade commander (metres).  ``None`` if unknown.
        Grants a bonus if within ``commander_range``.

    Returns
    -------
    float
        Total morale recovery for this step (non-negative).
    """
    bonus = config.base_recovery_rate

    # Distance-based recovery: scales from 0 at contact to full at safe_distance
    dist_factor = min(enemy_dist / config.safe_distance, 1.0)
    bonus += config.distance_recovery_bonus * dist_factor

    # Friendly support bonus
    if friendly_dist is not None and friendly_dist <= config.commander_range:
        bonus += config.friendly_support_bonus

    # Commander proximity bonus
    if commander_dist is not None and commander_dist <= config.commander_range:
        bonus += config.commander_proximity_bonus

    return bonus


# ---------------------------------------------------------------------------
# Cohesion modifier
# ---------------------------------------------------------------------------


def cohesion_modifier(morale: float, config: MoraleConfig) -> float:
    """Return the formation-cohesion effectiveness multiplier for the given morale.

    Above ``cohesion_threshold`` the unit is at full cohesion (returns ``1.0``).
    Below the threshold cohesion degrades linearly to ``0.5`` at morale zero,
    representing a disorganised but still partially functional unit.

    Parameters
    ----------
    morale:
        Current morale level ``[0, 1]``.
    config:
        :class:`MoraleConfig` with ``cohesion_threshold``.

    Returns
    -------
    float
        Multiplicative combat-effectiveness modifier in ``[0.5, 1.0]``.
    """
    morale = float(np.clip(morale, 0.0, 1.0))
    if morale >= config.cohesion_threshold:
        return 1.0
    # Linear degradation from 1.0 at threshold to 0.5 at morale=0
    ratio = morale / config.cohesion_threshold
    return 0.5 + 0.5 * ratio


# ---------------------------------------------------------------------------
# Full morale update
# ---------------------------------------------------------------------------


def update_morale(
    state: "CombatState",
    *,
    enemy_dist: float,
    config: MoraleConfig,
    flank_penalty: float = 0.0,
    friendly_dist: Optional[float] = None,
    commander_dist: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> bool:
    """Full per-step morale update incorporating stressors and distance recovery.

    This function is a drop-in replacement for
    :func:`~envs.sim.combat.morale_check` that adds:

    * Flanking / rear fire stressor via *flank_penalty*.
    * Distance-from-enemy recovery scaling.
    * Friendly-support and commander-proximity bonuses.
    * Configurable recovery rate via :class:`MoraleConfig`.

    It should be called **once per step** after all damage for that step has
    been accumulated via :func:`~envs.sim.combat.apply_casualties`.

    Parameters
    ----------
    state:
        The :class:`~envs.sim.combat.CombatState` of the battalion being updated.
    enemy_dist:
        Distance to the nearest enemy (metres).
    config:
        :class:`MoraleConfig` controlling thresholds and rates.
    flank_penalty:
        Additional morale hit from flanking/rear fire (as returned by
        :func:`compute_flank_stressor`).  Defaults to ``0.0``.
    friendly_dist:
        Distance to nearest friendly unit (metres), or ``None``.
    commander_dist:
        Distance to brigade commander (metres), or ``None``.
    rng:
        Optional :class:`numpy.random.Generator`.  A fresh generator is
        created if ``None`` is passed.

    Returns
    -------
    bool
        ``True`` if the unit is routing after this update, ``False`` otherwise.
    """
    from envs.sim.combat import MORALE_CASUALTY_WEIGHT  # avoid circular import

    if rng is None:
        rng = np.random.default_rng()

    # 1. Casualty-based morale hit (same formula as basic morale_check)
    morale_hit = state.accumulated_damage * MORALE_CASUALTY_WEIGHT

    # 2. Additional flank/rear penalty
    morale_hit += flank_penalty

    state.morale = max(0.0, state.morale - morale_hit)

    # 3. Recovery when not under fire
    if state.accumulated_damage == 0.0:
        recovery = compute_recovery(
            enemy_dist,
            config,
            friendly_dist=friendly_dist,
            commander_dist=commander_dist,
        )
        state.morale = min(1.0, state.morale + recovery)

    # 4. Rout/rally resolution
    if state.is_routing:
        rally_gate = config.rout_threshold * config.rally_threshold_multiplier
        if state.morale > rally_gate and rng.random() < config.rally_probability:
            state.is_routing = False
        return state.is_routing

    # 5. Fresh routing check when morale is below threshold
    if state.morale < config.rout_threshold:
        rout_prob = 1.0 - (state.morale / config.rout_threshold)
        if rng.random() < rout_prob:
            state.is_routing = True

    return state.is_routing


# ---------------------------------------------------------------------------
# Rout movement
# ---------------------------------------------------------------------------


def rout_velocity(
    unit_x: float,
    unit_y: float,
    enemy_x: float,
    enemy_y: float,
    max_speed: float,
    config: MoraleConfig,
) -> Tuple[float, float]:
    """Compute the velocity for a routing unit fleeing directly away from the enemy.

    The routing unit moves at ``max_speed × rout_speed_multiplier`` in the
    direction directly opposite the nearest enemy.  Callers should apply this
    velocity with ``Battalion.move(vx, vy, dt)`` instead of letting the agent
    or scripted opponent control the unit.

    Parameters
    ----------
    unit_x, unit_y:
        Current position of the routing unit (metres).
    enemy_x, enemy_y:
        Position of the nearest enemy unit (metres).
    max_speed:
        Unit's ``Battalion.max_speed`` (metres / step at dt=1.0).
    config:
        :class:`MoraleConfig` with ``rout_speed_multiplier``.

    Returns
    -------
    Tuple[float, float]
        ``(vx, vy)`` velocity components.  The speed equals
        ``max_speed × rout_speed_multiplier``.
    """
    dx = unit_x - enemy_x
    dy = unit_y - enemy_y
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 1e-6:
        # Unit is exactly on top of the enemy — flee east as a fallback
        return (max_speed * config.rout_speed_multiplier, 0.0)

    speed = max_speed * config.rout_speed_multiplier
    return (dx / dist * speed, dy / dist * speed)


# ---------------------------------------------------------------------------
# Dispersal check
# ---------------------------------------------------------------------------


def is_dispersed(state: "CombatState") -> bool:
    """Return ``True`` when the unit's morale has reached zero.

    A dispersed unit is considered permanently destroyed and should be
    removed from the simulation (equivalent to ``strength <= 0``).

    Parameters
    ----------
    state:
        The :class:`~envs.sim.combat.CombatState` of the battalion.
    """
    return state.morale <= 0.0
