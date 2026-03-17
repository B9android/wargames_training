# Firepower, damage, morale

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.sim.battalion import Battalion

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: Base fractional strength loss per volley at point-blank with intensity=1.
BASE_FIRE_DAMAGE: float = 0.05

#: How much each point of strength loss translates to a morale hit.
MORALE_CASUALTY_WEIGHT: float = 0.4

#: Morale below this value triggers a routing check.
MORALE_ROUT_THRESHOLD: float = 0.25

#: Passive morale recovery per simulation step when not under fire.
MORALE_RECOVERY_RATE: float = 0.01

#: Damage multiplier when the target is hit from a flanking angle (±45–135°).
FLANKED_DAMAGE_MULT: float = 1.5

#: Damage multiplier when the target is hit from the rear (>135°).
REAR_DAMAGE_MULT: float = 2.0

#: Module-level default random generator.  Used when callers do not pass an RNG.
#: Avoids the overhead of re-initializing RNG state on every call.
_DEFAULT_RNG: np.random.Generator = np.random.default_rng()


# ---------------------------------------------------------------------------
# Per-battalion combat state
# ---------------------------------------------------------------------------


@dataclass
class CombatState:
    """Mutable combat bookkeeping attached to a single battalion.

    One ``CombatState`` instance should be created per battalion and
    passed alongside the :class:`~envs.sim.battalion.Battalion` object
    to every combat function.
    """

    morale: float = 1.0
    """Current morale level in [0, 1].  Starts at full morale."""

    accumulated_damage: float = 0.0
    """Damage received *this simulation step*.  Reset by :meth:`reset_step_accumulators`."""

    total_casualties: float = 0.0
    """Cumulative strength lost since deployment (monotonically non-decreasing)."""

    is_routing: bool = False
    """``True`` when the unit has broken and is fleeing."""

    shots_fired: int = 0
    """Number of volleys this unit has fired.  Incremented by :func:`resolve_volley`."""

    def reset_step_accumulators(self) -> None:
        """Reset per-step counters.  Call at the *start* of each simulation step."""
        self.accumulated_damage = 0.0


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _angle_diff(a: float, b: float) -> float:
    """Return the smallest signed difference *a − b* wrapped to (−π, π]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _hit_angle_multiplier(shooter: Battalion, target: Battalion) -> float:
    """Return a damage multiplier based on the angle of attack.

    * Frontal hit  (within ±45° of target's facing direction)  → ``1.0``
    * Flanking hit (between 45° and 135° off the facing axis)  → ``FLANKED_DAMAGE_MULT``
    * Rear hit     (more than 135° from the facing direction)   → ``REAR_DAMAGE_MULT``
    """
    dx = target.x - shooter.x
    dy = target.y - shooter.y
    # Add π to reverse direction: gives the angle from which bullets arrive at the target
    # (opposite of the shooter-to-target direction)
    arrival_angle = np.arctan2(dy, dx) + np.pi
    diff = abs(_angle_diff(arrival_angle, target.theta))

    if diff < np.pi / 4:        # frontal ±45°
        return 1.0
    elif diff < 3 * np.pi / 4:  # flanking 45–135°
        return FLANKED_DAMAGE_MULT
    else:                        # rear >135°
        return REAR_DAMAGE_MULT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fire_damage(
    shooter: Battalion,
    target: Battalion,
    intensity: float,
) -> float:
    """Compute raw damage from a single volley.

    Parameters
    ----------
    shooter:
        The firing battalion.
    target:
        The battalion receiving fire.
    intensity:
        Agent-controlled fire intensity in ``[0, 1]``.  Higher values
        represent a larger volley (more ammunition spent).

    Returns
    -------
    float
        Fractional strength damage ``≥ 0.0``.  Returns ``0.0`` if the
        target is outside the shooter's range or fire arc.
    """
    if not shooter.can_fire_at(target):
        return 0.0

    # Clamp intensity to the documented [0, 1] range
    intensity = max(0.0, min(1.0, float(intensity)))

    dx = target.x - shooter.x
    dy = target.y - shooter.y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Linear range falloff: full damage at dist=0, zero damage at fire_range
    range_factor = max(0.0, 1.0 - dist / shooter.fire_range)

    # Angle-of-attack bonus
    angle_mult = _hit_angle_multiplier(shooter, target)

    # Weaker battalions produce less fire volume
    shooter_strength_factor = max(0.0, shooter.strength)

    base_damage = BASE_FIRE_DAMAGE * float(intensity)
    damage = base_damage * range_factor * angle_mult * shooter_strength_factor
    return float(damage)


def apply_casualties(
    target: Battalion,
    state: CombatState,
    damage: float,
) -> float:
    """Apply *damage* to *target* and update *state*.

    Clamps the result so ``target.strength`` stays in ``[0, 1]``.
    Updates both the step-level and cumulative damage accumulators.

    Parameters
    ----------
    target:
        The battalion receiving damage.
    state:
        The :class:`CombatState` belonging to *target*.
    damage:
        Raw damage value (as returned by :func:`compute_fire_damage`).

    Returns
    -------
    float
        Actual damage applied (may be less than *damage* if strength
        would otherwise go below zero).
    """
    actual = min(max(0.0, float(damage)), target.strength)
    # Clamp to [0, 1]: lower bound guards against floating-point underflow,
    # upper bound ensures a pre-existing over-strength value is corrected.
    target.strength = max(0.0, min(1.0, target.strength - actual))
    state.accumulated_damage += actual
    state.total_casualties += actual
    return actual


def morale_check(
    state: CombatState,
    rng: np.random.Generator | None = None,
) -> bool:
    """Update morale and check whether the unit routes.

    Should be called *once per step* after all damage for that step has
    been accumulated via :func:`apply_casualties`.

    Morale mechanics
    ~~~~~~~~~~~~~~~~
    * Each step's accumulated damage is converted to a morale hit using
      ``MORALE_CASUALTY_WEIGHT``.
    * When the unit is not under fire it recovers ``MORALE_RECOVERY_RATE``
      per step (applies whether or not the unit is currently routing).
    * A unit whose morale falls below ``MORALE_ROUT_THRESHOLD`` makes a
      probabilistic routing check; the lower morale is, the more likely
      it routs.
    * A routing unit has a small chance (5%) of rallying each step once
      morale recovers to ``2 × MORALE_ROUT_THRESHOLD``.

    Parameters
    ----------
    state:
        The :class:`CombatState` belonging to the battalion being checked.
    rng:
        Optional :class:`numpy.random.Generator`.  Falls back to the
        module-level ``_DEFAULT_RNG`` if ``None`` is passed.

    Returns
    -------
    bool
        ``True`` if the unit is routing after this check, ``False`` otherwise.
    """
    if rng is None:
        rng = _DEFAULT_RNG

    # Apply morale penalty from this step's casualties
    morale_hit = state.accumulated_damage * MORALE_CASUALTY_WEIGHT
    state.morale = max(0.0, state.morale - morale_hit)

    # Passive recovery when not under fire (applies even while routing so
    # routing units can eventually reach the rally gate)
    if state.accumulated_damage == 0.0:
        state.morale = min(1.0, state.morale + MORALE_RECOVERY_RATE)

    if state.is_routing:
        # Rally check: requires morale to have recovered above 2× threshold
        if state.morale > MORALE_ROUT_THRESHOLD * 2 and rng.random() < 0.05:
            state.is_routing = False
        return state.is_routing

    # Fresh rout check when morale is below threshold
    if state.morale < MORALE_ROUT_THRESHOLD:
        # Linear probability: 0% chance at the threshold, 100% chance at morale=0
        rout_probability = 1.0 - (state.morale / MORALE_ROUT_THRESHOLD)
        if rng.random() < rout_probability:
            state.is_routing = True

    return state.is_routing


def resolve_volley(
    shooter: Battalion,
    shooter_state: CombatState,
    target: Battalion,
    target_state: CombatState,
    intensity: float,
    rng: np.random.Generator | None = None,
) -> dict:
    """Convenience wrapper: compute damage, apply casualties, run morale check.

    Parameters
    ----------
    shooter, shooter_state:
        Firing unit and its combat state.
    target, target_state:
        Receiving unit and its combat state.
    intensity:
        Fire intensity in ``[0, 1]`` (agent action).
    rng:
        Optional random generator forwarded to :func:`morale_check`.

    Returns
    -------
    dict with keys:
        ``damage_dealt`` – actual damage applied to target strength.
        ``target_routing`` – whether the target is now routing.
        ``target_strength`` – target's strength after casualties.
        ``target_morale`` – target's morale after the check.
    """
    damage = compute_fire_damage(shooter, target, intensity)
    actual = apply_casualties(target, target_state, damage)
    shooter_state.shots_fired += 1
    routing = morale_check(target_state, rng=rng)
    return {
        "damage_dealt": actual,
        "target_routing": routing,
        "target_strength": target.strength,
        "target_morale": target_state.morale,
    }
