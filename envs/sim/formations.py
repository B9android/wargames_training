# envs/sim/formations.py
"""Formation system for Napoleonic battalion simulation.

Implements the four canonical Napoleonic formations as discrete states with
associated firepower, movement-speed, and vulnerability modifiers.  Formation
changes require a transition period, creating tactical trade-offs.

Formation summary
~~~~~~~~~~~~~~~~~
==========  ============  ===========  ==========  ================
Formation   Firepower     Speed        Morale      Cavalry resist
==========  ============  ===========  ==========  ================
LINE        1.0 (best)    0.8          1.0         0.5 (weak)
COLUMN      0.5           1.5 (best)   0.9         0.8
SQUARE      0.75          0.3 (slow)   1.5 (best)  5.0 (bounces)
SKIRMISH    0.6           1.2          1.1         0.6
==========  ============  ===========  ==========  ================

Formation attributes are calibrated against Nosworthy (1990) *The Anatomy
of Victory* and Nafziger (1988) *Napoleon's Army*.

Transition timing (steps) is also from Nosworthy (1990), ±20 %:

==========  ======  ========  ========  ==========
From \\ To  LINE    COLUMN    SQUARE    SKIRMISH
==========  ======  ========  ========  ==========
LINE        —       2         3         2
COLUMN      2       —         3         2
SQUARE      2       2         —         2
SKIRMISH    2       1         3         —
==========  ======  ========  ========  ==========

Key public API
~~~~~~~~~~~~~~
* :class:`Formation` — ``IntEnum`` with four states.
* :class:`FormationAttributes` — frozen dataclass of combat/movement modifiers.
* :data:`FORMATION_ATTRIBUTES` — mapping from :class:`Formation` to its attributes.
* :data:`TRANSITION_STEPS` — mapping from ``(from, to)`` pair to step count.
* :func:`get_transition_steps` — convenience accessor.
* :func:`get_attributes` — convenience accessor.
* :func:`resolve_cavalry_charge` — compute melee outcome for a cavalry charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Number of distinct formation states.
NUM_FORMATIONS: int = 4


# ---------------------------------------------------------------------------
# Formation enum
# ---------------------------------------------------------------------------


class Formation(IntEnum):
    """Discrete formation states for a Napoleonic infantry battalion.

    The integer values are stable indices used in observation vectors and
    action spaces; do **not** change them without updating downstream code.

    ==========  ===  ====================================================
    Formation   Int  Historical role
    ==========  ===  ====================================================
    LINE        0    Two- or three-rank line — maximum firepower front.
    COLUMN      1    Attack or march column — highest movement speed.
    SQUARE      2    Hollow square — impenetrable to unsupported cavalry.
    SKIRMISH    3    Extended order — loose screen, independent aimed fire.
    ==========  ===  ====================================================
    """

    LINE = 0
    COLUMN = 1
    SQUARE = 2
    SKIRMISH = 3


# ---------------------------------------------------------------------------
# Formation attributes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormationAttributes:
    """Combat and movement modifiers for a single formation.

    All multipliers are relative to the LINE baseline (1.0).

    Attributes
    ----------
    firepower_modifier:
        Multiplier on outgoing fire damage.  LINE = 1.0 (maximum effective
        firepower from a full two-rank line).
    speed_modifier:
        Multiplier on movement speed.  COLUMN = 1.5 (fastest; narrow front
        permits rapid advance); SQUARE = 0.3 (effectively stationary).
    morale_resilience:
        Multiplier that **reduces** incoming morale damage.  Values > 1.0
        mean the unit absorbs morale hits more robustly (SQUARE = 1.5 —
        tight formation and visible ranks instil confidence).
    vulnerability_modifier:
        Multiplier on incoming fire damage received.  SKIRMISH = 0.7
        (dispersed men are harder to hit); SQUARE = 1.3 (dense target for
        artillery).
    cavalry_resilience:
        Defence multiplier applied when resolving a cavalry charge
        (*melee*) against this formation.  SQUARE = 5.0 (cavalry bounces
        off bayonet hedge); LINE = 0.5 (very vulnerable in melee).
    """

    firepower_modifier: float
    speed_modifier: float
    morale_resilience: float
    vulnerability_modifier: float
    cavalry_resilience: float

    def __post_init__(self) -> None:
        for attr in (
            "firepower_modifier",
            "speed_modifier",
            "morale_resilience",
            "vulnerability_modifier",
            "cavalry_resilience",
        ):
            val = getattr(self, attr)
            if val <= 0.0:
                raise ValueError(
                    f"FormationAttributes.{attr} must be positive, got {val!r}"
                )


# ---------------------------------------------------------------------------
# Attribute table  (Nosworthy 1990 / Nafziger 1988 calibration)
# ---------------------------------------------------------------------------

#: Mapping from :class:`Formation` to its :class:`FormationAttributes`.
FORMATION_ATTRIBUTES: Dict[Formation, FormationAttributes] = {
    Formation.LINE: FormationAttributes(
        firepower_modifier=1.0,   # full two-rank volley
        speed_modifier=0.8,       # line dressing slows movement
        morale_resilience=1.0,    # baseline
        vulnerability_modifier=1.0,
        cavalry_resilience=0.5,   # line cannot quickly form square
    ),
    Formation.COLUMN: FormationAttributes(
        firepower_modifier=0.5,   # narrow front — few muskets bear
        speed_modifier=1.5,       # march column advances quickly
        morale_resilience=0.9,    # slightly less cohesive under fire
        vulnerability_modifier=1.1,  # narrow column, slightly easier to enfilade
        cavalry_resilience=0.8,
    ),
    Formation.SQUARE: FormationAttributes(
        firepower_modifier=0.75,  # all-round fire but fewer men per face
        speed_modifier=0.3,       # barely moves
        morale_resilience=1.5,    # tight formation, high morale
        vulnerability_modifier=1.3,  # dense mass — artillery bane
        cavalry_resilience=5.0,   # cavalry bounces off bayonet hedge
    ),
    Formation.SKIRMISH: FormationAttributes(
        firepower_modifier=0.6,   # independent aimed fire (fewer volleys)
        speed_modifier=1.2,       # loose order allows rapid movement
        morale_resilience=1.1,    # dispersed men less panicked by volley fire
        vulnerability_modifier=0.7,  # spread out — harder to hit
        cavalry_resilience=0.6,   # individual skirmishers are vulnerable
    ),
}


# ---------------------------------------------------------------------------
# Transition timing table  (Nosworthy 1990, ±20 %)
# ---------------------------------------------------------------------------

#: Number of simulation steps required to transition between formation pairs.
#: All transitions take 1–3 steps; forming square requires the most time.
TRANSITION_STEPS: Dict[Tuple[Formation, Formation], int] = {
    # From LINE
    (Formation.LINE, Formation.COLUMN): 2,
    (Formation.LINE, Formation.SQUARE): 3,
    (Formation.LINE, Formation.SKIRMISH): 2,
    # From COLUMN
    (Formation.COLUMN, Formation.LINE): 2,
    (Formation.COLUMN, Formation.SQUARE): 3,
    (Formation.COLUMN, Formation.SKIRMISH): 2,
    # From SQUARE
    (Formation.SQUARE, Formation.LINE): 2,
    (Formation.SQUARE, Formation.COLUMN): 2,
    (Formation.SQUARE, Formation.SKIRMISH): 2,
    # From SKIRMISH
    (Formation.SKIRMISH, Formation.LINE): 2,
    (Formation.SKIRMISH, Formation.COLUMN): 1,   # rally from skirmish is quick
    (Formation.SKIRMISH, Formation.SQUARE): 3,
}


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def get_transition_steps(
    from_formation: Formation, to_formation: Formation
) -> int:
    """Return the number of steps to transition between two formations.

    Returns ``0`` if ``from_formation == to_formation``.  Falls back to
    ``2`` for any pair not in :data:`TRANSITION_STEPS` (should not occur
    in normal use).

    Parameters
    ----------
    from_formation:
        The unit's current :class:`Formation`.
    to_formation:
        The desired target :class:`Formation`.

    Returns
    -------
    int
        Steps required (0–3).
    """
    if from_formation == to_formation:
        return 0
    return TRANSITION_STEPS.get((from_formation, to_formation), 2)


def get_attributes(formation: Formation) -> FormationAttributes:
    """Return the :class:`FormationAttributes` for *formation*.

    Parameters
    ----------
    formation:
        A :class:`Formation` value.

    Returns
    -------
    FormationAttributes
    """
    return FORMATION_ATTRIBUTES[formation]


# ---------------------------------------------------------------------------
# Cavalry charge resolution
# ---------------------------------------------------------------------------

#: Base melee damage dealt to a defender during one cavalry charge step.
BASE_CHARGE_DAMAGE: float = 0.15

#: Multiplier on charge damage when the attacker is in COLUMN formation
#: (the canonical attacking formation for Napoleonic cavalry).
COLUMN_CHARGE_MULT: float = 1.2

#: Minimum damage the attacker sustains from a charge (recoil, regardless of outcome).
BASE_ATTACKER_RECOIL_DAMAGE: float = 0.05


def resolve_cavalry_charge(
    attacker_formation: Formation,
    defender_formation: Formation,
    attacker_strength: float,
    defender_strength: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, str]:
    """Resolve a cavalry charge (melee contact) between two battalions.

    The *attacker* is the unit initiating the charge; the *defender* is the
    unit receiving it.  Typical usage: attacker in COLUMN, defender in LINE
    or SQUARE.

    Mechanics
    ~~~~~~~~~
    1. Base damage to the defender scales with ``attacker_strength`` and
       ``COLUMN_CHARGE_MULT`` if the attacker is in COLUMN formation.
    2. The defender's ``cavalry_resilience`` divides the damage received;
       SQUARE's ``cavalry_resilience=5.0`` means the charge effectively
       bounces off (trivial damage to defender, recoil damage to attacker).
    3. Attacker always takes a small recoil hit proportional to the
       ``cavalry_resilience`` of the formation they charged into.

    Parameters
    ----------
    attacker_formation:
        :class:`Formation` of the charging unit.
    defender_formation:
        :class:`Formation` of the defending unit.
    attacker_strength:
        Current strength of the attacker (``[0, 1]``).
    defender_strength:
        Current strength of the defender (``[0, 1]``).
    rng:
        Optional seeded :class:`numpy.random.Generator` for stochastic
        variance.  When ``None`` a fresh generator is used.

    Returns
    -------
    Tuple[float, float, str]
        ``(attacker_damage, defender_damage, outcome)`` where:

        * ``attacker_damage`` — fractional strength loss to the attacker.
        * ``defender_damage`` — fractional strength loss to the defender.
        * ``outcome`` — human-readable string (``"bounced"``, ``"repulsed"``,
          ``"broken"``).
    """
    if rng is None:
        rng = np.random.default_rng()

    def_attrs = get_attributes(defender_formation)

    # Charge impetus — attacker's strength scales the threat
    impetus = BASE_CHARGE_DAMAGE * max(0.0, float(attacker_strength))
    if attacker_formation == Formation.COLUMN:
        impetus *= COLUMN_CHARGE_MULT

    # Defender damage reduced by cavalry_resilience
    raw_defender_damage = impetus / def_attrs.cavalry_resilience

    # Attacker recoil — proportional to how well the defender resists
    recoil_factor = def_attrs.cavalry_resilience / (
        get_attributes(Formation.LINE).cavalry_resilience + def_attrs.cavalry_resilience
    )
    raw_attacker_damage = BASE_ATTACKER_RECOIL_DAMAGE * recoil_factor * max(
        0.0, float(defender_strength)
    )

    # Small stochastic variance (±10 %)
    noise = float(rng.uniform(0.9, 1.1))
    attacker_damage = float(np.clip(raw_attacker_damage * noise, 0.0, 1.0))
    defender_damage = float(np.clip(raw_defender_damage * noise, 0.0, 1.0))

    # Determine outcome label
    if def_attrs.cavalry_resilience >= get_attributes(Formation.SQUARE).cavalry_resilience:
        outcome = "bounced"   # charge fails completely (square)
    elif defender_damage > attacker_damage:
        outcome = "broken"    # defender takes worse than attacker
    else:
        outcome = "repulsed"  # charge slowed but not catastrophic

    return attacker_damage, defender_damage, outcome


# ---------------------------------------------------------------------------
# Formation transition state helper (called by BattalionEnv._advance_formation_transition)
# ---------------------------------------------------------------------------


def compute_transition_state(
    current_formation: Formation,
    target_formation: Optional[Formation],
    steps_remaining: int,
) -> Tuple[Formation, Optional[Formation], int]:
    """Advance the formation transition state machine by one step.

    Call once per simulation step on the unit's formation fields.

    Parameters
    ----------
    current_formation:
        The unit's active :class:`Formation`.
    target_formation:
        The :class:`Formation` the unit is transitioning *into*, or ``None``
        if not transitioning.
    steps_remaining:
        Steps left until the transition completes.

    Returns
    -------
    Tuple[Formation, Optional[Formation], int]
        ``(new_current, new_target, new_steps_remaining)`` after this step.
        When the transition completes, ``new_current`` equals the former
        *target_formation* and ``new_target`` is ``None``.
    """
    if target_formation is None or steps_remaining <= 0:
        return current_formation, None, 0

    new_steps = steps_remaining - 1
    if new_steps <= 0:
        # Transition complete
        return target_formation, None, 0
    return current_formation, target_formation, new_steps
