# SPDX-License-Identifier: MIT
"""Historically-grounded weapon profiles for Napoleonic-era combat.

Provides weapon profiles (musket, rifle, cannon, howitzer) with:

* Range-band accuracy distributions calibrated to Nafziger (1988) data.
* Per-unit reload state machines (LOADED → FIRING → RELOADING → LOADED).
* Volley synchronisation helpers for coordinated fire windows.
* Artillery suppression effects (morale penalty without casualties).

Historical calibration (Nafziger, G. F., *Napoleon's Army*, 1988):

    Musket hit probability per aimed shot — trained line infantry:

    ======  ============================
    Range   Hit probability (Nafziger)
    ======  ============================
    50 m    ~55 %
    150 m   ~25 %
    300 m   ~8 %
    ======  ============================

    The exponential decay model ``P(r) = A · exp(−k · r)`` is fitted to the
    50 m and 150 m reference points; P(300) then falls within ±5 pp of the
    reference value.

Usage example::

    from envs.sim.weapons import MUSKET, hit_probability, ReloadMachine, RangeBand

    prob = hit_probability(MUSKET, distance=100.0)
    machine = ReloadMachine(MUSKET)
    fired = machine.fire()   # True — unit was loaded
    machine.step()           # advance simulation step
    print(machine.status)    # ReloadStatus.FIRING or RELOADING depending on fire_steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Nafziger (1988) reference values
# ---------------------------------------------------------------------------

#: Musket hit probability at 50 m (Nafziger, 1988).
NAFZIGER_MUSKET_50M: float = 0.55
#: Musket hit probability at 150 m (Nafziger, 1988).
NAFZIGER_MUSKET_150M: float = 0.25
#: Musket hit probability at 300 m (Nafziger, 1988).
NAFZIGER_MUSKET_300M: float = 0.08

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WeaponType(Enum):
    """Broad weapon category."""

    MUSKET = auto()
    RIFLE = auto()
    CANNON = auto()
    HOWITZER = auto()


class RangeBand(Enum):
    """Named engagement range bands.

    ============  =====================================================
    Band          Typical effect
    ============  =====================================================
    CLOSE         Maximum hit probability; muskets most effective.
    EFFECTIVE     Standard engagement band for volleys.
    EXTREME       Reduced accuracy; only marksmen / artillery fire.
    OUT_OF_RANGE  Target beyond weapon's maximum range; no effect.
    ============  =====================================================
    """

    CLOSE = "close"
    EFFECTIVE = "effective"
    EXTREME = "extreme"
    OUT_OF_RANGE = "out_of_range"


class ReloadStatus(Enum):
    """Reload state machine states."""

    LOADED = "loaded"        # Ready to fire.
    FIRING = "firing"        # Executing fire action (lasts ``fire_steps`` steps).
    RELOADING = "reloading"  # Reloading in progress (lasts ``reload_steps`` steps).


# ---------------------------------------------------------------------------
# Weapon profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeaponProfile:
    """Immutable description of a weapon's combat characteristics.

    Parameters
    ----------
    weapon_type:
        The broad category of the weapon.
    max_range:
        Maximum range in metres beyond which the weapon has no effect.
    close_range:
        Upper boundary of the CLOSE range band (metres).
    effective_range:
        Upper boundary of the EFFECTIVE range band (metres).
        Must satisfy ``close_range < effective_range < max_range``.
    base_accuracy:
        Coefficient *A* in the exponential accuracy model
        ``P(r) = A · exp(−decay_rate · r)``.  Represents the theoretical
        hit probability at zero range (before clamping to [0, 1]).
    decay_rate:
        Decay constant *k* (units: 1/metres).  Larger values cause
        accuracy to fall off faster with range.
    formation_accuracy_bonus:
        Multiplicative modifier applied to hit probability when the
        firing unit is in formed-line formation.  1.0 means no bonus.
    reload_steps:
        Number of simulation steps required to reload after firing.
        Muskets: 2–4; artillery: 6–10.
    fire_steps:
        Number of simulation steps occupied by the firing action itself.
        Typically 1 for muskets; may be higher for artillery.
    suppression_radius:
        Radius (metres) within which near-misses cause a morale penalty
        without inflicting casualties.  0.0 disables suppression (small
        arms).
    suppression_morale_penalty:
        Morale deducted per step from units within *suppression_radius*.
        Only meaningful when ``suppression_radius > 0``.
    """

    weapon_type: WeaponType
    max_range: float
    close_range: float
    effective_range: float
    base_accuracy: float
    decay_rate: float
    formation_accuracy_bonus: float = 1.0
    reload_steps: int = 3
    fire_steps: int = 1
    suppression_radius: float = 0.0
    suppression_morale_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {self.max_range}")
        if self.close_range <= 0.0:
            raise ValueError(f"close_range must be positive, got {self.close_range}")
        if self.effective_range <= self.close_range:
            raise ValueError(
                f"effective_range ({self.effective_range}) must exceed "
                f"close_range ({self.close_range})"
            )
        if self.effective_range >= self.max_range:
            raise ValueError(
                f"effective_range ({self.effective_range}) must be less than "
                f"max_range ({self.max_range})"
            )
        if self.base_accuracy <= 0.0:
            raise ValueError(f"base_accuracy must be positive, got {self.base_accuracy}")
        if self.decay_rate <= 0.0:
            raise ValueError(f"decay_rate must be positive, got {self.decay_rate}")
        if self.reload_steps < 1:
            raise ValueError(f"reload_steps must be >= 1, got {self.reload_steps}")
        if self.fire_steps < 1:
            raise ValueError(f"fire_steps must be >= 1, got {self.fire_steps}")
        if self.suppression_radius < 0.0:
            raise ValueError(
                f"suppression_radius must be >= 0, got {self.suppression_radius}"
            )


# ---------------------------------------------------------------------------
# Pre-defined weapon profiles
# ---------------------------------------------------------------------------

#: Standard Napoleonic smoothbore musket (e.g. Brown Bess, Charleville 1777).
#:
#: Accuracy parameters fitted to Nafziger (1988):
#:   A = 0.8158, k = 0.007885 → P(50) ≈ 0.55, P(150) ≈ 0.25, P(300) ≈ 0.077.
MUSKET: WeaponProfile = WeaponProfile(
    weapon_type=WeaponType.MUSKET,
    max_range=300.0,
    close_range=75.0,
    effective_range=150.0,
    base_accuracy=0.8158,
    decay_rate=0.007885,
    formation_accuracy_bonus=1.0,
    reload_steps=3,
    fire_steps=1,
    suppression_radius=0.0,
    suppression_morale_penalty=0.0,
)

#: Rifle (Baker rifle, Jäger, voltigeur): longer range, better accuracy, slower reload.
RIFLE: WeaponProfile = WeaponProfile(
    weapon_type=WeaponType.RIFLE,
    max_range=400.0,
    close_range=100.0,
    effective_range=200.0,
    base_accuracy=0.90,
    decay_rate=0.005,
    formation_accuracy_bonus=1.0,
    reload_steps=4,
    fire_steps=1,
    suppression_radius=0.0,
    suppression_morale_penalty=0.0,
)

#: 6- or 12-pounder field cannon (direct fire, solid shot / canister).
CANNON: WeaponProfile = WeaponProfile(
    weapon_type=WeaponType.CANNON,
    max_range=700.0,
    close_range=200.0,
    effective_range=450.0,
    base_accuracy=0.70,
    decay_rate=0.002,
    formation_accuracy_bonus=1.0,
    reload_steps=7,
    fire_steps=1,
    suppression_radius=50.0,
    suppression_morale_penalty=0.15,
)

#: Howitzer (indirect plunging fire, high-trajectory shell).
HOWITZER: WeaponProfile = WeaponProfile(
    weapon_type=WeaponType.HOWITZER,
    max_range=900.0,
    close_range=300.0,
    effective_range=600.0,
    base_accuracy=0.55,
    decay_rate=0.0015,
    formation_accuracy_bonus=1.0,
    reload_steps=9,
    fire_steps=1,
    suppression_radius=75.0,
    suppression_morale_penalty=0.20,
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def hit_probability(
    weapon: WeaponProfile,
    distance: float,
    formation_modifier: float = 1.0,
) -> float:
    """Return the probability of a hit at *distance* metres.

    Uses an exponential decay model:
    ``P(r) = base_accuracy · exp(−decay_rate · r) · formation_modifier``

    Parameters
    ----------
    weapon:
        The :class:`WeaponProfile` of the firing unit.
    distance:
        Range to the target in metres (non-negative).
    formation_modifier:
        External formation modifier (e.g. from cohesion state).  This is
        *multiplied* by ``weapon.formation_accuracy_bonus`` to get the
        effective modifier.  Defaults to 1.0.

    Returns
    -------
    float
        Hit probability in ``[0, 1]``.  Returns ``0.0`` if *distance*
        exceeds ``weapon.max_range``.
    """
    distance = float(distance)
    if distance < 0.0:
        distance = 0.0
    if distance > weapon.max_range:
        return 0.0
    raw = weapon.base_accuracy * np.exp(-weapon.decay_rate * distance)
    effective_modifier = formation_modifier * weapon.formation_accuracy_bonus
    return float(np.clip(raw * effective_modifier, 0.0, 1.0))


def get_range_band(weapon: WeaponProfile, distance: float) -> RangeBand:
    """Classify *distance* into a :class:`RangeBand` for *weapon*.

    Parameters
    ----------
    weapon:
        The firing weapon.
    distance:
        Range in metres (non-negative).

    Returns
    -------
    RangeBand
        ``OUT_OF_RANGE`` if distance > ``weapon.max_range``;
        otherwise ``CLOSE``, ``EFFECTIVE``, or ``EXTREME``.
    """
    distance = float(distance)
    if distance < 0.0:
        distance = 0.0
    if distance > weapon.max_range:
        return RangeBand.OUT_OF_RANGE
    if distance <= weapon.close_range:
        return RangeBand.CLOSE
    if distance <= weapon.effective_range:
        return RangeBand.EFFECTIVE
    return RangeBand.EXTREME


def suppression_morale_penalty(
    weapon: WeaponProfile,
    distance: float,
) -> float:
    """Compute the artillery suppression morale penalty at *distance*.

    Suppression represents near-misses and explosions that depress morale
    without inflicting casualties.  Only cannon and howitzer weapons
    produce suppression.

    The penalty falls off linearly from ``weapon.suppression_morale_penalty``
    at distance 0 to 0 at ``weapon.suppression_radius``.

    Parameters
    ----------
    weapon:
        The firing weapon profile.
    distance:
        Distance from the impact point in metres.

    Returns
    -------
    float
        Morale penalty in ``[0, weapon.suppression_morale_penalty]``.
        Returns ``0.0`` for small-arms weapons or targets outside the
        suppression radius.
    """
    if weapon.weapon_type not in (WeaponType.CANNON, WeaponType.HOWITZER):
        return 0.0
    if weapon.suppression_radius <= 0.0:
        return 0.0
    distance = max(0.0, float(distance))
    if distance >= weapon.suppression_radius:
        return 0.0
    factor = 1.0 - (distance / weapon.suppression_radius)
    return float(weapon.suppression_morale_penalty * factor)


# ---------------------------------------------------------------------------
# Reload state machine
# ---------------------------------------------------------------------------


@dataclass
class ReloadMachine:
    """Per-unit reload state machine.

    Manages the firing cycle for a single weapon:
    ``LOADED → FIRING → RELOADING → LOADED``

    Parameters
    ----------
    weapon:
        The :class:`WeaponProfile` whose ``fire_steps`` and
        ``reload_steps`` govern cycle timing.

    Examples
    --------
    >>> from envs.sim.weapons import MUSKET, ReloadMachine, ReloadStatus
    >>> machine = ReloadMachine(MUSKET)
    >>> machine.is_loaded
    True
    >>> machine.fire()
    True
    >>> machine.status
    <ReloadStatus.FIRING: 'firing'>
    >>> machine.fire()  # blocked during firing
    False
    """

    weapon: WeaponProfile
    _status: ReloadStatus = field(default=ReloadStatus.LOADED, repr=False, init=False)
    _steps_remaining: int = field(default=0, repr=False, init=False)

    @property
    def status(self) -> ReloadStatus:
        """Current reload state."""
        return self._status

    @property
    def is_loaded(self) -> bool:
        """``True`` when the weapon is ready to fire."""
        return self._status == ReloadStatus.LOADED

    @property
    def steps_remaining(self) -> int:
        """Steps remaining in the current state (0 when LOADED)."""
        return self._steps_remaining

    def fire(self) -> bool:
        """Attempt to fire.

        If the weapon is loaded, transitions to FIRING and returns
        ``True``.  If it is already firing or reloading returns ``False``
        without changing state — this is the mechanism that blocks firing
        during the reload cycle.

        Returns
        -------
        bool
            ``True`` if the fire action was accepted; ``False`` if the
            weapon is not ready.
        """
        if self._status != ReloadStatus.LOADED:
            return False
        self._status = ReloadStatus.FIRING
        self._steps_remaining = self.weapon.fire_steps
        return True

    def step(self) -> ReloadStatus:
        """Advance one simulation step.

        Decrements the step counter and advances the state machine when
        the counter reaches zero:

        * ``FIRING`` → ``RELOADING`` (counter reset to ``reload_steps``)
        * ``RELOADING`` → ``LOADED`` (counter reset to 0)

        Returns
        -------
        ReloadStatus
            The state *after* this step has been processed.
        """
        if self._status == ReloadStatus.LOADED:
            return self._status

        self._steps_remaining = max(0, self._steps_remaining - 1)

        if self._steps_remaining == 0:
            if self._status == ReloadStatus.FIRING:
                self._status = ReloadStatus.RELOADING
                self._steps_remaining = self.weapon.reload_steps
            elif self._status == ReloadStatus.RELOADING:
                self._status = ReloadStatus.LOADED

        return self._status

    def reset(self) -> None:
        """Reset to the initial LOADED state."""
        self._status = ReloadStatus.LOADED
        self._steps_remaining = 0


# ---------------------------------------------------------------------------
# Volley synchronisation
# ---------------------------------------------------------------------------


def synchronized_volley(machines: Sequence[ReloadMachine]) -> list[bool]:
    """Fire a coordinated volley across multiple units.

    All units that are in the LOADED state fire simultaneously.  Units
    that are still reloading do not participate.

    Parameters
    ----------
    machines:
        Sequence of :class:`ReloadMachine` instances representing the
        units in the firing line.

    Returns
    -------
    list[bool]
        Per-unit fire result — ``True`` if the unit fired, ``False`` if
        it was not loaded and therefore could not participate.
    """
    return [m.fire() for m in machines]


def volley_readiness(machines: Sequence[ReloadMachine]) -> float:
    """Return the fraction of units in *machines* that are loaded and ready.

    A value of 1.0 means all units are ready for a full volley; lower
    values indicate the line is partially reloaded.

    Parameters
    ----------
    machines:
        The reload machines for units in the firing line.

    Returns
    -------
    float
        Readiness fraction in ``[0, 1]``.  Returns ``0.0`` for an empty
        sequence.
    """
    if not machines:
        return 0.0
    return float(sum(1 for m in machines if m.is_loaded) / len(machines))
