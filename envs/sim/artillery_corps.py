# envs/sim/artillery_corps.py
"""Artillery corps simulation: grand battery, counter-battery, fortification, and siege.

Artillery batteries operate as an independent arm at the corps echelon.
They execute four distinct operational missions:

GRAND_BATTERY
    Concentrated artillery fire.  When ≥ 2 batteries are within
    *grand_battery_radius* of each other, they count as a combined battery
    and apply a stacking morale-damage bonus proportional to the number of
    guns participating.  A grand battery of 6+ guns breaks enemy lines
    rapidly.

COUNTER_BATTERY
    Prioritise silencing enemy artillery over targeting infantry.  Counter-
    battery guns seek the nearest alive enemy artillery unit; if none is
    within range they fall back to targeting enemy infantry.  Counter-battery
    fire deals an additional strength-damage bonus compared with standard
    fire, making it more effective at permanently silencing guns.

SIEGE
    Reduce enemy fortification hit-points over time.  Siege guns target
    the nearest enemy :class:`Fortification` within their maximum range and
    erode its structural integrity each step.  Fortifications are destroyed
    when their HP reaches zero.

FORTIFY
    Build earthwork fortifications.  A battery in FORTIFY mission stays
    stationary and accumulates construction progress.  After
    ``fortify_steps`` consecutive FORTIFY steps the earthwork is completed,
    permanently granting a ``cover_bonus`` to friendly units at that
    position.

Historical background
~~~~~~~~~~~~~~~~~~~~~
Napoleonic artillery was organised into *réserves d'artillerie* at the
corps and army level.  Napoleon's famous *grande batterie* at Borodino
(1812) massed 102 guns to pulverise the Russian centre before the
infantry assault.  Counter-battery fire became an art form at Waterloo
(1815) where Wellington's gunners successfully silenced the French 12-
pounders.

Key public API
~~~~~~~~~~~~~~
* :class:`ArtilleryMission` — IDLE, GRAND_BATTERY, COUNTER_BATTERY,
  SIEGE, FORTIFY.
* :class:`ArtilleryUnitConfig` — frozen per-battery parameters.
* :class:`ArtilleryUnit` — mutable per-battery state.
* :class:`Fortification` — earthwork construction state.
* :class:`ArtilleryReport` — per-step summary returned by
  :meth:`ArtilleryCorps.step`.
* :class:`ArtilleryCorps` — container managing all batteries and
  executing their missions each step.

Typical usage::

    from envs.sim.artillery_corps import (
        ArtilleryCorps, ArtilleryMission, ArtilleryUnitConfig,
    )

    # Build a six-battery grand battery for Blue
    art = ArtilleryCorps.generate_default(
        map_width=10_000.0, map_height=5_000.0, n_batteries=6, team=0
    )

    # Assign missions (done by the agent each step)
    for unit in art.units:
        unit.mission = ArtilleryMission.GRAND_BATTERY

    # Execute one step
    report = art.step(inner_env)
    print(report.morale_damage_dealt)   # total morale damage this step
    print(report.guns_silenced)         # enemy artillery silenced
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

__all__ = [
    "ArtilleryMission",
    "ArtilleryUnitConfig",
    "ArtilleryUnit",
    "Fortification",
    "ArtilleryReport",
    "ArtilleryCorps",
    "N_ARTILLERY_MISSIONS",
    "DEFAULT_ARTILLERY_RANGE",
    "DEFAULT_ARTILLERY_SPEED",
    "DEFAULT_BASE_FIRE_DAMAGE",
    "DEFAULT_GRAND_BATTERY_RADIUS",
    "DEFAULT_GRAND_BATTERY_BONUS",
    "DEFAULT_COUNTER_BATTERY_BONUS",
    "DEFAULT_SIEGE_DAMAGE_PER_STEP",
    "DEFAULT_FORTIFY_STEPS",
    "DEFAULT_FORT_COVER_BONUS",
    "DEFAULT_PURSUIT_DAMAGE",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Number of distinct artillery mission types.
N_ARTILLERY_MISSIONS: int = 5

#: Default maximum fire range (metres).
DEFAULT_ARTILLERY_RANGE: float = 600.0

#: Default maximum move speed (metres per step) — artillery is slow.
DEFAULT_ARTILLERY_SPEED: float = 30.0

#: Default base morale damage applied to a target per step of direct fire.
DEFAULT_BASE_FIRE_DAMAGE: float = 0.05

#: Radius within which friendly batteries count towards a grand battery (m).
DEFAULT_GRAND_BATTERY_RADIUS: float = 800.0

#: Additional morale-damage bonus per extra gun participating in a grand battery.
DEFAULT_GRAND_BATTERY_BONUS: float = 0.01

#: Additional *strength* damage bonus when targeting enemy artillery.
DEFAULT_COUNTER_BATTERY_BONUS: float = 0.06

#: Fortification HP removed per siege step.
DEFAULT_SIEGE_DAMAGE_PER_STEP: float = 0.06

#: Number of consecutive FORTIFY steps required to complete earthworks.
DEFAULT_FORTIFY_STEPS: int = 12

#: Cover bonus (added to terrain cover) provided by a completed fortification.
DEFAULT_FORT_COVER_BONUS: float = 0.5

#: Kept for backwards-compat with import expectations (same value as counter bonus).
DEFAULT_PURSUIT_DAMAGE: float = DEFAULT_COUNTER_BATTERY_BONUS


# ---------------------------------------------------------------------------
# ArtilleryMission
# ---------------------------------------------------------------------------


class ArtilleryMission(IntEnum):
    """Operational mission assigned to an artillery battery."""

    IDLE = 0
    GRAND_BATTERY = 1
    COUNTER_BATTERY = 2
    SIEGE = 3
    FORTIFY = 4


# ---------------------------------------------------------------------------
# Fortification
# ---------------------------------------------------------------------------


@dataclass
class Fortification:
    """Earthwork fortification at a fixed map position.

    Parameters
    ----------
    x, y:
        World position in metres.
    team:
        Owning team: ``0`` = Blue, ``1`` = Red.
    hp:
        Current hit-points, normalised to ``[0, 1]``.  Starts at ``0.0``
        (under construction) and grows towards ``1.0`` as the building crew
        works.  Decreases under siege fire.
    complete:
        ``True`` once *hp* reaches ``1.0``.  Incomplete fortifications still
        provide partial cover proportional to *hp*.
    cover_bonus:
        Maximum additional cover bonus added to terrain cover when the
        fortification is complete.
    build_step_size:
        HP added per FORTIFY step (``1 / fortify_steps``).
    """

    x: float
    y: float
    team: int
    hp: float = 0.0
    complete: bool = False
    cover_bonus: float = DEFAULT_FORT_COVER_BONUS
    build_step_size: float = field(default=1.0 / DEFAULT_FORTIFY_STEPS)

    def build_step(self) -> bool:
        """Advance construction by one step.

        Returns
        -------
        bool
            ``True`` if the fortification was completed by this step.
        """
        was_complete = self.complete
        self.hp = min(1.0, self.hp + self.build_step_size)
        # Use a small tolerance to handle floating-point accumulation
        if self.hp >= 1.0 - 1e-9 and not was_complete:
            self.hp = 1.0
            self.complete = True
            return True
        return False

    def take_siege_damage(self, amount: float) -> None:
        """Reduce fortification HP by *amount*.

        Parameters
        ----------
        amount:
            Non-negative HP reduction.
        """
        if amount < 0.0:
            raise ValueError(
                f"siege damage must be >= 0, got {amount!r}"
            )
        self.hp = max(0.0, self.hp - amount)
        if self.hp < 1.0:
            self.complete = False

    @property
    def alive(self) -> bool:
        """``True`` when the fortification has any HP remaining."""
        return self.hp > 0.0

    def effective_cover(self) -> float:
        """Partial cover bonus based on current construction progress."""
        return self.cover_bonus * self.hp


# ---------------------------------------------------------------------------
# ArtilleryUnitConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtilleryUnitConfig:
    """Immutable configuration parameters for a single artillery battery.

    Parameters
    ----------
    max_speed:
        Maximum movement speed in metres per step.  Artillery is much
        slower than cavalry or even line infantry.
    max_range:
        Maximum fire range in metres.
    base_fire_damage:
        Morale damage applied to the nearest enemy battalion per step
        when within range.
    grand_battery_radius:
        Radius within which friendly batteries count towards a combined
        grand battery.
    grand_battery_bonus:
        Additional morale damage per extra gun participating in the
        grand battery (stacks linearly).
    counter_battery_bonus:
        Additional *strength* damage bonus applied when targeting an enemy
        artillery unit instead of infantry.
    siege_damage_per_step:
        HP removed from the target fortification per SIEGE step.
    fortify_steps:
        Number of consecutive FORTIFY steps required to complete a
        fortification.
    cover_bonus:
        Cover bonus provided by a completed fortification built by this
        battery.
    team:
        Owning team: ``0`` = Blue, ``1`` = Red.
    """

    max_speed: float = DEFAULT_ARTILLERY_SPEED
    max_range: float = DEFAULT_ARTILLERY_RANGE
    base_fire_damage: float = DEFAULT_BASE_FIRE_DAMAGE
    grand_battery_radius: float = DEFAULT_GRAND_BATTERY_RADIUS
    grand_battery_bonus: float = DEFAULT_GRAND_BATTERY_BONUS
    counter_battery_bonus: float = DEFAULT_COUNTER_BATTERY_BONUS
    siege_damage_per_step: float = DEFAULT_SIEGE_DAMAGE_PER_STEP
    fortify_steps: int = DEFAULT_FORTIFY_STEPS
    cover_bonus: float = DEFAULT_FORT_COVER_BONUS
    team: int = 0

    def __post_init__(self) -> None:
        if self.max_speed <= 0.0:
            raise ValueError(
                f"max_speed must be > 0, got {self.max_speed!r}"
            )
        if self.max_range <= 0.0:
            raise ValueError(
                f"max_range must be > 0, got {self.max_range!r}"
            )
        if self.base_fire_damage < 0.0:
            raise ValueError(
                f"base_fire_damage must be >= 0, got {self.base_fire_damage!r}"
            )
        if self.grand_battery_radius <= 0.0:
            raise ValueError(
                f"grand_battery_radius must be > 0, got {self.grand_battery_radius!r}"
            )
        if self.grand_battery_bonus < 0.0:
            raise ValueError(
                f"grand_battery_bonus must be >= 0, got {self.grand_battery_bonus!r}"
            )
        if self.counter_battery_bonus < 0.0:
            raise ValueError(
                f"counter_battery_bonus must be >= 0, "
                f"got {self.counter_battery_bonus!r}"
            )
        if self.siege_damage_per_step < 0.0:
            raise ValueError(
                f"siege_damage_per_step must be >= 0, "
                f"got {self.siege_damage_per_step!r}"
            )
        if int(self.fortify_steps) < 1:
            raise ValueError(
                f"fortify_steps must be >= 1, got {self.fortify_steps!r}"
            )
        if self.cover_bonus < 0.0:
            raise ValueError(
                f"cover_bonus must be >= 0, got {self.cover_bonus!r}"
            )
        if self.team not in (0, 1):
            raise ValueError(
                f"team must be 0 or 1, got {self.team!r}"
            )


# ---------------------------------------------------------------------------
# ArtilleryUnit
# ---------------------------------------------------------------------------


@dataclass
class ArtilleryUnit:
    """Mutable state for a single artillery battery.

    Parameters
    ----------
    x, y:
        World position in metres.
    theta:
        Facing angle in radians.
    strength:
        Combat strength in ``[0, 1]``.
    team:
        Owning team (``0`` = Blue, ``1`` = Red).
    config:
        Frozen per-battery configuration.
    alive:
        ``False`` once :meth:`take_damage` reduces strength to zero.
    mission:
        Currently assigned :class:`ArtilleryMission`.
    _fortify_progress:
        Internal counter of consecutive FORTIFY steps accumulated.
    """

    x: float
    y: float
    theta: float
    strength: float
    team: int
    config: ArtilleryUnitConfig = field(default_factory=ArtilleryUnitConfig)
    alive: bool = True
    mission: ArtilleryMission = ArtilleryMission.IDLE
    _fortify_progress: int = field(default=0, repr=False)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from this battery's position to ``(x, y)``."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_towards(
        self,
        tx: float,
        ty: float,
        dt: float = 1.0,
        map_width: float = float("inf"),
        map_height: float = float("inf"),
    ) -> None:
        """Move one step towards ``(tx, ty)`` at :attr:`~ArtilleryUnitConfig.max_speed`.

        Parameters
        ----------
        tx, ty:
            Target world position in metres.
        dt:
            Time-step scalar (default ``1.0``).
        map_width, map_height:
            Map boundary for position clamping.
        """
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return
        speed = min(self.config.max_speed * dt, dist)
        self.x += (dx / dist) * speed
        self.y += (dy / dist) * speed
        self.theta = math.atan2(dy, dx)
        self.x = max(0.0, min(self.x, map_width))
        self.y = max(0.0, min(self.y, map_height))

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def take_damage(self, amount: float) -> None:
        """Reduce strength by *amount*, marking the battery dead at zero.

        Parameters
        ----------
        amount:
            Non-negative strength loss.
        """
        if amount < 0.0:
            raise ValueError(f"damage amount must be >= 0, got {amount!r}")
        self.strength = max(0.0, self.strength - amount)
        if self.strength <= 0.0:
            self.alive = False


# ---------------------------------------------------------------------------
# ArtilleryReport
# ---------------------------------------------------------------------------


@dataclass
class ArtilleryReport:
    """Per-step summary of artillery operational activity.

    Returned by :meth:`ArtilleryCorps.step` and exposed in the
    :class:`~envs.artillery_corps_env.ArtilleryCorpsEnv` info dict.

    Attributes
    ----------
    morale_damage_dealt:
        Total morale damage applied to enemy battalions this step.
    guns_silenced:
        Number of enemy artillery units whose strength reached zero
        during this step (counter-battery kills).
    fortification_damage:
        Total HP removed from enemy fortifications during siege fire.
    fortifications_completed:
        Number of friendly fortifications completed (hp reached 1.0)
        this step.
    """

    morale_damage_dealt: float
    guns_silenced: int
    fortification_damage: float
    fortifications_completed: int


# ---------------------------------------------------------------------------
# ArtilleryCorps
# ---------------------------------------------------------------------------


class ArtilleryCorps:
    """Manages a team's artillery batteries and executes their missions.

    Parameters
    ----------
    units:
        List of :class:`ArtilleryUnit` objects belonging to this corps.
    fortifications:
        List of :class:`Fortification` objects managed by this corps.
        Batteries in FORTIFY mission add new entries here.
    map_width, map_height:
        Map dimensions in metres.  Used for boundary clamping.
    """

    def __init__(
        self,
        units: List[ArtilleryUnit],
        map_width: float,
        map_height: float,
        fortifications: Optional[List[Fortification]] = None,
    ) -> None:
        self.units: List[ArtilleryUnit] = list(units)
        self.map_width: float = float(map_width)
        self.map_height: float = float(map_height)
        self.fortifications: List[Fortification] = (
            list(fortifications) if fortifications is not None else []
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all batteries to full strength, IDLE mission, and clear forts.

        Unit positions are *not* altered here; callers must set positions
        explicitly (e.g. in :meth:`ArtilleryCorpsEnv.reset`).
        """
        for unit in self.units:
            unit.strength = 1.0
            unit.alive = True
            unit.mission = ArtilleryMission.IDLE
            unit._fortify_progress = 0
        self.fortifications.clear()

    # ------------------------------------------------------------------
    # Grand battery helpers
    # ------------------------------------------------------------------

    def count_grand_battery_guns(self, unit: ArtilleryUnit) -> int:
        """Return the number of alive friendly batteries within grand-battery radius.

        Counts *all* alive friendly units within
        :attr:`~ArtilleryUnitConfig.grand_battery_radius` of *unit*,
        including *unit* itself.

        Parameters
        ----------
        unit:
            The battery initiating the grand battery check.

        Returns
        -------
        int
            Number of guns (≥ 1, always includes *unit* itself).
        """
        count = 0
        for other in self.units:
            if not other.alive:
                continue
            if other.team != unit.team:
                continue
            if other.distance_to(unit.x, unit.y) <= unit.config.grand_battery_radius:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Mission execution (internal helpers)
    # ------------------------------------------------------------------

    def _execute_grand_battery(
        self,
        unit: ArtilleryUnit,
        n_guns_in_battery: int,
        inner,
    ) -> float:
        """Fire on the nearest enemy battalion with concentrated grand-battery effect.

        Damage is amplified by the number of participating guns:
        ``morale_damage = base_fire_damage + (n_guns - 1) * grand_battery_bonus``.

        The nearest enemy battalion within :attr:`~ArtilleryUnitConfig.max_range`
        is targeted; if no enemy is in range the unit advances towards the
        nearest enemy.

        Parameters
        ----------
        unit:
            The battery executing this mission.
        n_guns_in_battery:
            Total guns participating in the combined battery (≥ 1).
        inner:
            :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance.

        Returns
        -------
        float
            Morale damage applied to the target this step (``0.0`` if
            no target was in range).
        """
        enemy_prefix = "red_" if unit.team == 0 else "blue_"

        # Locate nearest alive enemy battalion
        best_target = None
        best_dist = float("inf")
        for agent_id, b in inner._battalions.items():
            if not agent_id.startswith(enemy_prefix):
                continue
            if agent_id not in inner._alive:
                continue
            dist = unit.distance_to(b.x, b.y)
            if dist < best_dist:
                best_dist = dist
                best_target = b

        if best_target is None:
            return 0.0

        if best_dist > unit.config.max_range:
            # Advance to bring the target into range
            unit.move_towards(
                best_target.x, best_target.y,
                map_width=self.map_width, map_height=self.map_height,
            )
            return 0.0

        # Compute stacking morale damage
        n_extra = max(0, n_guns_in_battery - 1)
        morale_dmg = (
            unit.config.base_fire_damage
            + n_extra * unit.config.grand_battery_bonus
        )
        best_target.morale = max(0.0, best_target.morale - morale_dmg)
        if (
            best_target.morale <= best_target.routing_threshold
            and not best_target.routed
        ):
            best_target.routed = True
        return morale_dmg

    def _execute_counter_battery(
        self,
        unit: ArtilleryUnit,
        enemy_artillery: List[ArtilleryUnit],
        inner,
    ) -> Tuple[int, float]:
        """Target enemy artillery preferentially; fall back to enemy infantry.

        When a live enemy :class:`ArtilleryUnit` is within
        :attr:`~ArtilleryUnitConfig.max_range`, the battery fires at it
        with ``base_fire_damage + counter_battery_bonus`` strength damage.
        When no enemy artillery is in range, it fires morale damage at
        the nearest enemy infantry battalion.

        Parameters
        ----------
        unit:
            The battery executing counter-battery fire.
        enemy_artillery:
            List of alive enemy :class:`ArtilleryUnit` objects (may be empty).
        inner:
            :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance.

        Returns
        -------
        guns_silenced : int
            ``1`` if an enemy artillery unit was killed, ``0`` otherwise.
        morale_damage : float
            Morale damage applied to infantry fallback (``0.0`` when
            targeting enemy artillery).
        """
        # ── Try to find enemy artillery in range ─────────────────────
        best_art = None
        best_art_dist = float("inf")
        for ea in enemy_artillery:
            if not ea.alive:
                continue
            dist = unit.distance_to(ea.x, ea.y)
            if dist < best_art_dist:
                best_art_dist = dist
                best_art = ea

        if best_art is not None:
            if best_art_dist <= unit.config.max_range:
                # Fire counter-battery
                dmg = unit.config.base_fire_damage + unit.config.counter_battery_bonus
                silenced_before = not best_art.alive
                best_art.take_damage(dmg)
                newly_silenced = (not best_art.alive) and (not silenced_before)
                return int(newly_silenced), 0.0
            else:
                # Advance towards nearest enemy artillery
                unit.move_towards(
                    best_art.x, best_art.y,
                    map_width=self.map_width, map_height=self.map_height,
                )
                return 0, 0.0

        # ── Fallback: fire on nearest enemy infantry ──────────────────
        enemy_prefix = "red_" if unit.team == 0 else "blue_"
        best_inf = None
        best_inf_dist = float("inf")
        for agent_id, b in inner._battalions.items():
            if not agent_id.startswith(enemy_prefix):
                continue
            if agent_id not in inner._alive:
                continue
            dist = unit.distance_to(b.x, b.y)
            if dist < best_inf_dist:
                best_inf_dist = dist
                best_inf = b

        if best_inf is None:
            return 0, 0.0

        if best_inf_dist > unit.config.max_range:
            unit.move_towards(
                best_inf.x, best_inf.y,
                map_width=self.map_width, map_height=self.map_height,
            )
            return 0, 0.0

        best_inf.morale = max(0.0, best_inf.morale - unit.config.base_fire_damage)
        if (
            best_inf.morale <= best_inf.routing_threshold
            and not best_inf.routed
        ):
            best_inf.routed = True
        return 0, unit.config.base_fire_damage

    def _execute_siege(
        self,
        unit: ArtilleryUnit,
        enemy_fortifications: List[Fortification],
    ) -> float:
        """Reduce the nearest enemy fortification's HP.

        The unit moves towards the nearest alive enemy fortification and
        fires on it once within :attr:`~ArtilleryUnitConfig.max_range`.

        Parameters
        ----------
        unit:
            The battery executing siege fire.
        enemy_fortifications:
            List of :class:`Fortification` objects belonging to the enemy.

        Returns
        -------
        float
            HP damage applied to the target fortification this step.
        """
        # Find nearest alive enemy fortification
        best_fort = None
        best_dist = float("inf")
        for fort in enemy_fortifications:
            if not fort.alive:
                continue
            dist = unit.distance_to(fort.x, fort.y)
            if dist < best_dist:
                best_dist = dist
                best_fort = fort

        if best_fort is None:
            return 0.0

        if best_dist > unit.config.max_range:
            unit.move_towards(
                best_fort.x, best_fort.y,
                map_width=self.map_width, map_height=self.map_height,
            )
            return 0.0

        best_fort.take_siege_damage(unit.config.siege_damage_per_step)
        return unit.config.siege_damage_per_step

    def _execute_fortify(self, unit: ArtilleryUnit) -> int:
        """Advance earthwork construction at the battery's current position.

        Accumulates consecutive FORTIFY steps.  On completion a new
        :class:`Fortification` is appended to :attr:`fortifications` and
        the counter is reset.

        Parameters
        ----------
        unit:
            The battery digging in.

        Returns
        -------
        int
            ``1`` if a fortification was completed this step, ``0`` otherwise.
        """
        unit._fortify_progress += 1
        if unit._fortify_progress >= unit.config.fortify_steps:
            unit._fortify_progress = 0
            build_size = 1.0 / unit.config.fortify_steps
            fort = Fortification(
                x=unit.x,
                y=unit.y,
                team=unit.team,
                hp=1.0,
                complete=True,
                cover_bonus=unit.config.cover_bonus,
                build_step_size=build_size,
            )
            self.fortifications.append(fort)
            return 1
        return 0

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        inner,
        enemy_artillery: Optional[List[ArtilleryUnit]] = None,
        enemy_fortifications: Optional[List[Fortification]] = None,
    ) -> ArtilleryReport:
        """Execute one step of artillery operations.

        Iterates over all alive batteries, executes their assigned mission,
        and aggregates the results into an :class:`ArtilleryReport`.

        Parameters
        ----------
        inner:
            :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance
            (post-infantry-step state).
        enemy_artillery:
            Optional list of enemy :class:`ArtilleryUnit` objects for
            counter-battery targeting.  Defaults to an empty list.
        enemy_fortifications:
            Optional list of enemy :class:`Fortification` objects for
            siege targeting.  Defaults to an empty list.

        Returns
        -------
        :class:`ArtilleryReport` summarising activity this step.
        """
        if enemy_artillery is None:
            enemy_artillery = []
        if enemy_fortifications is None:
            enemy_fortifications = []

        total_morale_dmg: float = 0.0
        total_guns_silenced: int = 0
        total_fort_dmg: float = 0.0
        total_forts_completed: int = 0

        # Pre-compute grand battery sizes (before any state changes)
        grand_battery_counts = {
            i: self.count_grand_battery_guns(u)
            for i, u in enumerate(self.units)
            if u.alive and u.mission == ArtilleryMission.GRAND_BATTERY
        }

        for i, unit in enumerate(self.units):
            if not unit.alive:
                continue

            if unit.mission == ArtilleryMission.GRAND_BATTERY:
                n_guns = grand_battery_counts.get(i, 1)
                dmg = self._execute_grand_battery(unit, n_guns, inner)
                total_morale_dmg += dmg

            elif unit.mission == ArtilleryMission.COUNTER_BATTERY:
                silenced, morale_dmg = self._execute_counter_battery(
                    unit, enemy_artillery, inner
                )
                total_guns_silenced += silenced
                total_morale_dmg += morale_dmg

            elif unit.mission == ArtilleryMission.SIEGE:
                fort_dmg = self._execute_siege(unit, enemy_fortifications)
                total_fort_dmg += fort_dmg

            elif unit.mission == ArtilleryMission.FORTIFY:
                completed = self._execute_fortify(unit)
                total_forts_completed += completed

            # ArtilleryMission.IDLE: no action

        return ArtilleryReport(
            morale_damage_dealt=total_morale_dmg,
            guns_silenced=total_guns_silenced,
            fortification_damage=total_fort_dmg,
            fortifications_completed=total_forts_completed,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def generate_default(
        cls,
        map_width: float,
        map_height: float,
        n_batteries: int = 4,
        team: int = 0,
        config: Optional[ArtilleryUnitConfig] = None,
    ) -> "ArtilleryCorps":
        """Create a default artillery corps with evenly-spaced batteries.

        Batteries are placed at ``x = map_width * 0.25`` for Blue
        (``0.75`` for Red) and spread evenly along the Y-axis, representing
        a battery line in rear of the main infantry body.

        Parameters
        ----------
        map_width, map_height:
            Map dimensions in metres.
        n_batteries:
            Number of artillery batteries (default ``4``).
        team:
            Owning team: ``0`` = Blue, ``1`` = Red.
        config:
            Optional :class:`ArtilleryUnitConfig`.  Defaults to a standard
            config with the specified *team*.

        Returns
        -------
        :class:`ArtilleryCorps` ready for use.
        """
        if n_batteries < 1:
            raise ValueError(
                f"n_batteries must be >= 1, got {n_batteries!r}"
            )
        if config is None:
            config = ArtilleryUnitConfig(team=team)
        units: List[ArtilleryUnit] = []
        for i in range(n_batteries):
            y = map_height * (i + 1) / (n_batteries + 1)
            # Blue deploys at 25% map width, Red at 75%
            x = map_width * 0.25 if team == 0 else map_width * 0.75
            units.append(
                ArtilleryUnit(
                    x=x,
                    y=y,
                    theta=0.0,
                    strength=1.0,
                    team=team,
                    config=config,
                )
            )
        return cls(units=units, map_width=map_width, map_height=map_height)
