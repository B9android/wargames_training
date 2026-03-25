# SPDX-License-Identifier: MIT
# envs/sim/cavalry_corps.py
"""Cavalry corps simulation: reconnaissance, raiding, and pursuit missions.

Cavalry brigades operate as an independent maneuver force at the corps
echelon.  They are faster than line infantry and carry out three distinct
operational missions:

RECONNAISSANCE
    Cavalry scouts sweep ahead of the advance, revealing enemy unit
    positions within *recon_radius* metres.  This intelligence is fed back
    to the corps commander, reducing the fog of war for allied divisions
    that would otherwise receive a sentinel threat vector.

RAIDING
    Cavalry columns penetrate deep into enemy rear areas, seeking out and
    interdicting enemy supply depots.  Raiders discover depots as
    high-value targets through movement — no explicit reward shaping is
    applied for discovery; the operational effect (degraded enemy supply)
    is the incentive signal.

PURSUIT
    Cavalry exploits routing infantry.  When an enemy battalion's morale
    has collapsed and it enters a rout state, pursuit cavalry intercepts
    and inflicts additional strength damage, compounding the enemy's
    operational crisis.

Historical background
~~~~~~~~~~~~~~~~~~~~~
Napoleonic cavalry brigades averaged 1 000–2 000 sabres and could cover
40–50 km per day — roughly three times the march rate of line infantry.
The *corps de cavalerie* (cavalry corps) of 1812–1815 gave army commanders
an independent operational arm for strategic screening, deep exploitation,
and relentless pursuit after Waterloo-style victories.

Key public API
~~~~~~~~~~~~~~
* :class:`CavalryMission` — IDLE, RECONNAISSANCE, RAIDING, PURSUIT.
* :class:`CavalryUnitConfig` — frozen per-brigade parameters.
* :class:`CavalryUnit` — mutable per-brigade state.
* :class:`CavalryReport` — per-step summary returned by :meth:`CavalryCorps.step`.
* :class:`CavalryCorps` — container that manages all cavalry brigades and
  executes their missions each step.

Typical usage::

    from envs.sim.cavalry_corps import (
        CavalryCorps, CavalryMission, CavalryUnitConfig,
    )
    from envs.sim.supply_network import SupplyNetwork

    # Build a two-brigade cavalry corps for Blue
    cav = CavalryCorps.generate_default(
        map_width=10_000.0, map_height=5_000.0, n_brigades=2, team=0
    )

    # Assign missions (done by the agent each step)
    cav.units[0].mission = CavalryMission.RECONNAISSANCE
    cav.units[1].mission = CavalryMission.RAIDING

    # Execute one step
    report = cav.step(inner_env, supply_network)
    print(report.revealed_enemy_positions)   # [(x, y, strength, morale), …]
    print(report.depots_raided)              # int
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

__all__ = [
    "CavalryMission",
    "CavalryUnitConfig",
    "CavalryUnit",
    "CavalryReport",
    "CavalryCorps",
    "N_CAVALRY_MISSIONS",
    "DEFAULT_RECON_RADIUS",
    "DEFAULT_RAID_RADIUS",
    "DEFAULT_PURSUIT_RADIUS",
    "DEFAULT_CAVALRY_SPEED",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Number of distinct cavalry mission types.
N_CAVALRY_MISSIONS: int = 4

#: Default reconnaissance detection radius (metres).
DEFAULT_RECON_RADIUS: float = 2_000.0

#: Default depot-interdiction engagement radius (metres).
DEFAULT_RAID_RADIUS: float = 400.0

#: Default pursuit-engagement radius (metres).
DEFAULT_PURSUIT_RADIUS: float = 300.0

#: Default cavalry max speed (metres per step) — ~3× line infantry.
DEFAULT_CAVALRY_SPEED: float = 150.0

#: Strength damage inflicted on a routed unit per pursuit step.
DEFAULT_PURSUIT_DAMAGE: float = 0.02


# ---------------------------------------------------------------------------
# CavalryMission
# ---------------------------------------------------------------------------


class CavalryMission(IntEnum):
    """Operational mission assigned to a cavalry brigade."""

    IDLE = 0
    RECONNAISSANCE = 1
    RAIDING = 2
    PURSUIT = 3


# ---------------------------------------------------------------------------
# CavalryUnitConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CavalryUnitConfig:
    """Immutable configuration parameters for a cavalry brigade.

    Parameters
    ----------
    max_speed:
        Maximum movement speed in metres per step.  Cavalry are
        significantly faster than line infantry (default ~150 m/step vs.
        ~50 m/step).
    recon_radius:
        Detection radius for the RECONNAISSANCE mission (metres).  Enemy
        units within this distance of the scouting cavalry are revealed.
    raid_radius:
        Engagement radius for the RAIDING mission (metres).  A raiding
        unit interdicts the nearest enemy depot when it comes within this
        distance.
    pursuit_radius:
        Engagement radius for the PURSUIT mission (metres).  Pursuit
        cavalry deal :attr:`pursuit_damage_per_step` to a routed target
        once they close within this distance.
    pursuit_damage_per_step:
        Strength damage inflicted on a routed enemy per step when pursuit
        cavalry are within :attr:`pursuit_radius`.
    team:
        Owning team: ``0`` = Blue, ``1`` = Red.
    """

    max_speed: float = DEFAULT_CAVALRY_SPEED
    recon_radius: float = DEFAULT_RECON_RADIUS
    raid_radius: float = DEFAULT_RAID_RADIUS
    pursuit_radius: float = DEFAULT_PURSUIT_RADIUS
    pursuit_damage_per_step: float = DEFAULT_PURSUIT_DAMAGE
    team: int = 0

    def __post_init__(self) -> None:
        if self.max_speed <= 0.0:
            raise ValueError(
                f"max_speed must be > 0, got {self.max_speed!r}"
            )
        if self.recon_radius <= 0.0:
            raise ValueError(
                f"recon_radius must be > 0, got {self.recon_radius!r}"
            )
        if self.raid_radius <= 0.0:
            raise ValueError(
                f"raid_radius must be > 0, got {self.raid_radius!r}"
            )
        if self.pursuit_radius <= 0.0:
            raise ValueError(
                f"pursuit_radius must be > 0, got {self.pursuit_radius!r}"
            )
        if self.pursuit_damage_per_step < 0.0:
            raise ValueError(
                f"pursuit_damage_per_step must be >= 0, "
                f"got {self.pursuit_damage_per_step!r}"
            )
        if self.team not in (0, 1):
            raise ValueError(
                f"team must be 0 or 1, got {self.team!r}"
            )


# ---------------------------------------------------------------------------
# CavalryUnit
# ---------------------------------------------------------------------------


@dataclass
class CavalryUnit:
    """Mutable state for a single cavalry brigade.

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
        Frozen per-brigade configuration.  Defaults to a Blue cavalry
        config if not provided.
    alive:
        ``False`` once :meth:`take_damage` reduces strength to zero.
    mission:
        Currently assigned :class:`CavalryMission`.
    """

    x: float
    y: float
    theta: float
    strength: float
    team: int
    config: CavalryUnitConfig = field(default_factory=CavalryUnitConfig)
    alive: bool = True
    mission: CavalryMission = CavalryMission.IDLE

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from this unit's position to ``(x, y)``."""
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
        """Move one step towards ``(tx, ty)`` at :attr:`~CavalryUnitConfig.max_speed`.

        The unit travels at most :attr:`~CavalryUnitConfig.max_speed` × *dt*
        metres per call, clamped by the actual distance to the target.
        Position is clipped to the map boundary after moving.

        Parameters
        ----------
        tx, ty:
            Target world position in metres.
        dt:
            Time step scalar (default ``1.0``).
        map_width, map_height:
            Map boundary for position clamping.  Defaults to unbounded.
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
        # Clamp to map boundaries
        self.x = max(0.0, min(self.x, map_width))
        self.y = max(0.0, min(self.y, map_height))

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def take_damage(self, amount: float) -> None:
        """Reduce strength by *amount*, marking the unit dead at zero.

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
# CavalryReport
# ---------------------------------------------------------------------------


@dataclass
class CavalryReport:
    """Per-step summary of cavalry operational activity.

    Returned by :meth:`CavalryCorps.step` and exposed in the
    :class:`~envs.cavalry_corps_env.CavalryCorpsEnv` info dict.

    Attributes
    ----------
    revealed_enemy_positions:
        List of ``(x, y, strength, morale)`` tuples for enemy units
        spotted by RECONNAISSANCE cavalry this step.
    depots_raided:
        Number of enemy supply depots interdicted by RAIDING cavalry.
    routed_units_pursued:
        Number of routed enemy units engaged by PURSUIT cavalry.
    pursuit_damage:
        Total strength damage inflicted by PURSUIT cavalry.
    """

    revealed_enemy_positions: List[Tuple[float, float, float, float]]
    depots_raided: int
    routed_units_pursued: int
    pursuit_damage: float


# ---------------------------------------------------------------------------
# CavalryCorps
# ---------------------------------------------------------------------------


class CavalryCorps:
    """Manages a team's cavalry brigades and executes their assigned missions.

    Parameters
    ----------
    units:
        List of :class:`CavalryUnit` objects belonging to this corps.
    map_width, map_height:
        Map dimensions in metres.  Used for boundary clamping during
        movement.
    """

    def __init__(
        self,
        units: List[CavalryUnit],
        map_width: float,
        map_height: float,
    ) -> None:
        self.units: List[CavalryUnit] = list(units)
        self.map_width: float = float(map_width)
        self.map_height: float = float(map_height)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all cavalry units to full strength and IDLE mission.

        Unit positions are *not* altered here; callers should set
        positions explicitly or use :meth:`generate_default` and reset
        from there.
        """
        for unit in self.units:
            unit.strength = 1.0
            unit.alive = True
            unit.mission = CavalryMission.IDLE

    # ------------------------------------------------------------------
    # Intelligence
    # ------------------------------------------------------------------

    def get_revealed_enemies(self, inner) -> List[Tuple[float, float, float, float]]:
        """Return enemy positions revealed by RECONNAISSANCE units.

        Only cavalry on :attr:`~CavalryMission.RECONNAISSANCE` mission
        contribute.  Each alive enemy battalion whose position is within
        :attr:`~CavalryUnitConfig.recon_radius` of at least one
        reconnoitring cavalry unit is included exactly once.

        Parameters
        ----------
        inner:
            :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance
            holding current battalion state.

        Returns
        -------
        List of ``(x, y, strength, morale)`` for revealed enemy units.
        """
        recon_units = [
            u for u in self.units
            if u.alive and u.mission == CavalryMission.RECONNAISSANCE
        ]
        if not recon_units:
            return []

        # Determine which battalion IDs belong to the enemy team
        enemy_prefix = "red_" if recon_units[0].team == 0 else "blue_"

        revealed: List[Tuple[float, float, float, float]] = []
        seen_ids: set = set()

        for agent_id, b in inner._battalions.items():
            if agent_id in seen_ids:
                continue
            if agent_id not in inner._alive:
                continue
            if not agent_id.startswith(enemy_prefix):
                continue
            # Check if any recon unit is within detection range
            for scout in recon_units:
                if scout.distance_to(b.x, b.y) <= scout.config.recon_radius:
                    revealed.append((b.x, b.y, float(b.strength), float(b.morale)))
                    seen_ids.add(agent_id)
                    break

        return revealed

    # ------------------------------------------------------------------
    # Mission execution (internal helpers)
    # ------------------------------------------------------------------

    def _execute_reconnaissance(self, unit: CavalryUnit) -> None:
        """Advance a RECONNAISSANCE unit forward to screen enemy territory.

        Scouts push to roughly 60 % of map width at the unit's current
        Y-position, spreading the recon umbrella over the enemy's likely
        axis of advance.
        """
        target_x = self.map_width * 0.6
        target_y = unit.y  # maintain current Y — move along the east axis
        if unit.distance_to(target_x, target_y) > self.map_width * 0.01:
            unit.move_towards(
                target_x, target_y,
                map_width=self.map_width, map_height=self.map_height,
            )

    def _execute_raiding(
        self, unit: CavalryUnit, supply_network
    ) -> Tuple[int, float]:
        """Move a RAIDING unit towards the nearest enemy depot and raid it.

        The unit navigates towards the closest alive enemy depot.  When
        within :attr:`~CavalryUnitConfig.raid_radius`, the depot is
        interdicted (destroyed).  No explicit reward is given for depot
        discovery — the operational effect degrades enemy supply.

        Parameters
        ----------
        unit:
            The cavalry unit executing this mission.
        supply_network:
            :class:`~envs.sim.supply_network.SupplyNetwork` instance.

        Returns
        -------
        depots_raided : int — 1 if a depot was interdicted this step, 0 otherwise.
        damage : float — always 0.0 for raiding (damage is indirect).
        """
        enemy_team = 1 if unit.team == 0 else 0

        # Find the nearest alive enemy depot
        best_depot = None
        best_dist = float("inf")
        for depot in supply_network.depots:
            if depot.team != enemy_team or not depot.alive:
                continue
            dist = unit.distance_to(depot.x, depot.y)
            if dist < best_dist:
                best_dist = dist
                best_depot = depot

        if best_depot is None:
            return 0, 0.0

        # Move towards the depot
        unit.move_towards(
            best_depot.x, best_depot.y,
            map_width=self.map_width, map_height=self.map_height,
        )

        # Interdict if close enough (based on post-move position)
        new_dist = unit.distance_to(best_depot.x, best_depot.y)
        if new_dist <= unit.config.raid_radius and best_depot.alive:
            best_depot.interdict()
            return 1, 0.0

        return 0, 0.0

    def _execute_pursuit(
        self, unit: CavalryUnit, inner
    ) -> Tuple[int, float]:
        """Chase and damage the nearest routed enemy battalion.

        The unit moves towards the nearest enemy battalion that is in a
        routing state.  When within :attr:`~CavalryUnitConfig.pursuit_radius`,
        it deals :attr:`~CavalryUnitConfig.pursuit_damage_per_step` strength
        damage.

        Parameters
        ----------
        unit:
            The cavalry unit executing this mission.
        inner:
            :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance.

        Returns
        -------
        routed_engaged : int — 1 if a routed unit was engaged, 0 otherwise.
        damage : float — strength damage dealt this step.
        """
        enemy_prefix = "red_" if unit.team == 0 else "blue_"

        # Locate nearest routed enemy battalion
        # Routed battalions may have been removed from _alive already, so we
        # scan _battalions directly and check the routed flag.
        best_target = None
        best_dist = float("inf")

        for agent_id, b in inner._battalions.items():
            if not agent_id.startswith(enemy_prefix):
                continue
            if not b.routed:
                continue
            dist = unit.distance_to(b.x, b.y)
            if dist < best_dist:
                best_dist = dist
                best_target = b

        if best_target is None:
            return 0, 0.0

        # Move towards the routing unit
        unit.move_towards(
            best_target.x, best_target.y,
            map_width=self.map_width, map_height=self.map_height,
        )

        # Deal pursuit damage if within engagement range (after movement)
        post_move_dist = unit.distance_to(best_target.x, best_target.y)
        if post_move_dist <= unit.config.pursuit_radius and best_target.strength > 0.0:
            best_target.take_damage(unit.config.pursuit_damage_per_step)
            return 1, unit.config.pursuit_damage_per_step

        return 0, 0.0

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, inner, supply_network) -> CavalryReport:
        """Execute one step of cavalry operations.

        Iterates over all alive cavalry units, executes their assigned
        mission, and aggregates the results into a :class:`CavalryReport`.

        Parameters
        ----------
        inner:
            :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance
            (post-infantry-step state).
        supply_network:
            :class:`~envs.sim.supply_network.SupplyNetwork` instance.

        Returns
        -------
        :class:`CavalryReport` summarising activity this step.
        """
        # Collect reconnaissance intelligence first (uses current state)
        revealed = self.get_revealed_enemies(inner)

        depots_raided = 0
        routed_pursued = 0
        pursuit_dmg = 0.0

        for unit in self.units:
            if not unit.alive:
                continue

            if unit.mission == CavalryMission.RECONNAISSANCE:
                self._execute_reconnaissance(unit)

            elif unit.mission == CavalryMission.RAIDING:
                n, _ = self._execute_raiding(unit, supply_network)
                depots_raided += n

            elif unit.mission == CavalryMission.PURSUIT:
                n, d = self._execute_pursuit(unit, inner)
                routed_pursued += n
                pursuit_dmg += d

            # CavalryMission.IDLE: no movement or action

        return CavalryReport(
            revealed_enemy_positions=revealed,
            depots_raided=depots_raided,
            routed_units_pursued=routed_pursued,
            pursuit_damage=pursuit_dmg,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def generate_default(
        cls,
        map_width: float,
        map_height: float,
        n_brigades: int = 2,
        team: int = 0,
        config: Optional[CavalryUnitConfig] = None,
    ) -> "CavalryCorps":
        """Create a default cavalry corps with evenly-spaced brigades.

        Brigades are placed at ``x = map_width * 0.2`` (forward screen
        position) and spread evenly along the Y-axis.

        Parameters
        ----------
        map_width, map_height:
            Map dimensions in metres.
        n_brigades:
            Number of cavalry brigades to create (default ``2``).
        team:
            Owning team: ``0`` = Blue, ``1`` = Red.
        config:
            Optional :class:`CavalryUnitConfig`.  Defaults to a standard
            Blue (or Red) config with the specified *team*.

        Returns
        -------
        :class:`CavalryCorps` ready for use.
        """
        if n_brigades < 1:
            raise ValueError(f"n_brigades must be >= 1, got {n_brigades!r}")
        if config is None:
            config = CavalryUnitConfig(team=team)
        units: List[CavalryUnit] = []
        for i in range(n_brigades):
            y = map_height * (i + 1) / (n_brigades + 1)
            x = map_width * 0.2 if team == 0 else map_width * 0.8
            units.append(
                CavalryUnit(
                    x=x,
                    y=y,
                    theta=0.0,
                    strength=1.0,
                    team=team,
                    config=config,
                )
            )
        return cls(units=units, map_width=map_width, map_height=map_height)
