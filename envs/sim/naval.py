# SPDX-License-Identifier: MIT
# envs/sim/naval.py
"""Naval unit types and coastal operations for Napoleonic-era simulation.

Adds three ship categories operating on sea, river, and coastal map tiles.
Naval vessels provide ranged gunfire support against coastal positions and
serve as transport platforms for amphibious landings.

Historical background
~~~~~~~~~~~~~~~~~~~~~
Napoleonic naval power shaped every coastal and riverine campaign.
Ships-of-the-line (74–120 guns) dominated open-sea battles; frigates acted
as fast scouts and commerce raiders; shallow-draft gunboats extended naval
fire support into estuaries and rivers.  Amphibious operations—from the
Danish expedition of 1807 to the Walcheren campaign of 1809—combined
naval gunfire suppression with infantry landings on hostile beaches.

Key public API
~~~~~~~~~~~~~~
* :class:`ShipType` — SHIP_OF_THE_LINE, FRIGATE, GUNBOAT.
* :class:`WaterTileType` — SEA, RIVER, COASTAL_SHALLOW, COASTAL_DEEP.
* :class:`NavalVesselConfig` — frozen attributes for each ship class.
* :data:`SHIP_CONFIGS` — table mapping :class:`ShipType` to its config.
* :class:`NavalVessel` — mutable per-vessel combat state (position, strength …).
* :class:`CoastalMap` — grid tracking which cells are water/land/coastal
  and providing tile-type queries.
* :func:`naval_gunfire_damage` — compute bombardment damage with LOS check.
* :func:`can_bombard` — range, arc, and water-tile check for gunfire support.
* :class:`AmphibiousLanding` — state machine for ship → beach → infantry.
* :class:`RiverCrossing` — ford/bridge crossing mechanics.
* :func:`generate_coastal_map` — factory for a default coastal scenario map.

Typical usage::

    from envs.sim.naval import (
        ShipType, NavalVessel, CoastalMap,
        can_bombard, naval_gunfire_damage,
        AmphibiousLanding, LandingPhase,
        generate_coastal_map,
    )

    cmap = generate_coastal_map(width=5_000.0, height=3_000.0)
    ship = NavalVessel(x=500.0, y=1_500.0, theta=0.0, ship_type=ShipType.FRIGATE, team=0)

    if can_bombard(ship, target_x=1_200.0, target_y=1_500.0, coastal_map=cmap):
        dmg = naval_gunfire_damage(ship, target_x=1_200.0, target_y=1_500.0,
                                   target_strength=1.0, coastal_map=cmap)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    # Enumerations
    "ShipType",
    "WaterTileType",
    "LandingPhase",
    # Config / attributes
    "NavalVesselConfig",
    "SHIP_CONFIGS",
    # Core classes
    "NavalVessel",
    "CoastalMap",
    "AmphibiousLanding",
    "RiverCrossing",
    # Functions
    "can_bombard",
    "naval_gunfire_damage",
    "generate_coastal_map",
    # Constants
    "BASE_NAVAL_DAMAGE",
    "LANDING_STEPS_DEFAULT",
    "FORD_CROSSING_STEPS",
    "BRIDGE_CROSSING_STEPS",
]

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: Base fractional strength loss from a full-intensity naval broadside at
#: point-blank range.  Higher than infantry fire (0.05) to reflect the
#: destructive power of a cannon broadside.
BASE_NAVAL_DAMAGE: float = 0.12

#: Default number of simulation steps required to complete an amphibious
#: landing (ship → beach phase).
LANDING_STEPS_DEFAULT: int = 8

#: Steps to cross a river at an unimproved ford.
FORD_CROSSING_STEPS: int = 6

#: Steps to cross a river over a bridge (faster).
BRIDGE_CROSSING_STEPS: int = 3

#: Half-angle (radians) of a ship's broadside fire arc (±90° from beam).
BROADSIDE_FIRE_ARC: float = math.pi / 2.0

#: Minimum movement speed multiplier while crossing a river ford.
FORD_SPEED_MODIFIER: float = 0.3

#: Speed multiplier for a ship-of-the-line (slowest).
SOL_SPEED_MODIFIER: float = 0.6

#: Speed multiplier for a frigate.
FRIGATE_SPEED_MODIFIER: float = 1.0

#: Speed multiplier for a gunboat (river-capable, agile in narrows).
GUNBOAT_SPEED_MODIFIER: float = 0.8


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ShipType(IntEnum):
    """Naval vessel categories.

    ==================  ===  =================================================
    Type                Int  Role
    ==================  ===  =================================================
    SHIP_OF_THE_LINE    0    74–120 guns; maximum bombardment power; sea only.
    FRIGATE             1    30–50 guns; fast scout and fire-support platform.
    GUNBOAT             2    2–20 guns; river and coastal assault; shallow draft.
    ==================  ===  =================================================
    """

    SHIP_OF_THE_LINE = 0
    FRIGATE = 1
    GUNBOAT = 2


class WaterTileType(IntEnum):
    """Tile type for water and coastal cells in a :class:`CoastalMap`.

    ================  ===  ====================================================
    Type              Int  Description
    ================  ===  ====================================================
    LAND              0    Normal land tile — no naval access.
    COASTAL_SHALLOW   1    Shallow coastal water — gunboats/frigates only.
    COASTAL_DEEP      2    Deep coastal water — all ship types.
    SEA               3    Open sea — all ship types at full speed.
    RIVER             4    River tile — gunboats only.
    BEACH             5    Beach/landing zone — transition tile for landings.
    FORD              6    Shallow river ford — infantry can cross (slowly).
    BRIDGE            7    River bridge — infantry crosses at reduced penalty.
    ================  ===  ====================================================
    """

    LAND = 0
    COASTAL_SHALLOW = 1
    COASTAL_DEEP = 2
    SEA = 3
    RIVER = 4
    BEACH = 5
    FORD = 6
    BRIDGE = 7


class LandingPhase(IntEnum):
    """State of an amphibious landing operation.

    =============  ===  ========================================================
    Phase          Int  Description
    =============  ===  ========================================================
    EMBARKED       0    Infantry embarked — ship underway.
    APPROACHING    1    Ship anchored offshore — landing boats deployed.
    LANDING        2    Infantry in surf — partially ashore, vulnerable.
    ESTABLISHED    3    Beach-head established — infantry fully ashore.
    COMPLETE       4    Landing complete; infantry operating as normal.
    =============  ===  ========================================================
    """

    EMBARKED = 0
    APPROACHING = 1
    LANDING = 2
    ESTABLISHED = 3
    COMPLETE = 4


# ---------------------------------------------------------------------------
# Ship configuration table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NavalVesselConfig:
    """Immutable performance attributes for a ship class.

    Parameters
    ----------
    ship_type:
        The :class:`ShipType` this config describes.
    fire_range:
        Maximum bombardment range in metres.
    broadside_damage:
        Damage multiplier relative to :data:`BASE_NAVAL_DAMAGE`.  The
        ship-of-the-line has the largest broadside; the gunboat the smallest.
    max_speed:
        Maximum movement speed in metres per second (before any
        terrain modifier).
    can_enter_river:
        ``True`` if the vessel can navigate :attr:`WaterTileType.RIVER` tiles.
    min_tile:
        Minimum :class:`WaterTileType` integer value that this ship can
        enter (e.g. GUNBOAT can enter COASTAL_SHALLOW = 1).
    """

    ship_type: ShipType
    fire_range: float
    broadside_damage: float
    max_speed: float
    can_enter_river: bool
    min_tile: int  # minimum WaterTileType value the ship can navigate


#: Lookup table: ship type → its configuration.
SHIP_CONFIGS: dict[ShipType, NavalVesselConfig] = {
    ShipType.SHIP_OF_THE_LINE: NavalVesselConfig(
        ship_type=ShipType.SHIP_OF_THE_LINE,
        fire_range=600.0,
        broadside_damage=3.0,   # 3× base — 74+ guns
        max_speed=30.0,
        can_enter_river=False,
        min_tile=int(WaterTileType.COASTAL_DEEP),
    ),
    ShipType.FRIGATE: NavalVesselConfig(
        ship_type=ShipType.FRIGATE,
        fire_range=450.0,
        broadside_damage=1.5,   # 1.5× base — ~36 guns
        max_speed=50.0,
        can_enter_river=False,
        min_tile=int(WaterTileType.COASTAL_SHALLOW),
    ),
    ShipType.GUNBOAT: NavalVesselConfig(
        ship_type=ShipType.GUNBOAT,
        fire_range=250.0,
        broadside_damage=0.5,   # 0.5× base — few light cannon
        max_speed=40.0,
        can_enter_river=True,
        min_tile=int(WaterTileType.COASTAL_SHALLOW),
    ),
}


# ---------------------------------------------------------------------------
# NavalVessel
# ---------------------------------------------------------------------------


@dataclass
class NavalVessel:
    """A mutable naval combat unit operating on water tiles.

    Parameters
    ----------
    x, y:
        Current world position in metres.
    theta:
        Facing angle in radians.  ``0`` = pointing east (starboard = south).
        Broadside arcs are perpendicular to the heading (±90°).
    ship_type:
        The :class:`ShipType` category of this vessel.
    team:
        Owning side — ``0`` = Blue, ``1`` = Red.
    strength:
        Current combat effectiveness in ``[0, 1]``.  Reduced by incoming
        fire; ``0`` = sunk/destroyed.
    """

    x: float
    y: float
    theta: float
    ship_type: ShipType
    team: int
    strength: float = 1.0

    # Derived from ship_type — populated in __post_init__
    fire_range: float = field(default=0.0, init=False)
    broadside_damage: float = field(default=0.0, init=False)
    max_speed: float = field(default=0.0, init=False)
    can_enter_river: bool = field(default=False, init=False)
    min_tile: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        cfg = SHIP_CONFIGS[self.ship_type]
        self.fire_range = cfg.fire_range
        self.broadside_damage = cfg.broadside_damage
        self.max_speed = cfg.max_speed
        self.can_enter_river = cfg.can_enter_river
        self.min_tile = cfg.min_tile
        if self.team not in (0, 1):
            raise ValueError(f"team must be 0 or 1, got {self.team!r}")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(
                f"strength must be in [0, 1], got {self.strength!r}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_afloat(self) -> bool:
        """``True`` while the vessel has positive strength."""
        return self.strength > 0.0

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(self, vx: float, vy: float, dt: float = 0.1) -> None:
        """Apply velocity, clamped to :attr:`max_speed`.

        Parameters
        ----------
        vx, vy:
            Velocity components in metres per second.
        dt:
            Simulation timestep in seconds.
        """
        speed = math.sqrt(vx ** 2 + vy ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
        self.x += vx * dt
        self.y += vy * dt

    def rotate(self, delta_theta: float, max_turn_rate: float = 0.05) -> None:
        """Rotate the ship, clamped to *max_turn_rate* radians per step.

        Ships turn more slowly than land units.

        Parameters
        ----------
        delta_theta:
            Desired rotation in radians (positive = counter-clockwise).
        max_turn_rate:
            Maximum rotation per step (default 0.05 rad ≈ 3°).
        """
        delta_theta = float(np.clip(delta_theta, -max_turn_rate, max_turn_rate))
        self.theta = (self.theta + delta_theta) % (2.0 * math.pi)

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def broadside_arc_contains(self, target_x: float, target_y: float) -> bool:
        """Return ``True`` if ``(target_x, target_y)`` is in the broadside arc.

        A ship fires its broadside perpendicular to its heading (±90° off
        the beam on either side — i.e. the full ±90° from both port and
        starboard).  Targets directly ahead or astern are outside the arc.
        The arc check is equivalent to: the angle from ship to target
        must differ from *heading* by between 45° and 135° (either side).
        """
        dx = target_x - self.x
        dy = target_y - self.y
        if dx == 0.0 and dy == 0.0:
            return False
        angle_to_target = math.atan2(dy, dx)
        # Signed angular difference wrapped to (−π, π]
        diff = (angle_to_target - self.theta + math.pi) % (2.0 * math.pi) - math.pi
        abs_diff = abs(diff)
        # Broadside: between 45° (π/4) and 135° (3π/4) off the bow (either side)
        return math.pi / 4.0 <= abs_diff <= 3.0 * math.pi / 4.0

    def take_damage(self, damage: float) -> None:
        """Apply *damage* to this vessel's strength.

        Parameters
        ----------
        damage:
            Fractional strength loss (clamped to remaining strength).
        """
        damage = float(np.clip(damage, 0.0, self.strength))
        self.strength = max(0.0, self.strength - damage)

    def can_tile(self, tile_type: WaterTileType) -> bool:
        """Return ``True`` if this vessel can navigate *tile_type*.

        Parameters
        ----------
        tile_type:
            The :class:`WaterTileType` to test.
        """
        tile_val = int(tile_type)
        # LAND (0) and BEACH (5) are never navigable by ships
        if tile_val in (int(WaterTileType.LAND), int(WaterTileType.BEACH)):
            return False
        # FORD and BRIDGE are not water tiles ships can enter
        if tile_val in (int(WaterTileType.FORD), int(WaterTileType.BRIDGE)):
            return False
        # River tiles require can_enter_river
        if tile_val == int(WaterTileType.RIVER):
            return self.can_enter_river
        # Otherwise check minimum navigable tile depth
        return tile_val >= self.min_tile


# ---------------------------------------------------------------------------
# CoastalMap
# ---------------------------------------------------------------------------


@dataclass
class CoastalMap:
    """2-D grid that tracks terrain tile types for a coastal/riverine scenario.

    The map spans ``[0, width] × [0, height]`` in world coordinates.  Each
    grid cell holds a :class:`WaterTileType` value.  The default grid has
    all cells set to :attr:`WaterTileType.LAND`; use
    :meth:`set_tile` or :func:`generate_coastal_map` to populate it.

    Parameters
    ----------
    width, height:
        World-space dimensions in metres.
    rows, cols:
        Grid resolution.
    """

    width: float
    height: float
    rows: int
    cols: int
    _grid: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError(
                f"rows and cols must be >= 1, got rows={self.rows}, cols={self.cols}"
            )
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(
                f"width and height must be > 0, got {self.width}, {self.height}"
            )
        # Default: all land
        self._grid = np.zeros((self.rows, self.cols), dtype=np.int32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to (row, col) grid indices, clamped."""
        col = int(np.clip(x / self.width * self.cols, 0, self.cols - 1))
        row = int(np.clip(y / self.height * self.rows, 0, self.rows - 1))
        return row, col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tile(self, x: float, y: float) -> WaterTileType:
        """Return the :class:`WaterTileType` at world position ``(x, y)``."""
        row, col = self._to_grid(x, y)
        return WaterTileType(int(self._grid[row, col]))

    def set_tile(self, row: int, col: int, tile: WaterTileType) -> None:
        """Set the tile at grid cell ``(row, col)`` to *tile*.

        Parameters
        ----------
        row, col:
            Grid indices (0-based).
        tile:
            :class:`WaterTileType` value to assign.

        Raises
        ------
        IndexError
            If *row* or *col* are out of bounds.
        """
        if not (0 <= row < self.rows):
            raise IndexError(f"row {row} out of range [0, {self.rows})")
        if not (0 <= col < self.cols):
            raise IndexError(f"col {col} out of range [0, {self.cols})")
        self._grid[row, col] = int(tile)

    def is_water(self, x: float, y: float) -> bool:
        """Return ``True`` if the tile at ``(x, y)`` is any water type."""
        tile = self.get_tile(x, y)
        return tile not in (WaterTileType.LAND,)

    def is_navigable_by(self, x: float, y: float, vessel: NavalVessel) -> bool:
        """Return ``True`` if *vessel* can navigate the tile at ``(x, y)``."""
        tile = self.get_tile(x, y)
        return vessel.can_tile(tile)

    def is_beach(self, x: float, y: float) -> bool:
        """Return ``True`` if the tile at ``(x, y)`` is a beach landing zone."""
        return self.get_tile(x, y) == WaterTileType.BEACH

    def is_river_crossable(self, x: float, y: float) -> bool:
        """Return ``True`` if infantry can cross the river at ``(x, y)``.

        Crossings require a :attr:`WaterTileType.FORD` or
        :attr:`WaterTileType.BRIDGE` tile.
        """
        tile = self.get_tile(x, y)
        return tile in (WaterTileType.FORD, WaterTileType.BRIDGE)

    def has_line_of_sight(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        num_samples: int = 20,
    ) -> bool:
        """Return ``True`` if there is no land tile blocking the line from
        ``(x0, y0)`` to ``(x1, y1)``.

        Naval gunfire travels over water; a land tile anywhere along the
        trajectory blocks the shot (cliffs, headlands, etc.).

        Parameters
        ----------
        x0, y0:
            Firing position (must be a water tile for a valid naval shot).
        x1, y1:
            Target position.
        num_samples:
            Number of equally-spaced sample points along the line.
        """
        if num_samples < 2:
            raise ValueError(f"num_samples must be >= 2, got {num_samples}")
        for i in range(1, num_samples - 1):
            t = i / (num_samples - 1)
            xp = x0 + t * (x1 - x0)
            yp = y0 + t * (y1 - y0)
            tile = self.get_tile(xp, yp)
            if tile == WaterTileType.LAND:
                return False
        return True

    def grid_array(self) -> np.ndarray:
        """Return a copy of the underlying grid as a 2-D ``int32`` array."""
        return self._grid.copy()


# ---------------------------------------------------------------------------
# Naval gunfire support functions
# ---------------------------------------------------------------------------


def _dist(x0: float, y0: float, x1: float, y1: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def can_bombard(
    vessel: NavalVessel,
    target_x: float,
    target_y: float,
    coastal_map: CoastalMap,
    require_water_tile: bool = True,
) -> bool:
    """Return ``True`` if *vessel* can bombard position ``(target_x, target_y)``.

    Conditions checked:
    1. Vessel is afloat (``strength > 0``).
    2. Target is within the vessel's :attr:`~NavalVessel.fire_range`.
    3. Target falls within the broadside arc (±90° from beam).
    4. The vessel's current tile is a water tile (when *require_water_tile*).
    5. The line from vessel to target has no intervening land tiles
       (coastal map LOS check).

    Parameters
    ----------
    vessel:
        The firing naval vessel.
    target_x, target_y:
        World position of the bombardment target.
    coastal_map:
        :class:`CoastalMap` used for tile and LOS queries.
    require_water_tile:
        When ``True`` (default) the vessel must be on a water tile to fire.
        Set to ``False`` in tests that don't set up full coastal maps.

    Returns
    -------
    bool
    """
    if not vessel.is_afloat:
        return False

    dist = _dist(vessel.x, vessel.y, target_x, target_y)
    if dist > vessel.fire_range:
        return False

    if not vessel.broadside_arc_contains(target_x, target_y):
        return False

    if require_water_tile:
        vessel_tile = coastal_map.get_tile(vessel.x, vessel.y)
        if not vessel.can_tile(vessel_tile):
            return False

    if not coastal_map.has_line_of_sight(vessel.x, vessel.y, target_x, target_y):
        return False

    return True


def naval_gunfire_damage(
    vessel: NavalVessel,
    target_x: float,
    target_y: float,
    target_strength: float,
    coastal_map: CoastalMap,
    intensity: float = 1.0,
    require_water_tile: bool = True,
) -> float:
    """Compute raw damage from a naval broadside bombardment.

    Returns ``0.0`` if :func:`can_bombard` returns ``False``.

    Damage formula
    ~~~~~~~~~~~~~~
    ``damage = BASE_NAVAL_DAMAGE × broadside_multiplier × range_factor × intensity × vessel.strength``

    where ``range_factor = 1 − dist / fire_range`` (linear falloff identical
    to the infantry model).

    Parameters
    ----------
    vessel:
        The firing naval vessel.
    target_x, target_y:
        World position of the bombardment target.
    target_strength:
        Current strength of the target (unused in the damage formula but
        kept for API symmetry with land-combat functions).
    coastal_map:
        :class:`CoastalMap` for tile and LOS checks.
    intensity:
        Fire intensity in ``[0, 1]`` (agent action).
    require_water_tile:
        Forwarded to :func:`can_bombard`.

    Returns
    -------
    float
        Fractional strength damage ``≥ 0.0``.
    """
    if not can_bombard(vessel, target_x, target_y, coastal_map, require_water_tile):
        return 0.0

    intensity = float(np.clip(intensity, 0.0, 1.0))
    dist = _dist(vessel.x, vessel.y, target_x, target_y)
    range_factor = float(np.clip(1.0 - dist / vessel.fire_range, 0.0, 1.0))

    damage = (
        BASE_NAVAL_DAMAGE
        * vessel.broadside_damage
        * range_factor
        * intensity
        * max(0.0, vessel.strength)
    )
    return float(damage)


# ---------------------------------------------------------------------------
# Amphibious landing state machine
# ---------------------------------------------------------------------------


@dataclass
class AmphibiousLanding:
    """State machine for an amphibious assault: ship → beach → infantry.

    Represents the combined operation of one :class:`NavalVessel` carrying
    an infantry contingent toward a beach landing zone.

    Phases (see :class:`LandingPhase`):

    * **EMBARKED** — troops are aboard; ship is moving toward the beach.
    * **APPROACHING** — ship has anchored offshore; landing boats launched.
    * **LANDING** — infantry is in the surf / at the waterline; vulnerable.
    * **ESTABLISHED** — beach-head secured; infantry fully ashore but not yet free.
    * **COMPLETE** — infantry is operating normally; landing over.

    Call :meth:`step` each simulation step to advance the operation.

    Parameters
    ----------
    vessel:
        The naval transport vessel.
    infantry_strength:
        Starting strength of the embarked infantry in ``[0, 1]``.
    beach_x, beach_y:
        Target landing position (must be a :attr:`WaterTileType.BEACH` tile
        on the associated coastal map).
    approach_radius:
        Distance (metres) from the beach at which the ship anchors and
        landing boats are deployed.
    landing_steps:
        Total number of steps the LANDING phase lasts.  During this phase
        the infantry is particularly vulnerable.
    """

    vessel: NavalVessel
    infantry_strength: float
    beach_x: float
    beach_y: float
    approach_radius: float = 200.0
    landing_steps: int = LANDING_STEPS_DEFAULT

    phase: LandingPhase = field(default=LandingPhase.EMBARKED, init=False)
    _landing_step_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.infantry_strength <= 1.0):
            raise ValueError(
                f"infantry_strength must be in [0, 1], "
                f"got {self.infantry_strength!r}"
            )
        if self.approach_radius <= 0.0:
            raise ValueError(
                f"approach_radius must be > 0, got {self.approach_radius!r}"
            )
        if self.landing_steps < 1:
            raise ValueError(
                f"landing_steps must be >= 1, got {self.landing_steps!r}"
            )

    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """``True`` when the landing operation is fully complete."""
        return self.phase == LandingPhase.COMPLETE

    @property
    def infantry_ashore(self) -> bool:
        """``True`` once infantry has reached the ESTABLISHED phase."""
        return self.phase in (LandingPhase.ESTABLISHED, LandingPhase.COMPLETE)

    @property
    def is_vulnerable(self) -> bool:
        """``True`` during the LANDING phase — infantry takes extra damage."""
        return self.phase == LandingPhase.LANDING

    def vulnerability_modifier(self) -> float:
        """Damage multiplier applied to the landing infantry.

        Returns ``2.0`` during :attr:`LandingPhase.LANDING` (infantry in
        the surf is extremely exposed), ``1.0`` otherwise.
        """
        return 2.0 if self.is_vulnerable else 1.0

    def step(self) -> LandingPhase:
        """Advance the landing state machine by one simulation step.

        Transitions
        ~~~~~~~~~~~
        * EMBARKED → APPROACHING: ship comes within *approach_radius* of beach.
        * APPROACHING → LANDING: immediately on the next step after anchoring.
        * LANDING → ESTABLISHED: after *landing_steps* steps.
        * ESTABLISHED → COMPLETE: on the next step after establishment.

        Returns
        -------
        :class:`LandingPhase`
            The current phase after this step.
        """
        if self.phase == LandingPhase.COMPLETE:
            return self.phase

        if self.phase == LandingPhase.EMBARKED:
            dist = _dist(self.vessel.x, self.vessel.y, self.beach_x, self.beach_y)
            if dist <= self.approach_radius:
                self.phase = LandingPhase.APPROACHING

        elif self.phase == LandingPhase.APPROACHING:
            self.phase = LandingPhase.LANDING
            self._landing_step_counter = 0

        elif self.phase == LandingPhase.LANDING:
            self._landing_step_counter += 1
            if self._landing_step_counter >= self.landing_steps:
                self.phase = LandingPhase.ESTABLISHED

        elif self.phase == LandingPhase.ESTABLISHED:
            self.phase = LandingPhase.COMPLETE

        return self.phase

    def apply_landing_casualties(self, incoming_damage: float) -> float:
        """Apply casualties to the embarked infantry, modulated by vulnerability.

        Parameters
        ----------
        incoming_damage:
            Raw damage before vulnerability modifier.

        Returns
        -------
        float
            Actual damage applied (≤ remaining infantry strength).
        """
        actual = min(
            incoming_damage * self.vulnerability_modifier(),
            self.infantry_strength,
        )
        actual = max(0.0, actual)
        self.infantry_strength = max(0.0, self.infantry_strength - actual)
        return actual


# ---------------------------------------------------------------------------
# River crossing state machine
# ---------------------------------------------------------------------------


@dataclass
class RiverCrossing:
    """State machine for infantry crossing a river at a ford or bridge.

    Infantry crossing a river is slowed and is more vulnerable to flanking
    fire.  The crossing type (ford vs bridge) determines both the speed
    penalty and the number of steps required.

    Parameters
    ----------
    unit_x, unit_y:
        Starting position of the crossing unit in world coordinates.
    crossing_x, crossing_y:
        Position of the ford or bridge.
    coastal_map:
        Used to verify the crossing tile type.
    team:
        Owning side — ``0`` = Blue, ``1`` = Red.

    Raises
    ------
    ValueError
        If the tile at ``(crossing_x, crossing_y)`` is not a ford or bridge.
    """

    unit_x: float
    unit_y: float
    crossing_x: float
    crossing_y: float
    coastal_map: CoastalMap
    team: int

    _crossing_steps_total: int = field(default=0, init=False)
    _crossing_steps_done: int = field(default=0, init=False)
    _complete: bool = field(default=False, init=False)
    crossing_type: WaterTileType = field(default=WaterTileType.FORD, init=False)

    def __post_init__(self) -> None:
        if self.team not in (0, 1):
            raise ValueError(f"team must be 0 or 1, got {self.team!r}")
        tile = self.coastal_map.get_tile(self.crossing_x, self.crossing_y)
        if tile not in (WaterTileType.FORD, WaterTileType.BRIDGE):
            raise ValueError(
                f"Tile at ({self.crossing_x}, {self.crossing_y}) is {tile!r}; "
                f"must be FORD or BRIDGE for a river crossing."
            )
        self.crossing_type = tile
        if tile == WaterTileType.FORD:
            self._crossing_steps_total = FORD_CROSSING_STEPS
        else:
            self._crossing_steps_total = BRIDGE_CROSSING_STEPS

    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """``True`` when the crossing is finished."""
        return self._complete

    @property
    def progress(self) -> float:
        """Crossing progress in ``[0, 1]``."""
        if self._crossing_steps_total == 0:
            return 1.0
        return min(1.0, self._crossing_steps_done / self._crossing_steps_total)

    @property
    def speed_modifier(self) -> float:
        """Movement speed modifier while crossing.

        Ford crossings are slower (``FORD_SPEED_MODIFIER = 0.3``); bridge
        crossings impose a lesser penalty (``0.7``).
        """
        if self._complete:
            return 1.0
        if self.crossing_type == WaterTileType.FORD:
            return FORD_SPEED_MODIFIER
        return 0.7  # bridge — slower than normal but faster than ford

    @property
    def vulnerability_modifier(self) -> float:
        """Damage multiplier for units currently crossing.

        Units crossing a ford are highly exposed (flanking fire in the
        water); bridge crossings are slightly less dangerous.
        """
        if self._complete:
            return 1.0
        if self.crossing_type == WaterTileType.FORD:
            return 1.8
        return 1.3  # bridge — confined but less exposed than ford

    def step(self) -> bool:
        """Advance the crossing by one simulation step.

        Returns
        -------
        bool
            ``True`` if the crossing completed on this step.
        """
        if self._complete:
            return False
        self._crossing_steps_done += 1
        if self._crossing_steps_done >= self._crossing_steps_total:
            self._complete = True
            return True
        return False


# ---------------------------------------------------------------------------
# Scenario map factory
# ---------------------------------------------------------------------------


def generate_coastal_map(
    width: float = 5_000.0,
    height: float = 3_000.0,
    rows: int = 30,
    cols: int = 50,
    sea_cols: int = 10,
    river_row_start: int = 13,
    river_row_end: int = 17,
    ford_col: int = 35,
    bridge_col: int = 40,
    beach_col: int = 10,
) -> CoastalMap:
    """Generate a default coastal / riverine scenario map.

    Layout (left → right):
    * Columns 0 to *sea_cols*-1: open sea (:attr:`WaterTileType.SEA`).
    * Column *beach_col*: beach landing strip (:attr:`WaterTileType.BEACH`),
      flanked by shallow coastal water.  Pass ``beach_col=-1`` to omit the
      beach strip entirely (useful for pure river or naval scenarios).
    * Columns *sea_cols* to *beach_col*-1: coastal shallow water.
    * Columns *beach_col*+1 to *ford_col*-1: land.
    * Rows *river_row_start*–*river_row_end*, columns *beach_col*+1 to end:
      river (:attr:`WaterTileType.RIVER`).
    * Column *ford_col* at river rows: ford crossing.
    * Column *bridge_col* at river rows: bridge crossing.

    Parameters
    ----------
    width, height:
        World dimensions in metres.
    rows, cols:
        Grid resolution.
    sea_cols:
        Number of columns (from x=0) that are open sea.
    river_row_start, river_row_end:
        Row range (inclusive) for the river band.
    ford_col:
        Column index for the river ford crossing point.
    bridge_col:
        Column index for the river bridge crossing point.
    beach_col:
        Column index for the beach landing strip.  Pass ``-1`` (or any
        negative integer) to disable beach generation entirely.

    Returns
    -------
    :class:`CoastalMap`
    """
    cmap = CoastalMap(width=width, height=height, rows=rows, cols=cols)
    # Negative beach_col is a sentinel meaning "no beach strip"
    has_beach = beach_col >= 0

    for r in range(rows):
        for c in range(cols):
            # Open sea band
            if c < sea_cols:
                tile = WaterTileType.SEA
            # Coastal shallow — between sea and beach (only when beach is enabled)
            elif has_beach and c < beach_col:
                tile = WaterTileType.COASTAL_SHALLOW
            # Beach strip (only when enabled and not already in sea band)
            elif has_beach and c == beach_col:
                tile = WaterTileType.BEACH
            # Landward side — but check if within river band
            elif river_row_start <= r <= river_row_end:
                if c == ford_col:
                    tile = WaterTileType.FORD
                elif c == bridge_col:
                    tile = WaterTileType.BRIDGE
                else:
                    tile = WaterTileType.RIVER
            else:
                tile = WaterTileType.LAND
            cmap.set_tile(r, c, tile)

    return cmap
