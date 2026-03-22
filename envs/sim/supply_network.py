# envs/sim/supply_network.py
"""Strategic supply network: depot nodes, convoy routes, and supply radius.

Models a Napoleonic-era strategic logistics system for the corps-level
simulation.  Each team maintains supply *depots* (stockpiles of provisions
and ammunition) and optional *convoy routes* that replenish them from a rear
base.  Units within the *effective supply radius* of a friendly depot can draw
provisions; units outside all depot radii are considered *out of supply*.

Cutting supply lines (capturing or destroying an enemy depot) degrades that
depot's effective supply radius immediately and prevents its convoys from
reaching it.

Historical background
~~~~~~~~~~~~~~~~~~~~~
Napoleonic armies relied entirely on their logistics trains for ammunition,
food, and forage.  Corps operations rarely lasted more than three days'
march from the nearest depot without incurring acute supply shortages that
degraded both combat power and manoeuvre tempo.

Key public API
~~~~~~~~~~~~~~
* :class:`SupplyDepot` — a stockpile node with position, team, stock, and
  a supply radius that shrinks as stock is consumed.
* :class:`ConvoyRoute` — a shuttle route between two depots that slowly
  replenishes the destination from the source.
* :class:`SupplyNetwork` — container managing all depots and convoy routes
  for a single battlefield; exposes supply-level queries and interdiction.

Typical usage::

    from envs.sim.supply_network import SupplyNetwork

    net = SupplyNetwork.generate_default(map_width=10_000.0, map_height=5_000.0)

    # Query supply level for a unit at (2_000, 2_500) on team 0 (Blue)
    level = net.get_supply_level(2_000.0, 2_500.0, team=0)   # → [0, 1]

    # Per-step consumption from nearby depots
    net.consume_supply([(2_000.0, 2_500.0)], team=0)

    # Check if a Blue unit can interdict (capture) a Red depot
    interdicted = net.interdict_nearest_depot(8_500.0, 2_500.0, enemy_team=1,
                                               capture_radius=300.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "SupplyDepot",
    "ConvoyRoute",
    "SupplyNetwork",
    "DEFAULT_SUPPLY_RADIUS",
    "DEFAULT_CONSUMPTION_PER_STEP",
    "DEFAULT_CONVOY_TRANSFER_RATE",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default base supply radius (metres).  Covers roughly one division frontage
#: on the default 10 km × 5 km corps map.
DEFAULT_SUPPLY_RADIUS: float = 1_500.0

#: Fractional stock consumed per unit per step when the unit is in supply.
#: At this rate a depot feeding a single unit is exhausted in ~10 000 steps.
DEFAULT_CONSUMPTION_PER_STEP: float = 1e-4

#: Fractional stock transferred from the source depot to the destination
#: depot per convoy step.  Matches the single-unit consumption rate so that
#: one active convoy can sustain one unit indefinitely.
DEFAULT_CONVOY_TRANSFER_RATE: float = 1e-4


# ---------------------------------------------------------------------------
# SupplyDepot
# ---------------------------------------------------------------------------


@dataclass
class SupplyDepot:
    """A supply stockpile node on the corps map.

    Parameters
    ----------
    x, y:
        World position in metres.
    team:
        Owning team: ``0`` = Blue, ``1`` = Red.
    initial_stock:
        Starting stock level in ``[0, 1]``.  ``1.0`` = full provisions.
    base_supply_radius:
        Maximum supply radius (metres) when stock is full.  The
        *effective* radius scales linearly with current stock so that a
        fully depleted depot covers no area.
    """

    x: float
    y: float
    team: int
    initial_stock: float = 1.0
    base_supply_radius: float = DEFAULT_SUPPLY_RADIUS

    # Mutable episode state — not set by the caller
    stock: float = field(default=1.0, init=False)
    alive: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.initial_stock <= 1.0):
            raise ValueError(
                f"initial_stock must be in (0, 1], got {self.initial_stock!r}"
            )
        if self.base_supply_radius <= 0.0:
            raise ValueError(
                f"base_supply_radius must be > 0, got {self.base_supply_radius!r}"
            )
        if self.team not in (0, 1):
            raise ValueError(f"team must be 0 or 1, got {self.team!r}")
        self.stock = self.initial_stock

    # ------------------------------------------------------------------

    @property
    def effective_supply_radius(self) -> float:
        """Supply radius shrinks proportionally with remaining stock.

        Returns ``0.0`` when the depot is destroyed or fully depleted.
        """
        if not self.alive:
            return 0.0
        return self.base_supply_radius * self.stock

    def supply_level_at(self, x: float, y: float) -> float:
        """Return the supply level ``[0, 1]`` contributed by this depot.

        Uses a linear falloff from ``1.0`` at the depot position to
        ``0.0`` at :attr:`effective_supply_radius`.  Returns ``0.0``
        beyond the effective radius.

        Parameters
        ----------
        x, y:
            Query position in world coordinates (metres).
        """
        r = self.effective_supply_radius
        if r <= 0.0:
            return 0.0
        dist = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        return max(0.0, 1.0 - dist / r)

    def consume(self, amount: float) -> float:
        """Reduce stock by *amount*, clamped to the available stock.

        Parameters
        ----------
        amount:
            Fractional stock to consume (must be ≥ 0).

        Returns
        -------
        float — actual amount consumed (≤ *amount*).
        """
        if amount < 0.0:
            raise ValueError(f"consume amount must be >= 0, got {amount!r}")
        if not self.alive:
            return 0.0
        actual = min(amount, self.stock)
        self.stock -= actual
        self.stock = max(0.0, self.stock)
        return actual

    def replenish(self, amount: float) -> None:
        """Increase stock by *amount*, clamped to :attr:`initial_stock`.

        Parameters
        ----------
        amount:
            Fractional stock to add (must be ≥ 0).
        """
        if amount < 0.0:
            raise ValueError(f"replenish amount must be >= 0, got {amount!r}")
        if not self.alive:
            return
        self.stock = min(self.initial_stock, self.stock + amount)

    def interdict(self) -> None:
        """Destroy this depot (immediate effect).

        Sets :attr:`alive` to ``False`` and :attr:`stock` to ``0``,
        collapsing the effective supply radius to zero.
        """
        self.alive = False
        self.stock = 0.0

    def reset(self) -> None:
        """Restore the depot to its initial state for a new episode."""
        self.stock = self.initial_stock
        self.alive = True


# ---------------------------------------------------------------------------
# ConvoyRoute
# ---------------------------------------------------------------------------


@dataclass
class ConvoyRoute:
    """A resupply convoy route between two depots.

    The convoy transfers stock from the *source* depot to the *destination*
    depot at a fixed rate each step, simulating wagon trains shuttling
    between a rear base and a forward depot.  Transfer only occurs when both
    depots are alive.

    Parameters
    ----------
    source_idx:
        Index of the source depot in :attr:`SupplyNetwork.depots`.
    dest_idx:
        Index of the destination depot in :attr:`SupplyNetwork.depots`.
    transfer_rate:
        Fractional stock transferred per simulation step
        (default :data:`DEFAULT_CONVOY_TRANSFER_RATE`).
    """

    source_idx: int
    dest_idx: int
    transfer_rate: float = DEFAULT_CONVOY_TRANSFER_RATE

    def __post_init__(self) -> None:
        if self.transfer_rate <= 0.0:
            raise ValueError(
                f"transfer_rate must be > 0, got {self.transfer_rate!r}"
            )
        if self.source_idx == self.dest_idx:
            raise ValueError("source_idx and dest_idx must differ")

    def step(self, depots: List[SupplyDepot]) -> None:
        """Execute one step of convoy resupply.

        Parameters
        ----------
        depots:
            The full list of :class:`SupplyDepot` objects from the owning
            :class:`SupplyNetwork`.
        """
        src = depots[self.source_idx]
        dst = depots[self.dest_idx]
        if not src.alive or not dst.alive:
            return
        transferred = src.consume(self.transfer_rate)
        dst.replenish(transferred)


# ---------------------------------------------------------------------------
# SupplyNetwork
# ---------------------------------------------------------------------------


@dataclass
class SupplyNetwork:
    """Strategic supply network for one battlefield.

    Holds all :class:`SupplyDepot` nodes and :class:`ConvoyRoute` shuttles
    for both teams.  Exposes helpers for querying supply level at a world
    position, consuming supply, and interdicting enemy depots.

    Parameters
    ----------
    depots:
        All supply depots on the map (both teams).
    convoy_routes:
        Convoy resupply routes.  May be empty.
    consumption_per_step:
        Fractional stock consumed per unit per step when the unit is within
        supply range of a friendly depot
        (default :data:`DEFAULT_CONSUMPTION_PER_STEP`).
    """

    depots: List[SupplyDepot] = field(default_factory=list)
    convoy_routes: List[ConvoyRoute] = field(default_factory=list)
    consumption_per_step: float = DEFAULT_CONSUMPTION_PER_STEP

    # ------------------------------------------------------------------
    # Supply level queries
    # ------------------------------------------------------------------

    def get_supply_level(self, x: float, y: float, team: int) -> float:
        """Return the supply level ``[0, 1]`` available at ``(x, y)`` for *team*.

        The supply level is the maximum contribution from any alive depot
        belonging to *team*.  A value of ``1.0`` means the position is
        directly on a full depot; ``0.0`` means no friendly supply.

        Parameters
        ----------
        x, y:
            Query position in world coordinates (metres).
        team:
            Querying team: ``0`` = Blue, ``1`` = Red.
        """
        best = 0.0
        for depot in self.depots:
            if depot.team != team:
                continue
            level = depot.supply_level_at(x, y)
            if level > best:
                best = level
        return best

    def get_division_supply_levels(
        self,
        unit_positions: Sequence[Tuple[float, float]],
        team: int,
    ) -> List[float]:
        """Return the supply level for each unit position.

        Parameters
        ----------
        unit_positions:
            Sequence of ``(x, y)`` tuples.
        team:
            Owning team of the units.

        Returns
        -------
        List[float] of supply levels, one per position, in ``[0, 1]``.
        """
        return [self.get_supply_level(x, y, team) for x, y in unit_positions]

    # ------------------------------------------------------------------
    # Supply consumption
    # ------------------------------------------------------------------

    def consume_supply(
        self,
        unit_positions: Sequence[Tuple[float, float]],
        team: int,
        amount: Optional[float] = None,
    ) -> None:
        """Deduct supply stock for units positioned near friendly depots.

        For each unit, the nearest in-range friendly depot has
        *amount* stock consumed from it.

        Parameters
        ----------
        unit_positions:
            Sequence of ``(x, y)`` unit positions.
        team:
            Owning team of the units.
        amount:
            Per-unit consumption amount (fractional stock); defaults to
            :attr:`consumption_per_step`.
        """
        if amount is None:
            amount = self.consumption_per_step
        if not unit_positions:
            return

        team_depots = [
            (i, d) for i, d in enumerate(self.depots)
            if d.team == team and d.alive
        ]
        if not team_depots:
            return

        for ux, uy in unit_positions:
            # Find nearest alive depot within effective radius
            best_idx: Optional[int] = None
            best_dist = float("inf")
            for i, depot in team_depots:
                if depot.effective_supply_radius <= 0.0:
                    continue
                dist = math.sqrt((ux - depot.x) ** 2 + (uy - depot.y) ** 2)
                if dist <= depot.effective_supply_radius and dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                self.depots[best_idx].consume(amount)

    # ------------------------------------------------------------------
    # Interdiction
    # ------------------------------------------------------------------

    def interdict_depot(self, depot_idx: int) -> None:
        """Immediately destroy the depot at *depot_idx*.

        Parameters
        ----------
        depot_idx:
            Index into :attr:`depots`.

        Raises
        ------
        IndexError:
            If *depot_idx* is out of range.
        """
        if depot_idx < 0 or depot_idx >= len(self.depots):
            raise IndexError(
                f"depot_idx {depot_idx!r} out of range "
                f"[0, {len(self.depots) - 1}]"
            )
        self.depots[depot_idx].interdict()

    def interdict_nearest_depot(
        self,
        x: float,
        y: float,
        enemy_team: int,
        capture_radius: float,
    ) -> Optional[int]:
        """Interdict the nearest alive enemy depot within *capture_radius*.

        Parameters
        ----------
        x, y:
            Attacker position in world coordinates.
        enemy_team:
            Team of the target depot (``0`` or ``1``).
        capture_radius:
            Maximum distance (metres) within which capture is possible.

        Returns
        -------
        int — index of the interdicted depot, or ``None`` if no depot was
        within range.
        """
        best_idx: Optional[int] = None
        best_dist = float("inf")
        for i, depot in enumerate(self.depots):
            if depot.team != enemy_team or not depot.alive:
                continue
            dist = math.sqrt((x - depot.x) ** 2 + (y - depot.y) ** 2)
            if dist <= capture_radius and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            self.depots[best_idx].interdict()
        return best_idx

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(
        self,
        blue_positions: Sequence[Tuple[float, float]],
        red_positions: Sequence[Tuple[float, float]],
    ) -> None:
        """Advance the supply network by one simulation step.

        Performs:

        1. Supply consumption for both teams.
        2. Convoy resupply transfers between connected depots.

        Parameters
        ----------
        blue_positions:
            Positions of alive Blue units.
        red_positions:
            Positions of alive Red units.
        """
        self.consume_supply(blue_positions, team=0)
        self.consume_supply(red_positions, team=1)
        for route in self.convoy_routes:
            route.step(self.depots)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all depots to their initial state for a new episode."""
        for depot in self.depots:
            depot.reset()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_depots_for_team(self, team: int) -> List[SupplyDepot]:
        """Return all depots belonging to *team*."""
        return [d for d in self.depots if d.team == team]

    def any_alive(self, team: int) -> bool:
        """Return ``True`` if *team* has at least one alive depot."""
        return any(d.alive for d in self.depots if d.team == team)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def generate_default(
        cls,
        map_width: float,
        map_height: float,
        supply_radius: float = DEFAULT_SUPPLY_RADIUS,
        consumption_per_step: float = DEFAULT_CONSUMPTION_PER_STEP,
    ) -> "SupplyNetwork":
        """Generate a default bilateral supply network.

        Creates two Blue depots (in the western third of the map) and two
        Red depots (in the eastern third), connected by intra-team convoy
        routes.  Red's primary forward depot is placed at 80% of map width
        to match the :class:`~envs.corps_env.CorpsEnv` ``CUT_SUPPLY_LINE``
        objective position.

        Parameters
        ----------
        map_width, map_height:
            Map dimensions in metres.
        supply_radius:
            Base supply radius for all generated depots (metres).
        consumption_per_step:
            Per-unit per-step stock consumption rate.
        """
        depots: List[SupplyDepot] = [
            # Blue rear depot (base of supply)
            SupplyDepot(
                x=map_width * 0.1,
                y=map_height * 0.5,
                team=0,
                base_supply_radius=supply_radius * 1.5,
            ),
            # Blue forward depot
            SupplyDepot(
                x=map_width * 0.25,
                y=map_height * 0.5,
                team=0,
                base_supply_radius=supply_radius,
            ),
            # Red forward depot (matches CUT_SUPPLY_LINE objective at 80%)
            SupplyDepot(
                x=map_width * 0.8,
                y=map_height * 0.5,
                team=1,
                base_supply_radius=supply_radius,
            ),
            # Red rear depot (base of supply)
            SupplyDepot(
                x=map_width * 0.9,
                y=map_height * 0.5,
                team=1,
                base_supply_radius=supply_radius * 1.5,
            ),
        ]

        convoy_routes: List[ConvoyRoute] = [
            # Blue rear → Blue forward
            ConvoyRoute(source_idx=0, dest_idx=1),
            # Red rear → Red forward
            ConvoyRoute(source_idx=3, dest_idx=2),
        ]

        return cls(
            depots=depots,
            convoy_routes=convoy_routes,
            consumption_per_step=consumption_per_step,
        )

    def __len__(self) -> int:
        """Return the total number of depots."""
        return len(self.depots)
