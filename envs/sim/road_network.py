# envs/sim/road_network.py
"""Road network: road segments with movement-speed bonuses.

:class:`RoadNetwork` holds a list of :class:`RoadSegment` objects and
exposes helpers to check whether a world position lies on a road and to
compute the resulting speed modifier.  The default road speed bonus is
``1.5×`` (50 % faster than off-road movement).

Typical usage::

    from envs.sim.road_network import RoadNetwork

    rng = np.random.default_rng(42)
    net = RoadNetwork.generate_default(map_width=10_000.0, map_height=5_000.0)

    # Check whether a battalion at (3_000, 2_500) is on a road
    on_road = net.is_on_road(3_000.0, 2_500.0)
    modifier = net.get_speed_modifier(3_000.0, 2_500.0)  # 1.5 or 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "RoadSegment",
    "RoadNetwork",
    "ROAD_SPEED_BONUS",
]

#: Speed multiplier applied to units that are on a road.
ROAD_SPEED_BONUS: float = 1.5


@dataclass
class RoadSegment:
    """A straight road segment between two world positions.

    Parameters
    ----------
    x0, y0:
        Start point in world coordinates (metres).
    x1, y1:
        End point in world coordinates (metres).
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def distance_to_point(self, x: float, y: float) -> float:
        """Return the perpendicular distance from point ``(x, y)`` to this segment.

        If the nearest point on the infinite line lies outside the segment,
        the distance to the closer endpoint is returned instead.
        """
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            # Degenerate segment — treat as a point
            return math.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
        # Scalar projection onto segment, clamped to [0, 1]
        t = ((x - self.x0) * dx + (y - self.y0) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        proj_x = self.x0 + t * dx
        proj_y = self.y0 + t * dy
        return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)


@dataclass
class RoadNetwork:
    """A collection of road segments that grant movement-speed bonuses.

    Parameters
    ----------
    segments:
        List of :class:`RoadSegment` objects that make up the road network.
    road_half_width:
        Half-width of each road in metres.  A unit is considered to be *on*
        the road when its distance to the nearest segment is ≤ this value.
        Default is 30 m (approximately one battalion frontage).
    """

    segments: List[RoadSegment] = field(default_factory=list)
    road_half_width: float = 30.0

    # ------------------------------------------------------------------
    # Road queries
    # ------------------------------------------------------------------

    def is_on_road(self, x: float, y: float) -> bool:
        """Return ``True`` if ``(x, y)`` lies within *road_half_width* of any segment."""
        for seg in self.segments:
            if seg.distance_to_point(x, y) <= self.road_half_width:
                return True
        return False

    def get_speed_modifier(self, x: float, y: float) -> float:
        """Return :data:`ROAD_SPEED_BONUS` if on a road, otherwise ``1.0``."""
        return ROAD_SPEED_BONUS if self.is_on_road(x, y) else 1.0

    def fraction_on_road(
        self, positions: Sequence[Tuple[float, float]]
    ) -> float:
        """Return the fraction of *positions* that lie on a road.

        Parameters
        ----------
        positions:
            A sequence of ``(x, y)`` tuples.

        Returns
        -------
        float in ``[0, 1]``, or ``0.0`` for an empty sequence.
        """
        if not positions:
            return 0.0
        on_road = sum(1 for x, y in positions if self.is_on_road(x, y))
        return on_road / len(positions)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def generate_default(
        cls,
        map_width: float,
        map_height: float,
        n_roads: int = 3,
        road_half_width: float = 30.0,
    ) -> "RoadNetwork":
        """Generate a simple default road network for *map_width* × *map_height*.

        Creates *n_roads* horizontal roads at evenly-spaced y-positions that
        span the full map width, plus one central north–south road.  This
        gives units multiple route options and creates natural corridors.

        Parameters
        ----------
        map_width, map_height:
            Map dimensions in metres.
        n_roads:
            Number of horizontal roads (default 3).
        road_half_width:
            Half-width of roads in metres (default 30 m).
        """
        segments: List[RoadSegment] = []

        # Horizontal roads evenly spaced along the y-axis
        for i in range(n_roads):
            y = (i + 1) * map_height / (n_roads + 1)
            segments.append(RoadSegment(x0=0.0, y0=y, x1=map_width, y1=y))

        # Central north–south road
        cx = map_width / 2.0
        segments.append(RoadSegment(x0=cx, y0=0.0, x1=cx, y1=map_height))

        return cls(segments=segments, road_half_width=road_half_width)

    def __len__(self) -> int:
        return len(self.segments)
