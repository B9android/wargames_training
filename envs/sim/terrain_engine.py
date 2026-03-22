# envs/sim/terrain_engine.py
"""Terrain engine: heightmap loader, slope, and Bresenham line-of-sight.

This module provides :class:`TerrainEngine`, a higher-level terrain interface
built on top of :class:`~envs.sim.terrain.TerrainMap`.  It adds:

* **Slope** (gradient magnitude) computation via finite differences.
* **Bresenham-style line-of-sight** that visits every grid cell crossed by
  the ray, avoiding the sampling-gap artefacts of uniform sampling.
* **Combined movement-cost** query integrating elevation and local slope.
* :class:`HeightmapLoader` for loading heightmaps from NumPy arrays
  (compatible with GeoTIFF data read via ``rasterio`` or ``PIL``) and for
  procedural generation with a single *ruggedness* control parameter.

Typical usage::

    import numpy as np
    from envs.sim.terrain_engine import TerrainEngine, HeightmapLoader

    # Procedural terrain with medium ruggedness
    rng = np.random.default_rng(42)
    engine = TerrainEngine.generate_random(rng, width=1000.0, height=1000.0,
                                           ruggedness=0.6)

    # Bresenham LOS between two world positions
    visible = engine.bresenham_los(100.0, 500.0, 800.0, 500.0)

    # Slope at a position
    s = engine.slope(400.0, 400.0)

    # Load from arrays (e.g. a GeoTIFF read with rasterio)
    elev = np.load("elevation.npy")
    cov  = np.zeros_like(elev)
    engine2 = HeightmapLoader.from_array(1000.0, 1000.0, elev, cov)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generator

import numpy as np

from envs.sim.terrain import TerrainMap

__all__ = [
    "TerrainEngine",
    "HeightmapLoader",
]


# ---------------------------------------------------------------------------
# Bresenham cell traversal (module-private)
# ---------------------------------------------------------------------------


def _bresenham_cells(
    r0: int, c0: int, r1: int, c1: int
) -> Generator[tuple[int, int], None, None]:
    """Yield ``(row, col)`` for every grid cell on the line from ``(r0,c0)`` to ``(r1,c1)``.

    Uses the standard Bresenham line-drawing algorithm.  The two ``if``
    blocks (rather than ``if/elif``) are intentional: when both conditions
    are true simultaneously, both *r* and *c* advance in the same iteration,
    producing a diagonal step.  This is required for correct traversal of
    ~45-degree lines and is standard Bresenham behaviour.
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:   # step along row direction; may also step col below
            err -= dc
            r += sr
        if e2 < dr:    # step along col direction (both ifs can fire: diagonal step)
            err += dr
            c += sc


# ---------------------------------------------------------------------------
# TerrainEngine
# ---------------------------------------------------------------------------


@dataclass
class TerrainEngine:
    """Higher-level terrain engine wrapping :class:`~envs.sim.terrain.TerrainMap`.

    Adds slope computation, Bresenham line-of-sight, and a combined
    movement-cost query on top of the standard elevation/cover APIs.

    Use the class-method factories (:meth:`flat`, :meth:`from_arrays`,
    :meth:`generate_random`, :meth:`from_terrain_map`) rather than
    calling the constructor directly.
    """

    terrain_map: TerrainMap

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        """Map width in world units."""
        return self.terrain_map.width

    @property
    def height(self) -> float:
        """Map height in world units."""
        return self.terrain_map.height

    @property
    def elevation(self) -> np.ndarray | None:
        """Elevation grid, or ``None`` for flat terrain."""
        return self.terrain_map.elevation

    @property
    def cover(self) -> np.ndarray | None:
        """Cover grid, or ``None`` for featureless terrain."""
        return self.terrain_map.cover

    @property
    def max_elevation(self) -> float:
        """Maximum elevation across the grid (delegates to :class:`~envs.sim.terrain.TerrainMap`)."""
        return self.terrain_map.max_elevation

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_terrain_map(cls, terrain_map: TerrainMap) -> "TerrainEngine":
        """Wrap an existing :class:`~envs.sim.terrain.TerrainMap`."""
        return cls(terrain_map=terrain_map)

    @classmethod
    def flat(
        cls,
        width: float,
        height: float,
        rows: int = 1,
        cols: int = 1,
    ) -> "TerrainEngine":
        """Create a flat, open terrain engine with zero elevation and no cover."""
        return cls(terrain_map=TerrainMap.flat(width, height, rows, cols))

    @classmethod
    def from_arrays(
        cls,
        width: float,
        height: float,
        elevation: np.ndarray,
        cover: np.ndarray,
    ) -> "TerrainEngine":
        """Create a terrain engine from elevation and cover arrays.

        This factory accepts 2-D NumPy arrays, making it compatible with
        heightmap data read from any external source (e.g. GeoTIFF files
        via ``rasterio`` or ``PIL/Pillow``).  Both arrays must have the
        same 2-D shape.

        Parameters
        ----------
        width, height:
            Map dimensions in world units.
        elevation:
            2-D NumPy array of elevation values.  Pre-normalise to
            ``[0, 1]`` for best results with :meth:`slope` and
            :meth:`movement_cost`.
        cover:
            2-D NumPy array of cover values in ``[0, 1]``.
        """
        return cls(terrain_map=TerrainMap.from_arrays(width, height, elevation, cover))

    @classmethod
    def generate_random(
        cls,
        rng: np.random.Generator,
        width: float,
        height: float,
        rows: int = 20,
        cols: int = 20,
        ruggedness: float = 0.5,
        num_forests: int = 3,
        forest_cover: float = 0.5,
    ) -> "TerrainEngine":
        """Generate a procedural terrain with a controllable *ruggedness* parameter.

        *ruggedness* is the single dial that controls how hilly the map is:

        * ``0.0`` — completely flat (no hills).
        * ``0.5`` — moderate hills (≈ 5 Gaussian blobs, medium sharpness).
        * ``1.0`` — very rugged terrain (10 sharp Gaussian hills).

        Internally, *ruggedness* controls the number of hills and their
        Gaussian sigma (smaller sigma → sharper ridges → more rugged).

        Parameters
        ----------
        rng:
            A ``numpy.random.Generator`` (e.g. ``np.random.default_rng(42)``).
        width, height:
            Map dimensions in world units.
        rows, cols:
            Grid resolution (both must be ≥ 1).
        ruggedness:
            Terrain ruggedness in ``[0, 1]``.
        num_forests:
            Number of Gaussian forest-cover patches.  Must be ≥ 0.
        forest_cover:
            Peak cover value for forest patches.  Must be in ``[0, 1]``.

        Raises
        ------
        ValueError
            If *ruggedness* is not in ``[0, 1]``, or other parameters are
            out of range.
        """
        ruggedness = float(ruggedness)
        if not (0.0 <= ruggedness <= 1.0):
            raise ValueError(f"ruggedness must be in [0, 1], got {ruggedness}")
        rows = int(rows)
        cols = int(cols)
        if rows < 1 or cols < 1:
            raise ValueError(
                f"rows and cols must be >= 1, got rows={rows}, cols={cols}"
            )
        num_forests = int(num_forests)
        if num_forests < 0:
            raise ValueError(f"num_forests must be >= 0, got {num_forests}")
        forest_cover = float(forest_cover)
        if not (0.0 <= forest_cover <= 1.0):
            raise ValueError(f"forest_cover must be in [0, 1], got {forest_cover}")

        # Flat terrain — delegate directly to TerrainMap.generate_random
        if ruggedness == 0.0:
            tm = TerrainMap.generate_random(
                rng=rng,
                width=width,
                height=height,
                rows=rows,
                cols=cols,
                num_hills=0,
                num_forests=num_forests,
                forest_cover=forest_cover,
            )
            return cls(terrain_map=tm)

        # num_hills: 1 at ruggedness≈0.1, up to 10 at ruggedness=1.0
        num_hills = max(1, int(round(ruggedness * 10)))

        # Gaussian sigma: large (0.20) = gentle hills; small (0.03) = sharp ridges.
        # sigma_max decreases as ruggedness increases (sharper peaks).
        sigma_max = 0.20 - ruggedness * 0.17   # 0.20 → 0.03
        sigma_min = max(sigma_max * 0.3, 0.03)

        elevation = np.zeros((rows, cols), dtype=np.float32)
        cover_arr = np.zeros((rows, cols), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, cols, dtype=np.float64)
        ys = np.linspace(0.0, 1.0, rows, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys)

        for _ in range(num_hills):
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            sigma = rng.uniform(sigma_min, sigma_max)
            height_scale = rng.uniform(0.5, 1.0)
            blob = height_scale * np.exp(
                -((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * sigma ** 2)
            )
            elevation += blob.astype(np.float32)

        for _ in range(num_forests):
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            sigma_f = rng.uniform(0.05, 0.2)
            scale = rng.uniform(0.5, 1.0) * forest_cover
            blob = scale * np.exp(
                -((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * sigma_f ** 2)
            )
            cover_arr += blob.astype(np.float32)

        # Normalise elevation to [0, 1]
        max_e = float(elevation.max())
        if max_e > 0.0:
            elevation = (elevation / max_e).astype(np.float32)
        cover_arr = np.clip(cover_arr, 0.0, 1.0).astype(np.float32)

        tm = TerrainMap(
            width=width,
            height=height,
            elevation=elevation,
            cover=cover_arr,
        )
        return cls(terrain_map=tm)

    # ------------------------------------------------------------------
    # Elevation & cover queries (delegated to TerrainMap)
    # ------------------------------------------------------------------

    def get_elevation(self, x: float, y: float) -> float:
        """Return the elevation at world position ``(x, y)``."""
        return self.terrain_map.get_elevation(x, y)

    def get_cover(self, x: float, y: float) -> float:
        """Return the cover value in ``[0, 1]`` at world position ``(x, y)``."""
        return self.terrain_map.get_cover(x, y)

    def apply_cover_modifier(self, x: float, y: float, damage: float) -> float:
        """Reduce *damage* by terrain cover at ``(x, y)``."""
        return self.terrain_map.apply_cover_modifier(x, y, damage)

    def get_speed_modifier(
        self, x: float, y: float, hill_speed_factor: float = 0.5
    ) -> float:
        """Return an elevation-based speed multiplier in ``(0, 1]`` at ``(x, y)``."""
        return self.terrain_map.get_speed_modifier(x, y, hill_speed_factor)

    # ------------------------------------------------------------------
    # Slope
    # ------------------------------------------------------------------

    def slope(self, x: float, y: float) -> float:
        """Compute terrain slope (gradient magnitude) at world position ``(x, y)``.

        Uses central finite differences in the interior of the grid and
        forward/backward differences at the boundary.  Returns ``0.0`` on
        flat terrain (no elevation grid).

        The unit of the returned value is *elevation units per world unit*.
        For elevation grids normalised to ``[0, 1]`` this is typically a
        small number well below ``1.0`` for realistic terrains.

        Parameters
        ----------
        x, y:
            World position.
        """
        elev = self.terrain_map.elevation
        if elev is None or elev.size == 0:
            return 0.0
        if self.terrain_map.width <= 0.0 or self.terrain_map.height <= 0.0:
            return 0.0

        rows, cols = elev.shape
        r, c = self.terrain_map._to_grid_coords(x, y, elev)

        cell_w = self.terrain_map.width / cols
        cell_h = self.terrain_map.height / rows

        # Gradient along the column axis (world x direction)
        if cols == 1:
            dz_dx = 0.0
        elif c == 0:
            dz_dx = (float(elev[r, c + 1]) - float(elev[r, c])) / cell_w
        elif c == cols - 1:
            dz_dx = (float(elev[r, c]) - float(elev[r, c - 1])) / cell_w
        else:
            dz_dx = (float(elev[r, c + 1]) - float(elev[r, c - 1])) / (2.0 * cell_w)

        # Gradient along the row axis (world y direction)
        if rows == 1:
            dz_dy = 0.0
        elif r == 0:
            dz_dy = (float(elev[r + 1, c]) - float(elev[r, c])) / cell_h
        elif r == rows - 1:
            dz_dy = (float(elev[r, c]) - float(elev[r - 1, c])) / cell_h
        else:
            dz_dy = (float(elev[r + 1, c]) - float(elev[r - 1, c])) / (2.0 * cell_h)

        return math.sqrt(dz_dx ** 2 + dz_dy ** 2)

    # ------------------------------------------------------------------
    # Bresenham line-of-sight
    # ------------------------------------------------------------------

    def bresenham_los(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        """Check line-of-sight using Bresenham grid-cell traversal.

        Unlike :meth:`line_of_sight` (which samples at uniform intervals and
        can miss narrow ridges), this method visits **every grid cell** the
        line passes through.  For each intermediate cell the terrain elevation
        is compared against the straight-line interpolant connecting the
        endpoint elevations; if the cell elevation exceeds that value the
        line of sight is blocked.

        Returns ``True`` if the line from ``(x0, y0)`` to ``(x1, y1)`` is
        unobstructed.  Always returns ``True`` on flat terrain (no elevation
        grid).

        Parameters
        ----------
        x0, y0:
            Start position in world coordinates.
        x1, y1:
            End position in world coordinates.
        """
        elev = self.terrain_map.elevation
        if elev is None or elev.size == 0:
            return True

        r0, c0 = self.terrain_map._to_grid_coords(x0, y0, elev)
        r1, c1 = self.terrain_map._to_grid_coords(x1, y1, elev)

        elev0 = float(elev[r0, c0])
        elev1 = float(elev[r1, c1])

        # Number of visited cells is max(|dr|,|dc|)+1 for Bresenham traversal —
        # avoid materializing the full list just to get its length.
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        n = max(dr, dc) + 1

        # Single cell: start and end map to the same grid cell — always visible.
        if n <= 1:
            return True

        # Denominator for computing the parametric position of each visited
        # cell along the segment in grid space.  Guaranteed > 0 when n > 1.
        denom = float(dc * dc + dr * dr)

        for i, (r, c) in enumerate(_bresenham_cells(r0, c0, r1, c1)):
            if i == 0 or i == n - 1:
                continue  # endpoints are not checked against themselves

            # Parametric t: projected position of this cell along the line
            # segment in grid coordinates, avoiding the index-step bias that
            # occurs when diagonal steps are made.
            vx = float(c - c0)
            vy = float(r - r0)
            t = (vx * dc + vy * dr) / denom
            t = max(0.0, min(1.0, t))  # clamp for numerical robustness

            los_height = elev0 + t * (elev1 - elev0)
            if float(elev[r, c]) > los_height:
                return False
        return True

    # ------------------------------------------------------------------
    # Combined movement cost
    # ------------------------------------------------------------------

    def movement_cost(
        self, x: float, y: float, hill_speed_factor: float = 0.5
    ) -> float:
        """Return the movement-cost multiplier at ``(x, y)``.

        Combines the elevation-based speed modifier with a slope penalty,
        producing a value in ``(0, 1]``:

        * ``1.0`` — flat terrain, no penalty.
        * Values close to *hill_speed_factor* — steep, high-elevation terrain.

        Parameters
        ----------
        x, y:
            World position.
        hill_speed_factor:
            Speed multiplier at maximum elevation (passed to
            :meth:`get_speed_modifier`).  Must be in ``(0, 1]``.
        """
        elev_mod = self.terrain_map.get_speed_modifier(x, y, hill_speed_factor)
        s = self.slope(x, y)
        # Slope penalty: each unit of slope reduces speed by 50 %.  The
        # intermediate result is clamped to avoid dropping below hill_speed_factor.
        slope_mod = float(np.clip(1.0 - s * 0.5, hill_speed_factor, 1.0))
        # Clamp the combined product: on steep high-elevation ground the product
        # of elev_mod (already as low as hill_speed_factor) and slope_mod can
        # fall below hill_speed_factor.  Clamp to [hill_speed_factor, 1].
        return float(np.clip(elev_mod * slope_mod, hill_speed_factor, 1.0))

    # ------------------------------------------------------------------
    # Sampling-based LOS (delegates to TerrainMap)
    # ------------------------------------------------------------------

    def line_of_sight(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        num_samples: int = 20,
    ) -> bool:
        """Sampling-based LOS check (delegates to :class:`~envs.sim.terrain.TerrainMap`).

        For higher accuracy, prefer :meth:`bresenham_los`.
        """
        return self.terrain_map.line_of_sight(x0, y0, x1, y1, num_samples)


# ---------------------------------------------------------------------------
# HeightmapLoader
# ---------------------------------------------------------------------------


class HeightmapLoader:
    """Utility class for loading heightmap data into a :class:`TerrainEngine`.

    This class provides static factory methods that serve as the
    heightmap-data interface for the terrain engine.  The ``from_array``
    method is compatible with any external heightmap source that can
    produce a NumPy array, including GeoTIFF files read via ``rasterio``
    or ``PIL/Pillow``.

    All methods return a :class:`TerrainEngine` instance.
    """

    @staticmethod
    def from_array(
        width: float,
        height: float,
        elevation: np.ndarray,
        cover: np.ndarray,
    ) -> TerrainEngine:
        """Load a :class:`TerrainEngine` from existing NumPy arrays.

        This is the primary entry point for externally-sourced heightmap
        data (e.g. a GeoTIFF elevation raster read into a NumPy array).
        Both arrays must be 2-D with the same shape.

        Parameters
        ----------
        width, height:
            Map dimensions in world units.
        elevation:
            2-D elevation array.  Values can be in any numeric range;
            normalise to ``[0, 1]`` before passing for best results with
            slope and movement cost queries.
        cover:
            2-D cover array.  Values should be in ``[0, 1]``.
        """
        return TerrainEngine.from_arrays(width, height, elevation, cover)

    @staticmethod
    def from_procedural(
        rng: np.random.Generator,
        width: float,
        height: float,
        rows: int = 20,
        cols: int = 20,
        ruggedness: float = 0.5,
        num_forests: int = 3,
        forest_cover: float = 0.5,
    ) -> TerrainEngine:
        """Generate a procedural :class:`TerrainEngine` map.

        Delegates to :meth:`TerrainEngine.generate_random`.  See that
        method for full parameter documentation.
        """
        return TerrainEngine.generate_random(
            rng=rng,
            width=width,
            height=height,
            rows=rows,
            cols=cols,
            ruggedness=ruggedness,
            num_forests=num_forests,
            forest_cover=forest_cover,
        )
