# SPDX-License-Identifier: MIT
# Map generation

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# TerrainMap
# ---------------------------------------------------------------------------


@dataclass
class TerrainMap:
    """2-D terrain representation used by the simulation.

    The map conceptually covers continuous world coordinates in the
    rectangle ``[0, width] × [0, height]``.  Coordinates passed into
    query methods are clamped to this rectangle so that points on the
    upper/right boundary (``x == width`` or ``y == height``) map into
    the last column/row of the underlying grids.

    Terrain data is stored in 2-D NumPy arrays whose shape is
    ``(rows, cols)``.  Row 0 corresponds to ``y ≈ 0`` and the last row
    to ``y ≈ height``; column 0 corresponds to ``x ≈ 0``.

    Flat, fully open terrain is represented by zero-valued elevation and
    cover grids (as produced by :meth:`flat`), which correspond to a
    perfectly flat, fully open plain.

    Use :meth:`flat` or :meth:`from_arrays` instead of calling the
    constructor directly.
    """

    width: float
    """Map width in world units."""

    height: float
    """Map height in world units."""

    elevation: np.ndarray | None = field(default=None, repr=False)
    """Optional 2-D elevation grid (shape ``(rows, cols)``).
    Values are in arbitrary height units — only relative differences
    matter for line-of-sight checks."""

    cover: np.ndarray | None = field(default=None, repr=False)
    """Optional 2-D cover grid (shape ``(rows, cols)``).
    Values should be in ``[0, 1]``: 0 = no cover, 1 = full cover."""

    # Cached maximum elevation value — computed once in __post_init__ to
    # avoid repeated full-array reductions in get_speed_modifier().
    _max_elev: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._max_elev = (
            float(self.elevation.max())
            if self.elevation is not None and self.elevation.size > 0
            else 0.0
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def flat(
        cls,
        width: float,
        height: float,
        rows: int = 1,
        cols: int = 1,
    ) -> "TerrainMap":
        """Create a flat, open terrain with zero elevation and no cover.

        Parameters
        ----------
        width, height:
            Map dimensions in world units.
        rows, cols:
            Grid resolution.  Defaults to ``1×1`` (single cell), which
            is sufficient for purely flat terrain.
        """
        elevation = np.zeros((rows, cols), dtype=np.float32)
        cover = np.zeros((rows, cols), dtype=np.float32)
        return cls(width=width, height=height, elevation=elevation, cover=cover)

    @classmethod
    def from_arrays(
        cls,
        width: float,
        height: float,
        elevation: np.ndarray,
        cover: np.ndarray,
    ) -> "TerrainMap":
        """Create a terrain from existing elevation and cover arrays.

        Both arrays must be 2-D with the same shape.  Copies are taken so
        that the caller's arrays are not aliased.

        Parameters
        ----------
        width, height:
            Map dimensions in world units.
        elevation:
            2-D NumPy array of elevation values.
        cover:
            2-D NumPy array of cover values in ``[0, 1]``.
        """
        elevation = np.asarray(elevation, dtype=np.float32)
        cover = np.asarray(cover, dtype=np.float32)
        if elevation.ndim != 2 or cover.ndim != 2:
            raise ValueError("elevation and cover must be 2-D arrays")
        if elevation.shape != cover.shape:
            raise ValueError(
                f"elevation and cover shapes must match: "
                f"{elevation.shape} != {cover.shape}"
            )
        return cls(
            width=width,
            height=height,
            elevation=elevation.copy(),
            cover=cover.copy(),
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def max_elevation(self) -> float:
        """Maximum elevation value across the entire grid.

        Returns ``0.0`` for flat terrain (no elevation grid).  This is the
        same value used internally by :meth:`get_speed_modifier` to normalise
        elevation; exposing it as a public property avoids callers reaching
        into private state.
        """
        return self._max_elev

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_grid_coords(self, x: float, y: float, grid: np.ndarray) -> tuple[int, int]:
        """Convert world coordinates to (row, col) indices for *grid*.

        Coordinates are clamped to valid index ranges so that positions
        at or beyond the map boundary map to the last cell.  If *width*
        or *height* is zero the first cell (0, 0) is returned.
        """
        rows, cols = grid.shape
        if self.width <= 0.0 or self.height <= 0.0:
            return 0, 0
        col = int(np.clip(x / self.width * cols, 0, cols - 1))
        row = int(np.clip(y / self.height * rows, 0, rows - 1))
        return row, col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_elevation(self, x: float, y: float) -> float:
        """Return the elevation at world position ``(x, y)``.

        Uses cell lookup based on the containing grid cell (via
        :meth:`_to_grid_coords`).  Returns ``0.0`` when no elevation
        grid is present (flat terrain).
        """
        if self.elevation is None or self.elevation.size == 0:
            return 0.0
        row, col = self._to_grid_coords(x, y, self.elevation)
        return float(self.elevation[row, col])

    def get_cover(self, x: float, y: float) -> float:
        """Return the cover value in ``[0, 1]`` at world position ``(x, y)``.

        Uses cell lookup based on the containing grid cell (via
        :meth:`_to_grid_coords`).  Returns ``0.0`` (no cover) when no
        cover grid is present.
        """
        if self.cover is None or self.cover.size == 0:
            return 0.0
        row, col = self._to_grid_coords(x, y, self.cover)
        return float(np.clip(self.cover[row, col], 0.0, 1.0))

    def line_of_sight(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        num_samples: int = 20,
    ) -> bool:
        """Return ``True`` if ``(x0, y0)`` has unobstructed sight to ``(x1, y1)``.

        The check samples *num_samples* equally-spaced points along the
        straight line between the two positions.  At each interior sample
        the terrain elevation is compared against the height of the
        imaginary straight line connecting the two *endpoint* elevations.
        If any interior point's elevation strictly exceeds that line, LOS
        is considered blocked.

        On flat terrain (no elevation grid) this always returns ``True``.

        Parameters
        ----------
        x0, y0:
            Start position in world coordinates.
        x1, y1:
            End position in world coordinates.
        num_samples:
            Number of sample points including both endpoints.  Higher
            values give more accurate results at the cost of more lookups.
        """
        if num_samples < 2:
            raise ValueError(f"num_samples must be >= 2, got {num_samples}")

        if self.elevation is None:
            return True

        elev0 = self.get_elevation(x0, y0)
        elev1 = self.get_elevation(x1, y1)

        # Check only interior samples (skip i=0 and i=num_samples-1)
        for i in range(1, num_samples - 1):
            t = i / (num_samples - 1)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            elev_at = self.get_elevation(x, y)
            los_height = elev0 + t * (elev1 - elev0)
            if elev_at > los_height:
                return False
        return True

    def apply_cover_modifier(self, x: float, y: float, damage: float) -> float:
        """Reduce *damage* according to the cover level at ``(x, y)``.

        A cover value ``c`` scales damage by ``(1 − c)``, so:

        * ``c = 0`` (no cover)  → full damage returned unchanged.
        * ``c = 1`` (full cover) → zero damage returned.

        Parameters
        ----------
        x, y:
            World position of the defending unit.
        damage:
            Raw incoming damage value.
        """
        cover = self.get_cover(x, y)
        return float(damage) * (1.0 - cover)

    def get_speed_modifier(self, x: float, y: float, hill_speed_factor: float = 0.5) -> float:
        """Return a movement speed multiplier at world position ``(x, y)``.

        Speed is reduced on elevated terrain (hills).  On flat terrain
        (elevation = 0) the modifier is ``1.0`` (full speed).  At the
        highest point in the map the modifier equals *hill_speed_factor*.
        Intermediate elevations are interpolated linearly.

        Parameters
        ----------
        x, y:
            World position of the unit.
        hill_speed_factor:
            Speed multiplier applied at maximum elevation.  Must be in
            ``(0, 1]``; a value of ``1.0`` disables the hill penalty.

        Raises
        ------
        ValueError
            If *hill_speed_factor* is not in ``(0, 1]``.
        """
        hill_speed_factor = float(hill_speed_factor)
        if not (0.0 < hill_speed_factor <= 1.0):
            raise ValueError(
                f"hill_speed_factor must be in (0, 1], got {hill_speed_factor}"
            )
        if self.elevation is None or self.elevation.size == 0:
            return 1.0
        if self._max_elev <= 0.0:
            return 1.0
        elev = self.get_elevation(x, y)
        normalised = float(np.clip(elev / self._max_elev, 0.0, 1.0))
        return 1.0 - normalised * (1.0 - hill_speed_factor)

    @classmethod
    def generate_random(
        cls,
        rng: "np.random.Generator",
        width: float,
        height: float,
        rows: int = 20,
        cols: int = 20,
        num_hills: int = 3,
        num_forests: int = 3,
        forest_cover: float = 0.5,
    ) -> "TerrainMap":
        """Generate a random terrain map with hills and forest patches.

        Hills are represented as Gaussian elevation blobs; forests as
        Gaussian cover blobs.  The elevation grid is normalised to
        ``[0, 1]`` so that :meth:`get_speed_modifier` works correctly
        regardless of the number or height of hills.  Cover values are
        clipped to ``[0, 1]``.

        Parameters
        ----------
        rng:
            A ``numpy.random.Generator`` instance (e.g. from
            ``np.random.default_rng(seed)``).
        width, height:
            Map dimensions in world units.
        rows, cols:
            Grid resolution.  Both must be positive integers.
        num_hills:
            Number of Gaussian elevation blobs to place.  Must be >= 0.
        num_forests:
            Number of Gaussian cover blobs to place.  Must be >= 0.
        forest_cover:
            Peak cover value for forest patches.  Must be in ``[0, 1]``.

        Raises
        ------
        ValueError
            If any parameter is out of its valid range.
        """
        rows = int(rows)
        cols = int(cols)
        num_hills = int(num_hills)
        num_forests = int(num_forests)
        forest_cover = float(forest_cover)
        if rows < 1 or cols < 1:
            raise ValueError(
                f"rows and cols must be >= 1, got rows={rows}, cols={cols}"
            )
        if num_hills < 0:
            raise ValueError(f"num_hills must be >= 0, got {num_hills}")
        if num_forests < 0:
            raise ValueError(f"num_forests must be >= 0, got {num_forests}")
        if not (0.0 <= forest_cover <= 1.0):
            raise ValueError(
                f"forest_cover must be in [0, 1], got {forest_cover}"
            )
        elevation = np.zeros((rows, cols), dtype=np.float32)
        cover = np.zeros((rows, cols), dtype=np.float32)

        # Normalised grid coordinates in [0, 1]
        xs = np.linspace(0.0, 1.0, cols, dtype=np.float64)
        ys = np.linspace(0.0, 1.0, rows, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys)

        for _ in range(num_hills):
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            sigma = rng.uniform(0.05, 0.2)
            height_scale = rng.uniform(0.5, 1.0)
            blob = height_scale * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * sigma ** 2))
            elevation += blob.astype(np.float32)

        for _ in range(num_forests):
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            sigma = rng.uniform(0.05, 0.2)
            cover_scale = rng.uniform(0.5, 1.0) * float(forest_cover)
            blob = cover_scale * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * sigma ** 2))
            cover += blob.astype(np.float32)

        # Normalise elevation to [0, 1]
        max_elev = float(elevation.max())
        if max_elev > 0.0:
            elevation = (elevation / max_elev).astype(np.float32)

        cover = np.clip(cover, 0.0, 1.0).astype(np.float32)

        return cls(width=width, height=height, elevation=elevation, cover=cover)
