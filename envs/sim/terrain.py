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

    When both *elevation* and *cover* are ``None`` the terrain is treated
    as a perfectly flat, fully open plain (all methods still work and
    return sensible defaults).

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

        Uses nearest-cell lookup.  Returns ``0.0`` when no elevation
        grid is present (flat terrain).
        """
        if self.elevation is None or self.elevation.size == 0:
            return 0.0
        row, col = self._to_grid_coords(x, y, self.elevation)
        return float(self.elevation[row, col])

    def get_cover(self, x: float, y: float) -> float:
        """Return the cover value in ``[0, 1]`` at world position ``(x, y)``.

        Uses nearest-cell lookup.  Returns ``0.0`` (no cover) when no
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
