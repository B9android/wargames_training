# envs/rendering/web_renderer.py
"""Lightweight JSON-serialisable renderer for BattalionEnv.

:class:`WebRenderer` produces a plain Python dictionary describing the
current game state.  The dict is JSON-serialisable and suitable for:

* Headless testing and state logging.
* Pyodide / WebAssembly frontends — pass the dict to JavaScript after
  ``json.dumps``, e.g. ``js.globalThis.renderFrame(json.dumps(frame))``.
* Server-sent-events or WebSocket streaming.

No pygame or GUI dependency is required.

Usage::

    from envs.rendering.web_renderer import WebRenderer

    renderer = WebRenderer(map_width=1000.0, map_height=1000.0)
    frame = renderer.render_frame(blue, red, step=0)
    # frame is a plain dict, ready for json.dumps()
    import json
    print(json.dumps(frame, indent=2))
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from envs.sim.battalion import Battalion
    from envs.sim.terrain import TerrainMap

# Default resolution of the down-sampled terrain grid.
_TERRAIN_GRID_SIZE: int = 10


class WebRenderer:
    """JSON-serialisable renderer for BattalionEnv game states.

    Unlike :class:`~envs.rendering.renderer.BattalionRenderer`, this class
    has no pygame dependency and can be used in headless or web (Pyodide)
    contexts.

    Parameters
    ----------
    map_width, map_height:
        World dimensions in metres.  Used for normalising unit positions
        in the output dict.
    terrain_grid_size:
        Resolution of the down-sampled terrain elevation grid included in
        each frame dict.  Defaults to ``10`` (10 × 10 grid).
    """

    def __init__(
        self,
        map_width: float,
        map_height: float,
        terrain_grid_size: int = _TERRAIN_GRID_SIZE,
    ) -> None:
        map_w = float(map_width)
        map_h = float(map_height)
        grid_size = int(terrain_grid_size)

        if map_w <= 0.0:
            raise ValueError(f"map_width must be positive, got {map_width!r}")
        if map_h <= 0.0:
            raise ValueError(f"map_height must be positive, got {map_height!r}")
        if grid_size < 1:
            raise ValueError(
                f"terrain_grid_size must be an integer >= 1, got {terrain_grid_size!r}"
            )

        self._map_w = map_w
        self._map_h = map_h
        self._grid_size = grid_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_frame(
        self,
        blue: "Battalion",
        red: "Battalion",
        terrain: Optional["TerrainMap"] = None,
        step: int = 0,
        info: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Produce a JSON-serialisable snapshot of the current game state.

        Parameters
        ----------
        blue, red:
            Current battalion states.
        terrain:
            Optional terrain map.  When provided, a down-sampled elevation
            grid is included in ``terrain_summary``.
        step:
            Current step index.
        info:
            Optional info dict from :meth:`~envs.battalion_env.BattalionEnv.step`.

        Returns
        -------
        dict
            A JSON-serialisable dict with the following keys:

            ``step`` : int
                Current step index.
            ``map`` : dict
                ``width`` and ``height`` of the world in metres.
            ``blue`` : dict
                Blue battalion state (see :meth:`_battalion_to_dict`).
            ``red`` : dict
                Red battalion state.
            ``terrain_summary`` : list[list[float]] or None
                Down-sampled normalised elevation grid
                (``terrain_grid_size × terrain_grid_size``), or ``None``
                when no terrain is supplied.
            ``info`` : dict
                A copy of the *info* dict, or an empty dict when *info*
                is ``None``.
        """
        return {
            "step": int(step),
            "map": {
                "width": self._map_w,
                "height": self._map_h,
            },
            "blue": self._battalion_to_dict(blue),
            "red": self._battalion_to_dict(red),
            "terrain_summary": self._terrain_summary(terrain),
            "info": dict(info) if info else {},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _battalion_to_dict(self, battalion: "Battalion") -> dict[str, Any]:
        """Serialise a :class:`~envs.sim.battalion.Battalion` to a plain dict.

        All floating-point values are cast with ``float()`` to ensure they
        are native Python floats (JSON-serialisable even when the source
        is a NumPy scalar).
        """
        return {
            "x": float(battalion.x),
            "y": float(battalion.y),
            "x_norm": float(battalion.x / self._map_w),
            "y_norm": float(battalion.y / self._map_h),
            "theta": float(battalion.theta),
            "cos_theta": float(math.cos(battalion.theta)),
            "sin_theta": float(math.sin(battalion.theta)),
            "strength": float(battalion.strength),
            "morale": float(battalion.morale),
            "routed": bool(battalion.routed),
            "team": int(battalion.team),
        }

    def _terrain_summary(
        self, terrain: Optional["TerrainMap"]
    ) -> Optional[list[list[float]]]:
        """Return a down-sampled normalised elevation grid, or ``None``.

        The returned grid has shape ``(terrain_grid_size, terrain_grid_size)``
        with values in ``[0, 1]`` where ``0`` maps to the minimum elevation
        and ``1`` maps to the maximum elevation on the map
        (i.e. ``(elev - min) / (max - min)``).
        """
        if terrain is None:
            return None
        if terrain.elevation is None or terrain.elevation.size == 0:
            return None

        elev = terrain.elevation
        rows, cols = elev.shape
        min_e = float(elev.min())
        max_e = float(elev.max())
        if max_e <= min_e:
            return None

        # Down-sample to (grid_size × grid_size).
        n = self._grid_size
        row_step = max(1, rows // n)
        col_step = max(1, cols // n)
        sampled = elev[::row_step, ::col_step][:n, :n]
        return ((sampled - min_e) / (max_e - min_e)).tolist()
