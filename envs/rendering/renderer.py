# envs/rendering/renderer.py
"""Pygame-based visualiser for BattalionEnv episodes.

Usage (standalone)::

    from envs.rendering.renderer import BattalionRenderer

    renderer = BattalionRenderer(map_width=1000.0, map_height=1000.0)
    renderer.set_terrain(env.terrain)
    alive = renderer.render_frame(env.blue, env.red, step=0)
    renderer.close()

Set the ``SDL_VIDEODRIVER=dummy`` environment variable before importing to
run in a headless environment (e.g. CI).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from envs.sim.battalion import Battalion
    from envs.sim.terrain import TerrainMap

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW_W: int = 800
_DEFAULT_WINDOW_H: int = 800
_DEFAULT_FPS: int = 30

# Colour palette
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_BLUE_UNIT = (30, 80, 200)
_RED_UNIT = (200, 30, 30)
_GREEN = (0, 180, 0)
_YELLOW = (220, 180, 0)
_ORANGE = (255, 120, 0)
_GRAY = (150, 150, 150)
_BG_COLOR = (195, 215, 175)  # muted grass green

# Battalion line half-length in pixels
_HALF_LEN: int = 28
# Status bar dimensions in pixels
_BAR_W: int = 52
_BAR_H: int = 6


# ---------------------------------------------------------------------------
# BattalionRenderer
# ---------------------------------------------------------------------------


class BattalionRenderer:
    """Pygame window for live or replay rendering of :class:`BattalionEnv`.

    Parameters
    ----------
    map_width, map_height:
        World-coordinate dimensions of the simulation map (metres).  Used
        to convert battalion positions to screen pixels.
    window_w, window_h:
        Pixel dimensions of the pygame window.
    fps:
        Target frame-rate cap (frames per second).  Pass ``0`` to disable
        the cap (useful for tests or replay).
    """

    def __init__(
        self,
        map_width: float,
        map_height: float,
        window_w: int = _DEFAULT_WINDOW_W,
        window_h: int = _DEFAULT_WINDOW_H,
        fps: int = _DEFAULT_FPS,
    ) -> None:
        import pygame  # noqa: PLC0415 — lazy import so pygame is optional at module level

        self._pygame = pygame
        self._map_w = float(map_width)
        self._map_h = float(map_height)
        self._win_w = window_w
        self._win_h = window_h
        self._fps = fps
        self._terrain_surface: Optional[pygame.Surface] = None
        # Track the identity of the cached terrain so we can detect changes
        # (e.g. when BattalionEnv generates a new TerrainMap on each reset).
        self._cached_terrain_id: Optional[int] = None

        pygame.init()
        self._screen: pygame.Surface = pygame.display.set_mode(
            (window_w, window_h)
        )
        pygame.display.set_caption("Wargames Training")
        self._clock: pygame.time.Clock = pygame.time.Clock()
        self._font: pygame.font.Font = pygame.font.SysFont("monospace", 12)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (metres) to screen (pixels).

        The world y-axis points *up* (y=0 at the bottom of the map), while
        the screen y-axis points *down* (y=0 at the top of the window).
        Coordinates are clamped to ``[0, win_w-1]`` × ``[0, win_h-1]`` so
        that units exactly on the map boundary remain within the surface.
        """
        sx = int(x / self._map_w * self._win_w)
        sy = int((1.0 - y / self._map_h) * self._win_h)
        sx = max(0, min(self._win_w - 1, sx))
        sy = max(0, min(self._win_h - 1, sy))
        return sx, sy

    # ------------------------------------------------------------------
    # Terrain rendering
    # ------------------------------------------------------------------

    def _build_terrain_surface(self, terrain: "TerrainMap") -> "pygame.Surface":
        """Rasterise terrain elevation and cover into a pygame Surface."""
        pygame = self._pygame
        surf: pygame.Surface = pygame.Surface((self._win_w, self._win_h))
        surf.fill(_BG_COLOR)

        if terrain.elevation is not None and terrain.elevation.size > 0:
            rows, cols = terrain.elevation.shape
            cell_w = self._win_w / cols
            cell_h = self._win_h / rows
            max_e = float(terrain.elevation.max())
            if max_e <= 0.0:
                max_e = 1.0
            for r in range(rows):
                for c in range(cols):
                    elev_norm = float(terrain.elevation[r, c]) / max_e
                    # Higher elevation → browner/darker shade
                    base = 200
                    shade = int(base - elev_norm * 70)
                    shade = max(80, min(255, shade))
                    color = (shade, int(shade * 0.95), int(shade * 0.75))
                    # Flip row index so that terrain row 0 (y≈0, world bottom)
                    # appears at the bottom of the screen, matching the y-axis
                    # convention used by _world_to_screen().
                    screen_row = rows - 1 - r
                    rect = (
                        int(c * cell_w),
                        int(screen_row * cell_h),
                        max(1, int(cell_w) + 1),
                        max(1, int(cell_h) + 1),
                    )
                    pygame.draw.rect(surf, color, rect)

        if terrain.cover is not None and terrain.cover.size > 0:
            rows, cols = terrain.cover.shape
            cell_w = self._win_w / cols
            cell_h = self._win_h / rows
            for r in range(rows):
                for c in range(cols):
                    cov = float(terrain.cover[r, c])
                    if cov > 0.05:
                        alpha = min(255, int(cov * 120))
                        cover_cell = pygame.Surface(
                            (max(1, int(cell_w)), max(1, int(cell_h))),
                            pygame.SRCALPHA,
                        )
                        cover_cell.fill((30, 140, 30, alpha))
                        # Same y-flip as elevation
                        screen_row = rows - 1 - r
                        surf.blit(cover_cell, (int(c * cell_w), int(screen_row * cell_h)))

        return surf

    def set_terrain(self, terrain: "TerrainMap") -> None:
        """Pre-render *terrain* into a cached surface for fast blitting.

        Calling this again with a different terrain object clears the old
        cache and builds a new surface.
        """
        self._terrain_surface = self._build_terrain_surface(terrain)
        self._cached_terrain_id = id(terrain)

    # ------------------------------------------------------------------
    # Battalion drawing
    # ------------------------------------------------------------------

    def _draw_battalion(self, battalion: "Battalion") -> None:
        """Draw one battalion as an oriented line segment with status bars."""
        pygame = self._pygame
        sx, sy = self._world_to_screen(battalion.x, battalion.y)
        color = _BLUE_UNIT if battalion.team == 0 else _RED_UNIT

        # --- Line segment oriented by angle θ ---------------------------
        # Screen y-axis is inverted relative to world y, so negate sin(θ).
        dx = math.cos(battalion.theta) * _HALF_LEN
        dy = math.sin(battalion.theta) * _HALF_LEN
        x1 = int(sx - dx)
        y1 = int(sy + dy)   # inverted y
        x2 = int(sx + dx)
        y2 = int(sy - dy)   # inverted y
        pygame.draw.line(self._screen, color, (x1, y1), (x2, y2), 4)
        # Centre dot
        pygame.draw.circle(self._screen, color, (sx, sy), 5)

        # --- Strength bar (below the centre) ----------------------------
        bar_x = sx - _BAR_W // 2
        bar_y = sy + 12
        pygame.draw.rect(self._screen, _GRAY, (bar_x, bar_y, _BAR_W, _BAR_H))
        filled_w = max(0, int(battalion.strength * _BAR_W))
        if filled_w > 0:
            str_color = _GREEN if battalion.strength > 0.5 else _YELLOW
            pygame.draw.rect(
                self._screen, str_color, (bar_x, bar_y, filled_w, _BAR_H)
            )

        # --- Morale indicator (below strength bar) ----------------------
        morale_y = bar_y + _BAR_H + 2
        pygame.draw.rect(
            self._screen, _GRAY, (bar_x, morale_y, _BAR_W, _BAR_H)
        )
        morale_w = max(0, int(battalion.morale * _BAR_W))
        if morale_w > 0:
            pygame.draw.rect(
                self._screen, color, (bar_x, morale_y, morale_w, _BAR_H)
            )

        # --- Routing indicator ------------------------------------------
        if battalion.routed:
            label = self._font.render("ROUTED", True, _ORANGE)
            self._screen.blit(label, (bar_x, morale_y + _BAR_H + 2))

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
    ) -> bool:
        """Render a single frame.

        Parameters
        ----------
        blue, red:
            Current :class:`~envs.sim.battalion.Battalion` states.
        terrain:
            Terrain map used for the background overlay.  On the first call
            with a non-None terrain the surface is cached; subsequent calls
            with the same terrain object are essentially free.
        step:
            Current simulation step shown in the HUD.
        info:
            Optional info dict from the last :meth:`BattalionEnv.step` call.
            Currently unused but available for future HUD extensions.

        Returns
        -------
        bool
            ``False`` if the user closed the window (quit event received),
            ``True`` otherwise.
        """
        pygame = self._pygame

        # Process window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Background / terrain
        if terrain is not None and id(terrain) != self._cached_terrain_id:
            self.set_terrain(terrain)
        if self._terrain_surface is not None:
            self._screen.blit(self._terrain_surface, (0, 0))
        else:
            self._screen.fill(_BG_COLOR)

        # Units
        self._draw_battalion(blue)
        self._draw_battalion(red)

        # HUD
        hud = self._font.render(f"Step: {step}", True, _BLACK)
        self._screen.blit(hud, (4, 4))

        pygame.display.flip()
        if self._fps > 0:
            self._clock.tick(self._fps)

        return True

    def close(self) -> None:
        """Shut down pygame and destroy the window."""
        self._pygame.quit()
