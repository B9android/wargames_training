# envs/sim/engine.py
"""Thin simulation engine: ties combat.py and terrain.py into full episodes.

A *sim episode* runs two :class:`~envs.sim.battalion.Battalion` objects
against each other using the combat and terrain modules until one side is
routed, effectively destroyed, or the step limit is reached — with no
Gymnasium or RL dependencies required.

Typical usage::

    from envs.sim.battalion import Battalion
    from envs.sim.engine import SimEngine
    import math, numpy as np

    blue = Battalion(x=300.0, y=500.0, theta=0.0,     strength=1.0, team=0)
    red  = Battalion(x=450.0, y=500.0, theta=math.pi, strength=1.0, team=1)

    result = SimEngine(blue, red, rng=np.random.default_rng(42)).run()
    print(result.winner, result.steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from envs.sim.battalion import Battalion
from envs.sim.combat import (
    CombatState,
    apply_casualties,
    compute_fire_damage,
    morale_check,
)
from envs.sim.terrain import TerrainMap

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Strength at or below this value is treated as "unit effectively destroyed".
#: Acts as a safety-net termination condition when the morale-routing path
#: is not triggered (e.g., in very short-range point-blank scenarios).
DESTROYED_THRESHOLD: float = 0.01


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Summary returned by :meth:`SimEngine.run`.

    Attributes
    ----------
    winner:
        ``0`` if blue wins, ``1`` if red wins, ``None`` for a draw (both
        sides ended simultaneously, or max steps was reached).
    steps:
        Number of simulation steps taken.
    blue_strength, red_strength:
        Final strength values in ``[0, 1]``.
    blue_morale, red_morale:
        Final morale values in ``[0, 1]``.
    blue_routed, red_routed:
        Whether each side is routing at episode end.
    """

    winner: int | None
    steps: int
    blue_strength: float
    red_strength: float
    blue_morale: float
    red_morale: float
    blue_routed: bool
    red_routed: bool


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------


class SimEngine:
    """Run a 1v1 battalion episode to completion.

    Each :meth:`step`:

    1. Resets per-step damage accumulators on both :class:`CombatState` objects.
    2. Computes fire damage from both sides simultaneously (before applying
       either, so there are no ordering effects on damage calculation).
    3. Applies terrain cover to reduce incoming damage at each unit's position.
    4. Applies casualties and updates strength.
    5. Runs a morale check for each unit, potentially triggering routing.

    The episode ends (:meth:`is_over` returns ``True``) when:

    * Either side's :class:`CombatState` reports ``is_routing = True``, or
    * Either side's strength falls to :data:`DESTROYED_THRESHOLD` or below, or
    * ``step_count`` reaches ``max_steps``.

    Parameters
    ----------
    blue, red:
        The two opposing battalions.  Modified in-place by each step.
    terrain:
        Optional :class:`~envs.sim.terrain.TerrainMap`.  Defaults to a
        flat 1 km × 1 km open plain.
    max_steps:
        Hard cap on episode length (default 500, matching acceptance
        criterion AC-1 of Epic E1.2).
    rng:
        Seeded random generator.  Defaults to a fresh unseeded generator.
        Pass a seeded generator for reproducible results.
    """

    def __init__(
        self,
        blue: Battalion,
        red: Battalion,
        terrain: Optional[TerrainMap] = None,
        max_steps: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.blue = blue
        self.red = red
        self.terrain: TerrainMap = terrain if terrain is not None else TerrainMap.flat(1000.0, 1000.0)
        self.max_steps = max_steps
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        self.blue_state = CombatState()
        self.red_state = CombatState()
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_done(self, unit: Battalion, state: CombatState) -> bool:
        """Return ``True`` if *unit* is routed or effectively destroyed."""
        return state.is_routing or unit.strength <= DESTROYED_THRESHOLD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_over(self) -> bool:
        """Return ``True`` when the episode should end."""
        return (
            self._is_done(self.blue, self.blue_state)
            or self._is_done(self.red, self.red_state)
            or self.step_count >= self.max_steps
        )

    def step(self) -> dict:
        """Advance one simulation step.

        Returns
        -------
        dict with keys:
            ``blue_damage_dealt`` – actual damage blue dealt to red this step.
            ``red_damage_dealt``  – actual damage red dealt to blue this step.
            ``blue_routing``      – whether blue is routing after this step.
            ``red_routing``       – whether red is routing after this step.
        """
        # 1. Reset per-step accumulators
        self.blue_state.reset_step_accumulators()
        self.red_state.reset_step_accumulators()

        # 2. Compute raw damages simultaneously (uses each shooter's current
        #    strength so neither side benefits from a favourable firing order)
        raw_blue_to_red = compute_fire_damage(self.blue, self.red, intensity=1.0)
        raw_red_to_blue = compute_fire_damage(self.red, self.blue, intensity=1.0)

        # 3. Apply terrain cover at each *target's* position
        raw_blue_to_red = self.terrain.apply_cover_modifier(
            self.red.x, self.red.y, raw_blue_to_red
        )
        raw_red_to_blue = self.terrain.apply_cover_modifier(
            self.blue.x, self.blue.y, raw_red_to_blue
        )

        # 4. Apply casualties simultaneously
        actual_blue_to_red = apply_casualties(self.red, self.red_state, raw_blue_to_red)
        actual_red_to_blue = apply_casualties(self.blue, self.blue_state, raw_red_to_blue)

        # Track shots fired
        self.blue_state.shots_fired += 1
        self.red_state.shots_fired += 1

        # 5. Morale checks
        blue_routing = morale_check(self.blue_state, rng=self.rng)
        red_routing = morale_check(self.red_state, rng=self.rng)

        self.step_count += 1

        return {
            "blue_damage_dealt": actual_blue_to_red,
            "red_damage_dealt": actual_red_to_blue,
            "blue_routing": blue_routing,
            "red_routing": red_routing,
        }

    def run(self) -> EpisodeResult:
        """Run the episode to completion and return a result summary."""
        while not self.is_over():
            self.step()
        return self._make_result()

    def _make_result(self) -> EpisodeResult:
        blue_done = self._is_done(self.blue, self.blue_state)
        red_done = self._is_done(self.red, self.red_state)

        if blue_done and not red_done:
            winner: int | None = 1  # red wins
        elif red_done and not blue_done:
            winner = 0  # blue wins
        else:
            winner = None  # draw: simultaneous rout/destruction or time-out

        return EpisodeResult(
            winner=winner,
            steps=self.step_count,
            blue_strength=self.blue.strength,
            red_strength=self.red.strength,
            blue_morale=self.blue_state.morale,
            red_morale=self.red_state.morale,
            blue_routed=self.blue_state.is_routing,
            red_routed=self.red_state.is_routing,
        )
