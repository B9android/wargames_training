# envs/battalion_env.py
"""Gymnasium 1v1 battalion environment.

``BattalionEnv`` wraps the battalion simulation engine into a standard
``gymnasium.Env`` interface for reinforcement learning.

The agent controls the **Blue** battalion; **Red** is driven by a built-in
scripted opponent that faces Blue, advances to fire range, and fires at
full intensity every step.

Typical usage::

    import gymnasium as gym
    from envs.battalion_env import BattalionEnv

    env = BattalionEnv()
    obs, info = env.reset(seed=42)
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.reward import RewardWeights, compute_reward
from envs.sim.battalion import Battalion
from envs.sim.combat import (
    CombatState,
    apply_casualties,
    compute_fire_damage,
    morale_check,
)
from envs.sim.engine import DESTROYED_THRESHOLD
from envs.sim.terrain import TerrainMap

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default map width in metres.
MAP_WIDTH: float = 1_000.0
#: Default map height in metres.
MAP_HEIGHT: float = 1_000.0
#: Hard episode-length cap (steps).
MAX_STEPS: int = 500
#: Re-exported for convenience; authoritative value lives in envs.sim.engine.
__all__ = [
    "BattalionEnv",
    "DESTROYED_THRESHOLD",
    "MAP_WIDTH",
    "MAP_HEIGHT",
    "MAX_STEPS",
    "RewardWeights",
]
#: Simulation time step used for movement (seconds).
DT: float = 0.1

# Curriculum levels — control scripted Red opponent difficulty.
#: Number of curriculum levels supported.
NUM_CURRICULUM_LEVELS: int = 5

# Legacy reward shaping coefficients (kept for backward compatibility).
# When a RewardWeights instance is passed to BattalionEnv these are not used.
REWARD_DAMAGE_SCALE: float = 5.0    #: Reward per unit of damage dealt to Red.
REWARD_PENALTY_SCALE: float = 5.0   #: Penalty per unit of damage received by Blue.
REWARD_WIN: float = 10.0            #: Terminal bonus for routing/destroying Red.
REWARD_LOSS: float = -10.0          #: Terminal penalty for Blue routing/destroyed.
REWARD_STEP: float = -0.01          #: Per-step time penalty.


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BattalionEnv(gym.Env):
    """1v1 battalion RL environment.

    The agent controls the **Blue** battalion; **Red** is driven by a built-in
    scripted opponent whose behaviour is controlled by the *curriculum_level*
    parameter.

    Curriculum levels
    ~~~~~~~~~~~~~~~~~
    =====  =============================================
    Level  Red opponent behaviour
    =====  =============================================
    1      Stationary — Red does not move or fire.
    2      Turning only — Red faces Blue but stays put.
    3      Advance only — Red turns and advances; no fire.
    4      Soft fire — Red turns, advances, fires at 50 % intensity.
    5      Full combat — Red turns, advances, fires at 100 % intensity (default).
    =====  =============================================

    Observation space — ``Box(shape=(12,), dtype=float32)``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    =====  ===========================  =========
    Index  Feature                      Range
    =====  ===========================  =========
    0      blue x / map_width           [0, 1]
    1      blue y / map_height          [0, 1]
    2      cos(blue θ)                  [-1, 1]
    3      sin(blue θ)                  [-1, 1]
    4      blue strength                [0, 1]
    5      blue morale                  [0, 1]
    6      distance to red / diagonal   [0, 1]
    7      cos(bearing to red)          [-1, 1]
    8      sin(bearing to red)          [-1, 1]
    9      red strength                 [0, 1]
    10     red morale                   [0, 1]
    11     step / max_steps             [0, 1]
    =====  ===========================  =========

    Action space — ``Box(shape=(3,), dtype=float32)``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    =====  ===========  ========  ============================================
    Index  Name         Range     Effect
    =====  ===========  ========  ============================================
    0      move         [-1, 1]   Scale ``max_speed``; positive = forward
    1      rotate       [-1, 1]   Scale ``max_turn_rate``; positive = CCW
    2      fire         [0, 1]    Fire intensity this step
    =====  ===========  ========  ============================================

    Parameters
    ----------
    map_width, map_height:
        Map dimensions in metres (default 1 km × 1 km).
    max_steps:
        Episode length cap (default 500).
    terrain:
        Optional :class:`~envs.sim.terrain.TerrainMap`.  When supplied,
        *randomize_terrain* is forced to ``False`` and this fixed map is
        used for every episode.  Defaults to a flat open plain.
    randomize_terrain:
        When ``True`` (default) and no fixed *terrain* is supplied, a new
        procedural terrain is generated from the seeded RNG at the start
        of each episode.  Set to ``False`` to keep a static flat plain.
    hill_speed_factor:
        Movement speed multiplier applied to units on maximum-elevation
        terrain.  Must be in ``(0, 1]``.  A value of ``0.5`` means units
        on the highest hill travel at half their normal speed; ``1.0``
        disables the hill penalty entirely.
    curriculum_level:
        Scripted Red opponent difficulty (1–5).  Level 1 is the easiest
        (stationary target); level 5 is full combat.  Defaults to ``5``.
    reward_weights:
        :class:`~envs.reward.RewardWeights` instance with per-component
        multipliers.  Defaults to ``RewardWeights()`` (standard shaped
        reward with the legacy coefficients).
    render_mode:
        Render mode.  Must be ``None`` (the only currently supported value).
        Passing any other string raises ``ValueError``.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        map_width: float = MAP_WIDTH,
        map_height: float = MAP_HEIGHT,
        max_steps: int = MAX_STEPS,
        terrain: Optional[TerrainMap] = None,
        randomize_terrain: bool = True,
        hill_speed_factor: float = 0.5,
        curriculum_level: int = 5,
        reward_weights: Optional[RewardWeights] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Argument validation
        # ------------------------------------------------------------------
        if float(map_width) <= 0:
            raise ValueError(f"map_width must be positive, got {map_width}")
        if float(map_height) <= 0:
            raise ValueError(f"map_height must be positive, got {map_height}")
        if int(max_steps) < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if not (0.0 < float(hill_speed_factor) <= 1.0):
            raise ValueError(
                f"hill_speed_factor must be in (0, 1], got {hill_speed_factor}"
            )
        _curriculum_level = int(curriculum_level)
        if _curriculum_level not in range(1, NUM_CURRICULUM_LEVELS + 1):
            raise ValueError(
                f"curriculum_level must be in 1–{NUM_CURRICULUM_LEVELS}, "
                f"got {curriculum_level}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}. "
                f"Supported modes: {self.metadata['render_modes']}"
            )

        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.map_diagonal = math.sqrt(self.map_width ** 2 + self.map_height ** 2)
        self.max_steps = int(max_steps)
        self.hill_speed_factor = float(hill_speed_factor)
        self.curriculum_level = _curriculum_level
        self.reward_weights: RewardWeights = (
            reward_weights if reward_weights is not None else RewardWeights()
        )
        # When an explicit terrain is supplied, terrain randomisation is
        # disabled so the caller's fixed map is used every episode.
        self.randomize_terrain: bool = bool(randomize_terrain) and (terrain is None)
        self._supplied_terrain: Optional[TerrainMap] = terrain
        self.terrain: TerrainMap = (
            terrain if terrain is not None else TerrainMap.flat(map_width, map_height)
        )
        self.render_mode = render_mode

        # ------------------------------------------------------------------
        # Observation space — 12-dimensional, all normalised
        # ------------------------------------------------------------------
        obs_low = np.array(
            [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ------------------------------------------------------------------
        # Action space — (move, rotate, fire)
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state — populated by reset()
        self.blue: Battalion | None = None
        self.red: Battalion | None = None
        self.blue_state: CombatState | None = None
        self.red_state: CombatState | None = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial observation.

        When *randomize_terrain* is ``True`` (the default when no fixed
        terrain map is passed to ``__init__``), a new procedural terrain
        is generated from the seeded RNG on every call.  Passing the same
        *seed* therefore always produces the same terrain layout and unit
        positions.

        Blue spawns in the western half of the map facing roughly east;
        Red spawns in the eastern half facing roughly west.
        """
        super().reset(seed=seed)
        rng = self.np_random

        # Generate a fresh terrain map from the seeded RNG each episode.
        if self.randomize_terrain:
            self.terrain = TerrainMap.generate_random(
                rng=rng,
                width=self.map_width,
                height=self.map_height,
            )
        elif self._supplied_terrain is not None:
            self.terrain = self._supplied_terrain

        # Blue: western quarter, roughly eastward
        bx = float(rng.uniform(0.1 * self.map_width, 0.4 * self.map_width))
        by = float(rng.uniform(0.1 * self.map_height, 0.9 * self.map_height))
        b_theta = float(rng.uniform(-math.pi / 4, math.pi / 4))

        # Red: eastern quarter, roughly westward
        rx = float(rng.uniform(0.6 * self.map_width, 0.9 * self.map_width))
        ry = float(rng.uniform(0.1 * self.map_height, 0.9 * self.map_height))
        r_theta = float(math.pi + rng.uniform(-math.pi / 4, math.pi / 4))

        self.blue = Battalion(x=bx, y=by, theta=b_theta, strength=1.0, team=0)
        self.red = Battalion(x=rx, y=ry, theta=r_theta, strength=1.0, team=1)
        self.blue_state = CombatState()
        self.red_state = CombatState()
        self._step_count = 0

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment by one step.

        Parameters
        ----------
        action:
            Array of shape ``(3,)``: ``[move, rotate, fire]``.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        if self.blue is None or self.red is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        move_cmd   = float(np.clip(action[0], -1.0, 1.0))
        rotate_cmd = float(np.clip(action[1], -1.0, 1.0))
        fire_cmd   = float(np.clip(action[2],  0.0, 1.0))

        # --- Apply agent action to Blue ---
        # Rotation (Battalion.rotate clamps to max_turn_rate internally)
        self.blue.rotate(rotate_cmd * self.blue.max_turn_rate)
        # Forward/backward movement along current heading, slowed on hills
        speed_mod = self.terrain.get_speed_modifier(
            self.blue.x, self.blue.y, self.hill_speed_factor
        )
        vx = math.cos(self.blue.theta) * move_cmd * self.blue.max_speed * speed_mod
        vy = math.sin(self.blue.theta) * move_cmd * self.blue.max_speed * speed_mod
        self.blue.move(vx, vy, dt=DT)
        # Clamp to map bounds
        self.blue.x = float(np.clip(self.blue.x, 0.0, self.map_width))
        self.blue.y = float(np.clip(self.blue.y, 0.0, self.map_height))

        # --- Scripted Red opponent ---
        self._step_red()

        # --- Combat resolution (simultaneous) ---
        self.blue_state.reset_step_accumulators()
        self.red_state.reset_step_accumulators()

        raw_b2r = compute_fire_damage(self.blue, self.red, intensity=fire_cmd)
        raw_r2b = compute_fire_damage(self.red, self.blue, intensity=self._red_fire_intensity())

        # Apply terrain cover at each target's position
        raw_b2r = self.terrain.apply_cover_modifier(self.red.x, self.red.y, raw_b2r)
        raw_r2b = self.terrain.apply_cover_modifier(self.blue.x, self.blue.y, raw_r2b)

        # Apply casualties
        dmg_b2r = apply_casualties(self.red, self.red_state, raw_b2r)
        dmg_r2b = apply_casualties(self.blue, self.blue_state, raw_r2b)

        # Morale checks
        morale_check(self.blue_state, rng=self.np_random)
        morale_check(self.red_state, rng=self.np_random)

        # Sync Battalion flags from CombatState
        self.blue.morale = self.blue_state.morale
        self.red.morale  = self.red_state.morale
        self.blue.routed = self.blue_state.is_routing
        self.red.routed  = self.red_state.is_routing

        self._step_count += 1

        # --- Termination ---
        blue_done = (
            self.blue_state.is_routing or self.blue.strength <= DESTROYED_THRESHOLD
        )
        red_done = (
            self.red_state.is_routing or self.red.strength <= DESTROYED_THRESHOLD
        )
        terminated = blue_done or red_done
        truncated  = (not terminated) and (self._step_count >= self.max_steps)

        # --- Reward ---
        blue_won = red_done and not blue_done
        blue_lost = blue_done and not red_done
        reward_comps = compute_reward(
            dmg_b2r=dmg_b2r,
            dmg_r2b=dmg_r2b,
            blue_strength=float(self.blue.strength),
            blue_won=blue_won,
            blue_lost=blue_lost,
            weights=self.reward_weights,
        )

        info: dict = {
            "blue_damage_dealt": float(dmg_b2r),
            "red_damage_dealt":  float(dmg_r2b),
            "blue_routed":       self.blue_state.is_routing,
            "red_routed":        self.red_state.is_routing,
            "step_count":        self._step_count,
            **reward_comps.as_dict(),
        }

        return self._get_obs(), reward_comps.total, terminated, truncated, info

    def render(self) -> None:
        """Render stub — no-op for ``render_mode=None``."""

    def close(self) -> None:
        """Clean up resources (no-op)."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build and return the normalised 12-dimensional observation."""
        b = self.blue
        r = self.red

        dx = r.x - b.x
        dy = r.y - b.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        bearing = math.atan2(dy, dx)

        obs = np.array(
            [
                b.x / self.map_width,                    # [0] blue x norm
                b.y / self.map_height,                   # [1] blue y norm
                math.cos(b.theta),                       # [2] cos(blue θ)
                math.sin(b.theta),                       # [3] sin(blue θ)
                b.strength,                              # [4] blue strength
                b.morale,                                # [5] blue morale
                min(dist / self.map_diagonal, 1.0),      # [6] dist norm
                math.cos(bearing),                       # [7] cos(bearing)
                math.sin(bearing),                       # [8] sin(bearing)
                r.strength,                              # [9] red strength
                r.morale,                                # [10] red morale
                min(self._step_count / self.max_steps, 1.0),  # [11] step norm
            ],
            dtype=np.float32,
        )
        # Clip to declared bounds to guard against floating-point drift
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _step_red(self) -> None:
        """Scripted Red opponent: behaviour depends on *curriculum_level*.

        This method handles Red's **movement only**.  Red's fire is resolved
        in :meth:`step` using :meth:`_red_fire_intensity` so that damage
        computation remains centralised (simultaneous with Blue's fire).

        ========  ==========================================
        Level     Movement behaviour
        ========  ==========================================
        1         Stationary — Red does not move.
        2         Turning only — Red faces Blue; no advance.
        3–5       Red turns and advances to within 80 % of fire range.
        ========  ==========================================
        """
        level = self.curriculum_level

        # Level 1: Red stands completely still.
        if level == 1:
            return

        r = self.red
        b = self.blue

        dx = b.x - r.x
        dy = b.y - r.y
        target_angle = math.atan2(dy, dx)

        # Rotate toward Blue via the shortest arc (levels 2–5).
        delta = (target_angle - r.theta + math.pi) % (2 * math.pi) - math.pi
        r.rotate(delta)

        # Level 2: turn only, no advance.
        if level == 2:
            return

        # Advance if outside 80 % of fire range (levels 3–5).
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist > r.fire_range * 0.8:
            speed_mod = self.terrain.get_speed_modifier(
                r.x, r.y, self.hill_speed_factor
            )
            vx = math.cos(r.theta) * r.max_speed * speed_mod
            vy = math.sin(r.theta) * r.max_speed * speed_mod
            r.move(vx, vy, dt=DT)
            r.x = float(np.clip(r.x, 0.0, self.map_width))
            r.y = float(np.clip(r.y, 0.0, self.map_height))

    def _red_fire_intensity(self) -> float:
        """Return the fire intensity Red uses this step, based on curriculum level.

        ========  ==========================
        Level     Red fire intensity
        ========  ==========================
        1–3       0.0  (Red does not fire)
        4         0.5  (50 % intensity)
        5         1.0  (full intensity)
        ========  ==========================
        """
        level = self.curriculum_level
        if level <= 3:
            return 0.0
        if level == 4:
            return 0.5
        return 1.0
