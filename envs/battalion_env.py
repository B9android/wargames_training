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

#: Default map width in metres.
MAP_WIDTH: float = 1_000.0
#: Default map height in metres.
MAP_HEIGHT: float = 1_000.0
#: Hard episode-length cap (steps).
MAX_STEPS: int = 500
#: Strength at or below this value is treated as "unit effectively destroyed".
DESTROYED_THRESHOLD: float = 0.01
#: Simulation time step used for movement (seconds).
DT: float = 0.1

# Reward shaping coefficients
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

    The agent controls the **Blue** battalion; **Red** is a scripted
    opponent that always faces Blue, advances to within 80 % of its fire
    range, and fires at full intensity every step.

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
        Optional :class:`~envs.sim.terrain.TerrainMap`.  Defaults to a
        flat open plain.
    render_mode:
        Render mode.  Currently only ``None`` is supported.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        map_width: float = MAP_WIDTH,
        map_height: float = MAP_HEIGHT,
        max_steps: int = MAX_STEPS,
        terrain: Optional[TerrainMap] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.map_diagonal = math.sqrt(self.map_width ** 2 + self.map_height ** 2)
        self.max_steps = int(max_steps)
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

        Blue spawns in the western half of the map facing roughly east;
        Red spawns in the eastern half facing roughly west.
        """
        super().reset(seed=seed)
        rng = self.np_random

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
        # Forward/backward movement along current heading
        vx = math.cos(self.blue.theta) * move_cmd * self.blue.max_speed
        vy = math.sin(self.blue.theta) * move_cmd * self.blue.max_speed
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
        raw_r2b = compute_fire_damage(self.red, self.blue, intensity=1.0)

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
        reward = float(REWARD_STEP)
        reward += dmg_b2r * REWARD_DAMAGE_SCALE
        reward -= dmg_r2b * REWARD_PENALTY_SCALE
        if terminated:
            if red_done and not blue_done:
                reward += REWARD_WIN
            elif blue_done and not red_done:
                reward += REWARD_LOSS
            # simultaneous rout/destruction → no extra bonus/penalty

        info: dict = {
            "blue_damage_dealt": float(dmg_b2r),
            "red_damage_dealt":  float(dmg_r2b),
            "blue_routed":       self.blue_state.is_routing,
            "red_routed":        self.red_state.is_routing,
            "step_count":        self._step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

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
        """Scripted Red opponent: face Blue, advance to fire range, fire."""
        r = self.red
        b = self.blue

        dx = b.x - r.x
        dy = b.y - r.y
        target_angle = math.atan2(dy, dx)

        # Rotate toward Blue via the shortest arc
        # (Battalion.rotate clamps the delta to max_turn_rate internally)
        delta = (target_angle - r.theta + math.pi) % (2 * math.pi) - math.pi
        r.rotate(delta)

        # Advance if outside 80 % of fire range
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist > r.fire_range * 0.8:
            vx = math.cos(r.theta) * r.max_speed
            vy = math.sin(r.theta) * r.max_speed
            r.move(vx, vy, dt=DT)
            r.x = float(np.clip(r.x, 0.0, self.map_width))
            r.y = float(np.clip(r.y, 0.0, self.map_height))
