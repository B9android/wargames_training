# envs/opponents/scripted.py
"""Scripted (heuristic) opponent for BattalionEnv evaluation and curriculum.

:class:`ScriptedOpponent` provides a deterministic, difficulty-tiered
opponent that can be used:

* As the training adversary inside :class:`~envs.battalion_env.BattalionEnv`
  (Red is already scripted there — this module exposes the same logic as a
  stand-alone class for unit-testing and evaluation purposes).
* As a permanent baseline for win-rate benchmarking.

Difficulty levels
-----------------
``L1`` — stationary; fires at Blue every step.
``L2`` — advance toward Blue until within 80 % of fire range, then fire
         (matches the scripted Red inside :class:`~envs.battalion_env.BattalionEnv`).

Usage::

    import numpy as np
    from envs.battalion_env import BattalionEnv
    from envs.opponents.scripted import ScriptedOpponent

    env = BattalionEnv()
    opponent = ScriptedOpponent(level=2)
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        action = opponent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

import math

import numpy as np


class ScriptedOpponent:
    """Heuristic opponent that operates from the *Red* perspective.

    The ``act`` method receives the standard 12-dimensional BattalionEnv
    observation (Blue's perspective) and returns a 3-dimensional action
    array suitable for passing directly to ``env.step()``.

    .. note::
        Because ``BattalionEnv`` already embeds a scripted Red internally
        and advances it each step, this class is mainly useful for
        *stand-alone evaluation scripts* where you want to measure win rate
        against a scripted policy.

    Parameters
    ----------
    level:
        Difficulty level.  ``1`` = stationary+fire; ``2`` = advance+fire.
    """

    def __init__(self, level: int = 2) -> None:
        if level not in (1, 2):
            raise ValueError(f"level must be 1 or 2, got {level!r}")
        self.level = level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a 3-dimensional action given a BattalionEnv observation.

        The action is computed from the observation regardless of level —
        movement is suppressed for L1.

        Parameters
        ----------
        obs:
            12-dimensional observation vector from ``BattalionEnv``.

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` action: ``[move, rotate, fire]``.
        """
        obs = np.asarray(obs, dtype=np.float32)

        # Indices from BattalionEnv observation (Blue's perspective, so we
        # use bearing-to-red for rotation heuristic):
        #   6: normalised distance to red
        #   7: cos(bearing to red)
        #   8: sin(bearing to red)
        dist_norm: float = float(obs[6])
        cos_bearing: float = float(obs[7])
        sin_bearing: float = float(obs[8])

        # For a scripted *Blue-acting* policy the opponent rotates to face
        # the enemy (bearing = 0 means enemy is directly ahead), and fires.
        # bearing_error is the angle between current heading and the enemy.
        bearing_error = math.atan2(sin_bearing, cos_bearing)

        # Rotate command: proportional to bearing error, clamped to [-1, 1]
        rotate_cmd = float(np.clip(bearing_error / math.pi, -1.0, 1.0))

        # Fire at full intensity
        fire_cmd = 1.0

        # Move command: L1 stays still; L2 advances until within 80 % fire range
        if self.level == 1:
            move_cmd = 0.0
        else:
            # Advance if far; hold position when close (dist_norm < ~0.2)
            move_cmd = 1.0 if dist_norm > 0.2 else 0.0

        return np.array([move_cmd, rotate_cmd, fire_cmd], dtype=np.float32)

    def __repr__(self) -> str:
        return f"ScriptedOpponent(level={self.level})"
