# envs/human_env.py
"""Human action input adapter for BattalionEnv.

:class:`HumanEnv` wraps :class:`~envs.battalion_env.BattalionEnv` and
provides keyboard-driven action polling so a human player can control
the Blue battalion interactively via Pygame.

Controls
--------
W / Arrow Up
    Move forward.
S / Arrow Down
    Move backward.
A / Arrow Left
    Rotate counter-clockwise (turn left).
D / Arrow Right
    Rotate clockwise (turn right).
Space
    Fire.
Escape
    Quit / end episode.

Scenarios
---------
Three built-in scenarios are registered in :data:`SCENARIOS`:

``open_field``
    Flat terrain, curriculum level 5 (full AI opponent).

``mountain_pass``
    Randomised hilly terrain with a speed penalty, curriculum level 5.

``last_stand``
    Flat terrain; Blue starts at 60% strength vs. a level-3 opponent.

Usage::

    env = HumanEnv.from_scenario("open_field", difficulty=5)
    obs, info = env.reset(seed=42)
    while True:
        action, quit_requested = env.poll_action()
        if quit_requested:
            break
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
    env.close()
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from envs.battalion_env import BattalionEnv, RedPolicy

# ---------------------------------------------------------------------------
# Built-in scenario definitions
# ---------------------------------------------------------------------------

#: Named scenario configurations for human play.
#:
#: Each entry maps a scenario name to a dict containing:
#:
#: * All keyword arguments forwarded to :class:`~envs.battalion_env.BattalionEnv`
#:   (minus the meta-keys below).
#: * ``"description"`` — human-readable description shown in the lobby.
#: * ``"initial_blue_strength"`` — Blue's starting strength after
#:   :meth:`HumanEnv.reset` (default ``1.0``).  Values < 1 simulate a
#:   weakened battalion.
SCENARIOS: dict[str, dict] = {
    "open_field": {
        "description": "Open plains — classic 1v1 engagement on flat terrain.",
        "map_width": 1000.0,
        "map_height": 1000.0,
        "max_steps": 500,
        "randomize_terrain": False,
        "hill_speed_factor": 0.5,
        "curriculum_level": 5,
        "initial_blue_strength": 1.0,
    },
    "mountain_pass": {
        "description": "Hilly terrain — navigate the hills to outflank the enemy.",
        "map_width": 1000.0,
        "map_height": 1000.0,
        "max_steps": 500,
        "randomize_terrain": True,
        "hill_speed_factor": 0.3,
        "curriculum_level": 5,
        "initial_blue_strength": 1.0,
    },
    "last_stand": {
        "description": "Last Stand — Blue starts at 60% strength. Survive!",
        "map_width": 1000.0,
        "map_height": 1000.0,
        "max_steps": 500,
        "randomize_terrain": False,
        "hill_speed_factor": 0.5,
        "curriculum_level": 3,
        "initial_blue_strength": 0.6,
    },
}

# Keys in SCENARIOS entries that are *not* valid BattalionEnv __init__ kwargs.
_SCENARIO_META_KEYS: frozenset[str] = frozenset({"description", "initial_blue_strength"})


# ---------------------------------------------------------------------------
# HumanEnv
# ---------------------------------------------------------------------------


class HumanEnv(gym.Env):
    """Human-vs-AI wrapper around :class:`~envs.battalion_env.BattalionEnv`.

    The human controls the **Blue** battalion via keyboard; the AI opponent
    drives **Red** using the scripted heuristic (selected by *difficulty*)
    or an optional pre-trained *red_policy*.

    Parameters
    ----------
    scenario:
        Name of the built-in scenario to load (must be a key in
        :data:`SCENARIOS`).
    difficulty:
        Scripted Red opponent curriculum level (1–5).  Overrides the
        ``curriculum_level`` in the scenario configuration when provided.
        Ignored when a *red_policy* is supplied.
    red_policy:
        Optional policy object for driving Red.  Must satisfy the
        :class:`~envs.battalion_env.RedPolicy` protocol (i.e., expose
        ``predict(obs, deterministic) -> (action, state)``).
    """

    metadata: dict = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: str = "open_field",
        difficulty: Optional[int] = None,
        red_policy: Optional[RedPolicy] = None,
    ) -> None:
        super().__init__()

        if scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario!r}. "
                f"Available: {sorted(SCENARIOS)}"
            )

        cfg = dict(SCENARIOS[scenario])
        self._scenario_name: str = scenario
        self._scenario_description: str = cfg.pop("description", "")
        self._initial_blue_strength: float = float(
            cfg.pop("initial_blue_strength", 1.0)
        )

        # Allow difficulty to override the scenario's curriculum_level.
        if difficulty is not None:
            cfg["curriculum_level"] = int(difficulty)

        self._env = BattalionEnv(
            render_mode="human",
            red_policy=red_policy,
            **cfg,
        )

        # Expose the inner env's spaces directly.
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = "human"

        # Quit flag; set by poll_action() or render().
        self._quit_requested: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        After delegating to the inner :class:`~envs.battalion_env.BattalionEnv`,
        applies any scenario-specific initial-state modifications (e.g. the
        reduced Blue starting strength for the *last_stand* scenario).
        """
        obs, info = self._env.reset(seed=seed, options=options)

        # Apply scenario override: set Blue's starting strength.
        if self._initial_blue_strength < 1.0 and self._env.blue is not None:
            self._env.blue.strength = self._initial_blue_strength
            # Recompute observation to reflect the modified strength.
            obs = self._env._get_obs()

        self._quit_requested = False
        info["scenario"] = self._scenario_name
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Delegate to the inner :class:`~envs.battalion_env.BattalionEnv`."""
        return self._env.step(action)

    def render(self) -> None:
        """Render the current frame.

        Uses ``skip_events=True`` on the underlying renderer because
        :meth:`poll_action` already drains the pygame event queue each
        step.  Any QUIT event that arrives *during* rendering (between
        two :meth:`poll_action` calls) is detected here and recorded in
        :attr:`_quit_requested` so that the next :meth:`poll_action` call
        can signal it to the game loop.
        """
        env = self._env
        if env.blue is None or env.red is None:
            return

        if env._renderer is None:
            from envs.rendering.renderer import BattalionRenderer  # noqa: PLC0415

            env._renderer = BattalionRenderer(env.map_width, env.map_height)

        alive = env._renderer.render_frame(
            env.blue,
            env.red,
            terrain=env.terrain,
            step=env._step_count,
            skip_events=True,
        )
        if not alive:
            self._quit_requested = True

    def close(self) -> None:
        """Clean up resources (delegates to the inner env)."""
        self._env.close()

    # ------------------------------------------------------------------
    # Human input
    # ------------------------------------------------------------------

    def poll_action(self) -> tuple[np.ndarray, bool]:
        """Poll the keyboard and return ``(action, quit_requested)``.

        Drains the pygame event queue to detect quit events (window close
        or :kbd:`Escape`), then samples the currently-held keys to build
        the continuous action vector.

        This method is designed to be called **once per game-loop
        iteration**, before :meth:`step` and :meth:`render`.

        Returns
        -------
        action : np.ndarray, shape (3,)
            ``[move, rotate, fire]`` in the ranges defined by
            :attr:`action_space`:

            * ``move``   ∈ ``[-1, 1]``: +1 = forward (W/↑), -1 = backward (S/↓)
            * ``rotate`` ∈ ``[-1, 1]``: +1 = CCW/left (A/←), -1 = CW/right (D/→)
            * ``fire``   ∈ ``[0, 1]``:  1 = fire (Space), 0 = cease fire
        quit_requested : bool
            ``True`` if the player pressed :kbd:`Escape` or closed the
            window.
        """
        # Fast path: quit already signalled (e.g. by render()).
        if self._quit_requested:
            return np.zeros(3, dtype=np.float32), True

        try:
            import pygame  # noqa: PLC0415
        except ImportError:
            # Pygame not installed — return a zero action (safe for tests).
            return np.zeros(3, dtype=np.float32), False

        # Pygame must be initialised before event/key calls are meaningful.
        if not pygame.get_init():
            return np.zeros(3, dtype=np.float32), False

        # Drain the event queue; detect quit events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit_requested = True

        if self._quit_requested:
            return np.zeros(3, dtype=np.float32), True

        # Sample continuous key state → action components.
        keys = pygame.key.get_pressed()

        move = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move = 1.0
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move = -1.0

        rotate = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            rotate = 1.0   # counter-clockwise
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            rotate = -1.0  # clockwise

        fire = 1.0 if keys[pygame.K_SPACE] else 0.0

        return np.array([move, rotate, fire], dtype=np.float32), False

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._env._step_count

    @property
    def map_width(self) -> float:
        """Map width in metres."""
        return self._env.map_width

    @property
    def map_height(self) -> float:
        """Map height in metres."""
        return self._env.map_height

    @property
    def scenario_name(self) -> str:
        """Name of the active scenario."""
        return self._scenario_name

    @property
    def scenario_description(self) -> str:
        """Human-readable description of the active scenario."""
        return self._scenario_description

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_scenario(
        cls,
        scenario: str,
        difficulty: Optional[int] = None,
        red_policy: Optional[Any] = None,
    ) -> "HumanEnv":
        """Create a :class:`HumanEnv` from a named scenario.

        Parameters
        ----------
        scenario:
            One of the keys in :data:`SCENARIOS`.
        difficulty:
            Override the scenario's default curriculum level (1–5).
        red_policy:
            Optional trained policy to drive the AI opponent.
        """
        return cls(scenario=scenario, difficulty=difficulty, red_policy=red_policy)
