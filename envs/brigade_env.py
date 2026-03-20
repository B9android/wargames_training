"""Brigade Commander Environment — high-level MDP for hierarchical RL.

:class:`BrigadeEnv` is a :class:`gymnasium.Env` that sits above the battalion
simulation and implements the *options* layer of the HRL hierarchy described
in the E3.2 epic.

Architecture
------------
::

    BrigadeEnv (Gymnasium)           ← brigade PPO agent lives here
        │
        └─ MultiBattalionEnv         ← primitive continuous actions
              (PettingZoo ParallelEnv)

At every **macro-step** the brigade commander selects one option from the
six-element vocabulary (see :mod:`envs.options`) for **each** Blue battalion.
The :class:`BrigadeEnv` then runs the option execution loop internally,
accumulating rewards across primitive steps until every selected option
terminates (option condition fires, hard cap reached, or underlying env ends).

Observation space
-----------------
``Box(shape=(obs_dim,), dtype=float32)``  where
``obs_dim = 3 + 7 * n_blue + 1``:

=========================  =============================================  =========
Slice                      Feature                                        Range
=========================  =============================================  =========
``[0:3]``                  Sector control (3 equal vertical strips)       ``[0, 1]``
                           ``sector_control[s]`` = blue strength          
                           / (blue + red strength) in sector *s*.         
                           0.5 when no units occupy the sector.           
``[3 : 3+2*n_blue]``       Per-blue strength + morale                     ``[0, 1]``
                           ``[str_0, mor_0, str_1, mor_1, …]``            
``[3+2*nb : 3+7*nb]``      Per-blue enemy threat vector (5 per battalion) mixed
                           ``[dist/diag, cos_bear, sin_bear, e_str,``     
                           ``e_mor]`` — nearest alive red battalion.      
                           Sentinel ``[1,0,0,0,0]`` when no enemy alive.  
``[-1]``                   Step progress: ``step / max_steps``            ``[0, 1]``
=========================  =============================================  =========

Action space
------------
``MultiDiscrete([n_options] * n_blue)``

Each element selects a macro-action index ``[0, n_options)`` for the
corresponding Blue battalion.  The six standard options are defined in
:class:`~envs.options.MacroAction`.

Frozen battalion policy
-----------------------
Pass a loaded :class:`~models.mappo_policy.MAPPOPolicy` to the constructor
(or call :meth:`set_battalion_policy` afterwards) to drive Red agents with a
frozen v2 checkpoint.  All parameters on the policy are kept with
``requires_grad=False`` so no gradients flow back through it during brigade
training.

Typical usage::

    from envs.brigade_env import BrigadeEnv

    env = BrigadeEnv(n_blue=2, n_red=2)
    obs, _ = env.reset(seed=42)
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces
import gymnasium as gym

from envs.multi_battalion_env import MultiBattalionEnv, MAP_WIDTH, MAP_HEIGHT, MAX_STEPS
from envs.options import Option, make_default_options

__all__ = ["BrigadeEnv", "BRIGADE_OBS_DIM"]

# Number of map sectors (vertical strips along the x-axis)
N_SECTORS: int = 3
# Primitive action dimension: [move, rotate, fire]
_PRIM_ACT_DIM: int = 3


def _brigade_obs_dim(n_blue: int) -> int:
    """Return the flat observation dimension for a brigade of *n_blue* battalions."""
    # sector_control(3) + per_blue(strength + morale)(2*n_blue)
    # + per_blue threat_vector(5*n_blue) + step_progress(1)
    return N_SECTORS + 7 * n_blue + 1


#: Public constant for the default 2-battalion brigade observation size.
BRIGADE_OBS_DIM: int = _brigade_obs_dim(2)


class BrigadeEnv(gym.Env):
    """Gymnasium environment for a brigade-level HRL commander.

    Parameters
    ----------
    n_blue:
        Number of Blue battalions controlled by the brigade.
    n_red:
        Number of Red opponent battalions.
    map_width:
        Map width in metres.
    map_height:
        Map height in metres.
    max_steps:
        Maximum primitive-step episode length.
    options:
        Option vocabulary.  ``None`` uses :func:`~envs.options.make_default_options`.
        When ``None``, the vocabulary is built using ``temporal_ratio`` as the
        option ``max_steps`` cap.
    temporal_ratio:
        Number of primitive battalion steps per brigade macro-step (option
        duration cap).  Ignored when an explicit ``options`` list is supplied.
        Must be ``>= 1``.  Corresponds to the hyperparameter swept in E3.5.
    battalion_policy:
        Optional frozen :class:`~models.mappo_policy.MAPPOPolicy` used to
        drive Red agents.  All parameters are detached (``requires_grad=False``).
        When ``None``, Red agents are stationary (zero primitive actions).
    red_random:
        When ``True`` and no ``battalion_policy`` is set, Red agents take
        random primitive actions.  Ignored when ``battalion_policy`` is set.
    randomize_terrain:
        Pass-through to :class:`~envs.multi_battalion_env.MultiBattalionEnv`.
    visibility_radius:
        Pass-through to :class:`~envs.multi_battalion_env.MultiBattalionEnv`.
    render_mode:
        ``None`` or ``"human"`` — delegated to the inner env.
    """

    metadata: dict = {"render_modes": ["human"], "name": "brigade_v0"}

    def __init__(
        self,
        n_blue: int = 2,
        n_red: int = 2,
        map_width: float = MAP_WIDTH,
        map_height: float = MAP_HEIGHT,
        max_steps: int = MAX_STEPS,
        options: Optional[list[Option]] = None,
        temporal_ratio: int = 10,
        battalion_policy=None,
        red_random: bool = False,
        randomize_terrain: bool = True,
        visibility_radius: float = 600.0,
        render_mode: Optional[str] = None,
    ) -> None:
        if int(n_blue) < 1:
            raise ValueError(f"n_blue must be >= 1, got {n_blue}")
        if int(n_red) < 1:
            raise ValueError(f"n_red must be >= 1, got {n_red}")
        if int(temporal_ratio) < 1:
            raise ValueError(f"temporal_ratio must be >= 1, got {temporal_ratio}")

        self.n_blue = int(n_blue)
        self.n_red = int(n_red)
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.map_diagonal = math.hypot(self.map_width, self.map_height)
        self.max_steps = int(max_steps)
        self.red_random = bool(red_random)
        self.render_mode = render_mode
        self.temporal_ratio: int = int(temporal_ratio)

        # Option vocabulary — use temporal_ratio as max_steps when no custom
        # options are provided so the hyperparameter takes effect.
        self._options: list[Option] = (
            list(options) if options is not None
            else make_default_options(max_steps=self.temporal_ratio)
        )
        if len(self._options) == 0:
            raise ValueError("options must contain at least one Option.")
        self.n_options: int = len(self._options)

        # ── Action space ──────────────────────────────────────────────────
        # One option index per blue battalion
        self.action_space = spaces.MultiDiscrete(
            [self.n_options] * self.n_blue, dtype=np.int64
        )

        # ── Observation space ─────────────────────────────────────────────
        self._obs_dim: int = _brigade_obs_dim(self.n_blue)
        obs_low, obs_high = self._build_obs_bounds()
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ── Inner environment ─────────────────────────────────────────────
        self._inner = MultiBattalionEnv(
            n_blue=self.n_blue,
            n_red=self.n_red,
            map_width=self.map_width,
            map_height=self.map_height,
            max_steps=self.max_steps,
            randomize_terrain=randomize_terrain,
            visibility_radius=visibility_radius,
            render_mode=render_mode,
        )

        # ── Frozen battalion policy (optional) ────────────────────────────
        self._battalion_policy = None
        self._policy_device: str = "cpu"
        if battalion_policy is not None:
            self.set_battalion_policy(battalion_policy)

        # ── Episode state (populated by reset()) ─────────────────────────
        self._last_obs: dict[str, np.ndarray] = {}
        self._prim_steps: int = 0
        self._macro_steps: int = 0

        # ── Red option overrides (set externally by DivisionEnv) ──────────
        # Maps red agent_id → option index.  When non-empty, _get_red_action
        # executes the corresponding Option primitive policy instead of the
        # default battalion-policy / random / zero behaviour.
        self._forced_red_options: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Observation bounds
    # ------------------------------------------------------------------

    def _build_obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(obs_low, obs_high)`` arrays for the observation space."""
        lows: list[float] = []
        highs: list[float] = []

        # Sector control: [0, 1] × 3
        lows.extend([0.0] * N_SECTORS)
        highs.extend([1.0] * N_SECTORS)

        # Per-blue battalion strength + morale: [0, 1] each
        for _ in range(self.n_blue):
            lows.extend([0.0, 0.0])
            highs.extend([1.0, 1.0])

        # Per-blue threat vector: [dist, cos, sin, e_str, e_mor]
        for _ in range(self.n_blue):
            # dist / map_diagonal in [0, 1]
            lows.append(0.0)
            highs.append(1.0)
            # cos(bearing) in [-1, 1]
            lows.append(-1.0)
            highs.append(1.0)
            # sin(bearing) in [-1, 1]
            lows.append(-1.0)
            highs.append(1.0)
            # enemy strength in [0, 1]
            lows.append(0.0)
            highs.append(1.0)
            # enemy morale in [0, 1]
            lows.append(0.0)
            highs.append(1.0)

        # Step progress: [0, 1]
        lows.append(0.0)
        highs.append(1.0)

        return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Frozen battalion policy
    # ------------------------------------------------------------------

    def set_battalion_policy(self, policy) -> None:
        """Set (or clear) the frozen policy used to drive Red agents.

        When a policy is supplied its parameters are frozen
        (``requires_grad=False``) and placed in evaluation mode so
        no gradients flow through it during brigade training.

        Parameters
        ----------
        policy:
            A :class:`~models.mappo_policy.MAPPOPolicy` instance, or
            ``None`` to revert to the default stationary / random Red
            behaviour.
        """
        if policy is None:
            self._battalion_policy = None
            self._policy_device = "cpu"
            return

        # Freeze all parameters
        for param in policy.parameters():
            param.requires_grad_(False)
        policy.eval()
        self._battalion_policy = policy
        # Store the device so _get_red_action can move obs tensors to it
        try:
            self._policy_device = next(policy.parameters()).device.type
        except StopIteration:
            self._policy_device = "cpu"

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial brigade observation.

        Parameters
        ----------
        seed:
            RNG seed forwarded to the inner :class:`~envs.multi_battalion_env.MultiBattalionEnv`.
        options:
            Unused; present for Gymnasium API compatibility.

        Returns
        -------
        obs : np.ndarray of shape ``(obs_dim,)``
        info : dict
        """
        if seed is not None:
            super().reset(seed=seed)

        inner_obs, _ = self._inner.reset(seed=seed, options=options)
        self._last_obs = dict(inner_obs)
        self._prim_steps = 0
        self._macro_steps = 0
        # Cleared here (between episodes) and also after each step by DivisionEnv
        # to prevent stale commands leaking into subsequent macro-steps.
        self._forced_red_options = {}
        return self._get_brigade_obs(), {}

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(
        self,
        brigade_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one macro-step: dispatch options to Blue battalions.

        For each alive Blue battalion, the selected option runs for multiple
        primitive steps until the option terminates or the underlying
        episode ends.

        Parameters
        ----------
        brigade_action:
            Array of shape ``(n_blue,)`` with option indices.  Indices for
            dead battalions are ignored.

        Returns
        -------
        obs : np.ndarray — brigade observation after the macro-step
        reward : float — mean reward across all Blue battalions
        terminated : bool — True when the episode ended naturally
        truncated : bool — True when the episode was cut short
        info : dict — metadata including primitive step count and option names
        """
        brigade_action = np.asarray(brigade_action, dtype=np.int64)

        # Validate action shape
        if brigade_action.shape != (self.n_blue,):
            raise ValueError(
                f"brigade_action has shape {brigade_action.shape!r}, "
                f"expected ({self.n_blue},)."
            )

        # Active blue agents at the start of this macro-step
        current_blue = [
            f"blue_{i}" for i in range(self.n_blue)
            if f"blue_{i}" in self._inner.agents
        ]

        if not current_blue:
            # All Blue battalions already dead — episode over
            obs = self._get_brigade_obs()
            return obs, 0.0, True, False, {}

        # ── Dispatch options ──────────────────────────────────────────
        selected_options: dict[str, Option] = {}
        option_steps: dict[str, int] = {}
        option_names: dict[str, str] = {}
        for i in range(self.n_blue):
            agent_id = f"blue_{i}"
            if agent_id in current_blue:
                idx = int(brigade_action[i])
                if idx < 0 or idx >= self.n_options:
                    raise ValueError(
                        f"Invalid macro-action index {idx!r} for battalion {agent_id!r}; "
                        f"expected integer in [0, {self.n_options - 1}]."
                    )
                selected_options[agent_id] = self._options[idx]
                option_steps[agent_id] = 0
                option_names[agent_id] = self._options[idx].name

        option_done: dict[str, bool] = {a: False for a in current_blue}
        agg_rewards: dict[str, float] = {a: 0.0 for a in current_blue}
        ep_terminated: dict[str, bool] = {a: False for a in current_blue}
        ep_truncated: dict[str, bool] = {a: False for a in current_blue}
        # Track whether the inner env issued any truncation during this macro-step
        any_inner_truncated: bool = False

        # ── Inner primitive-step loop ─────────────────────────────────
        while any(not option_done[a] for a in current_blue):
            if not self._inner.agents:
                for a in current_blue:
                    option_done[a] = True
                break

            # Build primitive actions for all alive agents
            prim_actions: dict[str, np.ndarray] = {}
            for agent in self._inner.agents:
                if agent.startswith("blue_"):
                    if agent in current_blue and not option_done[agent]:
                        prim_actions[agent] = selected_options[agent].get_action(
                            self._last_obs[agent]
                        )
                    else:
                        prim_actions[agent] = np.zeros(_PRIM_ACT_DIM, dtype=np.float32)
                else:
                    # Red agent: driven by battalion policy, random, or zero
                    prim_actions[agent] = self._get_red_action(agent)

            # Primitive step
            obs, rewards, terminated, truncated, _ = self._inner.step(prim_actions)
            self._prim_steps += 1

            # Update latest observations
            for agent, ob in obs.items():
                self._last_obs[agent] = ob

            # Update option tracking — always record env-level
            # termination/truncation regardless of option_done state
            for agent in current_blue:
                env_term = bool(terminated.get(agent, False))
                env_trunc = bool(truncated.get(agent, False))

                if env_trunc:
                    any_inner_truncated = True
                # Always update episode-level flags
                if env_term:
                    ep_terminated[agent] = True
                if env_trunc:
                    ep_truncated[agent] = True

                # Skip option bookkeeping once the option is done
                if option_done[agent]:
                    continue

                agg_rewards[agent] += float(rewards.get(agent, 0.0))

                if env_term or env_trunc:
                    option_done[agent] = True
                else:
                    option_steps[agent] += 1
                    if selected_options[agent].should_terminate(
                        self._last_obs[agent], option_steps[agent]
                    ):
                        option_done[agent] = True

        self._macro_steps += 1

        # ── Episode termination ───────────────────────────────────────
        blue_alive = [
            f"blue_{i}" for i in range(self.n_blue)
            if f"blue_{i}" in self._inner._alive
        ]
        red_alive = [
            f"red_{i}" for i in range(self.n_red)
            if f"red_{i}" in self._inner._alive
        ]
        blue_wiped = len(blue_alive) == 0
        red_wiped = len(red_alive) == 0

        # Decisive combat outcome → terminated; time limit without decisive outcome → truncated
        if blue_wiped or red_wiped:
            episode_terminated = True
            episode_truncated = False
        elif any_inner_truncated:
            episode_terminated = False
            episode_truncated = True
        else:
            episode_terminated = False
            episode_truncated = False

        # ── Brigade reward ────────────────────────────────────────────
        reward_vals = [agg_rewards[a] for a in current_blue]
        brigade_reward = float(np.mean(reward_vals)) if reward_vals else 0.0

        # ── Info dict ─────────────────────────────────────────────────
        info: dict = {
            "macro_steps": self._macro_steps,
            "primitive_steps": self._prim_steps,
            "option_names": option_names,
            "option_steps": {a: option_steps.get(a, 0) for a in current_blue},
            "blue_rewards": {a: agg_rewards[a] for a in current_blue},
        }
        if episode_terminated or episode_truncated:
            if red_wiped and not blue_wiped:
                info["winner"] = "blue"
            elif blue_wiped and not red_wiped:
                info["winner"] = "red"
            else:
                info["winner"] = "draw"

        return (
            self._get_brigade_obs(),
            brigade_reward,
            episode_terminated,
            episode_truncated,
            info,
        )

    # ------------------------------------------------------------------
    # Brigade observation construction
    # ------------------------------------------------------------------

    def _get_brigade_obs(self) -> np.ndarray:
        """Build and return the normalised brigade observation vector."""
        parts: list[float] = []

        # ── 1. Sector control (3 vertical strips) ────────────────────
        sector_width = self.map_width / N_SECTORS
        for s in range(N_SECTORS):
            x_lo = s * sector_width
            x_hi = (s + 1) * sector_width
            blue_str = 0.0
            red_str = 0.0
            for agent_id, b in self._inner._battalions.items():
                if agent_id not in self._inner._alive:
                    continue
                if x_lo <= b.x < x_hi or (s == N_SECTORS - 1 and b.x == self.map_width):
                    if agent_id.startswith("blue_"):
                        blue_str += float(b.strength)
                    else:
                        red_str += float(b.strength)
            total = blue_str + red_str
            parts.append(blue_str / total if total > 0.0 else 0.5)

        # ── 2. Per-blue battalion strength + morale ───────────────────
        for i in range(self.n_blue):
            agent_id = f"blue_{i}"
            if agent_id in self._inner._battalions and agent_id in self._inner._alive:
                b = self._inner._battalions[agent_id]
                parts.append(float(b.strength))
                parts.append(float(b.morale))
            else:
                parts.extend([0.0, 0.0])

        # ── 3. Per-blue enemy threat vector ───────────────────────────
        # Alive red battalions
        alive_red = [
            (r_id, self._inner._battalions[r_id])
            for r_id in self._inner._alive
            if r_id.startswith("red_") and r_id in self._inner._battalions
        ]

        for i in range(self.n_blue):
            agent_id = f"blue_{i}"
            if agent_id not in self._inner._alive or agent_id not in self._inner._battalions:
                # Dead battalion — sentinel threat
                parts.extend([1.0, 0.0, 0.0, 0.0, 0.0])
                continue

            b = self._inner._battalions[agent_id]

            if not alive_red:
                # No enemies — maximum distance sentinel
                parts.extend([1.0, 0.0, 0.0, 0.0, 0.0])
                continue

            # Find nearest alive red battalion
            best_dist = float("inf")
            best_r = None
            for r_id, r_bat in alive_red:
                dx = r_bat.x - b.x
                dy = r_bat.y - b.y
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_dist:
                    best_dist = d
                    best_r = r_bat

            assert best_r is not None
            dx = best_r.x - b.x
            dy = best_r.y - b.y
            bearing = math.atan2(dy, dx)

            parts.append(min(best_dist / self.map_diagonal, 1.0))
            parts.append(math.cos(bearing))
            parts.append(math.sin(bearing))
            parts.append(float(best_r.strength))
            parts.append(float(best_r.morale))

        # ── 4. Step progress ─────────────────────────────────────────
        parts.append(min(self._inner._step_count / self.max_steps, 1.0))

        obs = np.array(parts, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    # Red action helper
    # ------------------------------------------------------------------

    def _get_red_action(self, agent_id: str) -> np.ndarray:
        """Return a primitive action for a Red agent.

        Priority:
        1. :attr:`_forced_red_options` (set by DivisionEnv) — execute the option's
           primitive policy directly.
        2. :attr:`_battalion_policy` (frozen MAPPOPolicy) — if set.
        3. Random primitive action — when ``red_random=True``.
        4. Zero (stationary) action — default.
        """
        if self._forced_red_options and agent_id in self._forced_red_options:
            opt_idx = int(self._forced_red_options[agent_id])
            if opt_idx < 0 or opt_idx >= self.n_options:
                raise ValueError(
                    f"Invalid forced option index {opt_idx!r} for Red agent {agent_id!r}; "
                    f"expected integer in [0, {self.n_options - 1}]."
                )
            obs = self._last_obs.get(
                agent_id, np.zeros(self._inner._obs_dim, dtype=np.float32)
            )
            return self._options[opt_idx].get_action(obs)

        if self._battalion_policy is not None:
            obs = self._last_obs.get(
                agent_id, np.zeros(self._inner._obs_dim, dtype=np.float32)
            )
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._policy_device)
            # Infer agent index from the agent_id
            try:
                agent_idx = int(agent_id.split("_")[1]) % self._battalion_policy.n_agents
            except (IndexError, ValueError):
                agent_idx = 0
            with torch.no_grad():
                acts_t, _ = self._battalion_policy.act(
                    obs_t, agent_idx=agent_idx, deterministic=False
                )
            act = acts_t[0].cpu().numpy()
            act_low = self._inner._act_space.low
            act_high = self._inner._act_space.high
            return np.clip(act, act_low, act_high).astype(np.float32)

        if self.red_random:
            return self._inner.action_space(agent_id).sample()

        return np.zeros(_PRIM_ACT_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API: render / close
    # ------------------------------------------------------------------

    def render(self):
        """Delegate rendering to the inner environment."""
        return self._inner.render()

    def close(self) -> None:
        """Delegate cleanup to the inner environment."""
        self._inner.close()
