"""Division Commander Environment — top-level MDP for three-echelon HRL.

:class:`DivisionEnv` is a :class:`gymnasium.Env` that sits above the brigade
layer and implements the *division* echelon of the HRL hierarchy.

Architecture
------------
::

    DivisionEnv (Gymnasium)                ← division PPO agent lives here
        │
        └─ BrigadeEnv (Gymnasium)          ← brigade command dispatcher
              │
              └─ MultiBattalionEnv         ← primitive continuous actions
                    (PettingZoo ParallelEnv)

At every **division macro-step** the division commander selects one
*operational command* from a six-element vocabulary for **each** Blue brigade.
The :class:`DivisionEnv` translates this command into a flat brigade-level
action (option indices for every Blue battalion in the brigade) and delegates
execution to the wrapped :class:`~envs.brigade_env.BrigadeEnv`.

Blue brigades are formed by grouping consecutive Blue battalions:
brigade *i* = battalions ``[i*n_blue_per_brigade … (i+1)*n_blue_per_brigade)``.
Red brigades follow the same grouping for the Red side.

Observation space
-----------------
``Box(shape=(obs_dim,), dtype=float32)``  where
``obs_dim = N_THEATRE_SECTORS + 8 * n_brigades + 1``:

=============================  ================================================  =========
Slice                          Feature                                           Range
=============================  ================================================  =========
``[0 : N_THEATRE_SECTORS]``    Theatre sector control (5 vertical strips)        ``[0, 1]``
                               ``sector_control[s]`` = blue strength             
                               / (blue + red strength) in sector *s*.            
                               0.5 when no units occupy the sector.              
``[5 : 5+3*n_brigades]``       Per-brigade status (3 per brigade)                ``[0, 1]``
                               ``[avg_strength, avg_morale, alive_ratio]``       
                               Zeros/0 for fully destroyed brigades.             
``[5+3*nb : 5+8*nb]``          Per-brigade threat vector (5 per brigade)         mixed
                               ``[dist/diag, cos_bear, sin_bear, e_avg_str,``    
                               ``e_avg_mor]`` — nearest alive Red brigade        
                               centroid. Sentinel ``[1,0,0,0,0]`` when none.     
``[-1]``                       Step progress: ``step / max_steps``               ``[0, 1]``
=============================  ================================================  =========

Action space
------------
``MultiDiscrete([n_div_options] * n_brigades)``

Each element selects an operational-command index ``[0, n_div_options)`` for
the corresponding Blue brigade.  Operational commands map 1-to-1 to the six
brigade macro-actions:

=====  =======================  ===============================================
Index  Division command          Translated brigade option
=====  =======================  ===============================================
0      ``advance_theatre``       ``advance_sector``   (0)
1      ``hold_position``         ``defend_position``  (1)
2      ``envelop_left``          ``flank_left``        (2)
3      ``envelop_right``         ``flank_right``       (3)
4      ``strategic_withdrawal``  ``withdraw``          (4)
5      ``mass_fires``            ``concentrate_fire``  (5)
=====  =======================  ===============================================

Frozen brigade policy
---------------------
Pass a loaded **SB3 PPO** brigade checkpoint (or any callable with a
``predict(obs, deterministic)`` method) to the constructor as
``brigade_policy`` to drive Red brigades with a frozen policy.

The frozen policy must have been trained on a :class:`~envs.brigade_env.BrigadeEnv`
of the same Red-side size (``n_red`` total battalions).  It receives a
:class:`~envs.brigade_env.BrigadeEnv`-compatible observation for the Red side
(shape ``3 + 7 * n_red + 1``, treating Red battalions as the "blue" side) and
returns a per-battalion action of shape ``(n_red,)`` with option indices in
``[0, n_options)``.  These are injected into the inner
:class:`~envs.brigade_env.BrigadeEnv` via
:attr:`~envs.brigade_env.BrigadeEnv._forced_red_options`.

All parameters on the policy should be kept with ``requires_grad=False``
so no gradients flow back through it during division training.

Typical usage::

    from envs.division_env import DivisionEnv

    env = DivisionEnv(n_brigades=2, n_blue_per_brigade=2, n_red_brigades=2)
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
from gymnasium import spaces
import gymnasium as gym

from envs.brigade_env import BrigadeEnv, N_SECTORS as _BRIGADE_N_SECTORS
from envs.multi_battalion_env import MAP_WIDTH, MAP_HEIGHT, MAX_STEPS

__all__ = ["DivisionEnv", "DIVISION_OBS_DIM", "N_THEATRE_SECTORS"]

# Number of theatre-level sectors (5 vertical strips for division-scale)
N_THEATRE_SECTORS: int = 5
# Floats per brigade in the status block
_BRIGADE_STATUS_DIM: int = 3   # avg_strength, avg_morale, alive_ratio
# Floats per brigade in the threat block
_BRIGADE_THREAT_DIM: int = 5   # dist, cos_bear, sin_bear, e_avg_str, e_avg_mor


def _division_obs_dim(n_brigades: int) -> int:
    """Return the flat observation dimension for a division of *n_brigades*."""
    # theatre_sectors(5) + per_brigade_status(3*nb) + per_brigade_threat(5*nb) + step(1)
    return N_THEATRE_SECTORS + (_BRIGADE_STATUS_DIM + _BRIGADE_THREAT_DIM) * n_brigades + 1


#: Public constant for the default 2-brigade division observation size.
DIVISION_OBS_DIM: int = _division_obs_dim(2)


class DivisionEnv(gym.Env):
    """Gymnasium environment for a division-level HRL commander.

    Parameters
    ----------
    n_brigades:
        Number of Blue brigades.  Each brigade is a group of
        *n_blue_per_brigade* battalions.
    n_blue_per_brigade:
        Number of Blue battalions per brigade.
    n_red_brigades:
        Number of Red brigades.  Defaults to *n_brigades*.
    n_red_per_brigade:
        Number of Red battalions per Red brigade.
        Defaults to *n_blue_per_brigade*.
    map_width:
        Map width in metres (passed through to inner env).
    map_height:
        Map height in metres (passed through to inner env).
    max_steps:
        Maximum primitive-step episode length.
    brigade_policy:
        Optional frozen brigade-level policy for Red brigades.
        Must expose ``predict(obs, deterministic) -> (action, state)``.
        When provided all its parameters should have ``requires_grad=False``.
    red_random:
        When ``True`` and no *brigade_policy* is set, Red battalions take
        random primitive actions.  Ignored when *brigade_policy* is set.
    randomize_terrain:
        Pass-through to :class:`~envs.brigade_env.BrigadeEnv`.
    visibility_radius:
        Fog-of-war visibility radius in metres.
    render_mode:
        ``None`` or ``"human"`` — delegated to the inner env.
    """

    metadata: dict = {"render_modes": ["human"], "name": "division_v0"}

    def __init__(
        self,
        n_brigades: int = 2,
        n_blue_per_brigade: int = 2,
        n_red_brigades: Optional[int] = None,
        n_red_per_brigade: Optional[int] = None,
        map_width: float = MAP_WIDTH,
        map_height: float = MAP_HEIGHT,
        max_steps: int = MAX_STEPS,
        brigade_policy=None,
        red_random: bool = False,
        randomize_terrain: bool = True,
        visibility_radius: float = 600.0,
        render_mode: Optional[str] = None,
    ) -> None:
        if int(n_brigades) < 1:
            raise ValueError(f"n_brigades must be >= 1, got {n_brigades}")
        if int(n_blue_per_brigade) < 1:
            raise ValueError(f"n_blue_per_brigade must be >= 1, got {n_blue_per_brigade}")

        self.n_brigades: int = int(n_brigades)
        self.n_blue_per_brigade: int = int(n_blue_per_brigade)
        self.n_red_brigades: int = int(n_brigades if n_red_brigades is None else n_red_brigades)
        self.n_red_per_brigade: int = int(
            n_blue_per_brigade if n_red_per_brigade is None else n_red_per_brigade
        )

        if self.n_red_brigades < 1:
            raise ValueError(f"n_red_brigades must be >= 1, got {self.n_red_brigades}")
        if self.n_red_per_brigade < 1:
            raise ValueError(f"n_red_per_brigade must be >= 1, got {self.n_red_per_brigade}")

        # Total battalion counts
        self.n_blue: int = self.n_brigades * self.n_blue_per_brigade
        self.n_red: int = self.n_red_brigades * self.n_red_per_brigade

        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.map_diagonal = math.hypot(self.map_width, self.map_height)
        self.max_steps = int(max_steps)
        self.red_random = bool(red_random)
        self.render_mode = render_mode

        # ── Inner BrigadeEnv ─────────────────────────────────────────────
        self._brigade = BrigadeEnv(
            n_blue=self.n_blue,
            n_red=self.n_red,
            map_width=self.map_width,
            map_height=self.map_height,
            max_steps=self.max_steps,
            red_random=red_random,
            randomize_terrain=randomize_terrain,
            visibility_radius=visibility_radius,
            render_mode=render_mode,
        )

        # n_div_options matches the brigade option count
        self.n_div_options: int = self._brigade.n_options

        # ── Action space ────────────────────────────────────────────────
        # One operational command per Blue brigade
        self.action_space = spaces.MultiDiscrete(
            [self.n_div_options] * self.n_brigades, dtype=np.int64
        )

        # ── Observation space ───────────────────────────────────────────
        self._obs_dim: int = _division_obs_dim(self.n_brigades)
        obs_low, obs_high = self._build_obs_bounds()
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ── Frozen brigade policy for Red (optional) ────────────────────
        self._red_brigade_policy = None
        if brigade_policy is not None:
            self.set_brigade_policy(brigade_policy)

        # ── Episode state ────────────────────────────────────────────────
        self._div_steps: int = 0

    # ------------------------------------------------------------------
    # Observation bounds
    # ------------------------------------------------------------------

    def _build_obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(obs_low, obs_high)`` arrays for the observation space."""
        lows: list[float] = []
        highs: list[float] = []

        # Theatre sector control: [0, 1] × N_THEATRE_SECTORS
        lows.extend([0.0] * N_THEATRE_SECTORS)
        highs.extend([1.0] * N_THEATRE_SECTORS)

        # Per-brigade status: [avg_strength, avg_morale, alive_ratio]
        for _ in range(self.n_brigades):
            lows.extend([0.0, 0.0, 0.0])
            highs.extend([1.0, 1.0, 1.0])

        # Per-brigade threat vector: [dist, cos, sin, e_str, e_mor]
        for _ in range(self.n_brigades):
            lows.append(0.0)    # dist / diagonal
            highs.append(1.0)
            lows.append(-1.0)   # cos(bearing)
            highs.append(1.0)
            lows.append(-1.0)   # sin(bearing)
            highs.append(1.0)
            lows.append(0.0)    # enemy avg_strength
            highs.append(1.0)
            lows.append(0.0)    # enemy avg_morale
            highs.append(1.0)

        # Step progress: [0, 1]
        lows.append(0.0)
        highs.append(1.0)

        return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Frozen brigade policy
    # ------------------------------------------------------------------

    def set_brigade_policy(self, policy) -> None:
        """Set (or clear) the frozen brigade policy for Red.

        When *policy* is supplied any PyTorch parameters are frozen
        (``requires_grad=False``) and placed in evaluation mode.

        Parameters
        ----------
        policy:
            An object with a ``predict(obs, deterministic)`` method
            (e.g. an SB3 :class:`~stable_baselines3.PPO` model), or
            ``None`` to revert to the default Red behaviour.
        """
        if policy is None:
            self._red_brigade_policy = None
            return

        # Freeze parameters if this is a PyTorch module
        if hasattr(policy, "parameters"):
            for param in policy.parameters():
                param.requires_grad_(False)
        if hasattr(policy, "eval"):
            policy.eval()

        # Freeze SB3 policy networks if accessible
        if hasattr(policy, "policy") and hasattr(policy.policy, "parameters"):
            for param in policy.policy.parameters():
                param.requires_grad_(False)

        self._red_brigade_policy = policy

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial division observation.

        Parameters
        ----------
        seed:
            RNG seed forwarded to the inner :class:`~envs.brigade_env.BrigadeEnv`.
        options:
            Unused; present for Gymnasium API compatibility.

        Returns
        -------
        obs : np.ndarray of shape ``(obs_dim,)``
        info : dict
        """
        if seed is not None:
            super().reset(seed=seed)

        self._brigade.reset(seed=seed, options=options)
        self._div_steps = 0
        return self._get_division_obs(), {}

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(
        self,
        div_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one division macro-step.

        Translates the division operational command for each brigade into
        brigade-level actions (option indices for all battalions in the
        brigade) and delegates to :meth:`~envs.brigade_env.BrigadeEnv.step`.

        Parameters
        ----------
        div_action:
            Array of shape ``(n_brigades,)`` with operational command indices.
            Indices for brigades whose battalions are all dead are ignored.

        Returns
        -------
        obs : np.ndarray — division observation after the macro-step
        reward : float — brigade reward (passed through from BrigadeEnv)
        terminated : bool
        truncated : bool
        info : dict — includes ``div_steps``, ``brigade_action``, and
            fields from :class:`~envs.brigade_env.BrigadeEnv`
        """
        div_action = np.asarray(div_action, dtype=np.int64)

        if div_action.shape != (self.n_brigades,):
            raise ValueError(
                f"div_action has shape {div_action.shape!r}, "
                f"expected ({self.n_brigades},)."
            )

        for i, cmd in enumerate(div_action):
            if int(cmd) < 0 or int(cmd) >= self.n_div_options:
                raise ValueError(
                    f"Invalid operational command {int(cmd)!r} for brigade {i}; "
                    f"expected integer in [0, {self.n_div_options - 1}]."
                )

        # ── Inject Red brigade commands (if frozen policy set) ────────
        if self._red_brigade_policy is not None:
            self._update_red_brigade_options()

        # ── Translate division commands → BrigadeEnv action ──────────
        # brigade_action[i*n_per + j] = div_action[i]  for j in range(n_per)
        brigade_action = self._translate_division_action(div_action)

        # ── Delegate to BrigadeEnv ────────────────────────────────────
        _brigade_obs, reward, terminated, truncated, brigade_info = (
            self._brigade.step(brigade_action)
        )

        # Clear forced Red options after the step
        self._brigade._forced_red_options = {}

        self._div_steps += 1

        info: dict = {
            "div_steps": self._div_steps,
            "brigade_action": brigade_action.tolist(),
        }
        info.update(brigade_info)

        return self._get_division_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Command translation
    # ------------------------------------------------------------------

    def _translate_division_action(self, div_action: np.ndarray) -> np.ndarray:
        """Expand a per-brigade action to a flat per-battalion brigade action.

        Parameters
        ----------
        div_action:
            Integer array of shape ``(n_brigades,)`` with operational command
            indices.

        Returns
        -------
        np.ndarray of shape ``(n_blue,)`` — option index for every battalion.
        """
        brigade_action = np.empty(self.n_blue, dtype=np.int64)
        for i in range(self.n_brigades):
            start = i * self.n_blue_per_brigade
            end = start + self.n_blue_per_brigade
            brigade_action[start:end] = int(div_action[i])
        return brigade_action

    # ------------------------------------------------------------------
    # Red brigade policy injection
    # ------------------------------------------------------------------

    def _update_red_brigade_options(self) -> None:
        """Compute Red brigade obs, query the frozen policy, and inject options.

        The frozen brigade policy receives a :class:`~envs.brigade_env.BrigadeEnv`-
        compatible observation for the Red side (shape ``3 + 7 * n_red + 1``,
        treating Red battalions as the "blue" side) and returns a per-battalion
        action of shape ``(n_red,)`` with option indices in ``[0, n_options)``.
        Each option index is injected directly into
        :attr:`~envs.brigade_env.BrigadeEnv._forced_red_options` for the
        corresponding Red battalion, bypassing the default Red action logic.
        """
        red_obs = self._get_red_brigade_obs()
        red_action, _ = self._red_brigade_policy.predict(
            red_obs, deterministic=False
        )
        red_action = np.asarray(red_action, dtype=np.int64).flatten()

        if len(red_action) < self.n_red:
            raise ValueError(
                f"Frozen brigade policy returned action of length {len(red_action)}, "
                f"but expected at least {self.n_red} (one per Red battalion)."
            )

        forced: dict[str, int] = {}
        for idx in range(self.n_red):
            cmd = int(np.clip(red_action[idx], 0, self.n_div_options - 1))
            forced[f"red_{idx}"] = cmd
        self._brigade._forced_red_options = forced

    # ------------------------------------------------------------------
    # Division observation construction
    # ------------------------------------------------------------------

    def _battalion_in_sector(self, b, s: int, sector_width: float) -> bool:
        """Return True if battalion *b* occupies theatre sector *s*."""
        x_lo = s * sector_width
        x_hi = (s + 1) * sector_width
        return b.x >= x_lo and (
            b.x < x_hi or (s == N_THEATRE_SECTORS - 1 and b.x == self.map_width)
        )

    def _get_theatre_sector_strengths(self, inner) -> list[tuple[float, float]]:
        """Return ``[(blue_str, red_str), ...]`` for each theatre sector."""
        sector_width = self.map_width / N_THEATRE_SECTORS
        result: list[tuple[float, float]] = []
        for s in range(N_THEATRE_SECTORS):
            blue_str = 0.0
            red_str = 0.0
            for agent_id, b in inner._battalions.items():
                if agent_id not in inner._alive:
                    continue
                if self._battalion_in_sector(b, s, sector_width):
                    if agent_id.startswith("blue_"):
                        blue_str += float(b.strength)
                    else:
                        red_str += float(b.strength)
            result.append((blue_str, red_str))
        return result

    def _get_division_obs(self) -> np.ndarray:
        """Build and return the normalised division observation vector."""
        parts: list[float] = []
        inner = self._brigade._inner

        # ── 1. Theatre sector control (5 vertical strips) ─────────────
        for blue_str, red_str in self._get_theatre_sector_strengths(inner):
            total = blue_str + red_str
            parts.append(blue_str / total if total > 0.0 else 0.5)

        # ── 2. Per-brigade status [avg_strength, avg_morale, alive_ratio] ─
        for i in range(self.n_brigades):
            strengths = []
            morales = []
            alive_count = 0
            for j in range(self.n_blue_per_brigade):
                agent_id = f"blue_{i * self.n_blue_per_brigade + j}"
                if agent_id in inner._battalions and agent_id in inner._alive:
                    b = inner._battalions[agent_id]
                    strengths.append(float(b.strength))
                    morales.append(float(b.morale))
                    alive_count += 1
            avg_str = float(np.mean(strengths)) if strengths else 0.0
            avg_mor = float(np.mean(morales)) if morales else 0.0
            alive_ratio = alive_count / self.n_blue_per_brigade
            parts.extend([avg_str, avg_mor, alive_ratio])

        # ── 3. Per-brigade threat vector ───────────────────────────────
        # Build list of alive Red brigade centroids
        red_brigade_centroids = self._get_red_brigade_centroids(inner)

        for i in range(self.n_brigades):
            # Compute centroid of this Blue brigade
            bx_list, by_list = [], []
            for j in range(self.n_blue_per_brigade):
                agent_id = f"blue_{i * self.n_blue_per_brigade + j}"
                if agent_id in inner._battalions and agent_id in inner._alive:
                    b = inner._battalions[agent_id]
                    bx_list.append(b.x)
                    by_list.append(b.y)

            if not bx_list or not red_brigade_centroids:
                # This brigade is dead or no Red brigades alive — sentinel
                parts.extend([1.0, 0.0, 0.0, 0.0, 0.0])
                continue

            cx = float(np.mean(bx_list))
            cy = float(np.mean(by_list))

            # Find nearest Red brigade by centroid
            best_dist = float("inf")
            best_centroid = None
            best_e_str = 0.0
            best_e_mor = 0.0
            for (rx, ry, e_str, e_mor) in red_brigade_centroids:
                dx = rx - cx
                dy = ry - cy
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_dist:
                    best_dist = d
                    best_centroid = (rx, ry)
                    best_e_str = e_str
                    best_e_mor = e_mor

            assert best_centroid is not None
            dx = best_centroid[0] - cx
            dy = best_centroid[1] - cy
            bearing = math.atan2(dy, dx)

            parts.append(min(best_dist / self.map_diagonal, 1.0))
            parts.append(math.cos(bearing))
            parts.append(math.sin(bearing))
            parts.append(best_e_str)
            parts.append(best_e_mor)

        # ── 4. Step progress ───────────────────────────────────────────
        parts.append(min(inner._step_count / self.max_steps, 1.0))

        obs = np.array(parts, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_red_brigade_centroids(
        self, inner
    ) -> list[tuple[float, float, float, float]]:
        """Return ``(cx, cy, avg_strength, avg_morale)`` for each alive Red brigade."""
        centroids = []
        for i in range(self.n_red_brigades):
            xs, ys, strs, mors = [], [], [], []
            for j in range(self.n_red_per_brigade):
                agent_id = f"red_{i * self.n_red_per_brigade + j}"
                if agent_id in inner._battalions and agent_id in inner._alive:
                    b = inner._battalions[agent_id]
                    xs.append(b.x)
                    ys.append(b.y)
                    strs.append(float(b.strength))
                    mors.append(float(b.morale))
            if xs:
                centroids.append((
                    float(np.mean(xs)),
                    float(np.mean(ys)),
                    float(np.mean(strs)),
                    float(np.mean(mors)),
                ))
        return centroids

    # ------------------------------------------------------------------
    # Red brigade observation (for frozen Red brigade policy)
    # ------------------------------------------------------------------

    def _get_red_brigade_obs(self) -> np.ndarray:
        """Build a :class:`~envs.brigade_env.BrigadeEnv`-compatible observation for Red.

        The observation mirrors the format that a brigade-level PPO policy
        was trained on, treating Red battalions as the "blue" side:

        * ``_BRIGADE_N_SECTORS`` (= 3) sector-control values — Red's strength
          share in each of 3 equal vertical strips.
        * Per-Red-battalion ``[strength, morale]`` — zeros for dead battalions.
        * Per-Red-battalion enemy threat ``[dist/diag, cos_bear, sin_bear,
          e_str, e_mor]`` — nearest alive Blue *battalion* (not centroid).
          Sentinel ``[1, 0, 0, 0, 0]`` when no Blue battalion is alive.
        * Step progress.

        The returned array has shape ``(_BRIGADE_N_SECTORS + 7 * n_red + 1,)``
        and is clipped using per-element bounds, independent of
        ``self.observation_space`` (which is sized for the Blue division obs).
        """
        parts: list[float] = []
        inner = self._brigade._inner

        # ── 1. Sector control — 3 strips, Red's share ─────────────────
        sector_width = self.map_width / _BRIGADE_N_SECTORS
        for s in range(_BRIGADE_N_SECTORS):
            x_lo = s * sector_width
            x_hi = (s + 1) * sector_width
            blue_str = 0.0
            red_str = 0.0
            for agent_id, b in inner._battalions.items():
                if agent_id not in inner._alive:
                    continue
                in_sector = (x_lo <= b.x < x_hi) or (
                    s == _BRIGADE_N_SECTORS - 1 and b.x == self.map_width
                )
                if in_sector:
                    if agent_id.startswith("blue_"):
                        blue_str += float(b.strength)
                    else:
                        red_str += float(b.strength)
            total = blue_str + red_str
            parts.append(red_str / total if total > 0.0 else 0.5)

        # ── 2. Per-Red-battalion strength + morale ─────────────────────
        for idx in range(self.n_red):
            agent_id = f"red_{idx}"
            if agent_id in inner._battalions and agent_id in inner._alive:
                b = inner._battalions[agent_id]
                parts.append(float(b.strength))
                parts.append(float(b.morale))
            else:
                parts.extend([0.0, 0.0])

        # ── 3. Per-Red-battalion enemy threat → nearest alive Blue ─────
        alive_blue = [
            (b_id, inner._battalions[b_id])
            for b_id in inner._alive
            if b_id.startswith("blue_") and b_id in inner._battalions
        ]

        for idx in range(self.n_red):
            agent_id = f"red_{idx}"
            if agent_id not in inner._alive or agent_id not in inner._battalions:
                parts.extend([1.0, 0.0, 0.0, 0.0, 0.0])
                continue

            rb = inner._battalions[agent_id]

            if not alive_blue:
                parts.extend([1.0, 0.0, 0.0, 0.0, 0.0])
                continue

            best_dist = float("inf")
            best_b = None
            for _b_id, bb in alive_blue:
                dx = bb.x - rb.x
                dy = bb.y - rb.y
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_dist:
                    best_dist = d
                    best_b = bb

            assert best_b is not None
            dx = best_b.x - rb.x
            dy = best_b.y - rb.y
            bearing = math.atan2(dy, dx)

            parts.append(min(best_dist / self.map_diagonal, 1.0))
            parts.append(math.cos(bearing))
            parts.append(math.sin(bearing))
            parts.append(float(best_b.strength))
            parts.append(float(best_b.morale))

        # ── 4. Step progress ───────────────────────────────────────────
        parts.append(min(inner._step_count / self.max_steps, 1.0))

        obs = np.array(parts, dtype=np.float32)

        # Build per-element bounds for the BrigadeEnv-compatible obs layout.
        # This is independent of self.observation_space (Blue division obs).
        lows: list[float] = [0.0] * _BRIGADE_N_SECTORS
        highs: list[float] = [1.0] * _BRIGADE_N_SECTORS
        for _ in range(self.n_red):   # strength, morale per battalion
            lows.extend([0.0, 0.0])
            highs.extend([1.0, 1.0])
        for _ in range(self.n_red):   # dist, cos, sin, e_str, e_mor per battalion
            lows.extend([0.0, -1.0, -1.0, 0.0, 0.0])
            highs.extend([1.0,  1.0,  1.0, 1.0, 1.0])
        lows.append(0.0)
        highs.append(1.0)
        return np.clip(obs, np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32))

    # ------------------------------------------------------------------
    # Gymnasium API: render / close
    # ------------------------------------------------------------------

    def render(self):
        """Delegate rendering to the inner brigade environment."""
        return self._brigade.render()

    def close(self) -> None:
        """Delegate cleanup to the inner brigade environment."""
        self._brigade.close()
