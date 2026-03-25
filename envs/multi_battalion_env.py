# SPDX-License-Identifier: MIT
# envs/multi_battalion_env.py
"""PettingZoo ParallelEnv for NvN battalion combat.

``MultiBattalionEnv`` wraps the battalion simulation engine in a fully
compliant :class:`pettingzoo.ParallelEnv` interface that supports variable
team sizes (``n_blue`` vs ``n_red`` battalions per side).

Each agent receives a **local observation** containing its own state plus
the relative positions, headings, strengths and morale of all other units
on the map.  Enemy units beyond ``visibility_radius`` are subject to fog
of war — their state information is hidden (distance shown as 1.0; other
fields zeroed) while their approximate distance is still available.

A **global state tensor** is exposed via :meth:`MultiBattalionEnv.state`
for use by centralized critics (e.g. MAPPO, QMIX).

Typical usage::

    from envs.multi_battalion_env import MultiBattalionEnv

    env = MultiBattalionEnv(n_blue=2, n_red=2)
    obs, infos = env.reset(seed=42)
    for _ in range(500):
        actions = {agent: env.action_space(agent).sample()
                   for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if not env.agents:
            break
    env.close()

PettingZoo compliance
---------------------
* ``parallel_api_test(MultiBattalionEnv())`` passes.
* ``observation_space(agent)`` / ``action_space(agent)`` return the same
  object on every call (via :func:`functools.lru_cache`).
* Angles are encoded as ``(cos θ, sin θ)`` — never raw radians.
* All observation values are normalised to ``[0, 1]`` or ``[-1, 1]``.
* Seeding via ``reset(seed=…)`` produces deterministic episodes.

Observation space — ``Box(shape=(obs_dim,), dtype=float32)``
-------------------------------------------------------------
``obs_dim = 6 + 5 * (n_blue + n_red - 1) + 1``

Layout for *any* agent (blue or red are symmetric):

=================  ==============================  =========
Slice              Feature                         Range
=================  ==============================  =========
``[0]``            self x / map_width              ``[0, 1]``
``[1]``            self y / map_height             ``[0, 1]``
``[2]``            cos(self θ)                     ``[-1, 1]``
``[3]``            sin(self θ)                     ``[-1, 1]``
``[4]``            self strength                   ``[0, 1]``
``[5]``            self morale                     ``[0, 1]``
``[6 : 6+5*Na]``   allies (Na = team_size − 1),    see below
                   sorted by agent index
``[6+5*Na :]``     enemies (Ne = other team size), see below
                   sorted by agent index
``[-1]``           step / max_steps                ``[0, 1]``
=================  ==============================  =========

Per-unit block (5 floats each):

=====  ======================  =========
Index  Feature                 Range
=====  ======================  =========
0      dist / map_diagonal     ``[0, 1]``
1      cos(bearing)            ``[-1, 1]``
2      sin(bearing)            ``[-1, 1]``
3      strength (0 if hidden)  ``[0, 1]``
4      morale   (0 if hidden)  ``[0, 1]``
=====  ======================  =========

For dead agents the full 5-float block is zeros.
For enemies outside ``visibility_radius`` the block is
``[1.0, 0.0, 0.0, 0.0, 0.0]`` (known-far, unknown state).

Action space — ``Box(shape=(3,), dtype=float32)``
-------------------------------------------------
=====  ========  ============================================
Index  Range     Effect
=====  ========  ============================================
0      [-1, 1]   Move intensity (positive = forward)
1      [-1, 1]   Rotate (positive = CCW)
2      [0,  1]   Fire intensity
=====  ========  ============================================

Global state — ``Box(shape=(state_dim,), dtype=float32)``
---------------------------------------------------------
``state_dim = 6 * (n_blue + n_red) + 1``

Flat concatenation of ``[x/w, y/h, cos θ, sin θ, strength, morale]`` for
each agent in ``possible_agents`` order, followed by ``step/max_steps``.
Dead agents appear as an all-zero 6-float block.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from envs.sim.battalion import Battalion
from envs.sim.combat import (
    CombatState,
    apply_casualties,
    compute_fire_damage,
    morale_check,
)
from envs.metrics.coordination import compute_all as _compute_coordination
from envs.sim.engine import DESTROYED_THRESHOLD
from envs.sim.road_network import RoadNetwork
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
#: Simulation time step used for movement (seconds).
DT: float = 0.1
#: Default enemy visibility radius for fog-of-war (metres).
VISIBILITY_RADIUS: float = 600.0

__all__ = [
    "MultiBattalionEnv",
    "MAP_WIDTH",
    "MAP_HEIGHT",
    "MAX_STEPS",
    "VISIBILITY_RADIUS",
]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MultiBattalionEnv(ParallelEnv):
    """PettingZoo ``ParallelEnv`` for NvN battalion combat.

    Parameters
    ----------
    n_blue:
        Number of Blue (team 0) battalions.  Must be ``≥ 1``.
    n_red:
        Number of Red (team 1) battalions.  Must be ``≥ 1``.
    map_width, map_height:
        Map dimensions in metres (default 1 km × 1 km).
    max_steps:
        Episode length cap (default 500).
    terrain:
        Optional fixed :class:`~envs.sim.terrain.TerrainMap`.  When
        supplied *randomize_terrain* is forced to ``False`` and this map
        is used for every episode.
    randomize_terrain:
        When ``True`` (default, if no fixed *terrain* is supplied) a new
        procedural terrain is generated from the seeded RNG at the start
        of each episode.
    hill_speed_factor:
        Movement speed multiplier on maximum-elevation terrain.  Must be
        in ``(0, 1]``; ``0.5`` means half speed on the highest hills.
    visibility_radius:
        Fog-of-war cutoff distance (metres).  Enemy units beyond this
        distance have their strength and morale hidden in observations.
    render_mode:
        Currently unused; reserved for future rendering support.
    """

    metadata: dict = {"render_modes": [], "name": "multi_battalion_v0"}

    def __init__(
        self,
        n_blue: int = 2,
        n_red: int = 2,
        map_width: float = MAP_WIDTH,
        map_height: float = MAP_HEIGHT,
        max_steps: int = MAX_STEPS,
        terrain: Optional[TerrainMap] = None,
        randomize_terrain: bool = True,
        hill_speed_factor: float = 0.5,
        visibility_radius: float = VISIBILITY_RADIUS,
        render_mode: Optional[str] = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Argument validation
        # ------------------------------------------------------------------
        if int(n_blue) < 1:
            raise ValueError(f"n_blue must be >= 1, got {n_blue}")
        if int(n_red) < 1:
            raise ValueError(f"n_red must be >= 1, got {n_red}")
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
        if float(visibility_radius) <= 0:
            raise ValueError(
                f"visibility_radius must be positive, got {visibility_radius}"
            )

        self.n_blue = int(n_blue)
        self.n_red = int(n_red)
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.map_diagonal = math.hypot(self.map_width, self.map_height)
        self.max_steps = int(max_steps)
        self.hill_speed_factor = float(hill_speed_factor)
        self.visibility_radius = float(visibility_radius)
        self.randomize_terrain = bool(randomize_terrain) and (terrain is None)
        self._supplied_terrain: Optional[TerrainMap] = terrain
        self.terrain: TerrainMap = (
            terrain if terrain is not None
            else TerrainMap.flat(map_width, map_height)
        )
        self.render_mode = render_mode

        # Optional road network — when set, battalions on roads gain a
        # movement-speed bonus (see :data:`~envs.sim.road_network.ROAD_SPEED_BONUS`).
        # Can be set after construction: ``env.road_network = my_network``.
        self.road_network: Optional[RoadNetwork] = None

        # ------------------------------------------------------------------
        # PettingZoo required: possible_agents (fixed for the lifetime of env)
        # ------------------------------------------------------------------
        self.possible_agents: list[str] = (
            [f"blue_{i}" for i in range(self.n_blue)]
            + [f"red_{i}" for i in range(self.n_red)]
        )

        # ------------------------------------------------------------------
        # Observation / action space construction
        # ------------------------------------------------------------------
        # obs_dim = 6 (self) + 5 * (n_total - 1) (others) + 1 (step_norm)
        n_total = self.n_blue + self.n_red
        self._obs_dim: int = 6 + 5 * (n_total - 1) + 1

        # Global state dim: 6 per agent + 1 step_norm
        self._state_dim: int = 6 * n_total + 1

        # Per-unit observation bounds
        # Self state: [x/w, y/h, cos, sin, strength, morale]
        self_low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        self_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        # Other unit block: [dist_norm, cos_bearing, sin_bearing, strength, morale]
        other_unit_low = np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        other_unit_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        n_others = n_total - 1
        obs_low = np.concatenate(
            [self_low, np.tile(other_unit_low, n_others),
             np.array([0.0], dtype=np.float32)]
        )
        obs_high = np.concatenate(
            [self_high, np.tile(other_unit_high, n_others),
             np.array([1.0], dtype=np.float32)]
        )
        # Single shared observation space instance (same for every agent)
        self._obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # Single shared action space instance
        self._act_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Internal episode state (populated by reset())
        # ------------------------------------------------------------------
        self._battalions: dict[str, Battalion] = {}
        self._combat_states: dict[str, CombatState] = {}
        self._alive: set[str] = set()   # set of agent IDs still alive
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        # PettingZoo: live agents list (modified by reset/step)
        self.agents: list[str] = []

    # ------------------------------------------------------------------
    # PettingZoo API: spaces (must return the same object each call)
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        """Return observation space for *agent* (same object every call)."""
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Box:
        """Return action space for *agent* (same object every call)."""
        return self._act_space

    # ------------------------------------------------------------------
    # PettingZoo API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the environment and return initial observations.

        Parameters
        ----------
        seed:
            RNG seed.  Passing the same seed always produces the same
            terrain layout and starting positions.
        options:
            Currently unused; accepted for API compatibility.

        Returns
        -------
        observations : dict[agent_id, np.ndarray]
        infos        : dict[agent_id, dict]
        """
        self._rng = np.random.default_rng(seed)

        # Terrain
        if self.randomize_terrain:
            self.terrain = TerrainMap.generate_random(
                rng=self._rng,
                width=self.map_width,
                height=self.map_height,
            )
        elif self._supplied_terrain is not None:
            self.terrain = self._supplied_terrain

        # Reset live agents
        self.agents = list(self.possible_agents)
        self._alive = set(self.possible_agents)
        self._step_count = 0
        self._battalions = {}
        self._combat_states = {}

        # Spawn Blue agents in the western half, facing roughly east
        for i in range(self.n_blue):
            agent_id = f"blue_{i}"
            x = float(self._rng.uniform(0.1 * self.map_width, 0.4 * self.map_width))
            y = float(self._rng.uniform(0.1 * self.map_height, 0.9 * self.map_height))
            theta = float(self._rng.uniform(-math.pi / 4, math.pi / 4))
            self._battalions[agent_id] = Battalion(
                x=x, y=y, theta=theta, strength=1.0, team=0
            )
            self._combat_states[agent_id] = CombatState()

        # Spawn Red agents in the eastern half, facing roughly west
        for i in range(self.n_red):
            agent_id = f"red_{i}"
            x = float(self._rng.uniform(0.6 * self.map_width, 0.9 * self.map_width))
            y = float(self._rng.uniform(0.1 * self.map_height, 0.9 * self.map_height))
            theta = float(math.pi + self._rng.uniform(-math.pi / 4, math.pi / 4))
            self._battalions[agent_id] = Battalion(
                x=x, y=y, theta=theta, strength=1.0, team=1
            )
            self._combat_states[agent_id] = CombatState()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos: dict[str, dict] = {agent: {} for agent in self.agents}
        return observations, infos

    # ------------------------------------------------------------------
    # PettingZoo API: step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Advance the environment one step.

        Parameters
        ----------
        actions:
            Dict mapping each live agent ID to its action array of shape
            ``(3,)``: ``[move, rotate, fire]``.  Agents absent from the
            dict are treated as no-op (zeros).

        Returns
        -------
        observations, rewards, terminations, truncations, infos
            All keyed by agent ID; contain entries for every agent that
            was alive at the **start** of this step.
        """
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = list(self.agents)

        # --- 1. Apply movement for each live agent ---
        for agent_id in current_agents:
            action = np.asarray(
                actions.get(agent_id, np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            )
            move_cmd = float(np.clip(action[0], -1.0, 1.0))
            rotate_cmd = float(np.clip(action[1], -1.0, 1.0))

            battalion = self._battalions[agent_id]
            battalion.rotate(rotate_cmd * battalion.max_turn_rate)
            terrain_mod = self.terrain.get_speed_modifier(
                battalion.x, battalion.y, self.hill_speed_factor
            )
            road_mod = 1.0
            if self.road_network is not None:
                road_mod = self.road_network.get_speed_modifier(battalion.x, battalion.y)
            speed_mod = terrain_mod * road_mod
            vx = math.cos(battalion.theta) * move_cmd * battalion.max_speed * speed_mod
            vy = math.sin(battalion.theta) * move_cmd * battalion.max_speed * speed_mod
            # Battalion.move() clamps velocity magnitude to battalion.max_speed.
            # Temporarily raise the effective cap so road bonuses > 1.0 take effect.
            original_max_speed = battalion.max_speed
            try:
                battalion.max_speed = original_max_speed * max(1.0, road_mod)
                battalion.move(vx, vy, dt=DT)
            finally:
                battalion.max_speed = original_max_speed
            battalion.x = float(np.clip(battalion.x, 0.0, self.map_width))
            battalion.y = float(np.clip(battalion.y, 0.0, self.map_height))

        # --- 2. Combat resolution (simultaneous) ---
        for cs in self._combat_states.values():
            cs.reset_step_accumulators()

        blue_agents = [a for a in current_agents if a.startswith("blue_")]
        red_agents = [a for a in current_agents if a.startswith("red_")]

        # Compute all raw damages first (simultaneous resolution)
        # raw_damage[attacker_id][target_id] = raw damage value
        raw_damage: dict[str, dict[str, float]] = {a: {} for a in current_agents}

        for attacker_id in blue_agents:
            fire_cmd = float(
                np.clip(
                    actions.get(attacker_id, np.zeros(3, dtype=np.float32))[2],
                    0.0, 1.0,
                )
            )
            for target_id in red_agents:
                raw = compute_fire_damage(
                    self._battalions[attacker_id],
                    self._battalions[target_id],
                    intensity=fire_cmd,
                )
                raw = self.terrain.apply_cover_modifier(
                    self._battalions[target_id].x,
                    self._battalions[target_id].y,
                    raw,
                )
                raw_damage[attacker_id][target_id] = raw

        for attacker_id in red_agents:
            fire_cmd = float(
                np.clip(
                    actions.get(attacker_id, np.zeros(3, dtype=np.float32))[2],
                    0.0, 1.0,
                )
            )
            for target_id in blue_agents:
                raw = compute_fire_damage(
                    self._battalions[attacker_id],
                    self._battalions[target_id],
                    intensity=fire_cmd,
                )
                raw = self.terrain.apply_cover_modifier(
                    self._battalions[target_id].x,
                    self._battalions[target_id].y,
                    raw,
                )
                raw_damage[attacker_id][target_id] = raw

        # Sum incoming damage per target and apply once (preserves simultaneity)
        damage_dealt: dict[str, float] = {a: 0.0 for a in current_agents}
        damage_received: dict[str, float] = {a: 0.0 for a in current_agents}

        for target_id in current_agents:
            total_raw: float = sum(
                raw_damage[att].get(target_id, 0.0)
                for att in current_agents
                if att != target_id
            )
            if total_raw <= 0.0:
                continue

            actual = apply_casualties(
                self._battalions[target_id],
                self._combat_states[target_id],
                total_raw,
            )
            damage_received[target_id] = actual

            # Credit each attacker proportionally
            for attacker_id in current_agents:
                attacker_raw = raw_damage[attacker_id].get(target_id, 0.0)
                if attacker_raw > 0.0:
                    damage_dealt[attacker_id] += actual * (attacker_raw / total_raw)

        # --- 3. Morale checks ---
        for agent_id in current_agents:
            morale_check(self._combat_states[agent_id], rng=self._rng)
            # Sync Battalion fields from CombatState
            battalion = self._battalions[agent_id]
            cs = self._combat_states[agent_id]
            battalion.morale = cs.morale
            battalion.routed = cs.is_routing

        self._step_count += 1

        # --- 4. Termination / truncation ---
        # Individual termination: routed or effectively destroyed
        individually_done: dict[str, bool] = {
            agent_id: (
                self._combat_states[agent_id].is_routing
                or self._battalions[agent_id].strength <= DESTROYED_THRESHOLD
            )
            for agent_id in current_agents
        }

        # Team-level: surviving-side members are also terminated when the
        # opposing team is completely eliminated this step.
        blue_all_done = all(individually_done.get(a, True) for a in blue_agents)
        red_all_done = all(individually_done.get(a, True) for a in red_agents)

        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        for agent_id in current_agents:
            # Terminate if individually eliminated OR the battle is decided
            terminated[agent_id] = (
                individually_done[agent_id]
                or (blue_all_done and bool(blue_agents))
                or (red_all_done and bool(red_agents))
            )
            # Truncate remaining live agents at max_steps
            truncated[agent_id] = (
                not terminated[agent_id]
                and self._step_count >= self.max_steps
            )

        # --- 5. Rewards ---
        rewards: dict[str, float] = {}
        for agent_id in current_agents:
            is_blue = agent_id.startswith("blue_")
            r = (
                damage_dealt[agent_id] * 5.0
                - damage_received[agent_id] * 5.0
                - 0.01  # time penalty
            )
            if terminated[agent_id] and not truncated[agent_id]:
                if is_blue:
                    if red_all_done and not blue_all_done:
                        r += 10.0   # blue wins
                    elif blue_all_done and not red_all_done:
                        r -= 10.0   # blue loses
                else:
                    if blue_all_done and not red_all_done:
                        r += 10.0   # red wins
                    elif red_all_done and not blue_all_done:
                        r -= 10.0   # red loses
            rewards[agent_id] = float(r)

        # --- 6. Update _alive so _get_obs sees newly-dead agents as zero blocks ---
        self._alive -= {
            a for a in current_agents if terminated[a] or truncated[a]
        }

        # --- 7. Build observations (after updating _alive so dead-agent blocks
        #         are correctly zeroed in the observations of surviving agents) ---
        observations = {agent: self._get_obs(agent) for agent in current_agents}
        infos: dict[str, dict] = {
            agent: {
                "damage_dealt": float(damage_dealt[agent]),
                "damage_received": float(damage_received[agent]),
                "step_count": self._step_count,
            }
            for agent in current_agents
        }

        # --- 8. Update live-agents list ---
        self.agents = [
            a for a in self.agents
            if not terminated.get(a, False) and not truncated.get(a, False)
        ]

        return observations, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # PettingZoo API: state
    # ------------------------------------------------------------------

    def state(self) -> np.ndarray:
        """Return the global state tensor for centralized critics.

        Concatenates ``[x/w, y/h, cos θ, sin θ, strength, morale]`` for
        every agent in ``possible_agents`` order, followed by
        ``step / max_steps``.  Dead-agent slots are all-zero.

        Returns
        -------
        np.ndarray of shape ``(6 * (n_blue + n_red) + 1,)`` and dtype
        ``float32``.
        """
        parts: list[float] = []
        for agent_id in self.possible_agents:
            if agent_id in self._battalions and agent_id in self._alive:
                b = self._battalions[agent_id]
                parts.extend([
                    b.x / self.map_width,
                    b.y / self.map_height,
                    math.cos(b.theta),
                    math.sin(b.theta),
                    float(b.strength),
                    float(b.morale),
                ])
            else:
                parts.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        parts.append(min(self._step_count / self.max_steps, 1.0))
        return np.array(parts, dtype=np.float32)

    def get_coordination_metrics(
        self,
        support_radius: float = 300.0,
    ) -> dict[str, float]:
        """Compute per-step coordination metrics for the current environment state.

        Calls :func:`~envs.metrics.coordination.compute_all` on the currently
        alive and non-routed Blue and Red battalions.

        Parameters
        ----------
        support_radius:
            Distance threshold (metres) for
            :func:`~envs.metrics.coordination.mutual_support_score`.

        Returns
        -------
        dict[str, float]
            Keys: ``"coordination/flanking_ratio"``,
            ``"coordination/fire_concentration"``,
            ``"coordination/mutual_support_score"``.
        """
        blue = [
            b
            for agent_id, b in self._battalions.items()
            if agent_id.startswith("blue_") and not b.routed and b.strength > DESTROYED_THRESHOLD
        ]
        red = [
            b
            for agent_id, b in self._battalions.items()
            if agent_id.startswith("red_") and not b.routed and b.strength > DESTROYED_THRESHOLD
        ]
        return _compute_coordination(blue, red, support_radius=support_radius)

    # ------------------------------------------------------------------
    # PettingZoo API: close / render (stubs)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""

    def render(self) -> None:
        """Rendering is not yet implemented."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Build the normalised observation vector for *agent_id*.

        Layout: self(6) | allies(5 each) | enemies(5 each) | step_norm(1)

        Allies are ordered by agent index; enemies likewise.
        Dead agents and fog-of-war enemies produce zero/clamped blocks.
        """
        b = self._battalions[agent_id]
        is_blue = agent_id.startswith("blue_")

        # Self state
        self_state = np.array(
            [
                b.x / self.map_width,
                b.y / self.map_height,
                math.cos(b.theta),
                math.sin(b.theta),
                float(b.strength),
                float(b.morale),
            ],
            dtype=np.float32,
        )

        # Other units: allies first (own team minus self), then enemies
        ally_prefix = "blue_" if is_blue else "red_"
        enemy_prefix = "red_" if is_blue else "blue_"

        allies = [
            a for a in self.possible_agents
            if a != agent_id and a.startswith(ally_prefix)
        ]
        enemies = [
            a for a in self.possible_agents
            if a.startswith(enemy_prefix)
        ]
        other_order = allies + enemies

        other_features: list[float] = []
        for other_id in other_order:
            # Dead unit → all zeros
            if other_id not in self._alive:
                other_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            other = self._battalions[other_id]
            dx = other.x - b.x
            dy = other.y - b.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            dist_norm = min(dist / self.map_diagonal, 1.0)
            bearing = math.atan2(dy, dx)
            is_enemy = other_id.startswith(enemy_prefix)

            if is_enemy and dist > self.visibility_radius:
                # Fog of war: report max distance, hide state
                other_features.extend([1.0, 0.0, 0.0, 0.0, 0.0])
            else:
                other_features.extend([
                    dist_norm,
                    math.cos(bearing),
                    math.sin(bearing),
                    float(other.strength),
                    float(other.morale),
                ])

        step_norm = min(self._step_count / self.max_steps, 1.0)
        obs = np.concatenate(
            [self_state, np.array(other_features, dtype=np.float32), [step_norm]],
            dtype=np.float32,
        )
        return np.clip(obs, self._obs_space.low, self._obs_space.high)
