# SPDX-License-Identifier: MIT
"""Corps Commander Environment — top-level MDP for corps-level HRL.

:class:`CorpsEnv` extends the HRL stack to the *corps* echelon.  It wraps
a single :class:`~envs.division_env.DivisionEnv` whose brigades are
logically grouped into *divisions*, and adds:

* A :class:`~envs.sim.road_network.RoadNetwork` that grants a 1.5× movement
  speed bonus to units on roads.
* :class:`OperationalObjective` instances (capture objective hex, cut supply
  line, and fix-and-flank) that replace tactical kill-counting as the
  primary reward signal.
* An *inter-division communication radius* — divisions that are too far apart
  lose shared threat information in the observation.

Architecture
------------
::

    CorpsEnv (Gymnasium)               ← corps PPO agent lives here
        │
        └─ DivisionEnv (Gymnasium)     ← division command dispatcher
              │
              └─ BrigadeEnv (Gymnasium)
                    │
                    └─ MultiBattalionEnv (PettingZoo ParallelEnv)

Scale
-----
* 3–5 divisions per side (configurable via *n_divisions*).
* Each division contains 3–4 brigades of 4–6 battalions.
* Default map: 10 km × 5 km = 50 km².

Observation space
-----------------
``Box(shape=(obs_dim,), dtype=float32)``

``obs_dim = N_CORPS_SECTORS + 8 * n_divisions + N_ROAD_FEATURES +
N_OBJECTIVES + n_divisions + 1`` where ``N_CORPS_SECTORS = 7``,
``N_ROAD_FEATURES = 2`` (blue + red road-usage fractions),
``N_OBJECTIVES = 3``.

==================================  =============================================  =========
Slice                               Feature                                        Range
==================================  =============================================  =========
``[0 : N_CORPS_SECTORS]``           Corps sector control (7 vertical strips)       ``[0, 1]``
``[7 : 7+3*nd]``                    Per-division status (3 per division)           ``[0, 1]``
                                    ``[avg_strength, avg_morale, alive_ratio]``
``[7+3*nd : 7+8*nd]``               Per-division threat vector (5 per division)    mixed
                                    ``[dist/diag, cos_bear, sin_bear, e_str,``
                                    ``e_mor]`` — nearest enemy division centroid.
                                    Sentinel ``[1,0,0,0,0]`` when out of range.
``[7+8*nd : 7+8*nd+2]``             Road usage: blue / red fraction on roads       ``[0, 1]``
``[7+8*nd+2 : 7+8*nd+5]``           Objective control (3 objectives)               ``[0, 1]``
``[7+8*nd+5 : 7+8*nd+5+nd]``        Supply level per Blue division                 ``[0, 1]``
                                    (avg supply level of units in the division).
``[-1]``                            Step progress: ``step / max_steps``            ``[0, 1]``
==================================  =============================================  =========

Action space
------------
``MultiDiscrete([n_corps_options] * n_divisions)``

Each element selects one of six operational commands for the corresponding
Blue division (same vocabulary as the division and brigade layers).

Operational objectives
----------------------
Three objectives are placed on the map at construction time:

0. **Capture objective hex** — centre of the map.
1. **Cut supply line** — Red's supply depot (deep in Red territory).
2. **Fix-and-flank** — dynamic: active when Blue has units in ≥ 2 horizontal
   zones simultaneously.

Inter-division communication radius
------------------------------------
When the nearest enemy division centroid exceeds *comm_radius*, the
per-division threat vector for that division is replaced with the sentinel
``[1, 0, 0, 0, 0]``.

Typical usage::

    from envs.corps_env import CorpsEnv

    env = CorpsEnv(n_divisions=3, n_brigades_per_division=3)
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
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np
from gymnasium import spaces
import gymnasium as gym

from envs.division_env import DivisionEnv
from envs.multi_battalion_env import MAX_STEPS
from envs.sim.road_network import RoadNetwork
from envs.sim.supply_network import SupplyNetwork

__all__ = [
    "CorpsEnv",
    "ObjectiveType",
    "OperationalObjective",
    "CORPS_OBS_DIM",
    "N_CORPS_SECTORS",
    "N_OBJECTIVES",
    "CORPS_MAP_WIDTH",
    "CORPS_MAP_HEIGHT",
    "N_ROAD_FEATURES",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Number of theatre-level sectors (7 vertical strips for corps-scale).
N_CORPS_SECTORS: int = 7
#: Number of operational objectives.
N_OBJECTIVES: int = 3
#: Floats per division in the status block.
_DIV_STATUS_DIM: int = 3   # avg_strength, avg_morale, alive_ratio
#: Floats per division in the threat block.
_DIV_THREAT_DIM: int = 5   # dist, cos_bear, sin_bear, e_str, e_mor
#: Road-usage features (blue fraction + red fraction).
N_ROAD_FEATURES: int = 2

#: Default corps map width (10 km).
CORPS_MAP_WIDTH: float = 10_000.0
#: Default corps map height (5 km).
CORPS_MAP_HEIGHT: float = 5_000.0

#: Default inter-division communication radius (metres).
DEFAULT_COMM_RADIUS: float = 3_000.0

#: Default visibility radius for the inner simulation (metres).
DEFAULT_VISIBILITY_RADIUS: float = 1_500.0

# Objective reward weights applied on top of division-level rewards
_OBJ_CAPTURE_REWARD: float = 1.0
_OBJ_SUPPLY_CUT_REWARD: float = 0.5
_OBJ_FIX_FLANK_REWARD: float = 0.5


# ---------------------------------------------------------------------------
# Operational objectives
# ---------------------------------------------------------------------------


class ObjectiveType(IntEnum):
    """Types of operational objectives for corps-level play."""

    CAPTURE_HEX = 0
    CUT_SUPPLY_LINE = 1
    FIX_AND_FLANK = 2


@dataclass
class OperationalObjective:
    """A positional operational objective on the corps map.

    Parameters
    ----------
    x, y:
        World position in metres.
    radius:
        Capture radius — units within this distance count towards control.
    obj_type:
        :class:`ObjectiveType` enum value.
    """

    x: float
    y: float
    radius: float
    obj_type: ObjectiveType
    _blue_units: int = field(default=0, repr=False, init=False)
    _red_units: int = field(default=0, repr=False, init=False)

    def reset(self) -> None:
        """Reset unit counts to neutral."""
        self._blue_units = 0
        self._red_units = 0

    def update(self, inner) -> None:
        """Recount friendly/enemy units within the capture radius.

        Parameters
        ----------
        inner:
            The :class:`~envs.multi_battalion_env.MultiBattalionEnv` instance
            that holds the current battalion positions.
        """
        blue_in = 0
        red_in = 0
        for agent_id, b in inner._battalions.items():
            if agent_id not in inner._alive:
                continue
            dist = math.sqrt((b.x - self.x) ** 2 + (b.y - self.y) ** 2)
            if dist <= self.radius:
                if agent_id.startswith("blue_"):
                    blue_in += 1
                else:
                    red_in += 1
        self._blue_units = blue_in
        self._red_units = red_in

    @property
    def control_value(self) -> float:
        """Blue control fraction in ``[0, 1]``; 0.5 means neutral."""
        total = self._blue_units + self._red_units
        if total <= 0:
            return 0.5
        return self._blue_units / total

    @property
    def is_blue_controlled(self) -> bool:
        """``True`` when Blue has majority control."""
        return self._blue_units > self._red_units

    @property
    def is_red_controlled(self) -> bool:
        """``True`` when Red has majority control."""
        return self._red_units > self._blue_units


# ---------------------------------------------------------------------------
# Observation dimension helper
# ---------------------------------------------------------------------------


def _corps_obs_dim(n_divisions: int) -> int:
    """Return the flat observation dimension for a corps of *n_divisions*."""
    return (
        N_CORPS_SECTORS
        + (_DIV_STATUS_DIM + _DIV_THREAT_DIM) * n_divisions
        + N_ROAD_FEATURES
        + N_OBJECTIVES
        + n_divisions  # supply_level per Blue division
        + 1  # step progress
    )


#: Public constant for the default 3-division corps observation size.
CORPS_OBS_DIM: int = _corps_obs_dim(3)


# ---------------------------------------------------------------------------
# Fix-and-flank detection helper
# ---------------------------------------------------------------------------


def _detect_fix_and_flank(inner, map_height: float) -> bool:
    """Return ``True`` when Blue has alive units in ≥ 2 horizontal zones.

    The map is divided into three equal horizontal bands (south / centre /
    north).  Fix-and-flank is active when Blue occupies at least two bands
    simultaneously, indicating multi-axis manoeuvre.
    """
    zone_height = map_height / 3.0
    occupied: set[int] = set()
    for agent_id, b in inner._battalions.items():
        if agent_id not in inner._alive:
            continue
        if not agent_id.startswith("blue_"):
            continue
        zone = int(b.y / zone_height)
        zone = min(zone, 2)  # clamp to [0, 2]
        occupied.add(zone)
        if len(occupied) >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# CorpsEnv
# ---------------------------------------------------------------------------


class CorpsEnv(gym.Env):
    """Gymnasium environment for a corps-level HRL commander.

    Parameters
    ----------
    n_divisions:
        Number of Blue divisions.  Must be ≥ 1.
    n_brigades_per_division:
        Number of Blue brigades per division.  Must be ≥ 1.
    n_blue_per_brigade:
        Number of Blue battalions per brigade.  Must be ≥ 1.
    n_red_divisions:
        Number of Red divisions.  Defaults to *n_divisions*.
    n_red_brigades_per_division:
        Red brigades per division.  Defaults to *n_brigades_per_division*.
    n_red_per_brigade:
        Red battalions per brigade.  Defaults to *n_blue_per_brigade*.
    map_width, map_height:
        Map dimensions in metres (default 10 km × 5 km = 50 km²).
    max_steps:
        Maximum primitive-step episode length.
    road_network:
        Optional :class:`~envs.sim.road_network.RoadNetwork`.  When
        ``None`` (the default) a standard network is generated via
        :meth:`~envs.sim.road_network.RoadNetwork.generate_default`.
    supply_network:
        Optional :class:`~envs.sim.supply_network.SupplyNetwork`.  When
        ``None`` (the default) a standard bilateral supply network is
        generated via
        :meth:`~envs.sim.supply_network.SupplyNetwork.generate_default`.
    objectives:
        Optional list of :class:`OperationalObjective`.  When ``None``,
        three default objectives are placed automatically.
    comm_radius:
        Inter-division communication radius in metres.  Enemy threat
        vectors beyond this distance are replaced with sentinels.
    red_random:
        When ``True`` Red takes random brigade actions.
    randomize_terrain:
        Pass-through to the inner env.
    visibility_radius:
        Fog-of-war visibility radius in metres.
    render_mode:
        ``None`` or ``"human"``.
    """

    metadata: dict = {"render_modes": ["human"], "name": "corps_v0"}

    def __init__(
        self,
        n_divisions: int = 3,
        n_brigades_per_division: int = 3,
        n_blue_per_brigade: int = 4,
        n_red_divisions: Optional[int] = None,
        n_red_brigades_per_division: Optional[int] = None,
        n_red_per_brigade: Optional[int] = None,
        map_width: float = CORPS_MAP_WIDTH,
        map_height: float = CORPS_MAP_HEIGHT,
        max_steps: int = MAX_STEPS,
        road_network: Optional[RoadNetwork] = None,
        supply_network: Optional[SupplyNetwork] = None,
        objectives: Optional[List[OperationalObjective]] = None,
        comm_radius: float = DEFAULT_COMM_RADIUS,
        red_random: bool = True,
        randomize_terrain: bool = True,
        visibility_radius: float = DEFAULT_VISIBILITY_RADIUS,
        render_mode: Optional[str] = None,
    ) -> None:
        # ── Validation ───────────────────────────────────────────────
        if int(n_divisions) < 1:
            raise ValueError(f"n_divisions must be >= 1, got {n_divisions}")
        if int(n_brigades_per_division) < 1:
            raise ValueError(
                f"n_brigades_per_division must be >= 1, got {n_brigades_per_division}"
            )
        if int(n_blue_per_brigade) < 1:
            raise ValueError(
                f"n_blue_per_brigade must be >= 1, got {n_blue_per_brigade}"
            )

        self.n_divisions: int = int(n_divisions)
        self.n_brigades_per_division: int = int(n_brigades_per_division)
        self.n_blue_per_brigade: int = int(n_blue_per_brigade)

        self.n_red_divisions: int = int(
            n_divisions if n_red_divisions is None else n_red_divisions
        )
        self.n_red_brigades_per_division: int = int(
            n_brigades_per_division
            if n_red_brigades_per_division is None
            else n_red_brigades_per_division
        )
        self.n_red_per_brigade: int = int(
            n_blue_per_brigade if n_red_per_brigade is None else n_red_per_brigade
        )

        if self.n_red_divisions < 1:
            raise ValueError(
                f"n_red_divisions must be >= 1, got {self.n_red_divisions}"
            )
        if self.n_red_brigades_per_division < 1:
            raise ValueError(
                f"n_red_brigades_per_division must be >= 1, "
                f"got {self.n_red_brigades_per_division}"
            )
        if self.n_red_per_brigade < 1:
            raise ValueError(
                f"n_red_per_brigade must be >= 1, got {self.n_red_per_brigade}"
            )

        # Derived counts
        self.n_blue_brigades: int = self.n_divisions * self.n_brigades_per_division
        self.n_red_brigades: int = (
            self.n_red_divisions * self.n_red_brigades_per_division
        )
        self.n_blue: int = self.n_blue_brigades * self.n_blue_per_brigade
        self.n_red: int = self.n_red_brigades * self.n_red_per_brigade

        self.map_width = float(map_width)
        self.map_height = float(map_height)
        if self.map_width <= 0.0 or self.map_height <= 0.0:
            raise ValueError(
                f"map_width and map_height must both be > 0, "
                f"got map_width={self.map_width}, map_height={self.map_height}"
            )
        self.map_diagonal = math.hypot(self.map_width, self.map_height)
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        self.comm_radius = float(comm_radius)
        self.render_mode = render_mode

        # ── Road network ─────────────────────────────────────────────
        self.road_network: RoadNetwork = (
            road_network
            if road_network is not None
            else RoadNetwork.generate_default(self.map_width, self.map_height)
        )

        # ── Supply network ────────────────────────────────────────────
        self.supply_network: SupplyNetwork = (
            supply_network
            if supply_network is not None
            else SupplyNetwork.generate_default(self.map_width, self.map_height)
        )

        # ── Operational objectives ───────────────────────────────────
        self.objectives: List[OperationalObjective] = (
            objectives
            if objectives is not None
            else self._default_objectives()
        )

        # ── Inner DivisionEnv ────────────────────────────────────────
        self._division = DivisionEnv(
            n_brigades=self.n_blue_brigades,
            n_blue_per_brigade=self.n_blue_per_brigade,
            n_red_brigades=self.n_red_brigades,
            n_red_per_brigade=self.n_red_per_brigade,
            map_width=self.map_width,
            map_height=self.map_height,
            max_steps=self.max_steps,
            red_random=red_random,
            randomize_terrain=randomize_terrain,
            visibility_radius=visibility_radius,
            render_mode=render_mode,
        )
        # Propagate road network to the innermost battalion simulation
        self._set_inner_road_network()

        # n_corps_options mirrors the division option count
        self.n_corps_options: int = self._division.n_div_options

        # ── Action space ────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete(
            [self.n_corps_options] * self.n_divisions, dtype=np.int64
        )

        # ── Observation space ───────────────────────────────────────
        self._obs_dim: int = _corps_obs_dim(self.n_divisions)
        obs_low, obs_high = self._build_obs_bounds()
        # Store base corps bounds for use in _get_corps_obs() — subclasses
        # may override observation_space, so clipping must use these stored
        # bounds rather than self.observation_space.
        self._corps_obs_low: np.ndarray = obs_low
        self._corps_obs_high: np.ndarray = obs_high
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Episode state
        self._corps_steps: int = 0

    # ------------------------------------------------------------------
    # Road network propagation
    # ------------------------------------------------------------------

    def _set_inner_road_network(self) -> None:
        """Attach the road network to the innermost MultiBattalionEnv."""
        inner = self._division._brigade._inner
        inner.road_network = self.road_network

    # ------------------------------------------------------------------
    # Supply network helpers
    # ------------------------------------------------------------------

    def _interdiction_radius(self) -> float:
        """Capture radius used for supply-depot interdiction.

        Matches the radius of the ``CUT_SUPPLY_LINE`` operational objective
        so that any Blue unit that can claim the objective can also capture
        the corresponding depot.
        """
        for obj in self.objectives:
            if obj.obj_type == ObjectiveType.CUT_SUPPLY_LINE:
                return obj.radius
        return min(self.map_width, self.map_height) * 0.05

    def _compute_division_supply_levels(self, inner) -> List[float]:
        """Return the average supply level for each Blue division.

        For each Blue division, computes the mean
        :meth:`~envs.sim.supply_network.SupplyNetwork.get_supply_level`
        across all alive units in that division.  Returns ``0.0`` for a
        division with no alive units.

        Parameters
        ----------
        inner:
            The :class:`~envs.multi_battalion_env.MultiBattalionEnv`
            instance.
        """
        total_per_div = self.n_brigades_per_division * self.n_blue_per_brigade
        levels: List[float] = []
        for i in range(self.n_divisions):
            div_levels: List[float] = []
            for j in range(total_per_div):
                agent_id = f"blue_{i * total_per_div + j}"
                if agent_id in inner._battalions and agent_id in inner._alive:
                    b = inner._battalions[agent_id]
                    div_levels.append(
                        self.supply_network.get_supply_level(b.x, b.y, team=0)
                    )
            levels.append(float(np.mean(div_levels)) if div_levels else 0.0)
        return levels

    # ------------------------------------------------------------------
    # Default objectives
    # ------------------------------------------------------------------

    def _default_objectives(self) -> List[OperationalObjective]:
        """Return three default operational objectives for this map."""
        capture_radius = min(self.map_width, self.map_height) * 0.05
        return [
            OperationalObjective(
                x=self.map_width * 0.5,
                y=self.map_height * 0.5,
                radius=capture_radius,
                obj_type=ObjectiveType.CAPTURE_HEX,
            ),
            OperationalObjective(
                x=self.map_width * 0.8,
                y=self.map_height * 0.5,
                radius=capture_radius,
                obj_type=ObjectiveType.CUT_SUPPLY_LINE,
            ),
            OperationalObjective(
                x=self.map_width * 0.5,
                y=self.map_height * 0.5,
                radius=self.map_height * 0.5,  # covers whole height
                obj_type=ObjectiveType.FIX_AND_FLANK,
            ),
        ]

    # ------------------------------------------------------------------
    # Observation bounds
    # ------------------------------------------------------------------

    def _build_obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lows: list[float] = []
        highs: list[float] = []

        # Corps sector control: [0, 1] × N_CORPS_SECTORS
        lows.extend([0.0] * N_CORPS_SECTORS)
        highs.extend([1.0] * N_CORPS_SECTORS)

        # Per-division status: [avg_strength, avg_morale, alive_ratio]
        for _ in range(self.n_divisions):
            lows.extend([0.0, 0.0, 0.0])
            highs.extend([1.0, 1.0, 1.0])

        # Per-division threat vector: [dist, cos, sin, e_str, e_mor]
        for _ in range(self.n_divisions):
            lows.extend([0.0, -1.0, -1.0, 0.0, 0.0])
            highs.extend([1.0, 1.0, 1.0, 1.0, 1.0])

        # Road usage: [blue_fraction, red_fraction]
        lows.extend([0.0, 0.0])
        highs.extend([1.0, 1.0])

        # Objective control: [0, 1] × N_OBJECTIVES
        lows.extend([0.0] * N_OBJECTIVES)
        highs.extend([1.0] * N_OBJECTIVES)

        # Supply level per Blue division: [0, 1]
        lows.extend([0.0] * self.n_divisions)
        highs.extend([1.0] * self.n_divisions)

        # Step progress: [0, 1]
        lows.append(0.0)
        highs.append(1.0)

        return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial corps observation.

        Parameters
        ----------
        seed:
            RNG seed forwarded to the inner :class:`~envs.division_env.DivisionEnv`.
        options:
            Unused; present for Gymnasium API compatibility.
        """
        if seed is not None:
            super().reset(seed=seed)

        self._division.reset(seed=seed, options=options)
        # Re-attach road network after inner reset (terrain is regenerated)
        self._set_inner_road_network()

        # Reset objective state
        for obj in self.objectives:
            obj.reset()

        # Reset supply network
        self.supply_network.reset()

        self._corps_steps = 0
        return self._get_corps_obs(), {}

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(
        self,
        corps_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one corps macro-step.

        Translates the per-division operational command into a division-level
        action and delegates to the inner :class:`~envs.division_env.DivisionEnv`.

        Parameters
        ----------
        corps_action:
            Integer array of shape ``(n_divisions,)`` with operational
            command indices in ``[0, n_corps_options)``.

        Returns
        -------
        obs : np.ndarray — corps observation after the macro-step
        reward : float — pass-through from DivisionEnv plus objective bonuses
        terminated : bool
        truncated : bool
        info : dict — includes ``corps_steps``, ``division_action``,
            ``objective_rewards``, plus all fields from DivisionEnv
        """
        corps_action = np.asarray(corps_action, dtype=np.int64)

        if corps_action.shape != (self.n_divisions,):
            raise ValueError(
                f"corps_action has shape {corps_action.shape!r}, "
                f"expected ({self.n_divisions},)."
            )

        for i, cmd in enumerate(corps_action):
            if int(cmd) < 0 or int(cmd) >= self.n_corps_options:
                raise ValueError(
                    f"Invalid corps command {int(cmd)!r} for division {i}; "
                    f"expected integer in [0, {self.n_corps_options - 1}]."
                )

        # Translate corps commands → DivisionEnv action
        division_action = self._translate_corps_action(corps_action)

        # Delegate to DivisionEnv
        _div_obs, base_reward, terminated, truncated, div_info = self._division.step(
            division_action
        )

        # Update objective states
        inner = self._division._brigade._inner
        for obj in self.objectives:
            if obj.obj_type != ObjectiveType.FIX_AND_FLANK:
                obj.update(inner)

        # ── Supply network step ───────────────────────────────────────
        # Collect alive unit positions for both teams
        blue_positions = []
        red_positions = []
        for agent_id, b in inner._battalions.items():
            if agent_id not in inner._alive:
                continue
            if agent_id.startswith("blue_"):
                blue_positions.append((b.x, b.y))
            else:
                red_positions.append((b.x, b.y))

        # Check supply-line interdiction BEFORE step() so that captured depots
        # are already dead when consumption and convoy transfers are computed
        # for this tick — ensuring immediate effect on the same step.
        capture_radius = self._interdiction_radius()
        for bx, by in blue_positions:
            self.supply_network.interdict_nearest_depot(
                bx, by, enemy_team=1, capture_radius=capture_radius
            )

        self.supply_network.step(blue_positions, red_positions)

        # Compute supply levels per Blue division for the info dict
        supply_levels = self._compute_division_supply_levels(inner)

        # Compute operational objective rewards
        obj_reward, obj_details = self._compute_objective_rewards(inner)
        total_reward = float(base_reward) + obj_reward

        self._corps_steps += 1

        # Count alive units for operational casualty tracking.
        blue_alive = sum(
            1 for aid in inner._battalions
            if aid in inner._alive and aid.startswith("blue_")
        )
        red_alive = sum(
            1 for aid in inner._battalions
            if aid in inner._alive and not aid.startswith("blue_")
        )

        info: dict = {
            "corps_steps": self._corps_steps,
            "division_action": division_action.tolist(),
            "objective_rewards": obj_details,
            "supply_levels": supply_levels,
            "blue_units_alive": blue_alive,
            "red_units_alive": red_alive,
        }
        info.update(div_info)

        return self._get_corps_obs(), total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Command translation
    # ------------------------------------------------------------------

    def _translate_corps_action(self, corps_action: np.ndarray) -> np.ndarray:
        """Expand a per-division command into a flat per-brigade division action.

        Division *i* covers brigades
        ``[i*n_brigades_per_division : (i+1)*n_brigades_per_division]``.

        Parameters
        ----------
        corps_action:
            Integer array of shape ``(n_divisions,)`` with command indices.

        Returns
        -------
        np.ndarray of shape ``(n_blue_brigades,)`` — option index per brigade.
        """
        division_action = np.empty(self.n_blue_brigades, dtype=np.int64)
        for i in range(self.n_divisions):
            start = i * self.n_brigades_per_division
            end = start + self.n_brigades_per_division
            division_action[start:end] = int(corps_action[i])
        return division_action

    # ------------------------------------------------------------------
    # Objective rewards
    # ------------------------------------------------------------------

    def _compute_objective_rewards(
        self, inner
    ) -> tuple[float, dict]:
        """Compute per-objective reward bonuses for the current step.

        Always returns a stable dict with all three canonical keys
        (``capture_hex``, ``cut_supply_line``, ``fix_and_flank``),
        defaulting to 0.0 for objective types not in ``self.objectives``.

        Returns
        -------
        total_bonus : float
        details : dict  mapping objective name → reward granted this step
        """
        bonus = 0.0
        details: dict[str, float] = {
            "capture_hex": 0.0,
            "cut_supply_line": 0.0,
            "fix_and_flank": 0.0,
        }

        for obj in self.objectives:
            if obj.obj_type == ObjectiveType.CAPTURE_HEX:
                if obj.is_blue_controlled:
                    details["capture_hex"] = _OBJ_CAPTURE_REWARD
                    bonus += _OBJ_CAPTURE_REWARD

            elif obj.obj_type == ObjectiveType.CUT_SUPPLY_LINE:
                if obj.is_blue_controlled:
                    details["cut_supply_line"] = _OBJ_SUPPLY_CUT_REWARD
                    bonus += _OBJ_SUPPLY_CUT_REWARD

            elif obj.obj_type == ObjectiveType.FIX_AND_FLANK:
                active = _detect_fix_and_flank(inner, self.map_height)
                if active:
                    details["fix_and_flank"] = _OBJ_FIX_FLANK_REWARD
                    bonus += _OBJ_FIX_FLANK_REWARD

        return bonus, details

    # ------------------------------------------------------------------
    # Fog-of-war radius hook
    # ------------------------------------------------------------------

    def _get_fog_radius(self) -> float:
        """Return the effective fog-of-war comm radius for threat vectors.

        Subclasses may override this method to implement cavalry
        reconnaissance or other intelligence assets that extend the
        effective communication range for threat vector building.
        """
        return self.comm_radius

    # ------------------------------------------------------------------
    # Corps observation construction
    # ------------------------------------------------------------------

    def _get_corps_obs(self) -> np.ndarray:
        """Build and return the normalised corps observation vector."""
        parts: list[float] = []
        inner = self._division._brigade._inner

        # ── 1. Corps sector control (7 vertical strips) ───────────────
        sector_width = self.map_width / N_CORPS_SECTORS
        for s in range(N_CORPS_SECTORS):
            x_lo = s * sector_width
            x_hi = (s + 1) * sector_width
            blue_str = 0.0
            red_str = 0.0
            for agent_id, b in inner._battalions.items():
                if agent_id not in inner._alive:
                    continue
                in_sector = (x_lo <= b.x < x_hi) or (
                    s == N_CORPS_SECTORS - 1 and b.x == self.map_width
                )
                if in_sector:
                    if agent_id.startswith("blue_"):
                        blue_str += float(b.strength)
                    else:
                        red_str += float(b.strength)
            total = blue_str + red_str
            parts.append(blue_str / total if total > 0.0 else 0.5)

        # ── 2. Per-division status [avg_strength, avg_morale, alive_ratio] ──
        for i in range(self.n_divisions):
            strengths: list[float] = []
            morales: list[float] = []
            alive_count = 0
            total_per_div = self.n_brigades_per_division * self.n_blue_per_brigade
            for j in range(total_per_div):
                agent_id = f"blue_{i * total_per_div + j}"
                if agent_id in inner._battalions and agent_id in inner._alive:
                    b = inner._battalions[agent_id]
                    strengths.append(float(b.strength))
                    morales.append(float(b.morale))
                    alive_count += 1
            avg_str = float(np.mean(strengths)) if strengths else 0.0
            avg_mor = float(np.mean(morales)) if morales else 0.0
            alive_ratio = alive_count / total_per_div
            parts.extend([avg_str, avg_mor, alive_ratio])

        # ── 3. Per-division threat vector ──────────────────────────────
        red_div_centroids = self._get_red_division_centroids(inner)
        blue_div_centroids = self._get_blue_division_centroids(inner)

        for i in range(self.n_divisions):
            cx, cy = blue_div_centroids[i] if blue_div_centroids[i] else (None, None)

            if cx is None or not red_div_centroids:
                # This division is dead or no Red divisions alive — sentinel
                parts.extend([1.0, 0.0, 0.0, 0.0, 0.0])
                continue

            # Find nearest Red division centroid
            best_dist = float("inf")
            best_centroid = None
            best_e_str = 0.0
            best_e_mor = 0.0
            for (rx, ry, e_str, e_mor) in red_div_centroids:
                dx = rx - cx
                dy = ry - cy
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_dist:
                    best_dist = d
                    best_centroid = (rx, ry)
                    best_e_str = e_str
                    best_e_mor = e_mor

            assert best_centroid is not None

            # Apply comm_radius gating (subclasses may override via _get_fog_radius)
            if best_dist > self._get_fog_radius():
                # Enemy beyond communication range — sentinel
                parts.extend([1.0, 0.0, 0.0, 0.0, 0.0])
                continue

            dx = best_centroid[0] - cx
            dy = best_centroid[1] - cy
            bearing = math.atan2(dy, dx)
            parts.append(min(best_dist / self.map_diagonal, 1.0))
            parts.append(math.cos(bearing))
            parts.append(math.sin(bearing))
            parts.append(best_e_str)
            parts.append(best_e_mor)

        # ── 4. Road usage ─────────────────────────────────────────────
        blue_positions: list[tuple[float, float]] = []
        red_positions: list[tuple[float, float]] = []
        for agent_id, b in inner._battalions.items():
            if agent_id not in inner._alive:
                continue
            if agent_id.startswith("blue_"):
                blue_positions.append((b.x, b.y))
            else:
                red_positions.append((b.x, b.y))
        parts.append(self.road_network.fraction_on_road(blue_positions))
        parts.append(self.road_network.fraction_on_road(red_positions))

        # ── 5. Objective control ──────────────────────────────────────
        # Always emit exactly N_OBJECTIVES (3) slots for a stable obs schema.
        # Aggregate by type: last objective of each type wins; missing → 0.5/0.0.
        obj_values: dict[ObjectiveType, float] = {}
        fix_flank_active = _detect_fix_and_flank(inner, self.map_height)
        for obj in self.objectives:
            if obj.obj_type == ObjectiveType.FIX_AND_FLANK:
                obj_values[ObjectiveType.FIX_AND_FLANK] = 1.0 if fix_flank_active else 0.0
            else:
                obj_values[obj.obj_type] = float(obj.control_value)

        parts.append(obj_values.get(ObjectiveType.CAPTURE_HEX, 0.5))
        parts.append(obj_values.get(ObjectiveType.CUT_SUPPLY_LINE, 0.5))
        parts.append(obj_values.get(ObjectiveType.FIX_AND_FLANK, 0.0))

        # ── 6. Supply level per Blue division ─────────────────────────
        supply_levels = self._compute_division_supply_levels(inner)
        parts.extend(supply_levels)

        # ── 7. Step progress ──────────────────────────────────────────
        parts.append(min(inner._step_count / self.max_steps, 1.0))

        obs = np.array(parts, dtype=np.float32)
        return np.clip(obs, self._corps_obs_low, self._corps_obs_high)

    # ------------------------------------------------------------------
    # Division centroid helpers
    # ------------------------------------------------------------------

    def _get_blue_division_centroids(
        self, inner
    ) -> list[Optional[tuple[float, float]]]:
        """Return ``(cx, cy)`` centroid for each Blue division, or ``None`` if dead."""
        centroids: list[Optional[tuple[float, float]]] = []
        total_per_div = self.n_brigades_per_division * self.n_blue_per_brigade
        for i in range(self.n_divisions):
            xs: list[float] = []
            ys: list[float] = []
            for j in range(total_per_div):
                agent_id = f"blue_{i * total_per_div + j}"
                if agent_id in inner._battalions and agent_id in inner._alive:
                    b = inner._battalions[agent_id]
                    xs.append(b.x)
                    ys.append(b.y)
            if xs:
                centroids.append((float(np.mean(xs)), float(np.mean(ys))))
            else:
                centroids.append(None)
        return centroids

    def _get_red_division_centroids(
        self, inner
    ) -> list[tuple[float, float, float, float]]:
        """Return ``(cx, cy, avg_strength, avg_morale)`` for each alive Red division."""
        centroids = []
        total_per_div = self.n_red_brigades_per_division * self.n_red_per_brigade
        for i in range(self.n_red_divisions):
            xs: list[float] = []
            ys: list[float] = []
            strs: list[float] = []
            mors: list[float] = []
            for j in range(total_per_div):
                agent_id = f"red_{i * total_per_div + j}"
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
    # Gymnasium API: close / render
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""
        self._division.close()

    def render(self) -> None:
        """Rendering is not yet implemented."""
