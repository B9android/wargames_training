# envs/cavalry_corps_env.py
"""Cavalry Corps Environment — CorpsEnv extended with an independent cavalry arm.

:class:`CavalryCorpsEnv` wraps :class:`~envs.corps_env.CorpsEnv` and
introduces an independent cavalry echelon at the corps level.  Cavalry
brigades execute three operational missions each step — reconnaissance,
deep raiding, and pursuit of routing infantry — and their intelligence is
woven into the corps observation to reduce the fog of war.

Architecture
------------
::

    CavalryCorpsEnv (Gymnasium)            ← cavalry + corps PPO agent
        │
        └─ CorpsEnv (Gymnasium)            ← base corps commander
              │
              └─ DivisionEnv / BrigadeEnv / MultiBattalionEnv

Observation space
-----------------
The observation is the base :class:`~envs.corps_env.CorpsEnv` observation
vector (see that module's docstring) extended with:

* Per-cavalry-brigade state (``_CAV_UNIT_OBS_DIM = 4`` floats each):
  ``[x_norm, y_norm, mission_norm, strength]``.
* Cavalry intelligence summary (``_CAV_INTEL_DIM = 3`` floats):
  ``[n_revealed_norm, centroid_x_norm, centroid_y_norm]``.

Additionally, when at least one cavalry brigade is on RECONNAISSANCE
mission and has revealed enemy positions, the fog-of-war comm_radius
gating for threat vectors is lifted — allied divisions receive accurate
enemy threat vectors rather than sentinels.

Action space
------------
``MultiDiscrete([n_corps_options] * n_divisions + [N_CAVALRY_MISSIONS] * n_cavalry_brigades)``

The first ``n_divisions`` elements are the standard corps operational
commands (forwarded to :class:`~envs.corps_env.CorpsEnv`).  The
remaining ``n_cavalry_brigades`` elements select the cavalry mission
(``0``=IDLE, ``1``=RECON, ``2``=RAIDING, ``3``=PURSUIT) for each brigade.

Typical usage::

    from envs.cavalry_corps_env import CavalryCorpsEnv

    env = CavalryCorpsEnv(n_divisions=3, n_brigades_per_division=3,
                          n_cavalry_brigades=2)
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
from typing import List, Optional

import numpy as np
from gymnasium import spaces

from envs.corps_env import (
    CorpsEnv,
    CORPS_MAP_WIDTH,
    CORPS_MAP_HEIGHT,
    _corps_obs_dim,
)
from envs.multi_battalion_env import MAX_STEPS
from envs.sim.cavalry_corps import (
    CavalryCorps,
    CavalryMission,
    CavalryReport,
    CavalryUnitConfig,
    N_CAVALRY_MISSIONS,
)
from envs.sim.road_network import RoadNetwork
from envs.sim.supply_network import SupplyNetwork

__all__ = [
    "CavalryCorpsEnv",
    "N_CAVALRY_MISSIONS",
    "_cav_obs_dim",
    "_CAV_UNIT_OBS_DIM",
    "_CAV_INTEL_DIM",
]

# ---------------------------------------------------------------------------
# Observation dimension helpers
# ---------------------------------------------------------------------------

#: Floats per cavalry brigade in the observation vector.
#: ``[x_norm, y_norm, mission_norm, strength]``
_CAV_UNIT_OBS_DIM: int = 4

#: Floats for the cavalry intelligence summary.
#: ``[n_revealed_norm, centroid_x_norm, centroid_y_norm]``
_CAV_INTEL_DIM: int = 3


def _cav_obs_dim(n_divisions: int, n_cavalry_brigades: int) -> int:
    """Return the flat observation dimension for a cavalry corps env."""
    return (
        _corps_obs_dim(n_divisions)
        + n_cavalry_brigades * _CAV_UNIT_OBS_DIM
        + _CAV_INTEL_DIM
    )


# ---------------------------------------------------------------------------
# CavalryCorpsEnv
# ---------------------------------------------------------------------------


class CavalryCorpsEnv(CorpsEnv):
    """Gymnasium environment for a corps commander with an independent cavalry arm.

    Extends :class:`~envs.corps_env.CorpsEnv` with cavalry brigades that
    execute reconnaissance, raiding, and pursuit missions each step.
    Cavalry intelligence reduces the fog-of-war for allied divisions
    when units are on RECONNAISSANCE mission.

    Parameters
    ----------
    n_divisions:
        Number of Blue infantry divisions.
    n_brigades_per_division:
        Blue infantry brigades per division.
    n_blue_per_brigade:
        Blue battalions per brigade.
    n_red_divisions, n_red_brigades_per_division, n_red_per_brigade:
        Red force composition (mirrors Blue by default).
    map_width, map_height:
        Map dimensions in metres.
    max_steps:
        Episode length cap.
    road_network, supply_network, objectives:
        Optional overrides passed through to :class:`~envs.corps_env.CorpsEnv`.
    comm_radius:
        Base inter-division communication radius.  Overridden to ``∞``
        when cavalry recon reveals enemy positions.
    red_random:
        When ``True`` Red takes random brigade actions.
    randomize_terrain:
        Pass-through to inner env.
    visibility_radius:
        Fog-of-war visibility radius for the inner simulation.
    render_mode:
        ``None`` or ``"human"``.
    n_cavalry_brigades:
        Number of Blue cavalry brigades (default ``2``).
    cavalry_corps:
        Optional pre-built :class:`~envs.sim.cavalry_corps.CavalryCorps`.
        If ``None``, a default corps is generated at construction time.
    cav_config:
        Optional :class:`~envs.sim.cavalry_corps.CavalryUnitConfig` used
        when generating the default cavalry corps.
    """

    metadata: dict = {"render_modes": ["human"], "name": "cavalry_corps_v0"}

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
        objectives=None,
        comm_radius: float = 3_000.0,
        red_random: bool = True,
        randomize_terrain: bool = True,
        visibility_radius: float = 1_500.0,
        render_mode: Optional[str] = None,
        n_cavalry_brigades: int = 2,
        cavalry_corps: Optional[CavalryCorps] = None,
        cav_config: Optional[CavalryUnitConfig] = None,
    ) -> None:
        # ── Build base CorpsEnv ──────────────────────────────────────
        super().__init__(
            n_divisions=n_divisions,
            n_brigades_per_division=n_brigades_per_division,
            n_blue_per_brigade=n_blue_per_brigade,
            n_red_divisions=n_red_divisions,
            n_red_brigades_per_division=n_red_brigades_per_division,
            n_red_per_brigade=n_red_per_brigade,
            map_width=map_width,
            map_height=map_height,
            max_steps=max_steps,
            road_network=road_network,
            supply_network=supply_network,
            objectives=objectives,
            comm_radius=comm_radius,
            red_random=red_random,
            randomize_terrain=randomize_terrain,
            visibility_radius=visibility_radius,
            render_mode=render_mode,
        )

        # ── Cavalry configuration ────────────────────────────────────
        if int(n_cavalry_brigades) < 1:
            raise ValueError(
                f"n_cavalry_brigades must be >= 1, got {n_cavalry_brigades!r}"
            )
        self.n_cavalry_brigades: int = int(n_cavalry_brigades)

        self._cavalry: CavalryCorps = (
            cavalry_corps
            if cavalry_corps is not None
            else CavalryCorps.generate_default(
                map_width=self.map_width,
                map_height=self.map_height,
                n_brigades=self.n_cavalry_brigades,
                team=0,
                config=cav_config,
            )
        )
        self._last_cav_report: CavalryReport = CavalryReport([], 0, 0, 0.0)

        # ── Override action space ────────────────────────────────────
        # Corps commands (n_divisions) + cavalry missions (n_cavalry_brigades)
        self.action_space = spaces.MultiDiscrete(
            [self.n_corps_options] * self.n_divisions
            + [N_CAVALRY_MISSIONS] * self.n_cavalry_brigades,
            dtype=np.int64,
        )

        # ── Override observation space ───────────────────────────────
        obs_low, obs_high = self._build_cav_obs_bounds()
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Fog-of-war hook (overrides CorpsEnv._get_fog_radius)
    # ------------------------------------------------------------------

    def _get_fog_radius(self) -> float:
        """Lift the comm_radius gating when cavalry recon has revealed enemies.

        When at least one enemy unit has been spotted by a RECONNAISSANCE
        cavalry brigade, the effective fog radius is set to ``math.inf`` —
        all allied divisions receive accurate threat vectors rather than
        sentinels.  When no enemies are revealed, the base ``comm_radius``
        is used.
        """
        if self._last_cav_report.revealed_enemy_positions:
            return math.inf
        return self.comm_radius

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial cavalry corps observation.

        Calls :meth:`~envs.corps_env.CorpsEnv.reset` on the base env,
        resets cavalry unit positions, and returns the extended observation.
        """
        # Reset base corps env (also resets inner env, road/supply networks)
        _, info = super().reset(seed=seed, options=options)

        # Reset cavalry positions to default spread
        for i, unit in enumerate(self._cavalry.units[: self.n_cavalry_brigades]):
            unit.x = self.map_width * 0.2
            unit.y = self.map_height * (i + 1) / (self.n_cavalry_brigades + 1)
            unit.theta = 0.0
            unit.strength = 1.0
            unit.alive = True
            unit.mission = CavalryMission.IDLE

        self._last_cav_report = CavalryReport([], 0, 0, 0.0)
        return self._build_cav_obs(), info

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one cavalry-corps macro-step.

        Splits the combined action into corps commands and cavalry mission
        assignments.  Infantry and supply logic executes first (via the
        base :class:`~envs.corps_env.CorpsEnv`); cavalry then acts on the
        updated battlefield state.

        Parameters
        ----------
        action:
            Integer array of shape
            ``(n_divisions + n_cavalry_brigades,)``.
            Elements ``[:n_divisions]`` are standard corps operational
            commands.  Elements ``[n_divisions:]`` assign a
            :class:`~envs.sim.cavalry_corps.CavalryMission` to each
            cavalry brigade.

        Returns
        -------
        obs : np.ndarray — extended cavalry corps observation
        reward : float — base corps reward (cavalry adds no extra reward)
        terminated : bool
        truncated : bool
        info : dict — base corps info plus ``"cavalry"`` sub-dict
        """
        action = np.asarray(action, dtype=np.int64)
        expected_len = self.n_divisions + self.n_cavalry_brigades
        if action.shape != (expected_len,):
            raise ValueError(
                f"action has shape {action.shape!r}, "
                f"expected ({expected_len},)."
            )

        # Split action
        corps_action = action[: self.n_divisions]
        cav_action = action[self.n_divisions :]

        # Assign cavalry missions for this step
        for i, unit in enumerate(self._cavalry.units[: self.n_cavalry_brigades]):
            if unit.alive:
                unit.mission = CavalryMission(int(cav_action[i]))

        # ── Execute base corps step (infantry + supply) ───────────────
        _, base_reward, terminated, truncated, info = super().step(corps_action)

        # ── Execute cavalry step on updated battlefield state ─────────
        inner = self._division._brigade._inner
        self._last_cav_report = self._cavalry.step(inner, self.supply_network)

        # ── Augment info with cavalry report ─────────────────────────
        info["cavalry"] = {
            "depots_raided": self._last_cav_report.depots_raided,
            "routed_units_pursued": self._last_cav_report.routed_units_pursued,
            "pursuit_damage": self._last_cav_report.pursuit_damage,
            "n_revealed_enemies": len(
                self._last_cav_report.revealed_enemy_positions
            ),
        }

        return self._build_cav_obs(), base_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_cav_obs(self) -> np.ndarray:
        """Build and return the full cavalry corps observation vector.

        Concatenates:
        1. Base corps observation (with cavalry-enhanced fog of war).
        2. Per-cavalry-brigade state.
        3. Cavalry intelligence summary.
        """
        # Base corps observation uses _get_fog_radius() via _get_corps_obs()
        base_obs: np.ndarray = self._get_corps_obs()

        # ── Per-cavalry-brigade state ─────────────────────────────────
        cav_parts: List[float] = []
        for unit in self._cavalry.units[: self.n_cavalry_brigades]:
            if unit.alive:
                cav_parts.extend(
                    [
                        unit.x / self.map_width,
                        unit.y / self.map_height,
                        float(int(unit.mission)) / (N_CAVALRY_MISSIONS - 1),
                        unit.strength,
                    ]
                )
            else:
                cav_parts.extend([0.0, 0.0, 0.0, 0.0])

        # ── Cavalry intelligence summary ──────────────────────────────
        revealed = self._last_cav_report.revealed_enemy_positions
        n_total = max(
            1,
            self.n_red_divisions
            * self.n_red_brigades_per_division
            * self.n_red_per_brigade,
        )
        n_revealed_norm = min(1.0, len(revealed) / n_total)

        if revealed:
            rx_norm = (
                float(np.mean([p[0] for p in revealed])) / self.map_width
            )
            ry_norm = (
                float(np.mean([p[1] for p in revealed])) / self.map_height
            )
        else:
            rx_norm, ry_norm = 0.0, 0.0

        cav_parts.extend([n_revealed_norm, rx_norm, ry_norm])

        cav_arr = np.array(cav_parts, dtype=np.float32)
        obs = np.concatenate([base_obs, cav_arr])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    # Observation bounds
    # ------------------------------------------------------------------

    def _build_cav_obs_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)`` bounds for the full cavalry corps observation."""
        base_low, base_high = self._build_obs_bounds()

        # Extra dims: all in [0, 1]
        n_extra = self.n_cavalry_brigades * _CAV_UNIT_OBS_DIM + _CAV_INTEL_DIM
        extra_low = np.zeros(n_extra, dtype=np.float32)
        extra_high = np.ones(n_extra, dtype=np.float32)

        return (
            np.concatenate([base_low, extra_low]),
            np.concatenate([base_high, extra_high]),
        )
