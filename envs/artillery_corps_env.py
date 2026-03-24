# envs/artillery_corps_env.py
"""Artillery Corps Environment — CorpsEnv extended with an independent artillery arm.

:class:`ArtilleryCorpsEnv` wraps :class:`~envs.corps_env.CorpsEnv` and
introduces an independent artillery echelon at the corps level.  Artillery
batteries execute four operational missions — grand battery, counter-battery,
siege, and fortification construction — and their effects are woven into the
corps observation.

Architecture
------------
::

    ArtilleryCorpsEnv (Gymnasium)          ← artillery + corps PPO agent
        │
        └─ CorpsEnv (Gymnasium)            ← base corps commander
              │
              └─ DivisionEnv / BrigadeEnv / MultiBattalionEnv

Observation space
-----------------
The observation is the base :class:`~envs.corps_env.CorpsEnv` observation
vector (see that module's docstring) extended with:

* Per-artillery-battery state (``_ART_UNIT_OBS_DIM = 4`` floats each):
  ``[x_norm, y_norm, mission_norm, strength]``.
* Artillery summary (``_ART_SUMMARY_DIM = 4`` floats):
  ``[morale_dmg_norm, guns_silenced_norm, fort_dmg_norm, forts_built_norm]``.
* Fortification state (``_ART_FORT_OBS_DIM = 3`` floats per fort slot,
  ``n_artillery_batteries`` slots):
  ``[x_norm, y_norm, hp]`` for each fortification slot (zero-padded).

Action space
------------
``MultiDiscrete([n_corps_options] * n_divisions + [N_ARTILLERY_MISSIONS] * n_artillery_batteries)``

The first ``n_divisions`` elements are the standard corps operational
commands (forwarded to :class:`~envs.corps_env.CorpsEnv`).  The remaining
``n_artillery_batteries`` elements select the artillery mission
(``0``=IDLE, ``1``=GRAND_BATTERY, ``2``=COUNTER_BATTERY, ``3``=SIEGE,
``4``=FORTIFY) for each battery.

Typical usage::

    from envs.artillery_corps_env import ArtilleryCorpsEnv

    env = ArtilleryCorpsEnv(n_divisions=3, n_brigades_per_division=3,
                            n_artillery_batteries=6)
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
from envs.sim.artillery_corps import (
    ArtilleryCorps,
    ArtilleryMission,
    ArtilleryReport,
    ArtilleryUnit,
    ArtilleryUnitConfig,
    Fortification,
    N_ARTILLERY_MISSIONS,
)
from envs.sim.road_network import RoadNetwork
from envs.sim.supply_network import SupplyNetwork

__all__ = [
    "ArtilleryCorpsEnv",
    "N_ARTILLERY_MISSIONS",
    "_art_obs_dim",
    "_ART_UNIT_OBS_DIM",
    "_ART_SUMMARY_DIM",
    "_ART_FORT_OBS_DIM",
]

# ---------------------------------------------------------------------------
# Observation dimension helpers
# ---------------------------------------------------------------------------

#: Floats per artillery battery in the observation vector.
#: ``[x_norm, y_norm, mission_norm, strength]``
_ART_UNIT_OBS_DIM: int = 4

#: Floats for the artillery operational summary.
#: ``[morale_dmg_norm, guns_silenced_norm, fort_dmg_norm, forts_built_norm]``
_ART_SUMMARY_DIM: int = 4

#: Floats per fortification slot in the observation vector.
#: ``[x_norm, y_norm, hp]``
_ART_FORT_OBS_DIM: int = 3

#: Normalisation denominator for morale damage summary (caps the range).
_MORALE_DMG_NORM: float = 2.0

#: Normalisation denominator for fortification HP damage summary.
_FORT_DMG_NORM: float = 1.0


def _art_obs_dim(n_divisions: int, n_artillery_batteries: int) -> int:
    """Return the flat observation dimension for an artillery corps env."""
    return (
        _corps_obs_dim(n_divisions)
        + n_artillery_batteries * _ART_UNIT_OBS_DIM
        + _ART_SUMMARY_DIM
        + n_artillery_batteries * _ART_FORT_OBS_DIM  # fort slots
    )


# ---------------------------------------------------------------------------
# ArtilleryCorpsEnv
# ---------------------------------------------------------------------------


class ArtilleryCorpsEnv(CorpsEnv):
    """Gymnasium environment for a corps commander with an independent artillery arm.

    Extends :class:`~envs.corps_env.CorpsEnv` with artillery batteries that
    execute grand battery, counter-battery, siege, and fortification missions
    each step.

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
        Base inter-division communication radius.
    red_random:
        When ``True`` Red takes random brigade actions.
    randomize_terrain:
        Pass-through to inner env.
    visibility_radius:
        Fog-of-war visibility radius for the inner simulation.
    render_mode:
        ``None`` or ``"human"``.
    n_artillery_batteries:
        Number of Blue artillery batteries (default ``4``).
    artillery_corps:
        Optional pre-built :class:`~envs.sim.artillery_corps.ArtilleryCorps`.
        If ``None``, a default corps is generated at construction time.
    art_config:
        Optional :class:`~envs.sim.artillery_corps.ArtilleryUnitConfig` used
        when generating the default artillery corps.
    n_red_artillery_batteries:
        Number of Red artillery batteries for counter-battery targeting
        (default equals *n_artillery_batteries*).
    """

    metadata: dict = {"render_modes": ["human"], "name": "artillery_corps_v0"}

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
        n_artillery_batteries: int = 4,
        artillery_corps: Optional[ArtilleryCorps] = None,
        art_config: Optional[ArtilleryUnitConfig] = None,
        n_red_artillery_batteries: Optional[int] = None,
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

        # ── Blue artillery configuration ─────────────────────────────
        if artillery_corps is not None:
            try:
                n_art_from_corps = len(artillery_corps.units)
            except AttributeError as exc:
                raise ValueError(
                    "Provided artillery_corps must be a valid ArtilleryCorps "
                    "instance with a 'units' attribute."
                ) from exc
            if n_art_from_corps < 1:
                raise ValueError(
                    "Provided artillery_corps must contain at least one battery."
                )
            self.n_artillery_batteries: int = int(n_art_from_corps)
            self._artillery: ArtilleryCorps = artillery_corps
        else:
            if int(n_artillery_batteries) < 1:
                raise ValueError(
                    f"n_artillery_batteries must be >= 1, "
                    f"got {n_artillery_batteries!r}"
                )
            self.n_artillery_batteries: int = int(n_artillery_batteries)
            self._artillery: ArtilleryCorps = ArtilleryCorps.generate_default(
                map_width=self.map_width,
                map_height=self.map_height,
                n_batteries=self.n_artillery_batteries,
                team=0,
                config=art_config,
            )

        # ── Red artillery (for counter-battery targeting) ────────────
        n_red_art = (
            n_red_artillery_batteries
            if n_red_artillery_batteries is not None
            else self.n_artillery_batteries
        )
        if int(n_red_art) < 1:
            raise ValueError(
                f"n_red_artillery_batteries must be >= 1, got {n_red_art!r}"
            )
        self._red_artillery: ArtilleryCorps = ArtilleryCorps.generate_default(
            map_width=self.map_width,
            map_height=self.map_height,
            n_batteries=int(n_red_art),
            team=1,
            config=ArtilleryUnitConfig(team=1),
        )

        self._last_art_report: ArtilleryReport = ArtilleryReport(
            morale_damage_dealt=0.0,
            guns_silenced=0,
            fortification_damage=0.0,
            fortifications_completed=0,
        )

        # ── Override action space ────────────────────────────────────
        self.action_space = spaces.MultiDiscrete(
            [self.n_corps_options] * self.n_divisions
            + [N_ARTILLERY_MISSIONS] * self.n_artillery_batteries,
            dtype=np.int64,
        )

        # ── Override observation space ───────────────────────────────
        obs_low, obs_high = self._build_art_obs_bounds()
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial artillery corps observation.

        Calls :meth:`~envs.corps_env.CorpsEnv.reset` on the base env,
        resets artillery positions, and returns the extended observation.
        """
        _, info = super().reset(seed=seed, options=options)

        # Reset Blue artillery to default positions
        for i, unit in enumerate(
            self._artillery.units[: self.n_artillery_batteries]
        ):
            unit.x = self.map_width * 0.25
            unit.y = (
                self.map_height
                * (i + 1)
                / (self.n_artillery_batteries + 1)
            )
            unit.theta = 0.0
            unit.strength = 1.0
            unit.alive = True
            unit.mission = ArtilleryMission.IDLE
            unit._fortify_progress = 0
        self._artillery.fortifications.clear()

        # Reset Red artillery
        for i, unit in enumerate(self._red_artillery.units):
            unit.x = self.map_width * 0.75
            unit.y = (
                self.map_height
                * (i + 1)
                / (len(self._red_artillery.units) + 1)
            )
            unit.theta = math.pi
            unit.strength = 1.0
            unit.alive = True
            unit.mission = ArtilleryMission.IDLE
            unit._fortify_progress = 0
        self._red_artillery.fortifications.clear()

        self._last_art_report = ArtilleryReport(
            morale_damage_dealt=0.0,
            guns_silenced=0,
            fortification_damage=0.0,
            fortifications_completed=0,
        )
        return self._build_art_obs(), info

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one artillery-corps macro-step.

        Splits the combined action into corps commands and artillery mission
        assignments.  Infantry and supply logic executes first (via the
        base :class:`~envs.corps_env.CorpsEnv`); artillery then acts on the
        updated battlefield state.

        Parameters
        ----------
        action:
            Integer array of shape
            ``(n_divisions + n_artillery_batteries,)``.
            Elements ``[:n_divisions]`` are standard corps operational
            commands.  Elements ``[n_divisions:]`` assign an
            :class:`~envs.sim.artillery_corps.ArtilleryMission` to each
            battery.

        Returns
        -------
        obs : np.ndarray — extended artillery corps observation
        reward : float — base corps reward + fortification reward shaping
        terminated : bool
        truncated : bool
        info : dict — base corps info plus ``"artillery"`` sub-dict
        """
        action = np.asarray(action, dtype=np.int64)
        expected_len = self.n_divisions + self.n_artillery_batteries
        if action.shape != (expected_len,):
            raise ValueError(
                f"action has shape {action.shape!r}, "
                f"expected ({expected_len},)."
            )

        # Split action
        corps_action = action[: self.n_divisions]
        art_action = action[self.n_divisions :]

        # Assign Blue artillery missions for this step
        for i, unit in enumerate(
            self._artillery.units[: self.n_artillery_batteries]
        ):
            if unit.alive:
                unit.mission = ArtilleryMission(int(art_action[i]))
            else:
                unit.mission = ArtilleryMission.IDLE

        # ── Execute base corps step (infantry + supply) ───────────────
        _, base_reward, terminated, truncated, info = super().step(corps_action)

        # ── Execute Blue artillery step ───────────────────────────────
        inner = self._division._brigade._inner
        self._last_art_report = self._artillery.step(
            inner,
            enemy_artillery=list(self._red_artillery.units),
            enemy_fortifications=list(self._red_artillery.fortifications),
        )

        # ── Reward shaping: bonus for completing fortifications ───────
        fort_bonus = self._last_art_report.fortifications_completed * 0.1
        total_reward = base_reward + fort_bonus

        # ── Augment info with artillery report ────────────────────────
        info["artillery"] = {
            "morale_damage_dealt": self._last_art_report.morale_damage_dealt,
            "guns_silenced": self._last_art_report.guns_silenced,
            "fortification_damage": self._last_art_report.fortification_damage,
            "fortifications_completed": self._last_art_report.fortifications_completed,
            "n_blue_forts": len(self._artillery.fortifications),
            "n_red_forts": len(self._red_artillery.fortifications),
        }

        return (
            self._build_art_obs(),
            total_reward,
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_art_obs(self) -> np.ndarray:
        """Build and return the full artillery corps observation vector.

        Concatenates:
        1. Base corps observation.
        2. Per-artillery-battery state.
        3. Artillery operational summary.
        4. Fortification slot states (Blue forts, zero-padded).
        """
        base_obs: np.ndarray = self._get_corps_obs()

        # ── Per-battery state ─────────────────────────────────────────
        art_parts: List[float] = []
        for unit in self._artillery.units[: self.n_artillery_batteries]:
            if unit.alive:
                art_parts.extend(
                    [
                        unit.x / self.map_width,
                        unit.y / self.map_height,
                        float(int(unit.mission)) / (N_ARTILLERY_MISSIONS - 1),
                        unit.strength,
                    ]
                )
            else:
                art_parts.extend([0.0, 0.0, 0.0, 0.0])

        # ── Artillery operational summary ─────────────────────────────
        r = self._last_art_report
        art_parts.extend(
            [
                min(1.0, r.morale_damage_dealt / _MORALE_DMG_NORM),
                min(1.0, float(r.guns_silenced) / max(1, len(self._red_artillery.units))),
                min(1.0, r.fortification_damage / _FORT_DMG_NORM),
                min(1.0, float(r.fortifications_completed)),
            ]
        )

        # ── Fortification slots (n_artillery_batteries slots, Blue forts) ──
        forts = self._artillery.fortifications
        for slot in range(self.n_artillery_batteries):
            if slot < len(forts):
                fort = forts[slot]
                art_parts.extend(
                    [
                        fort.x / self.map_width,
                        fort.y / self.map_height,
                        fort.hp,
                    ]
                )
            else:
                art_parts.extend([0.0, 0.0, 0.0])

        art_arr = np.array(art_parts, dtype=np.float32)
        obs = np.concatenate([base_obs, art_arr])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    # Observation bounds
    # ------------------------------------------------------------------

    def _build_art_obs_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)`` bounds for the full artillery corps observation."""
        base_low, base_high = self._build_obs_bounds()

        # Extra dims: all in [0, 1]
        n_extra = (
            self.n_artillery_batteries * _ART_UNIT_OBS_DIM
            + _ART_SUMMARY_DIM
            + self.n_artillery_batteries * _ART_FORT_OBS_DIM
        )
        extra_low = np.zeros(n_extra, dtype=np.float32)
        extra_high = np.ones(n_extra, dtype=np.float32)

        return (
            np.concatenate([base_low, extra_low]),
            np.concatenate([base_high, extra_high]),
        )
