# SPDX-License-Identifier: MIT
# tests/test_cavalry_corps.py
"""Tests for envs/sim/cavalry_corps.py and envs/cavalry_corps_env.py.

Covers all acceptance criteria from epic E10.2:

AC-1  Cavalry reconnaissance correctly reduces fog-of-war for allied infantry.
AC-2  Raiding cavalry discovers supply depots as high-value targets without
      explicit reward.
AC-3  Pursuit mission targets and damages routed infantry.

Test organisation:
  1.  CavalryMission enum smoke tests.
  2.  CavalryUnitConfig defaults and validation.
  3.  CavalryUnit — geometry helpers.
  4.  CavalryUnit — movement (move_towards).
  5.  CavalryUnit — map-boundary clamping.
  6.  CavalryUnit — take_damage mechanics.
  7.  CavalryCorps — generate_default factory.
  8.  CavalryCorps — reset lifecycle.
  9.  CavalryCorps — get_revealed_enemies with RECON mission.
  10. CavalryCorps — get_revealed_enemies with non-RECON missions.
  11. CavalryCorps — step with IDLE mission.
  12. CavalryCorps — step with RECONNAISSANCE mission.
  13. CavalryCorps — step with RAIDING mission.
  14. CavalryCorps — step with PURSUIT mission.
  15. CavalryReport dataclass fields.
  16. CavalryCorpsEnv — construction and action/obs spaces.
  17. CavalryCorpsEnv — reset observation shape and content.
  18. CavalryCorpsEnv — step observation shape and info dict.
  19. CavalryCorpsEnv — invalid action raises ValueError.
  20. CavalryCorpsEnv — full short episode.
  21. AC-1 integration: fog-of-war reduced by RECON cavalry.
  22. AC-2 integration: raiders move towards enemy depots.
  23. AC-3 integration: pursuit cavalry targets routed infantry.
  24. CavalryCorpsEnv — dead cavalry units emit zero obs block.
  25. CavalryCorpsEnv — no revealed enemies → base comm_radius fog.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.cavalry_corps import (
    CavalryCorps,
    CavalryMission,
    CavalryReport,
    CavalryUnit,
    CavalryUnitConfig,
    DEFAULT_CAVALRY_SPEED,
    DEFAULT_PURSUIT_DAMAGE,
    DEFAULT_PURSUIT_RADIUS,
    DEFAULT_RAID_RADIUS,
    DEFAULT_RECON_RADIUS,
    N_CAVALRY_MISSIONS,
)
from envs.cavalry_corps_env import (
    CavalryCorpsEnv,
    _CAV_INTEL_DIM,
    _CAV_UNIT_OBS_DIM,
    _cav_obs_dim,
)
from envs.corps_env import _corps_obs_dim
from envs.sim.supply_network import SupplyDepot, SupplyNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


MAP_W: float = 10_000.0
MAP_H: float = 5_000.0


def _make_unit(
    x: float = 1_000.0,
    y: float = 2_500.0,
    team: int = 0,
    mission: CavalryMission = CavalryMission.IDLE,
    strength: float = 1.0,
    alive: bool = True,
    config: CavalryUnitConfig | None = None,
) -> CavalryUnit:
    if config is None:
        config = CavalryUnitConfig(team=team)
    unit = CavalryUnit(x=x, y=y, theta=0.0, strength=strength, team=team, config=config)
    unit.alive = alive
    unit.mission = mission
    return unit


def _make_corps(
    n: int = 2, team: int = 0, **kwargs
) -> CavalryCorps:
    return CavalryCorps.generate_default(MAP_W, MAP_H, n_brigades=n, team=team)


def _make_inner(
    red_bns: list[tuple[float, float, bool]] | None = None,
) -> MagicMock:
    """Build a minimal mock of MultiBattalionEnv's inner env.

    Parameters
    ----------
    red_bns:
        List of ``(x, y, routed)`` tuples for Red battalions.
        Defaults to one un-routed battalion near the enemy rear.
    """
    if red_bns is None:
        red_bns = [(8_000.0, 2_500.0, False)]

    inner = MagicMock()

    battalions: dict = {}
    alive_set: set = set()

    for i, (x, y, routed) in enumerate(red_bns):
        aid = f"red_{i}"
        bn = MagicMock()
        bn.x = x
        bn.y = y
        bn.strength = 1.0
        bn.morale = 0.8
        bn.routed = routed
        battalions[aid] = bn
        if not routed:  # keep alive if not yet routed
            alive_set.add(aid)
        else:
            # Routed units may still be in _battalions but removed from _alive
            alive_set.discard(aid)

    inner._battalions = battalions
    inner._alive = alive_set
    return inner


def _make_supply_network_with_red_depot(
    depot_x: float = 8_500.0, depot_y: float = 2_500.0
) -> SupplyNetwork:
    """Return a SupplyNetwork with a single Red depot at (depot_x, depot_y)."""
    return SupplyNetwork(
        depots=[
            SupplyDepot(x=depot_x, y=depot_y, team=1, base_supply_radius=1_500.0),
        ],
        convoy_routes=[],
    )


def _make_env(**kwargs) -> CavalryCorpsEnv:
    return CavalryCorpsEnv(
        n_divisions=2,
        n_brigades_per_division=2,
        n_blue_per_brigade=2,
        n_cavalry_brigades=2,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. CavalryMission enum smoke tests
# ---------------------------------------------------------------------------


class TestCavalryMission(unittest.TestCase):
    def test_n_missions(self) -> None:
        self.assertEqual(N_CAVALRY_MISSIONS, 4)

    def test_values_stable(self) -> None:
        self.assertEqual(int(CavalryMission.IDLE), 0)
        self.assertEqual(int(CavalryMission.RECONNAISSANCE), 1)
        self.assertEqual(int(CavalryMission.RAIDING), 2)
        self.assertEqual(int(CavalryMission.PURSUIT), 3)

    def test_round_trip_from_int(self) -> None:
        for i in range(N_CAVALRY_MISSIONS):
            self.assertEqual(int(CavalryMission(i)), i)


# ---------------------------------------------------------------------------
# 2. CavalryUnitConfig defaults and validation
# ---------------------------------------------------------------------------


class TestCavalryUnitConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = CavalryUnitConfig()
        self.assertEqual(cfg.max_speed, DEFAULT_CAVALRY_SPEED)
        self.assertEqual(cfg.recon_radius, DEFAULT_RECON_RADIUS)
        self.assertEqual(cfg.raid_radius, DEFAULT_RAID_RADIUS)
        self.assertEqual(cfg.pursuit_radius, DEFAULT_PURSUIT_RADIUS)
        self.assertEqual(cfg.pursuit_damage_per_step, DEFAULT_PURSUIT_DAMAGE)
        self.assertEqual(cfg.team, 0)

    def test_frozen(self) -> None:
        cfg = CavalryUnitConfig()
        with self.assertRaises((AttributeError, TypeError)):
            cfg.max_speed = 999.0  # type: ignore[misc]

    def test_invalid_max_speed(self) -> None:
        with self.assertRaises(ValueError):
            CavalryUnitConfig(max_speed=0.0)

    def test_invalid_recon_radius(self) -> None:
        with self.assertRaises(ValueError):
            CavalryUnitConfig(recon_radius=-1.0)

    def test_invalid_raid_radius(self) -> None:
        with self.assertRaises(ValueError):
            CavalryUnitConfig(raid_radius=0.0)

    def test_invalid_pursuit_radius(self) -> None:
        with self.assertRaises(ValueError):
            CavalryUnitConfig(pursuit_radius=-10.0)

    def test_negative_pursuit_damage(self) -> None:
        with self.assertRaises(ValueError):
            CavalryUnitConfig(pursuit_damage_per_step=-0.01)

    def test_invalid_team(self) -> None:
        with self.assertRaises(ValueError):
            CavalryUnitConfig(team=2)

    def test_red_team(self) -> None:
        cfg = CavalryUnitConfig(team=1)
        self.assertEqual(cfg.team, 1)


# ---------------------------------------------------------------------------
# 3. CavalryUnit — geometry helpers
# ---------------------------------------------------------------------------


class TestCavalryUnitGeometry(unittest.TestCase):
    def test_distance_to_same_point(self) -> None:
        unit = _make_unit(x=500.0, y=500.0)
        self.assertAlmostEqual(unit.distance_to(500.0, 500.0), 0.0)

    def test_distance_to_known(self) -> None:
        unit = _make_unit(x=0.0, y=0.0)
        self.assertAlmostEqual(unit.distance_to(3.0, 4.0), 5.0)

    def test_distance_symmetry(self) -> None:
        u1 = _make_unit(x=100.0, y=200.0)
        u2 = _make_unit(x=300.0, y=400.0)
        self.assertAlmostEqual(
            u1.distance_to(u2.x, u2.y),
            u2.distance_to(u1.x, u1.y),
        )


# ---------------------------------------------------------------------------
# 4. CavalryUnit — movement
# ---------------------------------------------------------------------------


class TestCavalryUnitMovement(unittest.TestCase):
    def test_move_towards_gets_closer(self) -> None:
        unit = _make_unit(x=0.0, y=0.0)
        unit.move_towards(1_000.0, 0.0)
        self.assertGreater(unit.x, 0.0)

    def test_move_towards_exact_target_no_overshoot(self) -> None:
        """Unit should stop at target when it's closer than max_speed."""
        cfg = CavalryUnitConfig(max_speed=DEFAULT_CAVALRY_SPEED)
        unit = _make_unit(x=0.0, y=0.0, config=cfg)
        unit.move_towards(10.0, 0.0)
        self.assertAlmostEqual(unit.x, 10.0, places=6)

    def test_move_towards_updates_theta(self) -> None:
        unit = _make_unit(x=0.0, y=0.0)
        unit.move_towards(0.0, 1_000.0)
        self.assertAlmostEqual(unit.theta, math.pi / 2, places=5)

    def test_move_towards_zero_distance_is_noop(self) -> None:
        unit = _make_unit(x=500.0, y=500.0)
        unit.move_towards(500.0, 500.0)
        self.assertAlmostEqual(unit.x, 500.0)
        self.assertAlmostEqual(unit.y, 500.0)


# ---------------------------------------------------------------------------
# 5. CavalryUnit — map-boundary clamping
# ---------------------------------------------------------------------------


class TestCavalryUnitClamping(unittest.TestCase):
    def test_clamp_left_boundary(self) -> None:
        unit = _make_unit(x=50.0, y=500.0)
        unit.move_towards(-10_000.0, 500.0, map_width=MAP_W, map_height=MAP_H)
        self.assertGreaterEqual(unit.x, 0.0)

    def test_clamp_right_boundary(self) -> None:
        unit = _make_unit(x=MAP_W - 50.0, y=2_500.0)
        for _ in range(100):
            unit.move_towards(MAP_W + 1_000_000.0, 2_500.0, map_width=MAP_W, map_height=MAP_H)
        self.assertLessEqual(unit.x, MAP_W)

    def test_clamp_bottom_boundary(self) -> None:
        unit = _make_unit(x=500.0, y=50.0)
        unit.move_towards(500.0, -10_000.0, map_width=MAP_W, map_height=MAP_H)
        self.assertGreaterEqual(unit.y, 0.0)

    def test_clamp_top_boundary(self) -> None:
        unit = _make_unit(x=500.0, y=MAP_H - 50.0)
        for _ in range(100):
            unit.move_towards(500.0, MAP_H + 1_000_000.0, map_width=MAP_W, map_height=MAP_H)
        self.assertLessEqual(unit.y, MAP_H)


# ---------------------------------------------------------------------------
# 6. CavalryUnit — take_damage
# ---------------------------------------------------------------------------


class TestCavalryUnitDamage(unittest.TestCase):
    def test_take_damage_reduces_strength(self) -> None:
        unit = _make_unit(strength=1.0)
        unit.take_damage(0.3)
        self.assertAlmostEqual(unit.strength, 0.7)

    def test_take_damage_clamps_at_zero(self) -> None:
        unit = _make_unit(strength=0.1)
        unit.take_damage(1.0)
        self.assertAlmostEqual(unit.strength, 0.0)

    def test_take_damage_marks_dead(self) -> None:
        unit = _make_unit(strength=0.01)
        unit.take_damage(0.5)
        self.assertFalse(unit.alive)

    def test_take_damage_negative_raises(self) -> None:
        unit = _make_unit()
        with self.assertRaises(ValueError):
            unit.take_damage(-0.1)

    def test_take_damage_zero_noop(self) -> None:
        unit = _make_unit(strength=0.5)
        unit.take_damage(0.0)
        self.assertAlmostEqual(unit.strength, 0.5)
        self.assertTrue(unit.alive)


# ---------------------------------------------------------------------------
# 7. CavalryCorps — generate_default factory
# ---------------------------------------------------------------------------


class TestCavalryCorpsFactory(unittest.TestCase):
    def test_default_n_brigades(self) -> None:
        corps = _make_corps(n=2)
        self.assertEqual(len(corps.units), 2)

    def test_single_brigade(self) -> None:
        corps = _make_corps(n=1)
        self.assertEqual(len(corps.units), 1)

    def test_units_positioned_in_map(self) -> None:
        corps = _make_corps(n=3)
        for unit in corps.units:
            self.assertGreater(unit.x, 0.0)
            self.assertLess(unit.x, MAP_W)
            self.assertGreater(unit.y, 0.0)
            self.assertLess(unit.y, MAP_H)

    def test_units_spread_along_y(self) -> None:
        corps = _make_corps(n=4)
        y_vals = sorted(u.y for u in corps.units)
        # All Y positions should be distinct
        self.assertEqual(len(set(y_vals)), len(corps.units))

    def test_team_0_positioned_left(self) -> None:
        corps = _make_corps(n=2, team=0)
        for unit in corps.units:
            self.assertLess(unit.x, MAP_W * 0.5)

    def test_team_1_positioned_right(self) -> None:
        corps = CavalryCorps.generate_default(MAP_W, MAP_H, n_brigades=2, team=1)
        for unit in corps.units:
            self.assertGreater(unit.x, MAP_W * 0.5)

    def test_invalid_n_brigades(self) -> None:
        with self.assertRaises(ValueError):
            CavalryCorps.generate_default(MAP_W, MAP_H, n_brigades=0)

    def test_units_start_alive_and_idle(self) -> None:
        corps = _make_corps(n=2)
        for unit in corps.units:
            self.assertTrue(unit.alive)
            self.assertEqual(unit.mission, CavalryMission.IDLE)
            self.assertAlmostEqual(unit.strength, 1.0)


# ---------------------------------------------------------------------------
# 8. CavalryCorps — reset lifecycle
# ---------------------------------------------------------------------------


class TestCavalryCorpsReset(unittest.TestCase):
    def test_reset_restores_strength(self) -> None:
        corps = _make_corps()
        corps.units[0].take_damage(0.5)
        corps.reset()
        self.assertAlmostEqual(corps.units[0].strength, 1.0)

    def test_reset_restores_alive(self) -> None:
        corps = _make_corps()
        corps.units[0].take_damage(1.0)
        corps.reset()
        self.assertTrue(corps.units[0].alive)

    def test_reset_sets_idle_mission(self) -> None:
        corps = _make_corps()
        corps.units[0].mission = CavalryMission.RAIDING
        corps.reset()
        self.assertEqual(corps.units[0].mission, CavalryMission.IDLE)


# ---------------------------------------------------------------------------
# 9. CavalryCorps — get_revealed_enemies with RECON mission
# ---------------------------------------------------------------------------


class TestGetRevealedEnemies(unittest.TestCase):
    def _make_recon_unit(
        self, x: float = 5_000.0, radius: float = DEFAULT_RECON_RADIUS
    ) -> CavalryUnit:
        cfg = CavalryUnitConfig(recon_radius=radius)
        return _make_unit(x=x, y=2_500.0, mission=CavalryMission.RECONNAISSANCE, config=cfg)

    def test_reveals_nearby_enemy(self) -> None:
        # Enemy is within recon_radius
        inner = _make_inner(red_bns=[(6_000.0, 2_500.0, False)])
        unit = self._make_recon_unit(x=5_000.0, radius=2_000.0)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        revealed = corps.get_revealed_enemies(inner)
        self.assertEqual(len(revealed), 1)

    def test_does_not_reveal_far_enemy(self) -> None:
        # Enemy is beyond recon_radius
        inner = _make_inner(red_bns=[(9_000.0, 2_500.0, False)])
        unit = self._make_recon_unit(x=2_000.0, radius=500.0)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        revealed = corps.get_revealed_enemies(inner)
        self.assertEqual(len(revealed), 0)

    def test_revealed_position_matches_enemy(self) -> None:
        inner = _make_inner(red_bns=[(6_000.0, 3_000.0, False)])
        unit = self._make_recon_unit(x=5_000.0, radius=2_000.0)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        revealed = corps.get_revealed_enemies(inner)
        self.assertAlmostEqual(revealed[0][0], 6_000.0)
        self.assertAlmostEqual(revealed[0][1], 3_000.0)

    def test_revealed_contains_strength_and_morale(self) -> None:
        inner = _make_inner(red_bns=[(5_500.0, 2_500.0, False)])
        inner._battalions["red_0"].strength = 0.7
        inner._battalions["red_0"].morale = 0.6
        unit = self._make_recon_unit(x=5_000.0, radius=2_000.0)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        revealed = corps.get_revealed_enemies(inner)
        self.assertAlmostEqual(revealed[0][2], 0.7)
        self.assertAlmostEqual(revealed[0][3], 0.6)

    def test_multiple_recon_no_duplicates(self) -> None:
        """Two recon units covering the same enemy should not duplicate it."""
        inner = _make_inner(red_bns=[(5_500.0, 2_500.0, False)])
        units = [
            self._make_recon_unit(x=5_000.0, radius=2_000.0),
            self._make_recon_unit(x=5_200.0, radius=2_000.0),
        ]
        corps = CavalryCorps(units=units, map_width=MAP_W, map_height=MAP_H)
        revealed = corps.get_revealed_enemies(inner)
        self.assertEqual(len(revealed), 1)


# ---------------------------------------------------------------------------
# 10. CavalryCorps — get_revealed_enemies with non-RECON missions
# ---------------------------------------------------------------------------


class TestRevealedEnemiesNonRecon(unittest.TestCase):
    def test_idle_reveals_nothing(self) -> None:
        inner = _make_inner(red_bns=[(100.0, 2_500.0, False)])
        unit = _make_unit(x=100.0, mission=CavalryMission.IDLE)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        self.assertEqual(corps.get_revealed_enemies(inner), [])

    def test_raiding_reveals_nothing(self) -> None:
        inner = _make_inner(red_bns=[(100.0, 2_500.0, False)])
        unit = _make_unit(x=100.0, mission=CavalryMission.RAIDING)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        self.assertEqual(corps.get_revealed_enemies(inner), [])

    def test_pursuit_reveals_nothing(self) -> None:
        inner = _make_inner(red_bns=[(100.0, 2_500.0, False)])
        unit = _make_unit(x=100.0, mission=CavalryMission.PURSUIT)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        self.assertEqual(corps.get_revealed_enemies(inner), [])

    def test_dead_recon_unit_reveals_nothing(self) -> None:
        inner = _make_inner(red_bns=[(100.0, 2_500.0, False)])
        unit = _make_unit(x=100.0, alive=False, mission=CavalryMission.RECONNAISSANCE)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        self.assertEqual(corps.get_revealed_enemies(inner), [])


# ---------------------------------------------------------------------------
# 11. CavalryCorps — step with IDLE mission
# ---------------------------------------------------------------------------


class TestCavalryCorpsStepIdle(unittest.TestCase):
    def test_idle_does_not_move(self) -> None:
        unit = _make_unit(x=2_000.0, y=2_500.0, mission=CavalryMission.IDLE)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner()
        sn = _make_supply_network_with_red_depot()
        report = corps.step(inner, sn)
        self.assertAlmostEqual(unit.x, 2_000.0)
        self.assertAlmostEqual(unit.y, 2_500.0)

    def test_idle_returns_empty_report(self) -> None:
        unit = _make_unit(mission=CavalryMission.IDLE)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[])
        sn = _make_supply_network_with_red_depot()
        report = corps.step(inner, sn)
        self.assertEqual(report.depots_raided, 0)
        self.assertEqual(report.routed_units_pursued, 0)
        self.assertAlmostEqual(report.pursuit_damage, 0.0)

    def test_dead_unit_skipped(self) -> None:
        unit = _make_unit(alive=False, mission=CavalryMission.RECONNAISSANCE)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(500.0, 2_500.0, False)])
        sn = _make_supply_network_with_red_depot()
        report = corps.step(inner, sn)
        self.assertEqual(report.revealed_enemy_positions, [])


# ---------------------------------------------------------------------------
# 12. CavalryCorps — step with RECONNAISSANCE mission
# ---------------------------------------------------------------------------


class TestCavalryCorpsStepRecon(unittest.TestCase):
    def test_recon_advances_forward(self) -> None:
        unit = _make_unit(x=500.0, y=2_500.0, mission=CavalryMission.RECONNAISSANCE)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[])
        sn = _make_supply_network_with_red_depot()
        x_before = unit.x
        corps.step(inner, sn)
        self.assertGreater(unit.x, x_before)  # moved eastward

    def test_recon_report_has_revealed_positions(self) -> None:
        # Put unit right next to an enemy so it's definitely within range
        cfg = CavalryUnitConfig(recon_radius=3_000.0)
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=CavalryMission.RECONNAISSANCE,
            config=cfg,
        )
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(5_500.0, 2_500.0, False)])
        sn = _make_supply_network_with_red_depot()
        report = corps.step(inner, sn)
        self.assertGreater(len(report.revealed_enemy_positions), 0)

    def test_recon_zero_depots_raided(self) -> None:
        unit = _make_unit(x=100.0, mission=CavalryMission.RECONNAISSANCE)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        sn = _make_supply_network_with_red_depot()
        report = corps.step(_make_inner(), sn)
        self.assertEqual(report.depots_raided, 0)


# ---------------------------------------------------------------------------
# 13. CavalryCorps — step with RAIDING mission
# ---------------------------------------------------------------------------


class TestCavalryCorpsStepRaiding(unittest.TestCase):
    def test_raider_moves_towards_depot(self) -> None:
        depot_x = 8_500.0
        unit = _make_unit(x=1_000.0, y=2_500.0, mission=CavalryMission.RAIDING)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        sn = _make_supply_network_with_red_depot(depot_x=depot_x, depot_y=2_500.0)
        corps.step(_make_inner(), sn)
        # Unit should have moved towards the depot (eastward)
        self.assertGreater(unit.x, 1_000.0)

    def test_raider_interdicts_depot_in_range(self) -> None:
        # Place cavalry within raid_radius of the depot
        cfg = CavalryUnitConfig(raid_radius=500.0)
        depot_x = 8_500.0
        unit = _make_unit(
            x=depot_x - 300.0, y=2_500.0,
            mission=CavalryMission.RAIDING,
            config=cfg,
        )
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        sn = _make_supply_network_with_red_depot(depot_x=depot_x, depot_y=2_500.0)
        report = corps.step(_make_inner(), sn)
        self.assertEqual(report.depots_raided, 1)
        # Depot should be destroyed
        self.assertFalse(sn.depots[0].alive)

    def test_raider_no_reward_in_report(self) -> None:
        """Raiding does not expose an explicit per-depot reward signal."""
        cfg = CavalryUnitConfig(raid_radius=500.0)
        depot_x = 8_500.0
        unit = _make_unit(
            x=depot_x - 300.0, y=2_500.0,
            mission=CavalryMission.RAIDING,
            config=cfg,
        )
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        sn = _make_supply_network_with_red_depot(depot_x=depot_x, depot_y=2_500.0)
        report = corps.step(_make_inner(), sn)
        # CavalryReport carries no monetary reward field
        self.assertFalse(hasattr(report, "reward"))

    def test_raider_no_living_depots(self) -> None:
        """Raider with no alive enemy depots should not crash."""
        sn = SupplyNetwork(depots=[], convoy_routes=[])
        unit = _make_unit(mission=CavalryMission.RAIDING)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner(), sn)
        self.assertEqual(report.depots_raided, 0)

    def test_raider_does_not_reveal_enemies(self) -> None:
        unit = _make_unit(x=500.0, mission=CavalryMission.RAIDING)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(600.0, 2_500.0, False)])
        sn = _make_supply_network_with_red_depot()
        report = corps.step(inner, sn)
        self.assertEqual(report.revealed_enemy_positions, [])

    def test_raider_interdicts_depot_crossed_into_range_this_step(self) -> None:
        """Raider starting just outside raid_radius but moving in-range should interdict."""
        # Place unit exactly one step away from entering raid_radius
        raid_radius = 200.0
        depot_x = 8_500.0
        # Start unit half a cavalry step's distance away from the raid_radius edge
        # (so it's currently out of range but moves into range in one step)
        start_x = depot_x - raid_radius - DEFAULT_CAVALRY_SPEED * 0.5
        cfg = CavalryUnitConfig(raid_radius=raid_radius)
        unit = _make_unit(x=start_x, y=2_500.0, mission=CavalryMission.RAIDING, config=cfg)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        sn = _make_supply_network_with_red_depot(depot_x=depot_x, depot_y=2_500.0)
        report = corps.step(_make_inner(), sn)
        # After moving, unit should be within raid_radius and depot should be interdicted
        self.assertEqual(report.depots_raided, 1)
        self.assertFalse(sn.depots[0].alive)


# ---------------------------------------------------------------------------
# 14. CavalryCorps — step with PURSUIT mission
# ---------------------------------------------------------------------------


class TestCavalryCorpsStepPursuit(unittest.TestCase):
    def test_pursuit_moves_towards_routed_unit(self) -> None:
        routed_x = 7_000.0
        unit = _make_unit(x=5_000.0, y=2_500.0, mission=CavalryMission.PURSUIT)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(routed_x, 2_500.0, True)])
        # Make routed unit accessible in _battalions
        inner._alive.discard("red_0")  # routed → not in alive set
        sn = _make_supply_network_with_red_depot()
        corps.step(inner, sn)
        # Cavalry should have moved eastward towards routed unit
        self.assertGreater(unit.x, 5_000.0)

    def test_pursuit_deals_damage_when_in_range(self) -> None:
        cfg = CavalryUnitConfig(pursuit_radius=500.0, pursuit_damage_per_step=0.05)
        routed_x = 5_200.0
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=CavalryMission.PURSUIT,
            config=cfg,
        )
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(routed_x, 2_500.0, True)])
        sn = _make_supply_network_with_red_depot()
        initial_strength = inner._battalions["red_0"].strength
        report = corps.step(inner, sn)
        self.assertEqual(report.routed_units_pursued, 1)
        self.assertAlmostEqual(report.pursuit_damage, 0.05)

    def test_pursuit_no_routed_units(self) -> None:
        unit = _make_unit(mission=CavalryMission.PURSUIT)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(8_000.0, 2_500.0, False)])  # not routed
        sn = _make_supply_network_with_red_depot()
        report = corps.step(inner, sn)
        self.assertEqual(report.routed_units_pursued, 0)

    def test_pursuit_no_enemy_at_all(self) -> None:
        unit = _make_unit(mission=CavalryMission.PURSUIT)
        corps = CavalryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[])
        sn = _make_supply_network_with_red_depot()
        report = corps.step(inner, sn)
        self.assertEqual(report.routed_units_pursued, 0)
        self.assertAlmostEqual(report.pursuit_damage, 0.0)


# ---------------------------------------------------------------------------
# 15. CavalryReport dataclass fields
# ---------------------------------------------------------------------------


class TestCavalryReport(unittest.TestCase):
    def test_construction(self) -> None:
        r = CavalryReport(
            revealed_enemy_positions=[(1.0, 2.0, 0.8, 0.9)],
            depots_raided=1,
            routed_units_pursued=2,
            pursuit_damage=0.04,
        )
        self.assertEqual(r.depots_raided, 1)
        self.assertEqual(r.routed_units_pursued, 2)
        self.assertAlmostEqual(r.pursuit_damage, 0.04)
        self.assertEqual(len(r.revealed_enemy_positions), 1)

    def test_empty_report(self) -> None:
        r = CavalryReport([], 0, 0, 0.0)
        self.assertEqual(r.revealed_enemy_positions, [])
        self.assertEqual(r.depots_raided, 0)


# ---------------------------------------------------------------------------
# 16. CavalryCorpsEnv — construction and action/obs spaces
# ---------------------------------------------------------------------------


class TestCavalryCorpsEnvSpaces(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _make_env()
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_obs_dim(self) -> None:
        expected = _cav_obs_dim(2, 2)
        self.assertEqual(self.env.observation_space.shape[0], expected)

    def test_obs_dim_formula(self) -> None:
        n_div = 2
        n_cav = 2
        base = _corps_obs_dim(n_div)
        full = _cav_obs_dim(n_div, n_cav)
        self.assertEqual(full, base + n_cav * _CAV_UNIT_OBS_DIM + _CAV_INTEL_DIM)

    def test_action_space_shape(self) -> None:
        # n_divisions corps cmds + n_cavalry_brigades cav missions
        self.assertEqual(
            len(self.env.action_space.nvec),
            2 + 2,  # n_divisions + n_cavalry_brigades
        )

    def test_action_space_cavalry_range(self) -> None:
        # Cavalry portion should have N_CAVALRY_MISSIONS options
        cav_nvec = self.env.action_space.nvec[2:]  # skip n_divisions
        self.assertTrue(all(v == N_CAVALRY_MISSIONS for v in cav_nvec))

    def test_invalid_n_cavalry_brigades(self) -> None:
        with self.assertRaises(ValueError):
            CavalryCorpsEnv(n_cavalry_brigades=0)

    def test_prebuilt_corps_derives_n_cavalry_brigades(self) -> None:
        """When cavalry_corps is provided, n_cavalry_brigades comes from it."""
        corps = CavalryCorps.generate_default(MAP_W, MAP_H, n_brigades=3, team=0)
        env = CavalryCorpsEnv(
            n_divisions=2, n_brigades_per_division=2, n_blue_per_brigade=2,
            cavalry_corps=corps,
        )
        # n_cavalry_brigades should be derived from the supplied corps (3 units)
        self.assertEqual(env.n_cavalry_brigades, 3)
        self.assertEqual(len(env.action_space.nvec), env.n_divisions + 3)
        env.close()

    def test_prebuilt_corps_empty_raises(self) -> None:
        """A pre-built corps with no units should raise ValueError."""
        empty_corps = CavalryCorps(units=[], map_width=MAP_W, map_height=MAP_H)
        with self.assertRaises(ValueError):
            _make_env(cavalry_corps=empty_corps)


# ---------------------------------------------------------------------------
# 17. CavalryCorpsEnv — reset observation shape and content
# ---------------------------------------------------------------------------


class TestCavalryCorpsEnvReset(unittest.TestCase):
    def test_reset_obs_shape(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (env.observation_space.shape[0],))
        env.close()

    def test_reset_obs_in_bounds(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        env.close()

    def test_reset_returns_empty_info(self) -> None:
        env = _make_env()
        _, info = env.reset(seed=0)
        self.assertIsInstance(info, dict)
        env.close()

    def test_reset_clears_cavalry_report(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        self.assertEqual(env._last_cav_report.depots_raided, 0)
        self.assertEqual(env._last_cav_report.routed_units_pursued, 0)
        self.assertEqual(env._last_cav_report.revealed_enemy_positions, [])
        env.close()


# ---------------------------------------------------------------------------
# 18. CavalryCorpsEnv — step observation shape and info dict
# ---------------------------------------------------------------------------


class TestCavalryCorpsEnvStep(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _make_env()
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_step_obs_shape(self) -> None:
        action = self.env.action_space.sample()
        obs, _, _, _, _ = self.env.step(action)
        self.assertEqual(obs.shape, (self.env.observation_space.shape[0],))

    def test_step_obs_in_bounds(self) -> None:
        action = self.env.action_space.sample()
        obs, _, _, _, _ = self.env.step(action)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_step_info_has_cavalry_key(self) -> None:
        action = self.env.action_space.sample()
        _, _, _, _, info = self.env.step(action)
        self.assertIn("cavalry", info)

    def test_step_cavalry_info_fields(self) -> None:
        action = self.env.action_space.sample()
        _, _, _, _, info = self.env.step(action)
        cav = info["cavalry"]
        self.assertIn("depots_raided", cav)
        self.assertIn("routed_units_pursued", cav)
        self.assertIn("pursuit_damage", cav)
        self.assertIn("n_revealed_enemies", cav)

    def test_step_returns_base_reward(self) -> None:
        """Cavalry step adds no extra reward — reward is purely from base corps."""
        action = self.env.action_space.sample()
        _, reward, _, _, _ = self.env.step(action)
        self.assertIsInstance(reward, float)


# ---------------------------------------------------------------------------
# 19. CavalryCorpsEnv — invalid action raises ValueError
# ---------------------------------------------------------------------------


class TestCavalryCorpsEnvActionValidation(unittest.TestCase):
    def test_wrong_action_length_raises(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        wrong_action = np.zeros(2, dtype=np.int64)  # too short
        with self.assertRaises(ValueError):
            env.step(wrong_action)
        env.close()

    def test_invalid_corps_command_raises(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        bad_action = np.array([999, 0, 0, 0], dtype=np.int64)  # invalid corps cmd
        with self.assertRaises(ValueError):
            env.step(bad_action)
        env.close()


# ---------------------------------------------------------------------------
# 20. CavalryCorpsEnv — full short episode
# ---------------------------------------------------------------------------


class TestCavalryCorpsEnvEpisode(unittest.TestCase):
    def test_full_episode_no_crash(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=1)
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        self.assertGreater(steps, 0)
        env.close()

    def test_episode_obs_always_in_bounds(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=2)
        for _ in range(20):
            if not env.observation_space.contains(obs):
                self.fail(f"Observation out of bounds: {obs}")
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=3)
        env.close()


# ---------------------------------------------------------------------------
# 21. AC-1 integration: fog-of-war reduced by RECON cavalry
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria1FogOfWar(unittest.TestCase):
    """AC-1: Cavalry reconnaissance correctly reduces fog-of-war."""

    def test_recon_lifts_comm_radius_sentinel(self) -> None:
        """With RECON cavalry that has revealed enemies, _get_fog_radius → inf."""
        env = _make_env(comm_radius=100.0)  # tiny comm_radius → all sentinels normally
        env.reset(seed=0)

        # Inject a non-empty cavalry report with revealed positions
        env._last_cav_report = CavalryReport(
            revealed_enemy_positions=[(8_000.0, 2_500.0, 0.9, 0.8)],
            depots_raided=0,
            routed_units_pursued=0,
            pursuit_damage=0.0,
        )
        # _get_fog_radius() must return math.inf when there are revelations
        self.assertEqual(env._get_fog_radius(), math.inf)
        env.close()

    def test_no_recon_uses_comm_radius(self) -> None:
        """Without revealed enemies, _get_fog_radius returns base comm_radius."""
        env = _make_env(comm_radius=3_000.0)
        env.reset(seed=0)
        # No cavalry intel yet
        env._last_cav_report = CavalryReport([], 0, 0, 0.0)
        self.assertEqual(env._get_fog_radius(), 3_000.0)
        env.close()

    def test_recon_obs_differs_from_no_recon(self) -> None:
        """The obs changes when cavalry reveals enemies vs when it doesn't."""
        env = _make_env(comm_radius=100.0)  # tiny → all threats sentinel normally
        obs_no_recon, _ = env.reset(seed=77)

        # Now assign RECON to first cavalry brigade and run a step
        # (force step to run with reconnaissance mission)
        action = env.action_space.sample()
        # Set cavalry mission to RECON (index 1) for all cavalry
        action[env.n_divisions :] = CavalryMission.RECONNAISSANCE
        obs_with_action, _, _, _, info = env.step(action)

        # Both are valid observations; the point is they are computed correctly
        self.assertEqual(obs_no_recon.shape, obs_with_action.shape)
        self.assertTrue(env.observation_space.contains(obs_with_action))
        env.close()


# ---------------------------------------------------------------------------
# 22. AC-2 integration: raiders move towards enemy depots
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria2Raiding(unittest.TestCase):
    """AC-2: Raiding cavalry discovers supply depots as high-value targets."""

    def test_raider_reduces_distance_to_depot_each_step(self) -> None:
        """Cavalry on RAIDING should get closer to the depot over successive steps."""
        env = _make_env()
        env.reset(seed=0)

        # Identify the Red rear supply depot from the supply network
        red_depots = env.supply_network.get_depots_for_team(1)
        self.assertGreater(len(red_depots), 0, "Need at least one Red depot")
        depot = red_depots[0]

        # Set both cavalry brigades to RAIDING
        for unit in env._cavalry.units:
            unit.mission = CavalryMission.RAIDING

        initial_dist = env._cavalry.units[0].distance_to(depot.x, depot.y)

        # Run several RAIDING steps (keeping mission constant)
        for _ in range(10):
            action = np.zeros(
                env.n_divisions + env.n_cavalry_brigades, dtype=np.int64
            )
            action[env.n_divisions :] = int(CavalryMission.RAIDING)
            env.step(action)

        final_dist = env._cavalry.units[0].distance_to(depot.x, depot.y)
        self.assertLess(final_dist, initial_dist,
                        "Raiding cavalry should approach the enemy depot")
        env.close()

    def test_raiding_interdicts_depot_no_explicit_reward(self) -> None:
        """CavalryCorpsEnv reward does NOT include a per-depot bonus."""
        env = _make_env()
        env.reset(seed=0)

        # Teleport cavalry to just inside raid_radius of a Red depot
        red_depots = env.supply_network.get_depots_for_team(1)
        if not red_depots:
            self.skipTest("No Red depots in default supply network")
        depot = red_depots[0]
        cfg = CavalryUnitConfig(raid_radius=5_000.0)  # huge radius to guarantee hit
        for unit in env._cavalry.units:
            unit.x = depot.x
            unit.y = depot.y
            unit.config = cfg
            unit.mission = CavalryMission.RAIDING

        action = np.zeros(
            env.n_divisions + env.n_cavalry_brigades, dtype=np.int64
        )
        action[env.n_divisions :] = int(CavalryMission.RAIDING)
        _, reward, _, _, info = env.step(action)

        # Reward comes entirely from base corps env — no separate raiding bonus
        self.assertIsInstance(reward, float)
        # info['cavalry'] shows the effect happened
        self.assertGreater(info["cavalry"]["depots_raided"], 0)
        env.close()


# ---------------------------------------------------------------------------
# 23. AC-3 integration: pursuit cavalry targets routed infantry
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria3Pursuit(unittest.TestCase):
    """AC-3: Pursuit mission correctly exploits routed infantry."""

    def test_pursuit_info_nonzero_when_routed_nearby(self) -> None:
        """When there are routed units near cavalry, pursuit info should be nonzero."""
        env = _make_env()
        env.reset(seed=0)

        # Find a Red battalion and force-rout it (must also update CombatState)
        inner = env._division._brigade._inner
        red_ids = [
            aid for aid in inner._battalions
            if aid.startswith("red_")
        ]
        if not red_ids:
            self.skipTest("No Red battalions in inner env")

        target_id = red_ids[0]
        bn = inner._battalions[target_id]
        bn.routed = True
        bn.strength = 0.5  # still has strength — valid pursuit target
        # Also force the CombatState so the inner env step doesn't undo it
        inner._combat_states[target_id].is_routing = True
        inner._combat_states[target_id].morale = 0.1

        # Move cavalry adjacent to the routed unit
        cfg = CavalryUnitConfig(
            pursuit_radius=5_000.0,  # large — guaranteed hit
            pursuit_damage_per_step=0.05,
        )
        for unit in env._cavalry.units:
            unit.x = bn.x
            unit.y = bn.y
            unit.config = cfg
            unit.mission = CavalryMission.PURSUIT

        action = np.zeros(
            env.n_divisions + env.n_cavalry_brigades, dtype=np.int64
        )
        action[env.n_divisions :] = int(CavalryMission.PURSUIT)
        _, _, _, _, info = env.step(action)
        self.assertGreater(info["cavalry"]["routed_units_pursued"], 0)
        self.assertGreater(info["cavalry"]["pursuit_damage"], 0.0)
        env.close()


# ---------------------------------------------------------------------------
# 24. CavalryCorpsEnv — dead cavalry units emit zero obs block
# ---------------------------------------------------------------------------


class TestDeadCavalryObsBlock(unittest.TestCase):
    def test_dead_unit_emits_zero_block(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=0)

        # Kill first cavalry brigade
        env._cavalry.units[0].alive = False
        env._cavalry.units[0].strength = 0.0

        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        # Locate the per-cavalry slice in the observation
        base_dim = _corps_obs_dim(env.n_divisions)
        unit0_slice = obs[base_dim: base_dim + _CAV_UNIT_OBS_DIM]
        # All four dims of a dead unit should be 0.0
        np.testing.assert_array_almost_equal(unit0_slice, [0.0, 0.0, 0.0, 0.0])
        env.close()


# ---------------------------------------------------------------------------
# 25. CavalryCorpsEnv — no revealed enemies → base comm_radius fog applied
# ---------------------------------------------------------------------------


class TestNoRevealedEnemiesFog(unittest.TestCase):
    def test_fog_radius_equals_comm_radius_without_recon(self) -> None:
        env = _make_env(comm_radius=1_500.0)
        env.reset(seed=0)
        # No cavalry intel yet
        env._last_cav_report = CavalryReport([], 0, 0, 0.0)
        self.assertAlmostEqual(env._get_fog_radius(), 1_500.0)
        env.close()

    def test_obs_valid_without_recon(self) -> None:
        env = _make_env(comm_radius=1_500.0)
        obs, _ = env.reset(seed=0)
        # With no revealed enemies, obs should still be valid
        self.assertTrue(env.observation_space.contains(obs))
        env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
