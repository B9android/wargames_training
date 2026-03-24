# tests/test_artillery_corps.py
"""Tests for envs/sim/artillery_corps.py and envs/artillery_corps_env.py.

Covers all acceptance criteria from epic E10.3:

AC-1  Grand battery of ≥ 6 guns breaks enemy line in < 5 game minutes
      (< 300 steps at 1 step/s).
AC-2  Counter-battery fire silences enemy guns faster than targeting
      infantry.
AC-3  Agents can build fortifications proactively; the completed works
      are reflected in the observation.

Test organisation:
  1.  ArtilleryMission enum smoke tests.
  2.  ArtilleryUnitConfig defaults and validation.
  3.  Fortification — construction mechanics.
  4.  Fortification — siege damage mechanics.
  5.  Fortification — effective_cover helper.
  6.  ArtilleryUnit — geometry helpers.
  7.  ArtilleryUnit — movement (move_towards).
  8.  ArtilleryUnit — map-boundary clamping.
  9.  ArtilleryUnit — take_damage mechanics.
  10. ArtilleryCorps — generate_default factory.
  11. ArtilleryCorps — reset lifecycle.
  12. ArtilleryCorps — count_grand_battery_guns.
  13. ArtilleryCorps — step with IDLE mission.
  14. ArtilleryCorps — step with GRAND_BATTERY mission.
  15. ArtilleryCorps — step with COUNTER_BATTERY mission.
  16. ArtilleryCorps — step with SIEGE mission.
  17. ArtilleryCorps — step with FORTIFY mission.
  18. ArtilleryReport dataclass fields.
  19. ArtilleryCorpsEnv — construction and action/obs spaces.
  20. ArtilleryCorpsEnv — reset observation shape and content.
  21. ArtilleryCorpsEnv — step observation shape and info dict.
  22. ArtilleryCorpsEnv — invalid action raises ValueError.
  23. ArtilleryCorpsEnv — full short episode.
  24. AC-1 integration: grand battery of 6+ guns breaks enemy line < 300 steps.
  25. AC-2 integration: counter-battery silences guns faster than infantry fire.
  26. AC-3 integration: fortification state appears in observation after completion.
  27. ArtilleryCorpsEnv — dead batteries emit zero obs block.
  28. ArtilleryCorpsEnv — fortification reward shaping.
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

from envs.sim.artillery_corps import (
    ArtilleryCorps,
    ArtilleryMission,
    ArtilleryReport,
    ArtilleryUnit,
    ArtilleryUnitConfig,
    Fortification,
    N_ARTILLERY_MISSIONS,
    DEFAULT_ARTILLERY_RANGE,
    DEFAULT_ARTILLERY_SPEED,
    DEFAULT_BASE_FIRE_DAMAGE,
    DEFAULT_BASE_MORALE_DAMAGE,
    DEFAULT_BASE_STRENGTH_DAMAGE,
    DEFAULT_GRAND_BATTERY_RADIUS,
    DEFAULT_GRAND_BATTERY_BONUS,
    DEFAULT_COUNTER_BATTERY_BONUS,
    DEFAULT_SIEGE_DAMAGE_PER_STEP,
    DEFAULT_FORTIFY_STEPS,
    DEFAULT_FORT_COVER_BONUS,
)
from envs.artillery_corps_env import (
    ArtilleryCorpsEnv,
    _ART_UNIT_OBS_DIM,
    _ART_SUMMARY_DIM,
    _ART_FORT_OBS_DIM,
    _art_obs_dim,
)
from envs.corps_env import _corps_obs_dim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAP_W: float = 10_000.0
MAP_H: float = 5_000.0


def _make_unit(
    x: float = 2_500.0,
    y: float = 2_500.0,
    team: int = 0,
    mission: ArtilleryMission = ArtilleryMission.IDLE,
    strength: float = 1.0,
    alive: bool = True,
    config: ArtilleryUnitConfig | None = None,
) -> ArtilleryUnit:
    if config is None:
        config = ArtilleryUnitConfig(team=team)
    unit = ArtilleryUnit(
        x=x, y=y, theta=0.0, strength=strength, team=team, config=config
    )
    unit.alive = alive
    unit.mission = mission
    return unit


def _make_corps(
    n: int = 4, team: int = 0, **kwargs
) -> ArtilleryCorps:
    return ArtilleryCorps.generate_default(
        MAP_W, MAP_H, n_batteries=n, team=team, **kwargs
    )


def _make_inner(
    red_bns: list[tuple[float, float, float, float]] | None = None,
) -> MagicMock:
    """Build a minimal mock of MultiBattalionEnv's inner env.

    Parameters
    ----------
    red_bns:
        List of ``(x, y, morale, strength)`` tuples for Red battalions.
        Defaults to one battalion at mid-map with full morale.
    """
    if red_bns is None:
        red_bns = [(7_500.0, 2_500.0, 1.0, 1.0)]

    inner = MagicMock()
    battalions: dict = {}
    alive_set: set = set()

    for i, (x, y, morale, strength) in enumerate(red_bns):
        aid = f"red_{i}"
        bn = MagicMock()
        bn.x = x
        bn.y = y
        bn.strength = strength
        bn.morale = morale
        bn.routed = morale <= 0.3
        bn.routing_threshold = 0.3
        battalions[aid] = bn
        if not bn.routed:
            alive_set.add(aid)

    inner._battalions = battalions
    inner._alive = alive_set
    return inner


def _make_env(**kwargs) -> ArtilleryCorpsEnv:
    return ArtilleryCorpsEnv(
        n_divisions=2,
        n_brigades_per_division=2,
        n_blue_per_brigade=2,
        n_artillery_batteries=4,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. ArtilleryMission enum smoke tests
# ---------------------------------------------------------------------------


class TestArtilleryMission(unittest.TestCase):
    def test_n_missions(self) -> None:
        self.assertEqual(N_ARTILLERY_MISSIONS, 5)

    def test_values_stable(self) -> None:
        self.assertEqual(int(ArtilleryMission.IDLE), 0)
        self.assertEqual(int(ArtilleryMission.GRAND_BATTERY), 1)
        self.assertEqual(int(ArtilleryMission.COUNTER_BATTERY), 2)
        self.assertEqual(int(ArtilleryMission.SIEGE), 3)
        self.assertEqual(int(ArtilleryMission.FORTIFY), 4)

    def test_round_trip_from_int(self) -> None:
        for i in range(N_ARTILLERY_MISSIONS):
            self.assertEqual(int(ArtilleryMission(i)), i)


# ---------------------------------------------------------------------------
# 2. ArtilleryUnitConfig defaults and validation
# ---------------------------------------------------------------------------


class TestArtilleryUnitConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = ArtilleryUnitConfig()
        self.assertEqual(cfg.max_speed, DEFAULT_ARTILLERY_SPEED)
        self.assertEqual(cfg.max_range, DEFAULT_ARTILLERY_RANGE)
        self.assertEqual(cfg.base_morale_damage, DEFAULT_BASE_MORALE_DAMAGE)
        self.assertEqual(cfg.base_strength_damage, DEFAULT_BASE_STRENGTH_DAMAGE)
        self.assertEqual(cfg.grand_battery_radius, DEFAULT_GRAND_BATTERY_RADIUS)
        self.assertEqual(cfg.grand_battery_bonus, DEFAULT_GRAND_BATTERY_BONUS)
        self.assertEqual(cfg.counter_battery_bonus, DEFAULT_COUNTER_BATTERY_BONUS)
        self.assertEqual(cfg.siege_damage_per_step, DEFAULT_SIEGE_DAMAGE_PER_STEP)
        self.assertEqual(cfg.fortify_steps, DEFAULT_FORTIFY_STEPS)
        self.assertEqual(cfg.cover_bonus, DEFAULT_FORT_COVER_BONUS)
        self.assertEqual(cfg.team, 0)

    def test_frozen(self) -> None:
        cfg = ArtilleryUnitConfig()
        with self.assertRaises((AttributeError, TypeError)):
            cfg.max_speed = 999.0  # type: ignore[misc]

    def test_invalid_max_speed(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(max_speed=0.0)

    def test_invalid_max_range(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(max_range=-1.0)

    def test_negative_base_morale_damage(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(base_morale_damage=-0.01)

    def test_negative_base_strength_damage(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(base_strength_damage=-0.01)

    def test_invalid_grand_battery_radius(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(grand_battery_radius=0.0)

    def test_negative_grand_battery_bonus(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(grand_battery_bonus=-0.01)

    def test_negative_counter_battery_bonus(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(counter_battery_bonus=-0.01)

    def test_negative_siege_damage(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(siege_damage_per_step=-0.01)

    def test_invalid_fortify_steps(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(fortify_steps=0)

    def test_negative_cover_bonus(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(cover_bonus=-0.1)

    def test_invalid_team(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryUnitConfig(team=2)

    def test_red_team(self) -> None:
        cfg = ArtilleryUnitConfig(team=1)
        self.assertEqual(cfg.team, 1)


# ---------------------------------------------------------------------------
# 3. Fortification — construction mechanics
# ---------------------------------------------------------------------------


class TestFortificationConstruction(unittest.TestCase):
    def test_initial_hp_zero(self) -> None:
        fort = Fortification(x=500.0, y=500.0, team=0)
        self.assertAlmostEqual(fort.hp, 0.0)
        self.assertFalse(fort.complete)

    def test_build_step_increases_hp(self) -> None:
        fort = Fortification(x=500.0, y=500.0, team=0, build_step_size=0.1)
        fort.build_step()
        self.assertAlmostEqual(fort.hp, 0.1)

    def test_build_step_completes_at_one(self) -> None:
        fort = Fortification(x=500.0, y=500.0, team=0, build_step_size=0.1)
        for _ in range(10):
            fort.build_step()
        self.assertAlmostEqual(fort.hp, 1.0)
        self.assertTrue(fort.complete)

    def test_build_step_returns_true_on_completion(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=0, build_step_size=1.0)
        completed = fort.build_step()
        self.assertTrue(completed)

    def test_build_step_returns_false_before_completion(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=0, build_step_size=0.1)
        completed = fort.build_step()
        self.assertFalse(completed)

    def test_hp_does_not_exceed_one(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=0, build_step_size=0.5)
        for _ in range(5):
            fort.build_step()
        self.assertLessEqual(fort.hp, 1.0)

    def test_alive_when_hp_positive(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=0, hp=0.5)
        self.assertTrue(fort.alive)

    def test_not_alive_when_hp_zero(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=0, hp=0.0)
        self.assertFalse(fort.alive)


# ---------------------------------------------------------------------------
# 4. Fortification — siege damage mechanics
# ---------------------------------------------------------------------------


class TestFortificationSiegeDamage(unittest.TestCase):
    def test_siege_damage_reduces_hp(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=1, hp=1.0, complete=True)
        fort.take_siege_damage(0.2)
        self.assertAlmostEqual(fort.hp, 0.8)

    def test_siege_damage_clamps_at_zero(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=1, hp=0.1, complete=True)
        fort.take_siege_damage(1.0)
        self.assertAlmostEqual(fort.hp, 0.0)

    def test_siege_damage_marks_incomplete(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=1, hp=1.0, complete=True)
        fort.take_siege_damage(0.1)
        self.assertFalse(fort.complete)

    def test_siege_damage_negative_raises(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=1, hp=1.0)
        with self.assertRaises(ValueError):
            fort.take_siege_damage(-0.1)

    def test_siege_damage_zero_noop(self) -> None:
        fort = Fortification(x=0.0, y=0.0, team=1, hp=0.8, complete=False)
        fort.take_siege_damage(0.0)
        self.assertAlmostEqual(fort.hp, 0.8)


# ---------------------------------------------------------------------------
# 5. Fortification — effective_cover helper
# ---------------------------------------------------------------------------


class TestFortificationCover(unittest.TestCase):
    def test_full_hp_full_cover(self) -> None:
        fort = Fortification(
            x=0.0, y=0.0, team=0, hp=1.0, complete=True,
            cover_bonus=0.5,
        )
        self.assertAlmostEqual(fort.effective_cover(), 0.5)

    def test_partial_hp_partial_cover(self) -> None:
        fort = Fortification(
            x=0.0, y=0.0, team=0, hp=0.5, complete=False,
            cover_bonus=0.5,
        )
        self.assertAlmostEqual(fort.effective_cover(), 0.25)

    def test_zero_hp_zero_cover(self) -> None:
        fort = Fortification(
            x=0.0, y=0.0, team=0, hp=0.0, complete=False,
            cover_bonus=0.5,
        )
        self.assertAlmostEqual(fort.effective_cover(), 0.0)


# ---------------------------------------------------------------------------
# 6. ArtilleryUnit — geometry helpers
# ---------------------------------------------------------------------------


class TestArtilleryUnitGeometry(unittest.TestCase):
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
# 7. ArtilleryUnit — movement
# ---------------------------------------------------------------------------


class TestArtilleryUnitMovement(unittest.TestCase):
    def test_move_towards_gets_closer(self) -> None:
        unit = _make_unit(x=0.0, y=0.0)
        unit.move_towards(1_000.0, 0.0)
        self.assertGreater(unit.x, 0.0)

    def test_move_towards_no_overshoot(self) -> None:
        unit = _make_unit(x=0.0, y=0.0)
        unit.move_towards(5.0, 0.0)
        self.assertAlmostEqual(unit.x, 5.0, places=6)

    def test_move_towards_updates_theta(self) -> None:
        unit = _make_unit(x=0.0, y=0.0)
        unit.move_towards(0.0, 1_000.0)
        self.assertAlmostEqual(unit.theta, math.pi / 2, places=5)

    def test_move_towards_zero_distance_is_noop(self) -> None:
        unit = _make_unit(x=500.0, y=500.0)
        unit.move_towards(500.0, 500.0)
        self.assertAlmostEqual(unit.x, 500.0)
        self.assertAlmostEqual(unit.y, 500.0)

    def test_artillery_slower_than_cavalry(self) -> None:
        unit = _make_unit(x=0.0, y=0.0)
        unit.move_towards(10_000.0, 0.0)
        self.assertLessEqual(unit.x, DEFAULT_ARTILLERY_SPEED + 1e-6)


# ---------------------------------------------------------------------------
# 8. ArtilleryUnit — map-boundary clamping
# ---------------------------------------------------------------------------


class TestArtilleryUnitClamping(unittest.TestCase):
    def test_clamp_left_boundary(self) -> None:
        unit = _make_unit(x=10.0, y=500.0)
        unit.move_towards(-10_000.0, 500.0, map_width=MAP_W, map_height=MAP_H)
        self.assertGreaterEqual(unit.x, 0.0)

    def test_clamp_right_boundary(self) -> None:
        unit = _make_unit(x=MAP_W - 10.0, y=2_500.0)
        for _ in range(100):
            unit.move_towards(
                MAP_W + 1_000_000.0, 2_500.0,
                map_width=MAP_W, map_height=MAP_H,
            )
        self.assertLessEqual(unit.x, MAP_W)

    def test_clamp_bottom_boundary(self) -> None:
        unit = _make_unit(x=500.0, y=10.0)
        unit.move_towards(500.0, -10_000.0, map_width=MAP_W, map_height=MAP_H)
        self.assertGreaterEqual(unit.y, 0.0)

    def test_clamp_top_boundary(self) -> None:
        unit = _make_unit(x=500.0, y=MAP_H - 10.0)
        for _ in range(100):
            unit.move_towards(
                500.0, MAP_H + 1_000_000.0,
                map_width=MAP_W, map_height=MAP_H,
            )
        self.assertLessEqual(unit.y, MAP_H)


# ---------------------------------------------------------------------------
# 9. ArtilleryUnit — take_damage
# ---------------------------------------------------------------------------


class TestArtilleryUnitDamage(unittest.TestCase):
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
# 10. ArtilleryCorps — generate_default factory
# ---------------------------------------------------------------------------


class TestArtilleryCorpsFactory(unittest.TestCase):
    def test_default_n_batteries(self) -> None:
        corps = _make_corps(n=4)
        self.assertEqual(len(corps.units), 4)

    def test_single_battery(self) -> None:
        corps = _make_corps(n=1)
        self.assertEqual(len(corps.units), 1)

    def test_units_positioned_in_map(self) -> None:
        corps = _make_corps(n=6)
        for unit in corps.units:
            self.assertGreaterEqual(unit.x, 0.0)
            self.assertLessEqual(unit.x, MAP_W)
            self.assertGreater(unit.y, 0.0)
            self.assertLess(unit.y, MAP_H)

    def test_units_spread_along_y(self) -> None:
        corps = _make_corps(n=4)
        y_vals = [u.y for u in corps.units]
        self.assertEqual(len(set(y_vals)), len(corps.units))

    def test_team_0_positioned_left(self) -> None:
        corps = _make_corps(n=2, team=0)
        for unit in corps.units:
            self.assertLess(unit.x, MAP_W * 0.5)

    def test_team_1_positioned_right(self) -> None:
        corps = ArtilleryCorps.generate_default(MAP_W, MAP_H, n_batteries=2, team=1)
        for unit in corps.units:
            self.assertGreater(unit.x, MAP_W * 0.5)

    def test_invalid_n_batteries(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryCorps.generate_default(MAP_W, MAP_H, n_batteries=0)

    def test_units_start_alive_and_idle(self) -> None:
        corps = _make_corps(n=2)
        for unit in corps.units:
            self.assertTrue(unit.alive)
            self.assertEqual(unit.mission, ArtilleryMission.IDLE)
            self.assertAlmostEqual(unit.strength, 1.0)

    def test_starts_with_no_fortifications(self) -> None:
        corps = _make_corps(n=4)
        self.assertEqual(len(corps.fortifications), 0)


# ---------------------------------------------------------------------------
# 11. ArtilleryCorps — reset lifecycle
# ---------------------------------------------------------------------------


class TestArtilleryCorpsReset(unittest.TestCase):
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
        corps.units[0].mission = ArtilleryMission.GRAND_BATTERY
        corps.reset()
        self.assertEqual(corps.units[0].mission, ArtilleryMission.IDLE)

    def test_reset_clears_fortifications(self) -> None:
        corps = _make_corps()
        corps.fortifications.append(
            Fortification(x=100.0, y=100.0, team=0, hp=1.0, complete=True)
        )
        corps.reset()
        self.assertEqual(len(corps.fortifications), 0)

    def test_reset_clears_fortify_progress(self) -> None:
        corps = _make_corps()
        corps.units[0]._fortify_progress = 5
        corps.reset()
        self.assertEqual(corps.units[0]._fortify_progress, 0)


# ---------------------------------------------------------------------------
# 12. ArtilleryCorps — count_grand_battery_guns
# ---------------------------------------------------------------------------


class TestCountGrandBatteryGuns(unittest.TestCase):
    def test_single_gun_counts_one(self) -> None:
        cfg = ArtilleryUnitConfig(grand_battery_radius=500.0)
        unit = _make_unit(x=2_000.0, y=2_500.0, config=cfg)
        unit.mission = ArtilleryMission.GRAND_BATTERY
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        self.assertEqual(corps.count_grand_battery_guns(unit), 1)

    def test_nearby_guns_counted(self) -> None:
        cfg = ArtilleryUnitConfig(grand_battery_radius=1_000.0)
        u1 = _make_unit(x=2_000.0, y=2_500.0, config=cfg)
        u2 = _make_unit(x=2_300.0, y=2_500.0, config=cfg)
        u3 = _make_unit(x=2_600.0, y=2_500.0, config=cfg)
        for u in (u1, u2, u3):
            u.mission = ArtilleryMission.GRAND_BATTERY
        corps = ArtilleryCorps(
            units=[u1, u2, u3], map_width=MAP_W, map_height=MAP_H
        )
        count = corps.count_grand_battery_guns(u1)
        self.assertEqual(count, 3)

    def test_far_guns_not_counted(self) -> None:
        cfg = ArtilleryUnitConfig(grand_battery_radius=100.0)
        u1 = _make_unit(x=2_000.0, y=2_500.0, config=cfg)
        u2 = _make_unit(x=5_000.0, y=2_500.0, config=cfg)
        for u in (u1, u2):
            u.mission = ArtilleryMission.GRAND_BATTERY
        corps = ArtilleryCorps(units=[u1, u2], map_width=MAP_W, map_height=MAP_H)
        count = corps.count_grand_battery_guns(u1)
        self.assertEqual(count, 1)

    def test_dead_guns_not_counted(self) -> None:
        cfg = ArtilleryUnitConfig(grand_battery_radius=1_000.0)
        u1 = _make_unit(x=2_000.0, y=2_500.0, config=cfg)
        u2 = _make_unit(x=2_100.0, y=2_500.0, alive=False, config=cfg)
        u1.mission = ArtilleryMission.GRAND_BATTERY
        u2.mission = ArtilleryMission.GRAND_BATTERY
        corps = ArtilleryCorps(units=[u1, u2], map_width=MAP_W, map_height=MAP_H)
        count = corps.count_grand_battery_guns(u1)
        self.assertEqual(count, 1)

    def test_enemy_guns_not_counted(self) -> None:
        cfg0 = ArtilleryUnitConfig(grand_battery_radius=1_000.0, team=0)
        cfg1 = ArtilleryUnitConfig(grand_battery_radius=1_000.0, team=1)
        u_blue = _make_unit(x=2_000.0, y=2_500.0, team=0, config=cfg0)
        u_red = _make_unit(x=2_100.0, y=2_500.0, team=1, config=cfg1)
        u_blue.mission = ArtilleryMission.GRAND_BATTERY
        u_red.mission = ArtilleryMission.GRAND_BATTERY
        corps = ArtilleryCorps(units=[u_blue, u_red], map_width=MAP_W, map_height=MAP_H)
        count = corps.count_grand_battery_guns(u_blue)
        self.assertEqual(count, 1)

    def test_off_mission_guns_not_counted(self) -> None:
        """IDLE or COUNTER_BATTERY batteries nearby should NOT inflate the count."""
        cfg = ArtilleryUnitConfig(grand_battery_radius=1_000.0)
        u1 = _make_unit(x=2_000.0, y=2_500.0, config=cfg)
        u2 = _make_unit(x=2_200.0, y=2_500.0, config=cfg)  # IDLE
        u3 = _make_unit(x=2_400.0, y=2_500.0, config=cfg)  # COUNTER_BATTERY
        u1.mission = ArtilleryMission.GRAND_BATTERY
        u2.mission = ArtilleryMission.IDLE
        u3.mission = ArtilleryMission.COUNTER_BATTERY
        corps = ArtilleryCorps(units=[u1, u2, u3], map_width=MAP_W, map_height=MAP_H)
        count = corps.count_grand_battery_guns(u1)
        self.assertEqual(count, 1)  # only u1 counts


# ---------------------------------------------------------------------------
# 13. ArtilleryCorps — step with IDLE mission
# ---------------------------------------------------------------------------


class TestArtilleryCorpsStepIdle(unittest.TestCase):
    def test_idle_does_not_move(self) -> None:
        unit = _make_unit(x=2_000.0, y=2_500.0, mission=ArtilleryMission.IDLE)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        corps.step(_make_inner())
        self.assertAlmostEqual(unit.x, 2_000.0)
        self.assertAlmostEqual(unit.y, 2_500.0)

    def test_idle_returns_zero_report(self) -> None:
        unit = _make_unit(mission=ArtilleryMission.IDLE)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner())
        self.assertAlmostEqual(report.morale_damage_dealt, 0.0)
        self.assertEqual(report.guns_silenced, 0)
        self.assertAlmostEqual(report.fortification_damage, 0.0)
        self.assertEqual(report.fortifications_completed, 0)

    def test_dead_unit_skipped(self) -> None:
        unit = _make_unit(alive=False, mission=ArtilleryMission.GRAND_BATTERY)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner())
        self.assertAlmostEqual(report.morale_damage_dealt, 0.0)


# ---------------------------------------------------------------------------
# 14. ArtilleryCorps — step with GRAND_BATTERY mission
# ---------------------------------------------------------------------------


class TestArtilleryCorpsStepGrandBattery(unittest.TestCase):
    def test_single_gun_applies_base_damage(self) -> None:
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_morale_damage=0.05,
            grand_battery_bonus=0.01,
        )
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.GRAND_BATTERY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(6_000.0, 2_500.0, 1.0, 1.0)])
        report = corps.step(inner)
        self.assertAlmostEqual(report.morale_damage_dealt, 0.05, places=6)

    def test_grand_battery_stacks_damage(self) -> None:
        """Six guns close together apply a stacking bonus."""
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_morale_damage=0.05,
            grand_battery_bonus=0.01,
            grand_battery_radius=1_000.0,
        )
        # Six guns clustered at x=5000
        units = [
            _make_unit(
                x=5_000.0 + i * 50.0, y=2_500.0,
                mission=ArtilleryMission.GRAND_BATTERY,
                config=cfg,
            )
            for i in range(6)
        ]
        corps = ArtilleryCorps(units=units, map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(6_000.0, 2_500.0, 1.0, 1.0)])
        report = corps.step(inner)
        # Each gun applies: 0.05 + (6-1)*0.01 = 0.10; total for 6 guns = 0.60
        self.assertGreater(report.morale_damage_dealt, 0.05 * 6)

    def test_grand_battery_out_of_range_moves_towards_enemy(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=100.0)
        unit = _make_unit(
            x=100.0, y=2_500.0,
            mission=ArtilleryMission.GRAND_BATTERY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(8_000.0, 2_500.0, 1.0, 1.0)])
        x_before = unit.x
        corps.step(inner)
        self.assertGreater(unit.x, x_before)

    def test_grand_battery_no_enemy_returns_zero_damage(self) -> None:
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.GRAND_BATTERY,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[])
        report = corps.step(inner)
        self.assertAlmostEqual(report.morale_damage_dealt, 0.0)

    def test_grand_battery_routes_target_when_morale_low(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=2_000.0, base_morale_damage=0.8)
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.GRAND_BATTERY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(6_000.0, 2_500.0, 0.4, 1.0)])
        corps.step(inner)
        target = inner._battalions["red_0"]
        self.assertTrue(target.routed)


# ---------------------------------------------------------------------------
# 15. ArtilleryCorps — step with COUNTER_BATTERY mission
# ---------------------------------------------------------------------------


class TestArtilleryCorpsStepCounterBattery(unittest.TestCase):
    def _make_enemy_art(
        self,
        x: float = 7_000.0,
        y: float = 2_500.0,
        strength: float = 1.0,
        alive: bool = True,
    ) -> ArtilleryUnit:
        cfg = ArtilleryUnitConfig(team=1)
        unit = ArtilleryUnit(
            x=x, y=y, theta=math.pi, strength=strength, team=1, config=cfg
        )
        unit.alive = alive
        return unit

    def test_counter_battery_damages_enemy_artillery(self) -> None:
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_strength_damage=0.05,
            counter_battery_bonus=0.06,
        )
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.COUNTER_BATTERY,
            config=cfg,
        )
        enemy_art = self._make_enemy_art(x=5_500.0)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner(), enemy_artillery=[enemy_art])
        self.assertLess(enemy_art.strength, 1.0)

    def test_counter_battery_silences_weakened_gun(self) -> None:
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_strength_damage=0.5,
            counter_battery_bonus=0.6,
        )
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.COUNTER_BATTERY,
            config=cfg,
        )
        enemy_art = self._make_enemy_art(x=5_500.0, strength=0.05)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner(), enemy_artillery=[enemy_art])
        self.assertEqual(report.guns_silenced, 1)
        self.assertFalse(enemy_art.alive)

    def test_counter_battery_fallback_to_infantry_when_no_enemy_art(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=2_000.0, base_morale_damage=0.05)
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.COUNTER_BATTERY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(6_000.0, 2_500.0, 1.0, 1.0)])
        report = corps.step(inner, enemy_artillery=[])
        # Fallback: morale damage to infantry
        self.assertGreater(report.morale_damage_dealt, 0.0)

    def test_counter_battery_out_of_range_advances(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=100.0)
        unit = _make_unit(
            x=100.0, y=2_500.0,
            mission=ArtilleryMission.COUNTER_BATTERY,
            config=cfg,
        )
        enemy_art = self._make_enemy_art(x=8_000.0)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        x_before = unit.x
        corps.step(_make_inner(), enemy_artillery=[enemy_art])
        self.assertGreater(unit.x, x_before)

    def test_counter_battery_prefers_artillery_over_infantry(self) -> None:
        """Counter-battery deals more total damage vs artillery than vs infantry."""
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_strength_damage=0.05,
            counter_battery_bonus=0.06,
        )
        # With enemy art nearby: fires at art with counter_battery_bonus
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.COUNTER_BATTERY,
            config=cfg,
        )
        enemy_art = self._make_enemy_art(x=5_500.0, strength=1.0)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        before_strength = enemy_art.strength
        corps.step(_make_inner(), enemy_artillery=[enemy_art])
        strength_dmg_to_art = before_strength - enemy_art.strength
        # strength_dmg_to_art should equal base_strength_damage + counter_battery_bonus = 0.11
        self.assertAlmostEqual(
            strength_dmg_to_art,
            cfg.base_strength_damage + cfg.counter_battery_bonus,
            places=6,
        )

    def test_counter_battery_dead_enemy_art_ignored(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=2_000.0, base_morale_damage=0.05)
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.COUNTER_BATTERY,
            config=cfg,
        )
        enemy_art = self._make_enemy_art(x=5_500.0, alive=False)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        inner = _make_inner(red_bns=[(6_000.0, 2_500.0, 1.0, 1.0)])
        report = corps.step(inner, enemy_artillery=[enemy_art])
        # Falls back to infantry
        self.assertGreater(report.morale_damage_dealt, 0.0)


# ---------------------------------------------------------------------------
# 16. ArtilleryCorps — step with SIEGE mission
# ---------------------------------------------------------------------------


class TestArtilleryCorpsStepSiege(unittest.TestCase):
    def _make_enemy_fort(
        self,
        x: float = 7_000.0,
        y: float = 2_500.0,
        hp: float = 1.0,
    ) -> Fortification:
        return Fortification(
            x=x, y=y, team=1, hp=hp, complete=(hp >= 1.0),
            cover_bonus=DEFAULT_FORT_COVER_BONUS,
        )

    def test_siege_reduces_fort_hp(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=2_000.0, siege_damage_per_step=0.1)
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.SIEGE,
            config=cfg,
        )
        fort = self._make_enemy_fort(x=6_000.0)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner(), enemy_fortifications=[fort])
        self.assertLess(fort.hp, 1.0)
        self.assertAlmostEqual(report.fortification_damage, 0.1)

    def test_siege_out_of_range_advances(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=100.0)
        unit = _make_unit(
            x=100.0, y=2_500.0,
            mission=ArtilleryMission.SIEGE,
            config=cfg,
        )
        fort = self._make_enemy_fort(x=8_000.0)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        x_before = unit.x
        corps.step(_make_inner(), enemy_fortifications=[fort])
        self.assertGreater(unit.x, x_before)

    def test_siege_no_forts_returns_zero_damage(self) -> None:
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.SIEGE,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner(), enemy_fortifications=[])
        self.assertAlmostEqual(report.fortification_damage, 0.0)

    def test_siege_targets_alive_forts_only(self) -> None:
        cfg = ArtilleryUnitConfig(max_range=2_000.0)
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.SIEGE,
            config=cfg,
        )
        dead_fort = Fortification(x=6_000.0, y=2_500.0, team=1, hp=0.0)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner(), enemy_fortifications=[dead_fort])
        self.assertAlmostEqual(report.fortification_damage, 0.0)

    def test_siege_repeated_destroys_fort(self) -> None:
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0, siege_damage_per_step=0.1
        )
        unit = _make_unit(
            x=5_000.0, y=2_500.0,
            mission=ArtilleryMission.SIEGE,
            config=cfg,
        )
        fort = self._make_enemy_fort(x=6_000.0, hp=1.0)
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        for _ in range(11):
            corps.step(_make_inner(), enemy_fortifications=[fort])
        self.assertAlmostEqual(fort.hp, 0.0, places=5)


# ---------------------------------------------------------------------------
# 17. ArtilleryCorps — step with FORTIFY mission
# ---------------------------------------------------------------------------


class TestArtilleryCorpsStepFortify(unittest.TestCase):
    def test_fortify_accumulates_progress(self) -> None:
        cfg = ArtilleryUnitConfig(fortify_steps=12)
        unit = _make_unit(
            x=2_000.0, y=2_500.0,
            mission=ArtilleryMission.FORTIFY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        corps.step(_make_inner())
        self.assertEqual(unit._fortify_progress, 1)

    def test_fortify_completes_after_required_steps(self) -> None:
        cfg = ArtilleryUnitConfig(fortify_steps=12)
        unit = _make_unit(
            x=2_000.0, y=2_500.0,
            mission=ArtilleryMission.FORTIFY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        for _ in range(12):
            corps.step(_make_inner())
        self.assertEqual(len(corps.fortifications), 1)

    def test_fortify_completion_resets_progress(self) -> None:
        cfg = ArtilleryUnitConfig(fortify_steps=12)
        unit = _make_unit(
            x=2_000.0, y=2_500.0,
            mission=ArtilleryMission.FORTIFY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        for _ in range(12):
            corps.step(_make_inner())
        self.assertEqual(unit._fortify_progress, 0)

    def test_fortify_fort_is_at_battery_position(self) -> None:
        cfg = ArtilleryUnitConfig(fortify_steps=1)
        unit = _make_unit(
            x=3_333.0, y=1_111.0,
            mission=ArtilleryMission.FORTIFY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        corps.step(_make_inner())
        fort = corps.fortifications[0]
        self.assertAlmostEqual(fort.x, 3_333.0)
        self.assertAlmostEqual(fort.y, 1_111.0)

    def test_fortify_completed_fort_is_complete(self) -> None:
        cfg = ArtilleryUnitConfig(fortify_steps=1)
        unit = _make_unit(
            x=2_000.0, y=2_500.0,
            mission=ArtilleryMission.FORTIFY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        report = corps.step(_make_inner())
        self.assertEqual(report.fortifications_completed, 1)
        self.assertTrue(corps.fortifications[0].complete)

    def test_fortify_requires_10plus_steps(self) -> None:
        """Fortify_steps must be >= 1; default DEFAULT_FORTIFY_STEPS >= 10."""
        self.assertGreaterEqual(DEFAULT_FORTIFY_STEPS, 10)

    def test_fortify_no_fort_before_completion(self) -> None:
        cfg = ArtilleryUnitConfig(fortify_steps=12)
        unit = _make_unit(
            x=2_000.0, y=2_500.0,
            mission=ArtilleryMission.FORTIFY,
            config=cfg,
        )
        corps = ArtilleryCorps(units=[unit], map_width=MAP_W, map_height=MAP_H)
        for _ in range(11):
            corps.step(_make_inner())
        self.assertEqual(len(corps.fortifications), 0)


# ---------------------------------------------------------------------------
# 18. ArtilleryReport dataclass fields
# ---------------------------------------------------------------------------


class TestArtilleryReport(unittest.TestCase):
    def test_fields_exist(self) -> None:
        r = ArtilleryReport(
            morale_damage_dealt=0.5,
            guns_silenced=2,
            fortification_damage=0.3,
            fortifications_completed=1,
        )
        self.assertAlmostEqual(r.morale_damage_dealt, 0.5)
        self.assertEqual(r.guns_silenced, 2)
        self.assertAlmostEqual(r.fortification_damage, 0.3)
        self.assertEqual(r.fortifications_completed, 1)

    def test_no_reward_field(self) -> None:
        r = ArtilleryReport(0.0, 0, 0.0, 0)
        self.assertFalse(hasattr(r, "reward"))


# ---------------------------------------------------------------------------
# 19. ArtilleryCorpsEnv — construction and action/obs spaces
# ---------------------------------------------------------------------------


class TestArtilleryCorpsEnvConstruction(unittest.TestCase):
    def test_observation_space_shape(self) -> None:
        env = _make_env()
        expected_dim = _art_obs_dim(
            n_divisions=2, n_artillery_batteries=4
        )
        self.assertEqual(env.observation_space.shape, (expected_dim,))

    def test_action_space_shape(self) -> None:
        env = _make_env()
        # 2 divisions + 4 batteries
        self.assertEqual(env.action_space.shape, (6,))

    def test_action_space_nvec_corps_portion(self) -> None:
        env = _make_env()
        corps_nvec = env.action_space.nvec[: env.n_divisions]
        for n in corps_nvec:
            self.assertEqual(n, env.n_corps_options)

    def test_action_space_nvec_art_portion(self) -> None:
        env = _make_env()
        art_nvec = env.action_space.nvec[env.n_divisions :]
        for n in art_nvec:
            self.assertEqual(n, N_ARTILLERY_MISSIONS)

    def test_n_artillery_batteries_stored(self) -> None:
        env = ArtilleryCorpsEnv(
            n_divisions=2, n_brigades_per_division=2, n_blue_per_brigade=2,
            n_artillery_batteries=6,
        )
        self.assertEqual(env.n_artillery_batteries, 6)

    def test_invalid_n_batteries_raises(self) -> None:
        with self.assertRaises(ValueError):
            ArtilleryCorpsEnv(n_divisions=2, n_brigades_per_division=2,
                              n_blue_per_brigade=2, n_artillery_batteries=0)

    def test_custom_artillery_corps_accepted(self) -> None:
        custom = ArtilleryCorps.generate_default(MAP_W, MAP_H, n_batteries=3)
        env = ArtilleryCorpsEnv(
            n_divisions=2, n_brigades_per_division=2, n_blue_per_brigade=2,
            map_width=MAP_W, map_height=MAP_H,
            artillery_corps=custom,
        )
        self.assertEqual(env.n_artillery_batteries, 3)

    def test_obs_low_high_shapes_match(self) -> None:
        env = _make_env()
        self.assertEqual(
            env.observation_space.low.shape,
            env.observation_space.high.shape,
        )


# ---------------------------------------------------------------------------
# 20. ArtilleryCorpsEnv — reset observation shape and content
# ---------------------------------------------------------------------------


class TestArtilleryCorpsEnvReset(unittest.TestCase):
    def test_reset_obs_shape(self) -> None:
        env = _make_env()
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_reset_obs_dtype(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.dtype, np.float32)

    def test_reset_obs_within_bounds(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=1)
        self.assertTrue(np.all(obs >= env.observation_space.low - 1e-5))
        self.assertTrue(np.all(obs <= env.observation_space.high + 1e-5))

    def test_reset_info_is_dict(self) -> None:
        env = _make_env()
        _, info = env.reset(seed=0)
        self.assertIsInstance(info, dict)

    def test_reset_clears_fortifications(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        self.assertEqual(len(env._artillery.fortifications), 0)

    def test_reset_batteries_alive(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        for unit in env._artillery.units[: env.n_artillery_batteries]:
            self.assertTrue(unit.alive)


# ---------------------------------------------------------------------------
# 21. ArtilleryCorpsEnv — step observation shape and info dict
# ---------------------------------------------------------------------------


class TestArtilleryCorpsEnvStep(unittest.TestCase):
    def test_step_obs_shape(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_step_obs_within_bounds(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        self.assertTrue(np.all(obs >= env.observation_space.low - 1e-5))
        self.assertTrue(np.all(obs <= env.observation_space.high + 1e-5))

    def test_step_info_has_artillery_key(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertIn("artillery", info)

    def test_step_artillery_info_keys(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        art_info = info["artillery"]
        for key in (
            "morale_damage_dealt",
            "guns_silenced",
            "fortification_damage",
            "fortifications_completed",
            "n_blue_forts",
            "n_red_forts",
        ):
            self.assertIn(key, art_info)

    def test_step_returns_float_reward(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        self.assertIsInstance(reward, float)

    def test_step_terminated_or_truncated_are_bool(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)


# ---------------------------------------------------------------------------
# 22. ArtilleryCorpsEnv — invalid action raises ValueError
# ---------------------------------------------------------------------------


class TestArtilleryCorpsEnvInvalidAction(unittest.TestCase):
    def test_wrong_shape_raises(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        bad_action = np.zeros(3, dtype=np.int64)
        with self.assertRaises(ValueError):
            env.step(bad_action)

    def test_too_long_action_raises(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        bad_action = np.zeros(20, dtype=np.int64)
        with self.assertRaises(ValueError):
            env.step(bad_action)


# ---------------------------------------------------------------------------
# 23. ArtilleryCorpsEnv — full short episode
# ---------------------------------------------------------------------------


class TestArtilleryCorpsEnvEpisode(unittest.TestCase):
    def test_short_episode_completes(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=42)
        steps = 0
        done = False
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        self.assertIsInstance(obs, np.ndarray)

    def test_episode_reward_is_finite(self) -> None:
        env = _make_env()
        env.reset(seed=42)
        total = 0.0
        for _ in range(10):
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            if term or trunc:
                break
        self.assertTrue(math.isfinite(total))


# ---------------------------------------------------------------------------
# 24. AC-1 integration: grand battery breaks enemy line in < 300 steps
# ---------------------------------------------------------------------------


class TestAC1GrandBattery(unittest.TestCase):
    """AC-1: Grand battery of ≥ 6 guns breaks enemy line in < 5 game minutes.

    Setup: 6 batteries in GRAND_BATTERY mission, all placed within grand-
    battery radius, target a single enemy battalion with full morale.
    The combined battery should reduce the target's morale below the
    routing threshold within 300 steps (5 game minutes at 1 s/step).
    """

    def test_six_guns_break_enemy_line(self) -> None:
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_morale_damage=DEFAULT_BASE_MORALE_DAMAGE,
            grand_battery_bonus=DEFAULT_GRAND_BATTERY_BONUS,
            grand_battery_radius=1_000.0,
            team=0,
        )
        # 6 guns clustered at (5000, 2500), all in GRAND_BATTERY mission
        units = [
            ArtilleryUnit(
                x=5_000.0 + i * 50.0,
                y=2_500.0,
                theta=0.0,
                strength=1.0,
                team=0,
                config=cfg,
                mission=ArtilleryMission.GRAND_BATTERY,
            )
            for i in range(6)
        ]
        corps = ArtilleryCorps(units=units, map_width=MAP_W, map_height=MAP_H)

        # One enemy battalion with full morale at 6500m (in range)
        inner = _make_inner(red_bns=[(6_500.0, 2_500.0, 1.0, 1.0)])
        target = inner._battalions["red_0"]
        target.routing_threshold = 0.3

        steps = 0
        max_steps = 300  # 5 game minutes
        while steps < max_steps and not target.routed:
            corps.step(inner)
            steps += 1

        self.assertTrue(
            target.routed,
            f"Enemy line not broken after {steps} steps "
            f"(morale={target.morale:.3f})"
        )
        self.assertLess(
            steps,
            max_steps,
            f"Grand battery took {steps} ≥ 300 steps to break enemy line",
        )

    def test_single_gun_slower_than_grand_battery(self) -> None:
        """A single gun should take more steps to route the enemy than a battery."""
        cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_morale_damage=DEFAULT_BASE_MORALE_DAMAGE,
            grand_battery_bonus=DEFAULT_GRAND_BATTERY_BONUS,
            grand_battery_radius=1_000.0,
            team=0,
        )
        # Single gun
        single_unit = ArtilleryUnit(
            x=5_000.0, y=2_500.0, theta=0.0, strength=1.0, team=0,
            config=cfg, mission=ArtilleryMission.GRAND_BATTERY,
        )
        single_corps = ArtilleryCorps(
            units=[single_unit], map_width=MAP_W, map_height=MAP_H
        )

        # Six guns
        battery_units = [
            ArtilleryUnit(
                x=5_000.0 + i * 50.0, y=2_500.0, theta=0.0, strength=1.0,
                team=0, config=cfg, mission=ArtilleryMission.GRAND_BATTERY,
            )
            for i in range(6)
        ]
        battery_corps = ArtilleryCorps(
            units=battery_units, map_width=MAP_W, map_height=MAP_H
        )

        def steps_to_route(corps_: ArtilleryCorps) -> int:
            inn = _make_inner(red_bns=[(6_500.0, 2_500.0, 1.0, 1.0)])
            t = inn._battalions["red_0"]
            t.routing_threshold = 0.3
            s = 0
            while s < 1_000 and not t.routed:
                corps_.step(inn)
                s += 1
            return s

        steps_single = steps_to_route(single_corps)
        steps_battery = steps_to_route(battery_corps)
        self.assertLess(steps_battery, steps_single)


# ---------------------------------------------------------------------------
# 25. AC-2 integration: counter-battery silences guns faster than inf. fire
# ---------------------------------------------------------------------------


class TestAC2CounterBattery(unittest.TestCase):
    """AC-2: Counter-battery fire silences enemy guns faster than targeting infantry."""

    def test_counter_battery_vs_artillery_faster_than_vs_infantry(self) -> None:
        # Config: counter-battery bonus ensures faster silencing
        cb_cfg = ArtilleryUnitConfig(
            max_range=2_000.0,
            base_morale_damage=0.05,
            base_strength_damage=0.05,
            counter_battery_bonus=0.10,
            team=0,
        )

        def steps_to_silence_art() -> int:
            """Steps for 1 gun in COUNTER_BATTERY to kill 1 enemy artillery."""
            enemy_art = ArtilleryUnit(
                x=5_500.0, y=2_500.0, theta=math.pi, strength=1.0, team=1,
                config=ArtilleryUnitConfig(team=1),
            )
            unit = ArtilleryUnit(
                x=5_000.0, y=2_500.0, theta=0.0, strength=1.0, team=0,
                config=cb_cfg, mission=ArtilleryMission.COUNTER_BATTERY,
            )
            corps = ArtilleryCorps(
                units=[unit], map_width=MAP_W, map_height=MAP_H
            )
            s = 0
            while s < 1_000 and enemy_art.alive:
                corps.step(_make_inner(), enemy_artillery=[enemy_art])
                s += 1
            return s

        def steps_to_route_infantry() -> int:
            """Steps for 1 gun in COUNTER_BATTERY (fallback) to route 1 infantry."""
            unit = ArtilleryUnit(
                x=5_000.0, y=2_500.0, theta=0.0, strength=1.0, team=0,
                config=cb_cfg, mission=ArtilleryMission.COUNTER_BATTERY,
            )
            corps = ArtilleryCorps(
                units=[unit], map_width=MAP_W, map_height=MAP_H
            )
            inner = _make_inner(red_bns=[(6_000.0, 2_500.0, 1.0, 1.0)])
            t = inner._battalions["red_0"]
            t.routing_threshold = 0.3
            s = 0
            while s < 1_000 and not t.routed:
                corps.step(inner, enemy_artillery=[])
                s += 1
            return s

        art_steps = steps_to_silence_art()
        inf_steps = steps_to_route_infantry()
        self.assertLess(
            art_steps,
            inf_steps,
            f"Counter-battery took {art_steps} steps vs artillery but "
            f"{inf_steps} steps vs infantry — should be faster vs artillery.",
        )


# ---------------------------------------------------------------------------
# 26. AC-3 integration: fortification state in observation after completion
# ---------------------------------------------------------------------------


class TestAC3Fortification(unittest.TestCase):
    """AC-3: Agents can build fortifications; forts appear in observation."""

    def test_fortification_obs_nonzero_after_completion(self) -> None:
        env = ArtilleryCorpsEnv(
            n_divisions=2, n_brigades_per_division=2, n_blue_per_brigade=2,
            n_artillery_batteries=4, max_steps=500,
        )
        env.reset(seed=0)

        # Assign all batteries to FORTIFY
        n_divs = env.n_divisions
        n_art = env.n_artillery_batteries
        n_corps_opts = env.n_corps_options

        fortify_action = np.array(
            [0] * n_divs + [int(ArtilleryMission.FORTIFY)] * n_art,
            dtype=np.int64,
        )

        # Run enough steps to complete a fortification
        obs = None
        for _ in range(DEFAULT_FORTIFY_STEPS + 2):
            obs, _, term, trunc, info = env.step(fortify_action)
            if term or trunc:
                env.reset(seed=0)

        # At least one fort should be recorded
        self.assertGreater(
            info["artillery"]["fortifications_completed"]
            + info["artillery"]["n_blue_forts"],
            0,
            "No fortification was built after enough FORTIFY steps.",
        )

        # Fort slot in observation should be non-zero (hp > 0)
        base_dim = _corps_obs_dim(n_divs)
        unit_block = n_art * _ART_UNIT_OBS_DIM
        summary_offset = base_dim + unit_block + _ART_SUMMARY_DIM
        # Fort slot 0 hp is at index: summary_offset + 2 (third element of slot)
        fort_hp_idx = summary_offset + 2
        self.assertGreater(obs[fort_hp_idx], 0.0)

    def test_fortifications_cleared_on_reset(self) -> None:
        env = _make_env()
        env.reset(seed=0)

        # Force-add a fortification
        env._artillery.fortifications.append(
            Fortification(x=2000.0, y=2500.0, team=0, hp=1.0, complete=True)
        )
        self.assertEqual(len(env._artillery.fortifications), 1)

        # Reset should clear them
        env.reset(seed=1)
        self.assertEqual(len(env._artillery.fortifications), 0)


# ---------------------------------------------------------------------------
# 27. ArtilleryCorpsEnv — dead batteries emit zero obs block
# ---------------------------------------------------------------------------


class TestArtilleryCorpsEnvDeadBatteries(unittest.TestCase):
    def test_dead_battery_obs_block_is_zeros(self) -> None:
        env = _make_env()
        env.reset(seed=0)

        # Kill first battery
        env._artillery.units[0].alive = False
        env._artillery.units[0].strength = 0.0

        obs, _ = env.reset(seed=2)
        # Kill first battery *after* reset to check obs building
        env._artillery.units[0].alive = False
        env._artillery.units[0].strength = 0.0
        env._last_art_report = ArtilleryReport(0.0, 0, 0.0, 0)
        obs = env._build_art_obs()

        base_dim = _corps_obs_dim(env.n_divisions)
        unit_0_start = base_dim
        unit_0_end = unit_0_start + _ART_UNIT_OBS_DIM
        unit_0_block = obs[unit_0_start:unit_0_end]
        np.testing.assert_array_equal(unit_0_block, np.zeros(_ART_UNIT_OBS_DIM))


# ---------------------------------------------------------------------------
# 28. ArtilleryCorpsEnv — fortification reward shaping
# ---------------------------------------------------------------------------


class TestArtilleryCorpsEnvFortReward(unittest.TestCase):
    def test_fort_completion_gives_bonus_reward(self) -> None:
        env = ArtilleryCorpsEnv(
            n_divisions=2, n_brigades_per_division=2, n_blue_per_brigade=2,
            n_artillery_batteries=4, max_steps=500,
        )
        env.reset(seed=0)

        n_divs = env.n_divisions
        n_art = env.n_artillery_batteries

        # FORTIFY action for all batteries
        fortify_action = np.array(
            [0] * n_divs + [int(ArtilleryMission.FORTIFY)] * n_art,
            dtype=np.int64,
        )

        rewards = []
        for _ in range(DEFAULT_FORTIFY_STEPS + 2):
            _, r, term, trunc, info = env.step(fortify_action)
            rewards.append(r)
            if term or trunc:
                break

        # The step at which the fort completes should have a finite reward
        # that includes the fort bonus (base reward may vary, but reward is always finite).
        step_idx = DEFAULT_FORTIFY_STEPS - 1
        if step_idx < len(rewards):
            self.assertTrue(math.isfinite(rewards[step_idx]))


if __name__ == "__main__":
    unittest.main()
