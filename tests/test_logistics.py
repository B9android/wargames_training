# tests/test_logistics.py
"""Tests for envs/sim/logistics.py — supply, ammunition, and fatigue model.

Covers:
* LogisticsConfig validation — all parameter constraints
* LogisticsState properties — is_ammo_exhausted, is_starving, etc.
* SupplyWagon — take_damage, move_toward, is_alive
* consume_ammo — depletion, scaling, jams at zero
* consume_food — per-step drain
* update_fatigue — accumulation from movement/fire, recovery at halt
* check_resupply — radius check, alive check, enable flag, clamping to 1.0
* get_ammo_modifier — linear penalty below CRITICAL_AMMO_THRESHOLD
* get_fatigue_speed_modifier — onset threshold and linear degradation
* get_fatigue_accuracy_modifier — same formula as speed modifier
* BattalionEnv integration — enable_logistics=True obs space, info dict
* Acceptance criteria: ammo exhaustion in long episodes; fatigue penalises
  speed and accuracy; resupply mechanic respects disable flag
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.logistics import (
    CRITICAL_AMMO_THRESHOLD,
    DEFAULT_INITIAL_AMMO,
    DEFAULT_INITIAL_FOOD,
    FATIGUE_ONSET_THRESHOLD,
    LogisticsConfig,
    LogisticsState,
    SupplyWagon,
    check_resupply,
    consume_ammo,
    consume_food,
    get_ammo_modifier,
    get_fatigue_accuracy_modifier,
    get_fatigue_speed_modifier,
    update_fatigue,
)
from envs.battalion_env import BattalionEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_config() -> LogisticsConfig:
    return LogisticsConfig()


def _fresh_state() -> LogisticsState:
    return LogisticsState()


def _make_wagon(x: float = 300.0, y: float = 500.0, team: int = 0) -> SupplyWagon:
    return SupplyWagon(x=x, y=y, team=team)


# ---------------------------------------------------------------------------
# LogisticsConfig — validation
# ---------------------------------------------------------------------------


class TestLogisticsConfigValidation(unittest.TestCase):
    """Verify parameter constraints in LogisticsConfig.__post_init__."""

    def test_default_instantiation(self) -> None:
        cfg = LogisticsConfig()
        self.assertEqual(cfg.initial_ammo, DEFAULT_INITIAL_AMMO)
        self.assertEqual(cfg.initial_food, DEFAULT_INITIAL_FOOD)
        self.assertTrue(cfg.enable_resupply)

    def test_invalid_initial_ammo_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(initial_ammo=-0.1)

    def test_invalid_initial_ammo_above_one(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(initial_ammo=1.1)

    def test_invalid_initial_food_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(initial_food=-0.01)

    def test_invalid_initial_food_above_one(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(initial_food=1.5)

    def test_invalid_ammo_per_volley_negative(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(ammo_per_volley=-0.001)

    def test_invalid_food_per_step_negative(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(food_per_step=-0.001)

    def test_invalid_fatigue_per_move_negative(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(fatigue_per_move_step=-0.001)

    def test_invalid_fatigue_per_fire_negative(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(fatigue_per_fire_step=-0.001)

    def test_invalid_fatigue_recovery_negative(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(fatigue_recovery_per_halt_step=-0.001)

    def test_invalid_resupply_radius_zero(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(resupply_radius=0.0)

    def test_invalid_resupply_radius_negative(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(resupply_radius=-10.0)

    def test_invalid_ammo_resupply_rate_above_one(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(ammo_resupply_rate=1.1)

    def test_invalid_food_resupply_rate_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(food_resupply_rate=-0.01)

    def test_invalid_low_ammo_penalty_above_one(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(low_ammo_accuracy_penalty=1.5)

    def test_invalid_fatigue_speed_penalty_above_one(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(fatigue_speed_penalty=1.2)

    def test_invalid_fatigue_accuracy_penalty_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(fatigue_accuracy_penalty=-0.1)

    def test_invalid_wagon_speed_zero(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(wagon_speed=0.0)

    def test_invalid_wagon_max_strength_zero(self) -> None:
        with self.assertRaises(ValueError):
            LogisticsConfig(wagon_max_strength=0.0)

    def test_zero_ammo_per_volley_valid(self) -> None:
        """Zero ammo consumption is legal (resupply mode / no depletion)."""
        cfg = LogisticsConfig(ammo_per_volley=0.0)
        self.assertEqual(cfg.ammo_per_volley, 0.0)

    def test_disable_resupply_valid(self) -> None:
        cfg = LogisticsConfig(enable_resupply=False)
        self.assertFalse(cfg.enable_resupply)


# ---------------------------------------------------------------------------
# LogisticsState — properties
# ---------------------------------------------------------------------------


class TestLogisticsState(unittest.TestCase):
    """Verify LogisticsState properties and initial values."""

    def test_initial_values(self) -> None:
        s = LogisticsState()
        self.assertEqual(s.ammo, DEFAULT_INITIAL_AMMO)
        self.assertEqual(s.food, DEFAULT_INITIAL_FOOD)
        self.assertEqual(s.fatigue, 0.0)

    def test_is_ammo_exhausted_full(self) -> None:
        self.assertFalse(LogisticsState(ammo=1.0).is_ammo_exhausted)

    def test_is_ammo_exhausted_zero(self) -> None:
        self.assertTrue(LogisticsState(ammo=0.0).is_ammo_exhausted)

    def test_is_critically_low_ammo_above_threshold(self) -> None:
        self.assertFalse(LogisticsState(ammo=CRITICAL_AMMO_THRESHOLD).is_critically_low_ammo)

    def test_is_critically_low_ammo_below_threshold(self) -> None:
        self.assertTrue(LogisticsState(ammo=CRITICAL_AMMO_THRESHOLD - 0.01).is_critically_low_ammo)

    def test_is_starving_full(self) -> None:
        self.assertFalse(LogisticsState(food=1.0).is_starving)

    def test_is_starving_zero(self) -> None:
        self.assertTrue(LogisticsState(food=0.0).is_starving)

    def test_custom_initial_values(self) -> None:
        s = LogisticsState(ammo=0.5, food=0.3, fatigue=0.4)
        self.assertAlmostEqual(s.ammo, 0.5)
        self.assertAlmostEqual(s.food, 0.3)
        self.assertAlmostEqual(s.fatigue, 0.4)


# ---------------------------------------------------------------------------
# SupplyWagon
# ---------------------------------------------------------------------------


class TestSupplyWagon(unittest.TestCase):
    """Tests for SupplyWagon unit."""

    def test_is_alive_full_strength(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0)
        self.assertTrue(w.is_alive)

    def test_is_alive_destroyed(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0, strength=0.0)
        self.assertFalse(w.is_alive)

    def test_take_damage_reduces_strength(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0)
        w.take_damage(0.3)
        self.assertAlmostEqual(w.strength, 0.7)

    def test_take_damage_clamped_at_zero(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0, strength=0.2)
        w.take_damage(0.5)
        self.assertEqual(w.strength, 0.0)
        self.assertFalse(w.is_alive)

    def test_take_damage_cannot_exceed_one(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0)
        w.take_damage(-0.1)  # negative damage — no change
        self.assertAlmostEqual(w.strength, 1.0)

    def test_move_toward_reaches_target(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0)
        w.move_toward(5.0, 0.0, speed=100.0, dt=1.0)
        self.assertAlmostEqual(w.x, 5.0)
        self.assertAlmostEqual(w.y, 0.0)

    def test_move_toward_partial_step(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0)
        w.move_toward(100.0, 0.0, speed=10.0, dt=1.0)
        self.assertAlmostEqual(w.x, 10.0)

    def test_move_toward_diagonal(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0)
        w.move_toward(3.0, 4.0, speed=50.0, dt=1.0)
        # Full distance is 5m; one step moves 50*1.0=50m > 5m so reaches target
        self.assertAlmostEqual(w.x, 3.0)
        self.assertAlmostEqual(w.y, 4.0)

    def test_move_toward_updates_theta(self) -> None:
        w = SupplyWagon(x=0.0, y=0.0, team=0)
        w.move_toward(1.0, 0.0, speed=0.5, dt=1.0)
        self.assertAlmostEqual(w.theta, 0.0)

    def test_move_toward_self_no_op(self) -> None:
        w = SupplyWagon(x=5.0, y=5.0, team=0)
        w.move_toward(5.0, 5.0, speed=10.0, dt=1.0)
        self.assertAlmostEqual(w.x, 5.0)
        self.assertAlmostEqual(w.y, 5.0)

    def test_team_attribute(self) -> None:
        wb = SupplyWagon(x=0.0, y=0.0, team=0)
        wr = SupplyWagon(x=0.0, y=0.0, team=1)
        self.assertEqual(wb.team, 0)
        self.assertEqual(wr.team, 1)


# ---------------------------------------------------------------------------
# consume_ammo
# ---------------------------------------------------------------------------


class TestConsumeAmmo(unittest.TestCase):
    """Tests for the consume_ammo function."""

    def test_full_ammo_full_intensity(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.01)
        state = LogisticsState(ammo=1.0)
        effective = consume_ammo(state, 1.0, cfg)
        self.assertAlmostEqual(effective, 1.0)
        self.assertAlmostEqual(state.ammo, 0.99)

    def test_zero_intensity_no_consumption(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.01)
        state = LogisticsState(ammo=1.0)
        effective = consume_ammo(state, 0.0, cfg)
        self.assertEqual(effective, 0.0)
        self.assertAlmostEqual(state.ammo, 1.0)

    def test_half_intensity_half_cost(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.1)
        state = LogisticsState(ammo=1.0)
        effective = consume_ammo(state, 0.5, cfg)
        self.assertAlmostEqual(effective, 0.5)
        self.assertAlmostEqual(state.ammo, 0.95)  # cost = 0.5 * 0.1 = 0.05

    def test_ammo_exhausted_returns_zero(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.01)
        state = LogisticsState(ammo=0.0)
        effective = consume_ammo(state, 1.0, cfg)
        self.assertEqual(effective, 0.0)
        self.assertTrue(state.is_ammo_exhausted)

    def test_partial_ammo_scales_intensity(self) -> None:
        """When remaining ammo < full cost, intensity is scaled proportionally."""
        cfg = LogisticsConfig(ammo_per_volley=0.1)
        state = LogisticsState(ammo=0.05)  # half what a full volley costs
        effective = consume_ammo(state, 1.0, cfg)
        self.assertAlmostEqual(effective, 0.5, places=5)
        self.assertAlmostEqual(state.ammo, 0.0)

    def test_ammo_depleted_to_exactly_zero(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.1)
        state = LogisticsState(ammo=0.1)
        effective = consume_ammo(state, 1.0, cfg)
        self.assertAlmostEqual(effective, 1.0)
        self.assertAlmostEqual(state.ammo, 0.0)

    def test_weapon_jams_after_exhaustion(self) -> None:
        """Weapon is completely jammed after ammo = 0."""
        cfg = LogisticsConfig(ammo_per_volley=0.2)
        state = LogisticsState(ammo=0.2)
        consume_ammo(state, 1.0, cfg)  # depletes
        # Now jammed
        eff = consume_ammo(state, 1.0, cfg)
        self.assertEqual(eff, 0.0)

    def test_intensity_clamped_to_one(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.01)
        state = LogisticsState(ammo=1.0)
        effective = consume_ammo(state, 2.0, cfg)
        self.assertAlmostEqual(effective, 1.0)

    def test_zero_ammo_per_volley_no_depletion(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.0)
        state = LogisticsState(ammo=1.0)
        effective = consume_ammo(state, 1.0, cfg)
        self.assertAlmostEqual(effective, 1.0)
        self.assertAlmostEqual(state.ammo, 1.0)


# ---------------------------------------------------------------------------
# consume_food
# ---------------------------------------------------------------------------


class TestConsumeFood(unittest.TestCase):
    """Tests for per-step food consumption."""

    def test_reduces_food_by_rate(self) -> None:
        cfg = LogisticsConfig(food_per_step=0.01)
        state = LogisticsState(food=1.0)
        consume_food(state, cfg)
        self.assertAlmostEqual(state.food, 0.99)

    def test_food_clamped_at_zero(self) -> None:
        cfg = LogisticsConfig(food_per_step=0.5)
        state = LogisticsState(food=0.3)
        consume_food(state, cfg)
        self.assertAlmostEqual(state.food, 0.0)

    def test_zero_rate_no_change(self) -> None:
        cfg = LogisticsConfig(food_per_step=0.0)
        state = LogisticsState(food=0.8)
        consume_food(state, cfg)
        self.assertAlmostEqual(state.food, 0.8)

    def test_starving_after_depletion(self) -> None:
        cfg = LogisticsConfig(food_per_step=0.1)
        state = LogisticsState(food=0.05)
        consume_food(state, cfg)
        self.assertTrue(state.is_starving)


# ---------------------------------------------------------------------------
# update_fatigue
# ---------------------------------------------------------------------------


class TestUpdateFatigue(unittest.TestCase):
    """Tests for fatigue accumulation and recovery."""

    def test_moving_accumulates_fatigue(self) -> None:
        cfg = LogisticsConfig(fatigue_per_move_step=0.005)
        state = _fresh_state()
        update_fatigue(state, is_moving=True, is_firing=False, config=cfg)
        self.assertAlmostEqual(state.fatigue, 0.005)

    def test_firing_accumulates_fatigue(self) -> None:
        cfg = LogisticsConfig(fatigue_per_fire_step=0.003)
        state = _fresh_state()
        update_fatigue(state, is_moving=False, is_firing=True, config=cfg)
        self.assertAlmostEqual(state.fatigue, 0.003)

    def test_move_and_fire_both_contribute(self) -> None:
        cfg = LogisticsConfig(fatigue_per_move_step=0.005, fatigue_per_fire_step=0.003)
        state = _fresh_state()
        update_fatigue(state, is_moving=True, is_firing=True, config=cfg)
        self.assertAlmostEqual(state.fatigue, 0.008)

    def test_halting_recovers_fatigue(self) -> None:
        cfg = LogisticsConfig(fatigue_recovery_per_halt_step=0.01)
        state = LogisticsState(fatigue=0.5)
        update_fatigue(state, is_moving=False, is_firing=False, config=cfg)
        self.assertAlmostEqual(state.fatigue, 0.49)

    def test_fatigue_clamped_at_one(self) -> None:
        cfg = LogisticsConfig(fatigue_per_move_step=0.5)
        state = LogisticsState(fatigue=0.8)
        update_fatigue(state, is_moving=True, is_firing=False, config=cfg)
        self.assertAlmostEqual(state.fatigue, 1.0)

    def test_fatigue_clamped_at_zero(self) -> None:
        cfg = LogisticsConfig(fatigue_recovery_per_halt_step=0.5)
        state = LogisticsState(fatigue=0.1)
        update_fatigue(state, is_moving=False, is_firing=False, config=cfg)
        self.assertAlmostEqual(state.fatigue, 0.0)

    def test_fresh_unit_no_fatigue(self) -> None:
        cfg = _default_config()
        state = _fresh_state()
        update_fatigue(state, is_moving=False, is_firing=False, config=cfg)
        self.assertAlmostEqual(state.fatigue, 0.0)


# ---------------------------------------------------------------------------
# check_resupply
# ---------------------------------------------------------------------------


class TestCheckResupply(unittest.TestCase):
    """Tests for the resupply zone mechanic."""

    def test_resupply_in_range(self) -> None:
        cfg = LogisticsConfig(resupply_radius=100.0, ammo_resupply_rate=0.05)
        state = LogisticsState(ammo=0.5)
        wagon = _make_wagon(x=300.0, y=500.0)
        result = check_resupply(state, unit_x=350.0, unit_y=500.0, wagon=wagon, config=cfg)
        self.assertTrue(result)
        self.assertAlmostEqual(state.ammo, 0.55)

    def test_no_resupply_out_of_range(self) -> None:
        cfg = LogisticsConfig(resupply_radius=100.0)
        state = LogisticsState(ammo=0.5)
        wagon = _make_wagon(x=300.0, y=500.0)
        result = check_resupply(state, unit_x=600.0, unit_y=500.0, wagon=wagon, config=cfg)
        self.assertFalse(result)
        self.assertAlmostEqual(state.ammo, 0.5)

    def test_no_resupply_destroyed_wagon(self) -> None:
        cfg = LogisticsConfig(resupply_radius=100.0)
        state = LogisticsState(ammo=0.5)
        wagon = _make_wagon()
        wagon.take_damage(1.0)  # destroy wagon
        result = check_resupply(state, unit_x=300.0, unit_y=500.0, wagon=wagon, config=cfg)
        self.assertFalse(result)
        self.assertAlmostEqual(state.ammo, 0.5)

    def test_no_resupply_when_disabled(self) -> None:
        cfg = LogisticsConfig(enable_resupply=False)
        state = LogisticsState(ammo=0.5)
        wagon = _make_wagon()
        result = check_resupply(state, unit_x=300.0, unit_y=500.0, wagon=wagon, config=cfg)
        self.assertFalse(result)
        self.assertAlmostEqual(state.ammo, 0.5)

    def test_ammo_clamped_at_one_on_resupply(self) -> None:
        cfg = LogisticsConfig(resupply_radius=200.0, ammo_resupply_rate=0.5)
        state = LogisticsState(ammo=0.9)
        wagon = _make_wagon()
        check_resupply(state, unit_x=300.0, unit_y=500.0, wagon=wagon, config=cfg)
        self.assertLessEqual(state.ammo, 1.0)

    def test_food_also_resupplied(self) -> None:
        cfg = LogisticsConfig(resupply_radius=200.0, food_resupply_rate=0.03)
        state = LogisticsState(food=0.5)
        wagon = _make_wagon()
        check_resupply(state, unit_x=300.0, unit_y=500.0, wagon=wagon, config=cfg)
        self.assertAlmostEqual(state.food, 0.53)

    def test_resupply_at_exact_radius_boundary(self) -> None:
        """Unit at exactly the radius boundary DOES get resupply (dist <= radius)."""
        cfg = LogisticsConfig(resupply_radius=100.0, ammo_resupply_rate=0.05)
        state = LogisticsState(ammo=0.5)
        wagon = SupplyWagon(x=0.0, y=0.0, team=0)
        # Exactly at radius — inclusive boundary
        result = check_resupply(state, unit_x=100.0, unit_y=0.0, wagon=wagon, config=cfg)
        self.assertTrue(result)

    def test_resupply_just_outside_radius(self) -> None:
        """Unit just beyond the radius boundary should NOT get resupply."""
        cfg = LogisticsConfig(resupply_radius=100.0)
        state = LogisticsState(ammo=0.5)
        wagon = SupplyWagon(x=0.0, y=0.0, team=0)
        result = check_resupply(state, unit_x=100.1, unit_y=0.0, wagon=wagon, config=cfg)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# get_ammo_modifier
# ---------------------------------------------------------------------------


class TestGetAmmoModifier(unittest.TestCase):
    """Tests for the ammo accuracy modifier."""

    def test_full_ammo_no_penalty(self) -> None:
        cfg = _default_config()
        state = LogisticsState(ammo=1.0)
        self.assertAlmostEqual(get_ammo_modifier(state, cfg), 1.0)

    def test_at_critical_threshold_no_penalty(self) -> None:
        cfg = _default_config()
        state = LogisticsState(ammo=CRITICAL_AMMO_THRESHOLD)
        self.assertAlmostEqual(get_ammo_modifier(state, cfg), 1.0)

    def test_zero_ammo_full_penalty(self) -> None:
        cfg = LogisticsConfig(low_ammo_accuracy_penalty=0.5)
        state = LogisticsState(ammo=0.0)
        self.assertAlmostEqual(get_ammo_modifier(state, cfg), 0.5)

    def test_half_critical_threshold(self) -> None:
        cfg = LogisticsConfig(low_ammo_accuracy_penalty=0.5)
        state = LogisticsState(ammo=CRITICAL_AMMO_THRESHOLD / 2)
        mod = get_ammo_modifier(state, cfg)
        self.assertAlmostEqual(mod, 0.75)  # midpoint between 0.5 and 1.0

    def test_modifier_between_penalty_and_one(self) -> None:
        cfg = LogisticsConfig(low_ammo_accuracy_penalty=0.3)
        state = LogisticsState(ammo=CRITICAL_AMMO_THRESHOLD / 4)
        mod = get_ammo_modifier(state, cfg)
        self.assertGreaterEqual(mod, 0.3)
        self.assertLessEqual(mod, 1.0)


# ---------------------------------------------------------------------------
# get_fatigue_speed_modifier
# ---------------------------------------------------------------------------


class TestGetFatigueSpeedModifier(unittest.TestCase):
    """Tests for the fatigue speed modifier."""

    def test_zero_fatigue_no_penalty(self) -> None:
        cfg = _default_config()
        state = _fresh_state()
        self.assertAlmostEqual(get_fatigue_speed_modifier(state, cfg), 1.0)

    def test_at_onset_no_penalty(self) -> None:
        cfg = _default_config()
        state = LogisticsState(fatigue=FATIGUE_ONSET_THRESHOLD)
        self.assertAlmostEqual(get_fatigue_speed_modifier(state, cfg), 1.0)

    def test_full_fatigue_max_penalty(self) -> None:
        cfg = LogisticsConfig(fatigue_speed_penalty=0.3)
        state = LogisticsState(fatigue=1.0)
        self.assertAlmostEqual(get_fatigue_speed_modifier(state, cfg), 0.7)

    def test_half_post_onset_range(self) -> None:
        cfg = LogisticsConfig(fatigue_speed_penalty=0.4)
        onset = FATIGUE_ONSET_THRESHOLD
        half_post = onset + (1.0 - onset) / 2.0
        state = LogisticsState(fatigue=half_post)
        mod = get_fatigue_speed_modifier(state, cfg)
        self.assertAlmostEqual(mod, 1.0 - 0.4 * 0.5, places=5)

    def test_modifier_bounded_below_by_max_penalty(self) -> None:
        cfg = LogisticsConfig(fatigue_speed_penalty=0.5)
        state = LogisticsState(fatigue=1.0)
        self.assertGreaterEqual(get_fatigue_speed_modifier(state, cfg), 0.0)
        self.assertLessEqual(get_fatigue_speed_modifier(state, cfg), 1.0)

    def test_just_above_onset(self) -> None:
        cfg = LogisticsConfig(fatigue_speed_penalty=0.3)
        state = LogisticsState(fatigue=FATIGUE_ONSET_THRESHOLD + 0.001)
        mod = get_fatigue_speed_modifier(state, cfg)
        self.assertLess(mod, 1.0)
        self.assertGreater(mod, 0.7)


# ---------------------------------------------------------------------------
# get_fatigue_accuracy_modifier
# ---------------------------------------------------------------------------


class TestGetFatigueAccuracyModifier(unittest.TestCase):
    """Tests for the fatigue accuracy modifier."""

    def test_zero_fatigue_no_penalty(self) -> None:
        cfg = _default_config()
        state = _fresh_state()
        self.assertAlmostEqual(get_fatigue_accuracy_modifier(state, cfg), 1.0)

    def test_at_onset_no_penalty(self) -> None:
        cfg = _default_config()
        state = LogisticsState(fatigue=FATIGUE_ONSET_THRESHOLD)
        self.assertAlmostEqual(get_fatigue_accuracy_modifier(state, cfg), 1.0)

    def test_full_fatigue_max_penalty(self) -> None:
        cfg = LogisticsConfig(fatigue_accuracy_penalty=0.2)
        state = LogisticsState(fatigue=1.0)
        self.assertAlmostEqual(get_fatigue_accuracy_modifier(state, cfg), 0.8)

    def test_independent_of_speed_penalty(self) -> None:
        """Accuracy and speed penalties are separate config fields."""
        cfg = LogisticsConfig(fatigue_speed_penalty=0.5, fatigue_accuracy_penalty=0.1)
        state = LogisticsState(fatigue=1.0)
        speed_mod = get_fatigue_speed_modifier(state, cfg)
        acc_mod = get_fatigue_accuracy_modifier(state, cfg)
        self.assertAlmostEqual(speed_mod, 0.5)
        self.assertAlmostEqual(acc_mod, 0.9)


# ---------------------------------------------------------------------------
# BattalionEnv integration
# ---------------------------------------------------------------------------


class TestBattalionEnvLogisticsIntegration(unittest.TestCase):
    """Tests for logistics integration in BattalionEnv."""

    def _make_env(self, **kwargs) -> BattalionEnv:
        kwargs.setdefault("curriculum_level", 1)  # stationary Red — avoids early termination
        return BattalionEnv(
            enable_logistics=True,
            randomize_terrain=False,
            **kwargs,
        )

    # --- Observation space ---

    def test_obs_shape_logistics_only(self) -> None:
        """Base 17 + 3 logistics = 20 dims."""
        env = self._make_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape[0], 20)

    def test_obs_shape_formations_and_logistics(self) -> None:
        """Base 17 + 2 formations + 3 logistics = 22 dims."""
        env = self._make_env(enable_formations=True)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape[0], 22)

    def test_obs_shape_unchanged_without_logistics(self) -> None:
        """Backward compatibility: no extra dims when logistics disabled."""
        env = BattalionEnv(randomize_terrain=False)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape[0], 17)

    def test_obs_logistics_dims_start_at_full_ammo(self) -> None:
        env = self._make_env()
        obs, _ = env.reset(seed=0)
        # Last 3 dims: ammo, food, fatigue
        self.assertAlmostEqual(float(obs[-3]), 1.0, places=5)  # ammo
        self.assertAlmostEqual(float(obs[-2]), 1.0, places=5)  # food
        self.assertAlmostEqual(float(obs[-1]), 0.0, places=5)  # fatigue

    def test_obs_within_space_bounds(self) -> None:
        env = self._make_env()
        obs, _ = env.reset(seed=0)
        self.assertTrue(np.all(obs >= env.observation_space.low))
        self.assertTrue(np.all(obs <= env.observation_space.high))

    # --- Info dict ---

    def test_info_contains_logistics_keys(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        for key in ("blue_ammo", "blue_food", "blue_fatigue",
                    "red_ammo", "red_food", "red_fatigue"):
            self.assertIn(key, info, f"Key {key!r} missing from info")

    def test_info_logistics_keys_absent_without_enable(self) -> None:
        env = BattalionEnv(randomize_terrain=False)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.assertNotIn("blue_ammo", info)

    # --- Ammo depletion ---

    def test_ammo_decreases_with_firing(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        self.assertLess(info["blue_ammo"], 1.0)

    def test_ammo_unchanged_without_firing(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(info["blue_ammo"], 1.0, places=3)

    def test_ammo_exhaustion_in_long_episode(self) -> None:
        """Agents that fire continuously run out of ammo in long episodes."""
        cfg = LogisticsConfig(ammo_per_volley=0.01, enable_resupply=False)
        env = self._make_env(logistics_config=cfg, max_steps=200)
        env.reset(seed=99)
        exhausted = False
        for _ in range(200):
            action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            if info.get("blue_ammo", 1.0) <= 0.0:
                exhausted = True
                break
            if terminated or truncated:
                break
        self.assertTrue(exhausted, "Blue ammo should be exhausted after sustained firing")

    def test_ammo_exhaustion_rate_80_percent(self) -> None:
        """Acceptance criterion: ≥80% of long episodes exhaust ammo when no resupply."""
        cfg = LogisticsConfig(ammo_per_volley=0.01, enable_resupply=False)
        # Use curriculum_level=1 so episodes run full 150 steps (> 100 to exhaust)
        env = self._make_env(logistics_config=cfg, max_steps=150, curriculum_level=1)
        exhausted_count = 0
        N = 10
        for trial in range(N):
            env.reset(seed=trial * 100)
            for _ in range(150):
                action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                _, _, terminated, truncated, info = env.step(action)
                if info.get("blue_ammo", 1.0) <= 0.0:
                    exhausted_count += 1
                    break
                if terminated or truncated:
                    break
        self.assertGreaterEqual(
            exhausted_count / N, 0.8,
            f"Expected ≥80% ammo exhaustion, got {exhausted_count}/{N}",
        )

    # --- Fatigue ---

    def test_fatigue_accumulates_with_movement(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertGreater(info["blue_fatigue"], 0.0)

    def test_fatigue_reduces_speed(self) -> None:
        """A fatigued battalion covers less distance than a fresh one in one step."""
        from envs.sim.logistics import LogisticsConfig, LogisticsState

        cfg = LogisticsConfig(fatigue_speed_penalty=1.0)  # max penalty
        env = self._make_env(logistics_config=cfg)
        env.reset(seed=7)

        # Record starting position
        start_x = env.blue.x
        start_y = env.blue.y

        # Inject maximum fatigue directly
        env.blue_logistics.fatigue = 1.0

        # Move forward
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        dist_fatigued = math.sqrt(
            (env.blue.x - start_x) ** 2 + (env.blue.y - start_y) ** 2
        )

        # Reset and move without fatigue
        env.reset(seed=7)
        start_x2 = env.blue.x
        start_y2 = env.blue.y
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        dist_fresh = math.sqrt(
            (env.blue.x - start_x2) ** 2 + (env.blue.y - start_y2) ** 2
        )

        self.assertLess(dist_fatigued, dist_fresh)

    # --- Resupply ---

    def test_resupply_restores_ammo(self) -> None:
        """A battalion near its wagon gets ammo replenished."""
        cfg = LogisticsConfig(
            ammo_per_volley=0.1,
            resupply_radius=500.0,  # very large to guarantee proximity
            ammo_resupply_rate=0.05,
        )
        env = self._make_env(logistics_config=cfg)
        env.reset(seed=0)

        # Deplete ammo
        env.blue_logistics.ammo = 0.5

        # Stand still next to wagon (wagon spawned near Blue)
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        # Ammo should have increased (resupply rate > depletion from no firing)
        self.assertGreater(info["blue_ammo"], 0.5)

    def test_supply_wagons_exist_after_reset(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        self.assertIsNotNone(env.blue_wagon)
        self.assertIsNotNone(env.red_wagon)
        self.assertEqual(env.blue_wagon.team, 0)
        self.assertEqual(env.red_wagon.team, 1)

    def test_no_wagons_without_logistics(self) -> None:
        env = BattalionEnv(randomize_terrain=False)
        env.reset(seed=0)
        self.assertIsNone(env.blue_wagon)
        self.assertIsNone(env.red_wagon)

    def test_logistics_states_none_without_enable(self) -> None:
        env = BattalionEnv(randomize_terrain=False)
        env.reset(seed=0)
        self.assertIsNone(env.blue_logistics)
        self.assertIsNone(env.red_logistics)

    def test_resupply_disabled_config(self) -> None:
        """With enable_resupply=False, ammo should not recover even near wagon."""
        cfg = LogisticsConfig(
            ammo_per_volley=0.0,
            enable_resupply=False,
            resupply_radius=1000.0,
        )
        env = self._make_env(logistics_config=cfg)
        env.reset(seed=0)
        env.blue_logistics.ammo = 0.5
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(info["blue_ammo"], 0.5, places=4)

    # --- Custom LogisticsConfig ---

    def test_custom_logistics_config_respected(self) -> None:
        cfg = LogisticsConfig(ammo_per_volley=0.05)
        env = self._make_env(logistics_config=cfg)
        env.reset(seed=0)
        env.blue_logistics.ammo = 0.5
        env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        # Ammo should have gone down by ~0.05
        self.assertAlmostEqual(env.blue_logistics.ammo, 0.45, places=2)

    def test_multiple_resets_reinitialise_logistics(self) -> None:
        """Logistics state is freshly created on every reset()."""
        env = self._make_env()
        env.reset(seed=0)
        # Exhaust ammo
        env.blue_logistics.ammo = 0.0
        env.blue_logistics.fatigue = 0.9
        # Reset should restore fresh state
        env.reset(seed=0)
        self.assertAlmostEqual(env.blue_logistics.ammo, 1.0)
        self.assertAlmostEqual(env.blue_logistics.fatigue, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
