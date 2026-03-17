"""Unit tests for envs/sim/combat.py."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.battalion import Battalion
from envs.sim.combat import (
    BASE_DAMAGE_MULTIPLIER,
    BASE_FIRE_DAMAGE,
    MORALE_CASUALTY_WEIGHT,
    MORALE_RECOVERY_RATE,
    MORALE_ROUT_THRESHOLD,
    CombatState,
    apply_casualties,
    compute_fire_damage,
    in_fire_arc,
    in_fire_range,
    morale_check,
    range_factor,
    resolve_volley,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(dist: float = 100.0, theta: float = 0.0):
    """Return (shooter, target) facing each other along the x-axis."""
    shooter = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
    target = Battalion(x=dist, y=0.0, theta=np.pi, strength=1.0, team=1)
    shooter.theta = theta
    return shooter, target


def _make_attacker(x=0.0, y=0.0, theta=0.0, strength=1.0) -> Battalion:
    """Helper: attacker facing right (+x direction) at origin."""
    return Battalion(x=x, y=y, theta=theta, strength=strength, team=0)


def _make_target(x=0.0, y=0.0, theta=0.0, strength=1.0) -> Battalion:
    """Helper: a target battalion."""
    return Battalion(x=x, y=y, theta=theta, strength=strength, team=1)


def _create_seeded_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Standalone helpers: range_factor
# ---------------------------------------------------------------------------


class TestRangeFactor(unittest.TestCase):
    """Tests for the linear damage falloff (range_factor)."""

    def test_full_damage_at_zero_range(self) -> None:
        self.assertAlmostEqual(range_factor(0.0, 200.0), 1.0)

    def test_no_damage_at_max_range(self) -> None:
        self.assertAlmostEqual(range_factor(200.0, 200.0), 0.0)

    def test_half_damage_at_half_range(self) -> None:
        self.assertAlmostEqual(range_factor(100.0, 200.0), 0.5)

    def test_clamped_to_zero_beyond_max_range(self) -> None:
        self.assertAlmostEqual(range_factor(300.0, 200.0), 0.0)

    def test_clamped_to_one_at_negative_dist(self) -> None:
        # Negative distance is nonsensical but should not blow up.
        self.assertAlmostEqual(range_factor(-10.0, 200.0), 1.0)

    def test_three_quarter_damage_at_quarter_range(self) -> None:
        self.assertAlmostEqual(range_factor(50.0, 200.0), 0.75)

    def test_raises_for_zero_fire_range(self) -> None:
        with self.assertRaises(ValueError):
            range_factor(0.0, 0.0)

    def test_raises_for_negative_fire_range(self) -> None:
        with self.assertRaises(ValueError):
            range_factor(50.0, -100.0)


# ---------------------------------------------------------------------------
# Standalone helpers: in_fire_range
# ---------------------------------------------------------------------------


class TestInFireRange(unittest.TestCase):
    """Tests for range checks."""

    def test_target_within_range(self) -> None:
        attacker = _make_attacker()
        target = _make_target(x=100.0)
        self.assertTrue(in_fire_range(attacker, target))

    def test_target_exactly_at_max_range(self) -> None:
        attacker = _make_attacker()
        target = _make_target(x=200.0)  # fire_range = 200.0
        self.assertTrue(in_fire_range(attacker, target))

    def test_target_beyond_max_range(self) -> None:
        attacker = _make_attacker()
        target = _make_target(x=201.0)
        self.assertFalse(in_fire_range(attacker, target))

    def test_range_computed_in_2d(self) -> None:
        attacker = _make_attacker()
        # 3-4-5 triangle: dist = 5, well within range
        target = _make_target(x=3.0, y=4.0)
        self.assertTrue(in_fire_range(attacker, target))

    def test_target_far_diagonally_out_of_range(self) -> None:
        attacker = _make_attacker()
        # sqrt(150^2 + 150^2) ≈ 212 > 200
        target = _make_target(x=150.0, y=150.0)
        self.assertFalse(in_fire_range(attacker, target))


# ---------------------------------------------------------------------------
# Standalone helpers: in_fire_arc
# ---------------------------------------------------------------------------


class TestInFireArc(unittest.TestCase):
    """Tests for frontal arc checks (attacker facing +x, arc = ±45°)."""

    def test_target_directly_ahead_in_arc(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=100.0, y=0.0)
        self.assertTrue(in_fire_arc(attacker, target))

    def test_target_at_arc_boundary_inside(self) -> None:
        # Exactly at ±44° should be inside the ±45° arc.
        attacker = _make_attacker(theta=0.0)
        angle = math.radians(44)
        target = _make_target(x=100.0 * math.cos(angle), y=100.0 * math.sin(angle))
        self.assertTrue(in_fire_arc(attacker, target))

    def test_target_outside_arc(self) -> None:
        # Target is at 90° relative to facing (+y direction), outside ±45° arc.
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=0.0, y=100.0)
        self.assertFalse(in_fire_arc(attacker, target))

    def test_target_directly_behind_outside_arc(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=-100.0, y=0.0)
        self.assertFalse(in_fire_arc(attacker, target))

    def test_arc_rotates_with_facing(self) -> None:
        # Attacker faces +y (theta = π/2); target at (0, 100) should be in arc.
        attacker = _make_attacker(theta=math.pi / 2)
        target = _make_target(x=0.0, y=100.0)
        self.assertTrue(in_fire_arc(attacker, target))

    def test_arc_rotates_with_facing_miss(self) -> None:
        # Attacker faces +y; target at (100, 0) is 90° off — outside arc.
        attacker = _make_attacker(theta=math.pi / 2)
        target = _make_target(x=100.0, y=0.0)
        self.assertFalse(in_fire_arc(attacker, target))


# ---------------------------------------------------------------------------
# CombatState
# ---------------------------------------------------------------------------


class TestCombatState(unittest.TestCase):
    def test_defaults(self) -> None:
        s = CombatState()
        self.assertAlmostEqual(s.morale, 1.0)
        self.assertAlmostEqual(s.accumulated_damage, 0.0)
        self.assertAlmostEqual(s.total_casualties, 0.0)
        self.assertFalse(s.is_routing)

    def test_reset_step_accumulators_clears_accumulated_damage(self) -> None:
        s = CombatState(accumulated_damage=0.3, total_casualties=0.3)
        s.reset_step_accumulators()
        self.assertAlmostEqual(s.accumulated_damage, 0.0)
        # total_casualties must NOT be reset
        self.assertAlmostEqual(s.total_casualties, 0.3)


# ---------------------------------------------------------------------------
# compute_fire_damage
# ---------------------------------------------------------------------------


class TestComputeFireDamage(unittest.TestCase):
    def test_zero_when_out_of_range(self) -> None:
        shooter, target = _make_pair(dist=300.0)  # fire_range default is 200
        self.assertEqual(compute_fire_damage(shooter, target, 1.0), 0.0)

    def test_zero_when_outside_fire_arc(self) -> None:
        shooter = Battalion(x=0.0, y=0.0, theta=np.pi, strength=1.0, team=0)
        target = Battalion(x=100.0, y=0.0, theta=0.0, strength=1.0, team=1)
        # Shooter faces away from target — target is outside the frontal arc
        self.assertEqual(compute_fire_damage(shooter, target, 1.0), 0.0)

    def test_positive_at_close_frontal_range(self) -> None:
        shooter, target = _make_pair(dist=50.0)
        dmg = compute_fire_damage(shooter, target, 1.0)
        self.assertGreater(dmg, 0.0)

    def test_damage_decreases_with_range(self) -> None:
        shooter1, target1 = _make_pair(dist=50.0)
        shooter2, target2 = _make_pair(dist=150.0)
        dmg_close = compute_fire_damage(shooter1, target1, 1.0)
        dmg_far = compute_fire_damage(shooter2, target2, 1.0)
        self.assertGreater(dmg_close, dmg_far)

    def test_zero_intensity_yields_zero_damage(self) -> None:
        shooter, target = _make_pair(dist=50.0)
        self.assertAlmostEqual(compute_fire_damage(shooter, target, 0.0), 0.0)

    def test_flanking_damage_greater_than_frontal(self) -> None:
        dist = 50.0
        # Frontal attack: shooter at origin, target at (dist, 0) facing west
        shooter_front = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target_front = Battalion(x=dist, y=0.0, theta=np.pi, strength=1.0, team=1)

        # Flanking attack: shooter below target, target still facing west
        shooter_flank = Battalion(x=dist, y=-dist, theta=np.pi / 2, strength=1.0, team=0)
        target_flank = Battalion(x=dist, y=0.0, theta=np.pi, strength=1.0, team=1)

        dmg_front = compute_fire_damage(shooter_front, target_front, 1.0)
        dmg_flank = compute_fire_damage(shooter_flank, target_flank, 1.0)
        self.assertGreater(dmg_flank, dmg_front)

    def test_rear_damage_greater_than_flanking(self) -> None:
        dist = 50.0
        # Rear attack: target faces east (θ=0); shooter is to the WEST so bullets
        # arrive from behind (the target's back faces west).
        shooter_rear = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target_rear = Battalion(x=dist, y=0.0, theta=0.0, strength=1.0, team=1)

        # Flanking: shooter directly south of target at the same dist; target faces east
        shooter_flank = Battalion(x=dist, y=-dist, theta=np.pi / 2, strength=1.0, team=0)
        target_flank = Battalion(x=dist, y=0.0, theta=0.0, strength=1.0, team=1)

        dmg_rear = compute_fire_damage(shooter_rear, target_rear, 1.0)
        dmg_flank = compute_fire_damage(shooter_flank, target_flank, 1.0)
        self.assertGreater(dmg_rear, dmg_flank)

    def test_weak_shooter_deals_less_damage(self) -> None:
        shooter_full, target1 = _make_pair(dist=50.0)
        shooter_weak, target2 = _make_pair(dist=50.0)
        shooter_weak.strength = 0.3
        dmg_full = compute_fire_damage(shooter_full, target1, 1.0)
        dmg_weak = compute_fire_damage(shooter_weak, target2, 1.0)
        self.assertGreater(dmg_full, dmg_weak)

    def test_point_blank_frontal_formula(self) -> None:
        """At dist=0 with intensity=1, damage == BASE_FIRE_DAMAGE."""
        shooter = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        # Place target at x=1 (almost point-blank) facing shooter
        target = Battalion(x=1.0, y=0.0, theta=np.pi, strength=1.0, team=1)
        dist = 1.0
        range_factor = 1.0 - dist / shooter.fire_range
        expected = BASE_FIRE_DAMAGE * 1.0 * range_factor * 1.0 * 1.0
        self.assertAlmostEqual(compute_fire_damage(shooter, target, 1.0), expected, places=10)

    def test_negative_intensity_clamped_to_zero(self) -> None:
        """Negative intensity must not produce negative (healing) damage."""
        shooter, target = _make_pair(dist=50.0)
        self.assertAlmostEqual(compute_fire_damage(shooter, target, -1.0), 0.0)

    def test_intensity_above_one_clamped_to_one(self) -> None:
        """intensity > 1 should produce the same damage as intensity == 1."""
        shooter, target = _make_pair(dist=50.0)
        dmg_clamped = compute_fire_damage(shooter, target, 1.0)
        dmg_over = compute_fire_damage(shooter, target, 10.0)
        self.assertAlmostEqual(dmg_over, dmg_clamped)


# ---------------------------------------------------------------------------
# apply_casualties
# ---------------------------------------------------------------------------


class TestApplyCasualties(unittest.TestCase):
    def test_reduces_target_strength(self) -> None:
        _, target = _make_pair()
        state = CombatState()
        apply_casualties(target, state, 0.1)
        self.assertAlmostEqual(target.strength, 0.9)

    def test_strength_never_below_zero(self) -> None:
        _, target = _make_pair()
        state = CombatState()
        apply_casualties(target, state, 5.0)  # way more than strength=1.0
        self.assertAlmostEqual(target.strength, 0.0)

    def test_updates_accumulated_and_total_casualties(self) -> None:
        _, target = _make_pair()
        state = CombatState()
        apply_casualties(target, state, 0.2)
        self.assertAlmostEqual(state.accumulated_damage, 0.2)
        self.assertAlmostEqual(state.total_casualties, 0.2)

    def test_total_casualties_accumulates_across_calls(self) -> None:
        _, target = _make_pair()
        state = CombatState()
        apply_casualties(target, state, 0.1)
        state.reset_step_accumulators()
        apply_casualties(target, state, 0.2)
        self.assertAlmostEqual(state.total_casualties, 0.3)

    def test_returns_actual_damage_applied(self) -> None:
        _, target = _make_pair()
        state = CombatState()
        actual = apply_casualties(target, state, 0.15)
        self.assertAlmostEqual(actual, 0.15)

    def test_returns_clamped_damage_when_overkill(self) -> None:
        _, target = _make_pair()
        target.strength = 0.05
        state = CombatState()
        actual = apply_casualties(target, state, 0.5)
        self.assertAlmostEqual(actual, 0.05)

    def test_negative_damage_clamped_to_zero(self) -> None:
        """Negative damage must not increase strength."""
        _, target = _make_pair()
        state = CombatState()
        actual = apply_casualties(target, state, -0.5)
        self.assertAlmostEqual(actual, 0.0)
        self.assertAlmostEqual(target.strength, 1.0)
        self.assertAlmostEqual(state.accumulated_damage, 0.0)


# ---------------------------------------------------------------------------
# morale_check
# ---------------------------------------------------------------------------


class TestMoraleCheck(unittest.TestCase):
    def test_no_damage_triggers_recovery(self) -> None:
        state = CombatState(morale=0.8, accumulated_damage=0.0)
        morale_check(state, rng=_create_seeded_rng())
        self.assertAlmostEqual(state.morale, 0.8 + MORALE_RECOVERY_RATE)

    def test_morale_decreases_with_damage(self) -> None:
        state = CombatState(morale=1.0, accumulated_damage=0.2)
        morale_check(state, rng=_create_seeded_rng())
        expected_morale = 1.0 - 0.2 * MORALE_CASUALTY_WEIGHT
        self.assertAlmostEqual(state.morale, expected_morale)

    def test_morale_clamped_to_zero(self) -> None:
        state = CombatState(morale=0.1, accumulated_damage=0.5)
        morale_check(state, rng=_create_seeded_rng(0))
        self.assertGreaterEqual(state.morale, 0.0)

    def test_morale_clamped_to_one_on_recovery(self) -> None:
        state = CombatState(morale=0.999, accumulated_damage=0.0)
        morale_check(state, rng=_create_seeded_rng())
        self.assertLessEqual(state.morale, 1.0)

    def test_unit_routes_when_morale_zero(self) -> None:
        """With morale at 0, rout probability is 1.0 — must always route."""
        state = CombatState(morale=0.0, accumulated_damage=0.0)
        result = morale_check(state, rng=_create_seeded_rng())
        self.assertTrue(result)
        self.assertTrue(state.is_routing)

    def test_unit_does_not_route_when_morale_above_threshold(self) -> None:
        state = CombatState(morale=0.9, accumulated_damage=0.0)
        result = morale_check(state, rng=_create_seeded_rng())
        self.assertFalse(result)
        self.assertFalse(state.is_routing)

    def test_routing_unit_recovers_morale_when_not_under_fire(self) -> None:
        """Routing units must still recover morale so they can reach the rally gate."""
        state = CombatState(morale=0.4, accumulated_damage=0.0, is_routing=True)
        # Use a seeded RNG with very low probability to avoid accidental rally
        morale_check(state, rng=_create_seeded_rng(1))
        self.assertGreater(state.morale, 0.4)

    def test_routing_unit_can_rally(self) -> None:
        """Set morale > 2× threshold and use a seeded RNG that produces < 0.05."""
        # morale well above 2× threshold; accumulated_damage=0 so no further hit
        state = CombatState(
            morale=MORALE_ROUT_THRESHOLD * 3,
            accumulated_damage=0.0,
            is_routing=True,
        )
        # Drive the RNG so the first draw is < 0.05 → rally
        rng = np.random.default_rng(0)
        rallied = False
        for _ in range(200):
            state.is_routing = True
            state.morale = MORALE_ROUT_THRESHOLD * 3
            state.accumulated_damage = 0.0
            result = morale_check(state, rng=rng)
            if not result:
                rallied = True
                break
        self.assertTrue(rallied, "Unit never rallied in 200 attempts")

    def test_returns_routing_flag(self) -> None:
        state = CombatState(morale=0.0, accumulated_damage=0.0)
        result = morale_check(state, rng=_create_seeded_rng())
        self.assertEqual(result, state.is_routing)


# ---------------------------------------------------------------------------
# resolve_volley
# ---------------------------------------------------------------------------


class TestResolveVolley(unittest.TestCase):
    def test_out_of_range_returns_zero_damage(self) -> None:
        shooter, target = _make_pair(dist=300.0)
        s_state = CombatState()
        t_state = CombatState()
        result = resolve_volley(shooter, s_state, target, t_state, 1.0, rng=_create_seeded_rng())
        self.assertAlmostEqual(result["damage_dealt"], 0.0)
        self.assertAlmostEqual(result["target_strength"], 1.0)

    def test_result_keys_present(self) -> None:
        shooter, target = _make_pair(dist=50.0)
        s_state = CombatState()
        t_state = CombatState()
        result = resolve_volley(shooter, s_state, target, t_state, 1.0, rng=_create_seeded_rng())
        for key in ("damage_dealt", "target_routing", "target_strength", "target_morale"):
            self.assertIn(key, result)

    def test_damage_dealt_matches_target_strength_reduction(self) -> None:
        shooter, target = _make_pair(dist=50.0)
        initial_strength = target.strength
        s_state = CombatState()
        t_state = CombatState()
        result = resolve_volley(shooter, s_state, target, t_state, 1.0, rng=_create_seeded_rng())
        self.assertAlmostEqual(
            result["damage_dealt"],
            initial_strength - result["target_strength"],
        )

    def test_repeated_volleys_exhaust_target(self) -> None:
        shooter, target = _make_pair(dist=10.0)
        s_state = CombatState()
        t_state = CombatState()
        rng = _create_seeded_rng()
        for _ in range(200):
            t_state.reset_step_accumulators()
            resolve_volley(shooter, s_state, target, t_state, 1.0, rng=rng)
            if target.strength <= 0.0:
                break
        self.assertAlmostEqual(target.strength, 0.0)

    def test_shooter_state_shots_fired_increments(self) -> None:
        """shots_fired on the shooter's CombatState increments each volley."""
        shooter, target = _make_pair(dist=50.0)
        s_state = CombatState()
        t_state = CombatState()
        resolve_volley(shooter, s_state, target, t_state, 1.0, rng=_create_seeded_rng())
        self.assertEqual(s_state.shots_fired, 1)
        resolve_volley(shooter, s_state, target, t_state, 1.0, rng=_create_seeded_rng())
        self.assertEqual(s_state.shots_fired, 2)


if __name__ == "__main__":
    unittest.main()
