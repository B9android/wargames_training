import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.battalion import Battalion
from envs.sim import combat


def _make_attacker(x=0.0, y=0.0, theta=0.0, strength=1.0) -> Battalion:
    """Helper: attacker facing right (+x direction) at origin."""
    return Battalion(x=x, y=y, theta=theta, strength=strength, team=0)


def _make_target(x=0.0, y=0.0, theta=0.0, strength=1.0) -> Battalion:
    """Helper: a target battalion."""
    return Battalion(x=x, y=y, theta=theta, strength=strength, team=1)


class TestRangeFactor(unittest.TestCase):
    """Tests for the linear damage falloff (range_factor)."""

    def test_full_damage_at_zero_range(self) -> None:
        self.assertAlmostEqual(combat.range_factor(0.0, 200.0), 1.0)

    def test_no_damage_at_max_range(self) -> None:
        self.assertAlmostEqual(combat.range_factor(200.0, 200.0), 0.0)

    def test_half_damage_at_half_range(self) -> None:
        self.assertAlmostEqual(combat.range_factor(100.0, 200.0), 0.5)

    def test_clamped_to_zero_beyond_max_range(self) -> None:
        self.assertAlmostEqual(combat.range_factor(300.0, 200.0), 0.0)

    def test_clamped_to_one_at_negative_dist(self) -> None:
        # Negative distance is nonsensical but should not blow up.
        self.assertAlmostEqual(combat.range_factor(-10.0, 200.0), 1.0)

    def test_three_quarter_damage_at_quarter_range(self) -> None:
        self.assertAlmostEqual(combat.range_factor(50.0, 200.0), 0.75)


class TestInFireRange(unittest.TestCase):
    """Tests for range checks."""

    def test_target_within_range(self) -> None:
        attacker = _make_attacker()
        target = _make_target(x=100.0)
        self.assertTrue(combat.in_fire_range(attacker, target))

    def test_target_exactly_at_max_range(self) -> None:
        attacker = _make_attacker()
        target = _make_target(x=200.0)  # fire_range = 200.0
        self.assertTrue(combat.in_fire_range(attacker, target))

    def test_target_beyond_max_range(self) -> None:
        attacker = _make_attacker()
        target = _make_target(x=201.0)
        self.assertFalse(combat.in_fire_range(attacker, target))

    def test_range_computed_in_2d(self) -> None:
        attacker = _make_attacker()
        # 3-4-5 triangle: dist = 5, well within range
        target = _make_target(x=3.0, y=4.0)
        self.assertTrue(combat.in_fire_range(attacker, target))

    def test_target_far_diagonally_out_of_range(self) -> None:
        attacker = _make_attacker()
        # sqrt(150^2 + 150^2) ≈ 212 > 200
        target = _make_target(x=150.0, y=150.0)
        self.assertFalse(combat.in_fire_range(attacker, target))


class TestInFireArc(unittest.TestCase):
    """Tests for frontal arc checks (attacker facing +x, arc = ±45°)."""

    def test_target_directly_ahead_in_arc(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=100.0, y=0.0)
        self.assertTrue(combat.in_fire_arc(attacker, target))

    def test_target_at_arc_boundary_inside(self) -> None:
        # Exactly at ±44° should be inside the ±45° arc.
        attacker = _make_attacker(theta=0.0)
        angle = math.radians(44)
        target = _make_target(x=100.0 * math.cos(angle), y=100.0 * math.sin(angle))
        self.assertTrue(combat.in_fire_arc(attacker, target))

    def test_target_outside_arc(self) -> None:
        # Target is at 90° relative to facing (+y direction), outside ±45° arc.
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=0.0, y=100.0)
        self.assertFalse(combat.in_fire_arc(attacker, target))

    def test_target_directly_behind_outside_arc(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=-100.0, y=0.0)
        self.assertFalse(combat.in_fire_arc(attacker, target))

    def test_arc_rotates_with_facing(self) -> None:
        # Attacker faces +y (theta = π/2); target at (0, 100) should be in arc.
        attacker = _make_attacker(theta=math.pi / 2)
        target = _make_target(x=0.0, y=100.0)
        self.assertTrue(combat.in_fire_arc(attacker, target))

    def test_arc_rotates_with_facing_miss(self) -> None:
        # Attacker faces +y; target at (100, 0) is 90° off — outside arc.
        attacker = _make_attacker(theta=math.pi / 2)
        target = _make_target(x=100.0, y=0.0)
        self.assertFalse(combat.in_fire_arc(attacker, target))


class TestComputeDamage(unittest.TestCase):
    """Tests for the combined damage computation."""

    def test_no_damage_when_out_of_range(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=300.0)  # beyond fire_range=200
        self.assertAlmostEqual(combat.compute_damage(attacker, target, 1.0), 0.0)

    def test_no_damage_when_outside_arc(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=0.0, y=100.0)  # 90° off, outside ±45° arc
        self.assertAlmostEqual(combat.compute_damage(attacker, target, 1.0), 0.0)

    def test_full_intensity_close_range(self) -> None:
        # Target at dist ≈ 0 (very close): range_factor ≈ 1.0
        # damage = intensity * 1.0 * 0.05
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=1.0)  # 1 m away, facing +x
        dmg = combat.compute_damage(attacker, target, 1.0)
        expected = 1.0 * combat.range_factor(1.0, 200.0) * combat.BASE_DAMAGE_MULTIPLIER
        self.assertAlmostEqual(dmg, expected, places=6)

    def test_damage_falls_off_with_range(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target_near = _make_target(x=50.0)
        target_far = _make_target(x=150.0)
        dmg_near = combat.compute_damage(attacker, target_near, 1.0)
        dmg_far = combat.compute_damage(attacker, target_far, 1.0)
        self.assertGreater(dmg_near, dmg_far)

    def test_damage_scales_with_intensity(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=50.0)
        dmg_low = combat.compute_damage(attacker, target, 0.5)
        dmg_high = combat.compute_damage(attacker, target, 1.0)
        self.assertAlmostEqual(dmg_high, dmg_low * 2.0, places=6)

    def test_zero_intensity_gives_zero_damage(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=50.0)
        self.assertAlmostEqual(combat.compute_damage(attacker, target, 0.0), 0.0)


class TestResolveFire(unittest.TestCase):
    """Tests for resolve_fire which mutates target strength."""

    def test_reduces_target_strength(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=50.0, strength=1.0)
        dmg = combat.resolve_fire(attacker, target, 1.0)
        self.assertGreater(dmg, 0.0)
        self.assertAlmostEqual(target.strength, 1.0 - dmg, places=6)

    def test_strength_does_not_go_below_zero(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=1.0, strength=0.0)
        combat.resolve_fire(attacker, target, 1000.0)
        self.assertGreaterEqual(target.strength, 0.0)

    def test_no_damage_out_of_range_leaves_strength_unchanged(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=500.0, strength=0.8)
        combat.resolve_fire(attacker, target, 1.0)
        self.assertAlmostEqual(target.strength, 0.8)

    def test_returns_damage_dealt(self) -> None:
        attacker = _make_attacker(theta=0.0)
        target = _make_target(x=100.0, strength=1.0)
        dmg = combat.resolve_fire(attacker, target, 1.0)
        expected = combat.compute_damage(attacker, _make_target(x=100.0), 1.0)
        self.assertAlmostEqual(dmg, expected, places=6)


class TestMoralePenalty(unittest.TestCase):
    """Tests for morale penalties based on strength loss."""

    def test_no_loss_no_penalty(self) -> None:
        self.assertAlmostEqual(combat.morale_penalty(1.0, 1.0), 0.0)

    def test_small_loss_gives_small_penalty(self) -> None:
        penalty = combat.morale_penalty(1.0, 0.9)
        # loss = 0.1; penalty = 0.1 * 0.5 = 0.05
        self.assertAlmostEqual(penalty, 0.05)

    def test_large_loss_capped_at_one(self) -> None:
        # loss = 1.0; raw = 0.5; still below cap
        penalty = combat.morale_penalty(1.0, 0.0)
        self.assertAlmostEqual(penalty, 0.5)

    def test_extreme_loss_capped_at_one(self) -> None:
        # morale can never exceed 1.0 even for fabricated extreme values
        penalty = combat.morale_penalty(10.0, 0.0)
        self.assertAlmostEqual(penalty, 1.0)

    def test_negative_loss_clamped_to_zero(self) -> None:
        # strength_after > strength_before means healing — no morale penalty
        self.assertAlmostEqual(combat.morale_penalty(0.5, 1.0), 0.0)

    def test_penalty_proportional_to_loss(self) -> None:
        p1 = combat.morale_penalty(1.0, 0.8)  # loss 0.2
        p2 = combat.morale_penalty(1.0, 0.6)  # loss 0.4
        self.assertAlmostEqual(p2, p1 * 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
