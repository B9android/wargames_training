# SPDX-License-Identifier: MIT
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.battalion import Battalion


def make_battalion(**kwargs) -> Battalion:
    defaults = dict(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
    defaults.update(kwargs)
    return Battalion(**defaults)


class TestMoveBoundaryClamping(unittest.TestCase):
    """Tests for speed/velocity clamping in Battalion.move()."""

    def test_normal_move_applies_correctly(self) -> None:
        """Velocity within max_speed moves the battalion by vx*dt, vy*dt."""
        b = make_battalion()
        b.move(10.0, 20.0, dt=0.1)
        self.assertAlmostEqual(b.x, 1.0)
        self.assertAlmostEqual(b.y, 2.0)

    def test_move_at_exactly_max_speed_is_not_clamped(self) -> None:
        """Velocity equal to max_speed is not scaled down."""
        b = make_battalion()
        # Pure x velocity at exactly max_speed
        b.move(b.max_speed, 0.0, dt=0.1)
        self.assertAlmostEqual(b.x, b.max_speed * 0.1)
        self.assertAlmostEqual(b.y, 0.0)

    def test_excess_speed_is_clamped_to_max_speed(self) -> None:
        """Velocity magnitude exceeding max_speed is clamped."""
        b = make_battalion()
        # Speed is greater than max_speed; should be clamped to max_speed
        b.move(2.0 * b.max_speed, 0.0, dt=0.1)
        # After clamping: vx = max_speed, so x += max_speed * 0.1
        self.assertAlmostEqual(b.x, b.max_speed * 0.1)
        self.assertAlmostEqual(b.y, 0.0)

    def test_excess_diagonal_speed_is_clamped(self) -> None:
        """Diagonal velocity exceeding max_speed is clamped while preserving direction."""
        b = make_battalion()
        # Both components equal; speed is much larger than max_speed
        vx, vy = 100.0, 100.0
        b.move(vx, vy, dt=0.1)
        # Clamped to max_speed along (1/sqrt(2), 1/sqrt(2)) direction
        expected = (b.max_speed / np.sqrt(2)) * 0.1
        self.assertAlmostEqual(b.x, expected)
        self.assertAlmostEqual(b.y, expected)

    def test_clamped_velocity_direction_is_preserved(self) -> None:
        """After clamping, the direction of motion is unchanged."""
        b = make_battalion()
        vx, vy = 60.0, 80.0  # speed = 100.0
        b.move(vx, vy, dt=1.0)
        # Ratio x/y should match original direction
        self.assertAlmostEqual(b.x / b.y, vx / vy)

    def test_zero_velocity_does_not_move(self) -> None:
        """Zero velocity leaves position unchanged."""
        b = make_battalion(x=5.0, y=5.0)
        b.move(0.0, 0.0, dt=0.1)
        self.assertAlmostEqual(b.x, 5.0)
        self.assertAlmostEqual(b.y, 5.0)

    def test_negative_velocity_moves_correctly(self) -> None:
        """Negative velocity moves in the negative direction."""
        b = make_battalion(x=10.0, y=10.0)
        b.move(-20.0, -30.0, dt=0.1)
        self.assertAlmostEqual(b.x, 10.0 - 2.0)
        self.assertAlmostEqual(b.y, 10.0 - 3.0)

    def test_large_negative_speed_is_clamped(self) -> None:
        """Large negative velocity is also clamped to max_speed magnitude."""
        b = make_battalion()
        b.move(-4.0 * b.max_speed, 0.0, dt=0.1)
        # Clamped to -max_speed; x += -max_speed * 0.1
        self.assertAlmostEqual(b.x, -b.max_speed * 0.1)
        self.assertAlmostEqual(b.y, 0.0)

    def test_custom_dt_scales_movement(self) -> None:
        """Position delta scales linearly with dt."""
        b1 = make_battalion()
        b2 = make_battalion()
        b1.move(10.0, 0.0, dt=0.5)
        b2.move(10.0, 0.0, dt=1.0)
        self.assertAlmostEqual(b2.x, 2.0 * b1.x)


class TestRotationLimits(unittest.TestCase):
    """Tests for rotation clamping and angle wrapping in Battalion.rotate()."""

    def test_normal_rotation_is_applied(self) -> None:
        """Rotation within max_turn_rate changes theta by the exact amount."""
        b = make_battalion(theta=0.0)
        delta = 0.5 * b.max_turn_rate
        b.rotate(delta)
        self.assertAlmostEqual(b.theta, delta)

    def test_positive_excess_rotation_is_clamped(self) -> None:
        """delta_theta larger than max_turn_rate is clamped to max_turn_rate."""
        b = make_battalion(theta=0.0)
        b.rotate(1.0)  # >> max_turn_rate (0.1)
        self.assertAlmostEqual(b.theta, b.max_turn_rate)

    def test_negative_excess_rotation_is_clamped(self) -> None:
        """delta_theta smaller than -max_turn_rate is clamped to -max_turn_rate."""
        b = make_battalion(theta=1.0)
        b.rotate(-1.0)  # << -max_turn_rate
        self.assertAlmostEqual(b.theta, 1.0 - b.max_turn_rate)

    def test_zero_rotation_leaves_theta_unchanged(self) -> None:
        """Zero delta_theta does not change theta."""
        b = make_battalion(theta=0.5)
        b.rotate(0.0)
        self.assertAlmostEqual(b.theta, 0.5)

    def test_rotation_at_exact_limit_is_applied_fully(self) -> None:
        """Rotation exactly at max_turn_rate is not reduced."""
        b = make_battalion(theta=0.0)
        b.rotate(b.max_turn_rate)
        self.assertAlmostEqual(b.theta, b.max_turn_rate)

    def test_rotation_at_negative_exact_limit(self) -> None:
        """Rotation exactly at -max_turn_rate is not reduced."""
        b = make_battalion(theta=1.0)
        b.rotate(-b.max_turn_rate)
        self.assertAlmostEqual(b.theta, 1.0 - b.max_turn_rate)

    def test_theta_wraps_above_two_pi(self) -> None:
        """theta wraps back into [0, 2π) when it would exceed 2π."""
        b = make_battalion(theta=0.0)
        # Choose a delta based on max_turn_rate so this test remains valid if
        # max_turn_rate changes. Ensure the delta is strictly within the limit
        # so that clamping does not affect the wraparound behavior being tested.
        delta = 0.5 * b.max_turn_rate
        # Start slightly below 2π so that applying `delta` crosses the boundary
        # by a known amount (delta - start_offset).
        start_offset = delta / 2.0
        b.theta = 2 * np.pi - start_offset
        b.rotate(delta)
        self.assertGreaterEqual(b.theta, 0.0)
        self.assertLess(b.theta, 2 * np.pi)
        expected_theta = (2 * np.pi - start_offset + delta) % (2 * np.pi)
        self.assertAlmostEqual(b.theta, expected_theta)

    def test_theta_wraps_below_zero(self) -> None:
        """theta wraps back into [0, 2π) when it would go below 0."""
        b = make_battalion(theta=0.0)
        # Choose a delta based on max_turn_rate so this test remains valid if
        # max_turn_rate changes, and ensure it is within the limit.
        delta = 0.5 * b.max_turn_rate
        # Start slightly above 0 so that applying `-delta` crosses the boundary
        # by a known amount (delta - start_offset).
        start_offset = delta / 2.0
        b.theta = start_offset
        b.rotate(-delta)
        self.assertGreaterEqual(b.theta, 0.0)
        self.assertLess(b.theta, 2 * np.pi)
        expected_theta = (start_offset - delta) % (2 * np.pi)
        self.assertAlmostEqual(b.theta, expected_theta)

    def test_multiple_rotations_accumulate_correctly(self) -> None:
        """Successive rotations accumulate and stay within [0, 2π)."""
        b = make_battalion(theta=0.0)
        steps = int(2 * np.pi / b.max_turn_rate) + 5
        for _ in range(steps):
            b.rotate(b.max_turn_rate)
        self.assertGreaterEqual(b.theta, 0.0)
        self.assertLess(b.theta, 2 * np.pi)

    def test_custom_max_turn_rate_is_respected(self) -> None:
        """A battalion with a different max_turn_rate clamps to that value."""
        b = make_battalion(theta=0.0)
        b.max_turn_rate = 0.3
        b.rotate(1.0)
        self.assertAlmostEqual(b.theta, 0.3)


if __name__ == "__main__":
    unittest.main()
