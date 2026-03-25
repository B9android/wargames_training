# SPDX-License-Identifier: MIT
# tests/test_coordination_metrics.py
"""Unit tests for envs/metrics/coordination.py.

Tests verify the three coordination metrics with carefully constructed
Battalion fixtures whose expected outputs can be computed by hand.
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

from envs.sim.battalion import Battalion
from envs.metrics.coordination import (
    flanking_ratio,
    fire_concentration,
    mutual_support_score,
    compute_all,
)
from envs.multi_battalion_env import MultiBattalionEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_battalion(
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    strength: float = 1.0,
    team: int = 0,
    fire_range: float = 200.0,
    fire_arc: float = math.pi / 4,
) -> Battalion:
    """Create a Battalion with controlled parameters."""
    b = Battalion(x=x, y=y, theta=theta, strength=strength, team=team)
    b.fire_range = fire_range
    b.fire_arc = fire_arc
    return b


# ---------------------------------------------------------------------------
# flanking_ratio
# ---------------------------------------------------------------------------


class TestFlankingRatio(unittest.TestCase):
    """Tests for flanking_ratio()."""

    def test_empty_blue_returns_zero(self) -> None:
        red = [_make_battalion(x=100.0, team=1)]
        self.assertAlmostEqual(flanking_ratio([], red), 0.0)

    def test_empty_red_returns_zero(self) -> None:
        blue = [_make_battalion(x=0.0, team=0)]
        self.assertAlmostEqual(flanking_ratio(blue, []), 0.0)

    def test_no_in_range_pairs_returns_zero(self) -> None:
        """Blue units more than fire_range away → 0.0."""
        blue = [_make_battalion(x=0.0, team=0, fire_range=100.0)]
        red = [_make_battalion(x=500.0, theta=math.pi, team=1, fire_range=100.0)]
        self.assertAlmostEqual(flanking_ratio(blue, red), 0.0)

    def test_pure_frontal_attack_returns_zero(self) -> None:
        """Blue directly in front of Red → 0.0 flanking."""
        # Red faces east (theta=0), Blue is directly to the east (in front)
        # Blue position relative to Red: (100, 0) → angle = 0 rad from Red's perspective
        # Red fire_arc = pi/4, so angle_diff = 0 < pi/4 → frontal → not flanking
        blue = [_make_battalion(x=100.0, y=0.0, team=0)]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        self.assertAlmostEqual(flanking_ratio(blue, red), 0.0)

    def test_pure_rear_attack_returns_one(self) -> None:
        """Blue directly behind Red → 1.0 flanking."""
        # Red faces east (theta=0), Blue is to the west (behind)
        # Angle from Red to Blue = pi rad → angle_diff ≈ pi > pi/4 → flanking
        blue = [_make_battalion(x=-100.0, y=0.0, team=0)]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        result = flanking_ratio(blue, red)
        self.assertAlmostEqual(result, 1.0)

    def test_pure_flank_attack_returns_one(self) -> None:
        """Blue directly to the side of Red → 1.0 flanking."""
        # Red faces east (theta=0), Blue is directly north (90°)
        # angle_diff = pi/2 > pi/4 → flanking
        blue = [_make_battalion(x=0.0, y=100.0, team=0)]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        result = flanking_ratio(blue, red)
        self.assertAlmostEqual(result, 1.0)

    def test_mixed_frontal_and_flanking(self) -> None:
        """One frontal + one flanking → ratio = 0.5."""
        # Red faces east; blue_0 is frontal (east), blue_1 is flanking (west)
        blue = [
            _make_battalion(x=100.0, y=0.0, team=0),   # frontal
            _make_battalion(x=-100.0, y=0.0, team=0),  # rear → flanking
        ]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        result = flanking_ratio(blue, red)
        self.assertAlmostEqual(result, 0.5)

    def test_result_in_range(self) -> None:
        """Result is always in [0, 1]."""
        blue = [_make_battalion(x=x, team=0) for x in [50, 100, 150]]
        red = [_make_battalion(x=200.0, team=1)]
        r = flanking_ratio(blue, red)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

    def test_flanking_threshold(self) -> None:
        """flanking_ratio > 0.3 is considered meaningful flanking."""
        # Two blue units attacking a single red from the flanks/rear
        blue = [
            _make_battalion(x=-100.0, y=0.0, team=0),  # rear
            _make_battalion(x=0.0, y=100.0, team=0),   # flank
            _make_battalion(x=80.0, y=0.0, team=0),    # frontal
        ]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        result = flanking_ratio(blue, red)
        self.assertGreater(result, 0.3)


# ---------------------------------------------------------------------------
# fire_concentration
# ---------------------------------------------------------------------------


class TestFireConcentration(unittest.TestCase):
    """Tests for fire_concentration()."""

    def test_empty_blue_returns_zero(self) -> None:
        red = [_make_battalion(x=100.0, team=1)]
        self.assertAlmostEqual(fire_concentration([], red), 0.0)

    def test_empty_red_returns_zero(self) -> None:
        blue = [_make_battalion(x=0.0, team=0)]
        self.assertAlmostEqual(fire_concentration(blue, []), 0.0)

    def test_no_blue_can_fire_returns_zero(self) -> None:
        """Blue units all out of range → 0.0."""
        blue = [_make_battalion(x=0.0, team=0, fire_range=50.0)]
        red = [_make_battalion(x=500.0, team=1)]
        self.assertAlmostEqual(fire_concentration(blue, red), 0.0)

    def test_single_blue_single_red_returns_one(self) -> None:
        """One Blue in range → 100 % concentration."""
        blue = [_make_battalion(x=100.0, y=0.0, theta=math.pi, team=0)]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        result = fire_concentration(blue, red)
        self.assertAlmostEqual(result, 1.0)

    def test_all_blues_target_same_red(self) -> None:
        """All Blue units aim at the same (nearest) Red target → 1.0."""
        # Two blue units on the east side, one red in the middle (closer),
        # another red far away.  Both blues should pick the closer red.
        blue = [
            _make_battalion(x=100.0, y=0.0, theta=math.pi, team=0),
            _make_battalion(x=90.0, y=5.0, theta=math.pi, team=0),
        ]
        red_close = _make_battalion(x=0.0, y=0.0, theta=0.0, team=1)
        red_far = _make_battalion(x=-500.0, y=0.0, team=1)  # out of range
        result = fire_concentration(blue, [red_close, red_far])
        self.assertAlmostEqual(result, 1.0)

    def test_distributed_fire_returns_lower_concentration(self) -> None:
        """Blues split fire across two red targets → concentration < 1.0."""
        # blue_0 faces west and is close to red_0 which is to the west
        # blue_1 faces east and is close to red_1 which is to the east
        blue = [
            _make_battalion(x=0.0, y=0.0, theta=math.pi, team=0),  # faces west
            _make_battalion(x=200.0, y=0.0, theta=0.0, team=0),    # faces east
        ]
        red = [
            _make_battalion(x=-100.0, y=0.0, theta=0.0, team=1),   # to the west
            _make_battalion(x=300.0, y=0.0, theta=math.pi, team=1),# to the east
        ]
        result = fire_concentration(blue, red)
        self.assertLess(result, 1.0)

    def test_result_in_range(self) -> None:
        """Result is always in [0, 1]."""
        blue = [_make_battalion(x=x, y=0.0, theta=math.pi, team=0) for x in [50, 100]]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        r = fire_concentration(blue, red)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)


# ---------------------------------------------------------------------------
# mutual_support_score
# ---------------------------------------------------------------------------


class TestMutualSupportScore(unittest.TestCase):
    """Tests for mutual_support_score()."""

    def test_empty_blue_returns_zero(self) -> None:
        self.assertAlmostEqual(mutual_support_score([]), 0.0)

    def test_single_blue_returns_zero(self) -> None:
        blue = [_make_battalion(x=0.0, team=0)]
        self.assertAlmostEqual(mutual_support_score(blue), 0.0)

    def test_two_units_within_radius_returns_one(self) -> None:
        """Two Blues within support_radius → score = 1.0."""
        blue = [
            _make_battalion(x=0.0, y=0.0, team=0),
            _make_battalion(x=100.0, y=0.0, team=0),
        ]
        result = mutual_support_score(blue, support_radius=300.0)
        self.assertAlmostEqual(result, 1.0)

    def test_two_units_outside_radius_returns_zero(self) -> None:
        """Two Blues beyond support_radius → score = 0.0."""
        blue = [
            _make_battalion(x=0.0, y=0.0, team=0),
            _make_battalion(x=500.0, y=0.0, team=0),
        ]
        result = mutual_support_score(blue, support_radius=100.0)
        self.assertAlmostEqual(result, 0.0)

    def test_three_units_all_within_radius(self) -> None:
        """Three Blues all within mutual range → score = 1.0."""
        blue = [
            _make_battalion(x=0.0, y=0.0, team=0),
            _make_battalion(x=100.0, y=0.0, team=0),
            _make_battalion(x=50.0, y=50.0, team=0),
        ]
        result = mutual_support_score(blue, support_radius=300.0)
        self.assertAlmostEqual(result, 1.0)

    def test_three_units_one_isolated(self) -> None:
        """Two Blues close together, one far away → score between 0 and 1."""
        blue = [
            _make_battalion(x=0.0, y=0.0, team=0),
            _make_battalion(x=100.0, y=0.0, team=0),
            _make_battalion(x=900.0, y=0.0, team=0),  # isolated
        ]
        result = mutual_support_score(blue, support_radius=300.0)
        # unit 0: 1 nearby (unit 1) → 0.5
        # unit 1: 1 nearby (unit 0) → 0.5
        # unit 2: 0 nearby → 0.0
        # mean = (0.5 + 0.5 + 0.0) / 3 ≈ 0.333
        self.assertAlmostEqual(result, 1.0 / 3.0, places=5)

    def test_result_in_range(self) -> None:
        """Result is always in [0, 1]."""
        blue = [_make_battalion(x=x, team=0) for x in [0, 100, 800]]
        r = mutual_support_score(blue, support_radius=200.0)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

    def test_custom_support_radius(self) -> None:
        """Larger radius includes more units in support."""
        blue = [
            _make_battalion(x=0.0, y=0.0, team=0),
            _make_battalion(x=400.0, y=0.0, team=0),
        ]
        # Small radius → 0.0, large radius → 1.0
        self.assertAlmostEqual(mutual_support_score(blue, support_radius=100.0), 0.0)
        self.assertAlmostEqual(mutual_support_score(blue, support_radius=500.0), 1.0)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


class TestComputeAll(unittest.TestCase):
    """Tests for compute_all()."""

    def test_returns_all_three_keys(self) -> None:
        blue = [_make_battalion(x=0.0, y=100.0, team=0)]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        result = compute_all(blue, red)
        self.assertIn("coordination/flanking_ratio", result)
        self.assertIn("coordination/fire_concentration", result)
        self.assertIn("coordination/mutual_support_score", result)

    def test_all_values_in_unit_interval(self) -> None:
        blue = [
            _make_battalion(x=0.0, y=100.0, team=0),
            _make_battalion(x=0.0, y=200.0, team=0),
        ]
        red = [_make_battalion(x=0.0, y=0.0, theta=0.0, team=1)]
        result = compute_all(blue, red)
        for key, value in result.items():
            self.assertGreaterEqual(value, 0.0, msg=f"{key} < 0")
            self.assertLessEqual(value, 1.0, msg=f"{key} > 1")

    def test_empty_inputs(self) -> None:
        result = compute_all([], [])
        for v in result.values():
            self.assertAlmostEqual(v, 0.0)

    def test_custom_support_radius_propagated(self) -> None:
        """support_radius kwarg must be forwarded to mutual_support_score."""
        blue = [
            _make_battalion(x=0.0, y=0.0, team=0),
            _make_battalion(x=50.0, y=0.0, team=0),
        ]
        red: list[Battalion] = []
        # With large radius both units are within range → score = 1.0
        result_large = compute_all(blue, red, support_radius=500.0)
        self.assertAlmostEqual(
            result_large["coordination/mutual_support_score"], 1.0
        )
        # With tiny radius → score = 0.0
        result_small = compute_all(blue, red, support_radius=10.0)
        self.assertAlmostEqual(
            result_small["coordination/mutual_support_score"], 0.0
        )


# ---------------------------------------------------------------------------
# MultiBattalionEnv.get_coordination_metrics integration
# ---------------------------------------------------------------------------


class TestMultiBattalionEnvCoordinationMetrics(unittest.TestCase):
    """Integration tests for MultiBattalionEnv.get_coordination_metrics()."""

    def setUp(self) -> None:
        self.env = MultiBattalionEnv(n_blue=2, n_red=2)
        self.env.reset(seed=42)

    def tearDown(self) -> None:
        self.env.close()

    def test_method_exists(self) -> None:
        self.assertTrue(hasattr(self.env, "get_coordination_metrics"))

    def test_returns_expected_keys(self) -> None:
        metrics = self.env.get_coordination_metrics()
        self.assertIn("coordination/flanking_ratio", metrics)
        self.assertIn("coordination/fire_concentration", metrics)
        self.assertIn("coordination/mutual_support_score", metrics)

    def test_values_are_floats_in_unit_interval(self) -> None:
        metrics = self.env.get_coordination_metrics()
        for key, value in metrics.items():
            self.assertIsInstance(value, float, msg=f"{key} is not float")
            self.assertGreaterEqual(value, 0.0, msg=f"{key} < 0")
            self.assertLessEqual(value, 1.0, msg=f"{key} > 1")

    def test_custom_support_radius(self) -> None:
        """Passing support_radius kwarg does not raise."""
        metrics = self.env.get_coordination_metrics(support_radius=500.0)
        self.assertEqual(len(metrics), 3)

    def test_metrics_callable_after_step(self) -> None:
        """Metrics can be computed after env.step() without errors."""
        actions = {a: self.env.action_space(a).sample() for a in self.env.agents}
        self.env.step(actions)
        metrics = self.env.get_coordination_metrics()
        self.assertEqual(len(metrics), 3)

    def test_1v1_env_metrics(self) -> None:
        """1v1 environment returns valid metrics."""
        env = MultiBattalionEnv(n_blue=1, n_red=1)
        env.reset(seed=7)
        metrics = env.get_coordination_metrics()
        # mutual_support_score is 0 with only one blue unit
        self.assertAlmostEqual(metrics["coordination/mutual_support_score"], 0.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
