# SPDX-License-Identifier: MIT
# tests/test_elo.py
"""Tests for training/elo.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.elo import (
    DEFAULT_RATING,
    BASELINE_RATINGS,
    EloRegistry,
    expected_score,
    k_factor,
)


# ---------------------------------------------------------------------------
# expected_score
# ---------------------------------------------------------------------------


class TestExpectedScore(unittest.TestCase):
    """Tests for the expected_score() pure function."""

    def test_equal_ratings_returns_half(self) -> None:
        """Equally-rated agents each expect 0.5."""
        self.assertAlmostEqual(expected_score(1000.0, 1000.0), 0.5)

    def test_higher_rated_agent_expects_more(self) -> None:
        """Agent with higher rating expects a score > 0.5."""
        self.assertGreater(expected_score(1200.0, 1000.0), 0.5)

    def test_lower_rated_agent_expects_less(self) -> None:
        """Agent with lower rating expects a score < 0.5."""
        self.assertLess(expected_score(800.0, 1000.0), 0.5)

    def test_symmetry(self) -> None:
        """expected_score(a, b) + expected_score(b, a) == 1.0."""
        e_ab = expected_score(1100.0, 900.0)
        e_ba = expected_score(900.0, 1100.0)
        self.assertAlmostEqual(e_ab + e_ba, 1.0, places=10)

    def test_400_point_gap(self) -> None:
        """A 400-point gap yields expected_score ≈ 0.909 for the stronger agent."""
        # 10/(1+10) = 0.9090...
        es = expected_score(1400.0, 1000.0)
        self.assertAlmostEqual(es, 10.0 / 11.0, places=6)

    def test_output_range(self) -> None:
        """expected_score is always in (0, 1)."""
        for diff in [-1000, -400, -100, 0, 100, 400, 1000]:
            es = expected_score(1000.0 + diff, 1000.0)
            self.assertGreater(es, 0.0)
            self.assertLess(es, 1.0)


# ---------------------------------------------------------------------------
# k_factor
# ---------------------------------------------------------------------------


class TestKFactor(unittest.TestCase):
    """Tests for the k_factor() schedule."""

    def test_fresh_agent_gets_high_k(self) -> None:
        """A fresh agent (0 games) gets K = 40."""
        self.assertEqual(k_factor(0), 40.0)

    def test_boundary_29_games(self) -> None:
        """29 games still gives K = 40."""
        self.assertEqual(k_factor(29), 40.0)

    def test_boundary_30_games(self) -> None:
        """30 games gives K = 20."""
        self.assertEqual(k_factor(30), 20.0)

    def test_boundary_99_games(self) -> None:
        """99 games gives K = 20."""
        self.assertEqual(k_factor(99), 20.0)

    def test_boundary_100_games(self) -> None:
        """100 games gives K = 10."""
        self.assertEqual(k_factor(100), 10.0)

    def test_large_game_count(self) -> None:
        """Many games still gives K = 10 (does not go lower)."""
        self.assertEqual(k_factor(10_000), 10.0)

    def test_k_is_positive(self) -> None:
        """K-factor is always positive for any non-negative count."""
        for n in [0, 1, 29, 30, 99, 100, 500]:
            self.assertGreater(k_factor(n), 0.0)


# ---------------------------------------------------------------------------
# EloRegistry
# ---------------------------------------------------------------------------


class TestEloRegistry(unittest.TestCase):
    """Tests for the EloRegistry class."""

    def _registry(self, tmp: str) -> EloRegistry:
        return EloRegistry(path=Path(tmp) / "elo.json")

    # -- get_rating --

    def test_unknown_agent_gets_default_rating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            self.assertEqual(reg.get_rating("unknown_agent"), DEFAULT_RATING)

    def test_scripted_opponent_gets_baseline_rating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            for name, expected in BASELINE_RATINGS.items():
                self.assertAlmostEqual(reg.get_rating(name), expected)

    def test_random_opponent_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            self.assertAlmostEqual(reg.get_rating("random"), 500.0)

    # -- get_game_count --

    def test_game_count_starts_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            self.assertEqual(reg.get_game_count("new_agent"), 0)

    # -- update --

    def test_update_win_increases_rating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            old = reg.get_rating("agent_a")
            delta = reg.update("agent_a", "scripted_l3", outcome=1.0, n_games=50)
            self.assertGreater(delta, 0.0)
            self.assertGreater(reg.get_rating("agent_a"), old)

    def test_update_loss_decreases_rating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            old = reg.get_rating("agent_a")
            delta = reg.update("agent_a", "scripted_l5", outcome=0.0, n_games=50)
            self.assertLess(delta, 0.0)
            self.assertLess(reg.get_rating("agent_a"), old)

    def test_update_draw_moves_rating_by_small_amount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            # Against an equal-rated opponent a 50 % outcome produces zero delta.
            # scripted_l5 is 1000 == DEFAULT_RATING so expected == 0.5.
            delta = reg.update("agent_a", "scripted_l5", outcome=0.5, n_games=50)
            self.assertAlmostEqual(delta, 0.0, places=5)

    def test_update_increments_game_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            reg.update("agent_a", "scripted_l3", outcome=0.6, n_games=20)
            self.assertEqual(reg.get_game_count("agent_a"), 20)
            reg.update("agent_a", "scripted_l3", outcome=0.6, n_games=30)
            self.assertEqual(reg.get_game_count("agent_a"), 50)

    def test_update_baseline_opponent_rating_unchanged(self) -> None:
        """Scripted opponent ratings must never change."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            before = reg.get_rating("scripted_l3")
            reg.update("agent_a", "scripted_l3", outcome=0.8, n_games=100)
            after = reg.get_rating("scripted_l3")
            self.assertAlmostEqual(before, after)

    def test_update_invalid_outcome_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            with self.assertRaises(ValueError):
                reg.update("a", "scripted_l3", outcome=1.5)
            with self.assertRaises(ValueError):
                reg.update("a", "scripted_l3", outcome=-0.1)

    def test_update_invalid_n_games_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            with self.assertRaises(ValueError):
                reg.update("a", "scripted_l3", outcome=0.5, n_games=0)

    def test_update_baseline_agent_raises(self) -> None:
        """Attempting to update a baseline agent's rating raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            for name in BASELINE_RATINGS:
                with self.assertRaises(ValueError, msg=f"should raise for baseline '{name}'"):
                    reg.update(name, "scripted_l3", outcome=0.5, n_games=1)

    # -- save / load round-trip --

    def test_save_creates_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            reg.update("agent_x", "scripted_l3", outcome=0.7, n_games=10)
            reg.save()
            json_path = Path(tmp) / "elo.json"
            self.assertTrue(json_path.exists())
            with open(json_path) as fh:
                data = json.load(fh)
            self.assertIn("ratings", data)
            self.assertIn("game_counts", data)

    def test_load_restores_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg1 = self._registry(tmp)
            reg1.update("agent_x", "scripted_l3", outcome=0.8, n_games=20)
            rating1 = reg1.get_rating("agent_x")
            count1 = reg1.get_game_count("agent_x")
            reg1.save()

            reg2 = self._registry(tmp)
            self.assertAlmostEqual(reg2.get_rating("agent_x"), rating1, places=6)
            self.assertEqual(reg2.get_game_count("agent_x"), count1)

    def test_all_ratings_returns_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            reg.update("agent_a", "scripted_l5", outcome=0.5, n_games=10)
            ratings = reg.all_ratings()
            self.assertIn("agent_a", ratings)
            # Mutating the returned dict does not affect the registry.
            ratings["agent_a"] = 9999.0
            self.assertNotEqual(reg.get_rating("agent_a"), 9999.0)

    def test_save_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested_path = Path(tmp) / "a" / "b" / "c" / "elo.json"
            reg = EloRegistry(path=nested_path)
            reg.update("agent_x", "scripted_l3", outcome=0.5, n_games=5)
            reg.save()
            self.assertTrue(nested_path.exists())

    def test_in_memory_registry_path_none(self) -> None:
        """EloRegistry(path=None) works as in-memory registry."""
        reg = EloRegistry(path=None)
        # Against scripted_l5 (1000 = DEFAULT_RATING), a full win (1.0)
        # always increases the rating since outcome > expected (0.5).
        delta = reg.update("agent_a", "scripted_l5", outcome=1.0, n_games=10)
        self.assertIsInstance(delta, float)
        self.assertGreater(delta, 0.0)
        self.assertGreater(reg.get_rating("agent_a"), DEFAULT_RATING)

    def test_in_memory_registry_save_raises(self) -> None:
        """save() raises ValueError when path is None."""
        reg = EloRegistry(path=None)
        reg.update("agent_a", "scripted_l3", outcome=0.5, n_games=5)
        with self.assertRaises(ValueError):
            reg.save()

    def test_k_factor_decreases_over_games(self) -> None:
        """After many games the effective K-factor used by update() is small."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            # Accumulate 100+ games so K-factor drops to 10.
            for _ in range(11):
                reg.update("veteran", "scripted_l3", outcome=0.5, n_games=10)
            # At 110 games, K=10; a full win should delta < 10.
            reg2 = self._registry(tmp)
            reg2._ratings = dict(reg._ratings)
            reg2._game_counts = dict(reg._game_counts)
            delta = reg2.update("veteran", "scripted_l5", outcome=1.0, n_games=1)
            self.assertLess(abs(delta), 10.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
