# tests/test_evaluate_hrl.py
"""Tests for training/evaluate_hrl.py — HRL vs. flat MARL evaluation harness."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate_hrl import (
    TournamentResult,
    bootstrap_ci,
    run_flat_marl_episodes,
    run_hrl_episodes,
    run_tournament,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI(unittest.TestCase):
    """Tests for bootstrap_ci()."""

    def test_returns_two_floats(self) -> None:
        outcomes = [1, 0, 1, -1, 1]
        lo, hi = bootstrap_ci(outcomes, n_bootstrap=100, rng=np.random.default_rng(0))
        self.assertIsInstance(lo, float)
        self.assertIsInstance(hi, float)

    def test_ci_ordered(self) -> None:
        """Lower bound must not exceed upper bound."""
        outcomes = [1] * 7 + [-1] * 3
        lo, hi = bootstrap_ci(outcomes, n_bootstrap=200, rng=np.random.default_rng(1))
        self.assertLessEqual(lo, hi)

    def test_all_wins_ci_near_one(self) -> None:
        """When every episode is a win, the CI should be near [1.0, 1.0]."""
        outcomes = [1] * 20
        lo, hi = bootstrap_ci(outcomes, n_bootstrap=500, rng=np.random.default_rng(2))
        self.assertGreater(lo, 0.9)
        self.assertAlmostEqual(hi, 1.0)

    def test_all_losses_ci_near_zero(self) -> None:
        """When every episode is a loss, the CI should be near [0.0, 0.0]."""
        outcomes = [-1] * 20
        lo, hi = bootstrap_ci(outcomes, n_bootstrap=500, rng=np.random.default_rng(3))
        self.assertAlmostEqual(lo, 0.0)
        self.assertLess(hi, 0.1)

    def test_reproducible_with_same_seed(self) -> None:
        """Same rng seed → same CI."""
        outcomes = [1, 0, -1] * 10
        lo1, hi1 = bootstrap_ci(outcomes, n_bootstrap=100, rng=np.random.default_rng(7))
        lo2, hi2 = bootstrap_ci(outcomes, n_bootstrap=100, rng=np.random.default_rng(7))
        self.assertAlmostEqual(lo1, lo2)
        self.assertAlmostEqual(hi1, hi2)

    def test_empty_outcomes_raises(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_ci([], n_bootstrap=100)

    def test_n_bootstrap_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_ci([1, 0], n_bootstrap=0)

    def test_invalid_confidence_raises(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_ci([1, 0], n_bootstrap=100, confidence=1.5)
        with self.assertRaises(ValueError):
            bootstrap_ci([1, 0], n_bootstrap=100, confidence=0.0)


# ---------------------------------------------------------------------------
# run_hrl_episodes (random actions — no checkpoint required)
# ---------------------------------------------------------------------------


class TestRunHRLEpisodes(unittest.TestCase):
    """Smoke tests for run_hrl_episodes() without any checkpoint."""

    def test_returns_tournament_result(self) -> None:
        result = run_hrl_episodes(n_episodes=3, seed=0, n_bootstrap=50)
        self.assertIsInstance(result, TournamentResult)

    def test_episode_counts_consistent(self) -> None:
        result = run_hrl_episodes(n_episodes=5, seed=1, n_bootstrap=50)
        self.assertEqual(result.n_episodes, 5)
        self.assertEqual(result.wins + result.draws + result.losses, 5)

    def test_win_rate_in_range(self) -> None:
        result = run_hrl_episodes(n_episodes=4, seed=2, n_bootstrap=50)
        self.assertGreaterEqual(result.win_rate, 0.0)
        self.assertLessEqual(result.win_rate, 1.0)

    def test_ci_ordered(self) -> None:
        result = run_hrl_episodes(n_episodes=4, seed=3, n_bootstrap=50)
        self.assertLessEqual(result.ci_lower, result.ci_upper)

    def test_ci_confidence_stored(self) -> None:
        result = run_hrl_episodes(
            n_episodes=4, seed=4, n_bootstrap=50, ci_confidence=0.90
        )
        self.assertAlmostEqual(result.ci_confidence, 0.90)

    def test_zero_episodes_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_hrl_episodes(n_episodes=0)

    def test_label_stored(self) -> None:
        result = run_hrl_episodes(n_episodes=2, seed=5, n_bootstrap=20, label="TestHRL")
        self.assertEqual(result.label, "TestHRL")

    def test_to_dict(self) -> None:
        result = run_hrl_episodes(n_episodes=2, seed=6, n_bootstrap=20)
        d = result.to_dict()
        self.assertIn("win_rate", d)
        self.assertIn("ci_lower", d)
        self.assertIn("ci_upper", d)

    def test_red_random_mode(self) -> None:
        """red_random=True should not raise."""
        result = run_hrl_episodes(
            n_episodes=3, seed=7, n_bootstrap=30, red_random=True
        )
        self.assertEqual(result.n_episodes, 3)


# ---------------------------------------------------------------------------
# run_flat_marl_episodes (random actions — no checkpoint required)
# ---------------------------------------------------------------------------


class TestRunFlatMARLEpisodes(unittest.TestCase):
    """Smoke tests for run_flat_marl_episodes() without any checkpoint."""

    def test_returns_tournament_result(self) -> None:
        result = run_flat_marl_episodes(n_episodes=3, seed=10, n_bootstrap=50)
        self.assertIsInstance(result, TournamentResult)

    def test_episode_counts_consistent(self) -> None:
        result = run_flat_marl_episodes(n_episodes=5, seed=11, n_bootstrap=50)
        self.assertEqual(result.n_episodes, 5)
        self.assertEqual(result.wins + result.draws + result.losses, 5)

    def test_win_rate_in_range(self) -> None:
        result = run_flat_marl_episodes(n_episodes=4, seed=12, n_bootstrap=50)
        self.assertGreaterEqual(result.win_rate, 0.0)
        self.assertLessEqual(result.win_rate, 1.0)

    def test_ci_ordered(self) -> None:
        result = run_flat_marl_episodes(n_episodes=4, seed=13, n_bootstrap=50)
        self.assertLessEqual(result.ci_lower, result.ci_upper)

    def test_zero_episodes_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_flat_marl_episodes(n_episodes=0)

    def test_label_stored(self) -> None:
        result = run_flat_marl_episodes(
            n_episodes=2, seed=14, n_bootstrap=20, label="TestFlat"
        )
        self.assertEqual(result.label, "TestFlat")

    def test_red_random_mode(self) -> None:
        result = run_flat_marl_episodes(
            n_episodes=3, seed=15, n_bootstrap=30, red_random=True
        )
        self.assertEqual(result.n_episodes, 3)


# ---------------------------------------------------------------------------
# run_tournament
# ---------------------------------------------------------------------------


class TestRunTournament(unittest.TestCase):
    """Tests for run_tournament()."""

    def _make_result(
        self,
        wins: int,
        total: int = 10,
        ci_lower: float = 0.2,
        ci_upper: float = 0.8,
        label: str = "Test",
    ) -> TournamentResult:
        losses = total - wins
        return TournamentResult(
            wins=wins,
            draws=0,
            losses=losses,
            n_episodes=total,
            win_rate=wins / total,
            draw_rate=0.0,
            loss_rate=losses / total,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_confidence=0.95,
            label=label,
        )

    def test_returns_dict(self) -> None:
        hrl = self._make_result(7, label="HRL")
        flat = self._make_result(4, label="FlatMARL")
        summary = run_tournament(hrl, flat)
        self.assertIsInstance(summary, dict)
        self.assertIn("hrl", summary)
        self.assertIn("flat_marl", summary)
        self.assertIn("delta_win_rate", summary)
        self.assertIn("conclusion", summary)

    def test_delta_win_rate_correct(self) -> None:
        hrl = self._make_result(8, label="HRL")
        flat = self._make_result(3, label="FlatMARL")
        summary = run_tournament(hrl, flat)
        expected = round(0.8 - 0.3, 4)
        self.assertAlmostEqual(summary["delta_win_rate"], expected, places=4)

    def test_hrl_outperforms_conclusion(self) -> None:
        """Non-overlapping CIs with HRL higher → HRL outperforms."""
        hrl = self._make_result(9, ci_lower=0.7, ci_upper=0.95, label="HRL")
        flat = self._make_result(2, ci_lower=0.05, ci_upper=0.4, label="FlatMARL")
        summary = run_tournament(hrl, flat)
        self.assertFalse(summary["ci_overlap"])
        self.assertIn("HRL", summary["conclusion"])

    def test_flat_outperforms_conclusion(self) -> None:
        """Non-overlapping CIs with Flat MARL higher."""
        hrl = self._make_result(1, ci_lower=0.0, ci_upper=0.3, label="HRL")
        flat = self._make_result(9, ci_lower=0.7, ci_upper=1.0, label="FlatMARL")
        summary = run_tournament(hrl, flat)
        self.assertFalse(summary["ci_overlap"])
        self.assertIn("FlatMARL", summary["conclusion"])

    def test_inconclusive_when_ci_overlap(self) -> None:
        """Overlapping CIs → Inconclusive."""
        hrl = self._make_result(6, ci_lower=0.3, ci_upper=0.9, label="HRL")
        flat = self._make_result(5, ci_lower=0.2, ci_upper=0.8, label="FlatMARL")
        summary = run_tournament(hrl, flat)
        self.assertTrue(summary["ci_overlap"])
        self.assertIn("Inconclusive", summary["conclusion"])

    def test_json_serialisable(self) -> None:
        hrl = self._make_result(6, label="HRL")
        flat = self._make_result(5, label="FlatMARL")
        summary = run_tournament(hrl, flat)
        # Should not raise
        json.dumps(summary)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    """Smoke test the CLI entry point."""

    def test_main_stdout(self) -> None:
        """main() runs end-to-end and prints valid JSON to stdout."""
        import io
        from contextlib import redirect_stdout
        from training.evaluate_hrl import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--n-episodes", "2", "--seed", "99"])
        output = buf.getvalue()
        summary = json.loads(output)
        self.assertIn("hrl", summary)
        self.assertIn("flat_marl", summary)
        self.assertIn("conclusion", summary)

    def test_main_output_file(self) -> None:
        """main() writes results to a JSON file when --output is given."""
        from training.evaluate_hrl import main

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results.json"
            main(["--n-episodes", "2", "--seed", "42", "--output", str(out)])
            self.assertTrue(out.exists())
            summary = json.loads(out.read_text())
            self.assertIn("hrl", summary)

    def test_main_zero_episodes_raises(self) -> None:
        from training.evaluate_hrl import main

        with self.assertRaises(SystemExit):
            main(["--n-episodes", "0"])


if __name__ == "__main__":
    unittest.main()
