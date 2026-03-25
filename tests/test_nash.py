# SPDX-License-Identifier: MIT
# tests/test_nash.py
"""Tests for training/league/nash.py — Nash equilibrium approximation (E4.5)."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.league.nash import (
    build_payoff_matrix,
    compute_nash_distribution,
    nash_entropy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_distribution(dist: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if *dist* is a valid probability distribution."""
    return (
        dist.ndim == 1
        and len(dist) > 0
        and bool(np.all(dist >= -tol))
        and abs(dist.sum() - 1.0) < tol
    )


# ---------------------------------------------------------------------------
# compute_nash_distribution — basic properties
# ---------------------------------------------------------------------------


class TestComputeNashDistributionProperties(unittest.TestCase):
    """Verify that compute_nash_distribution returns valid distributions."""

    def test_single_agent_returns_uniform(self) -> None:
        M = np.array([[0.5]])
        dist = compute_nash_distribution(M)
        self.assertEqual(dist.shape, (1,))
        self.assertAlmostEqual(dist[0], 1.0, places=6)

    def test_two_agents_sums_to_one(self) -> None:
        M = np.array([[0.5, 0.6], [0.4, 0.5]])
        dist = compute_nash_distribution(M)
        self.assertTrue(_is_valid_distribution(dist))

    def test_three_agents_sums_to_one(self) -> None:
        # Symmetric zero-sum 3×3 game.
        M = np.array([
            [0.5, 0.8, 0.2],
            [0.2, 0.5, 0.8],
            [0.8, 0.2, 0.5],
        ])
        dist = compute_nash_distribution(M)
        self.assertTrue(_is_valid_distribution(dist))

    def test_output_shape_matches_n_agents(self) -> None:
        for n in [2, 3, 4, 5]:
            M = np.full((n, n), 0.5)
            np.fill_diagonal(M, 0.5)
            dist = compute_nash_distribution(M)
            self.assertEqual(dist.shape, (n,))

    def test_non_negative_probabilities(self) -> None:
        rng = np.random.default_rng(0)
        M = rng.random((4, 4))
        dist = compute_nash_distribution(M)
        self.assertTrue(np.all(dist >= 0.0))

    def test_invalid_non_square_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_nash_distribution(np.array([[0.5, 0.4, 0.3]]))

    def test_invalid_1d_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_nash_distribution(np.array([0.5, 0.5]))

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_nash_distribution(np.empty((0, 0)))


# ---------------------------------------------------------------------------
# compute_nash_distribution — known game solutions
# ---------------------------------------------------------------------------


class TestComputeNashDistributionKnownGames(unittest.TestCase):
    """Verify Nash solution on games with known equilibria."""

    def _assert_approx_uniform(
        self,
        dist: np.ndarray,
        n: int,
        atol: float = 0.05,
    ) -> None:
        expected = 1.0 / n
        for i, p in enumerate(dist):
            self.assertAlmostEqual(
                p, expected, delta=atol,
                msg=f"dist[{i}]={p:.4f} expected ≈ {expected:.4f}"
            )

    def test_2x2_balanced_game(self) -> None:
        """Balanced 2×2 game: any distribution is a valid Nash equilibrium."""
        # M = all 0.5 means both agents are equally matched; any σ is Nash.
        M = np.array([[0.5, 0.5], [0.5, 0.5]])
        dist = compute_nash_distribution(M)
        self.assertTrue(_is_valid_distribution(dist))
        # Both agents should have non-trivial weight (not one dominating).
        # (This is a soft check — the LP may pick any valid Nash.)
        self.assertAlmostEqual(dist.sum(), 1.0, places=6)

    def test_2x2_dominated_strategy(self) -> None:
        """Agent 0 strictly dominates agent 1: Nash puts all weight on agent 0."""
        # Agent 0 always beats agent 1: M[0,1] = 1, M[1,0] = 0.
        M = np.array([[0.5, 1.0], [0.0, 0.5]])
        dist = compute_nash_distribution(M)
        # Nash should heavily favour agent 0.
        self.assertGreater(dist[0], dist[1])
        self.assertGreater(dist[0], 0.8)

    def test_2x2_zero_sum_rps_style(self) -> None:
        """2×2 uniform game: any distribution is Nash; verify it is valid."""
        # For any M where all agents are symmetric, any σ is Nash.
        M = np.array([[0.5, 0.5], [0.5, 0.5]])
        dist = compute_nash_distribution(M)
        self.assertTrue(_is_valid_distribution(dist))

    def test_3x3_rock_paper_scissors(self) -> None:
        """RPS-like 3×3 game: Nash = uniform [1/3, 1/3, 1/3].

        Win-rate matrix for a cyclic game:
        - Agent 0 beats agent 1 (0.9), loses to agent 2 (0.1)
        - Agent 1 beats agent 2 (0.9), loses to agent 0 (0.1)
        - Agent 2 beats agent 0 (0.9), loses to agent 1 (0.1)
        """
        M = np.array([
            [0.5, 0.9, 0.1],
            [0.1, 0.5, 0.9],
            [0.9, 0.1, 0.5],
        ])
        dist = compute_nash_distribution(M)
        self.assertTrue(_is_valid_distribution(dist))
        self._assert_approx_uniform(dist, 3, atol=0.08)

    def test_3x3_one_dominant_agent(self) -> None:
        """One agent beats all others: Nash concentrates on dominant agent."""
        # Agent 0 beats everyone: M[0, 1] = M[0, 2] = 1, M[1,2] = 0.5.
        M = np.array([
            [0.5, 1.0, 1.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5],
        ])
        dist = compute_nash_distribution(M)
        self.assertGreater(dist[0], 0.7)

    def test_3x3_symmetric_cyclic(self) -> None:
        """Strictly cyclic 3-agent game: uniform Nash."""
        # Each agent beats the next (modulo 3) with rate 0.7.
        M = np.array([
            [0.5, 0.7, 0.3],
            [0.3, 0.5, 0.7],
            [0.7, 0.3, 0.5],
        ])
        dist = compute_nash_distribution(M)
        self._assert_approx_uniform(dist, 3, atol=0.08)

    def test_2x2_strictly_cyclic_unique_nash(self) -> None:
        """2×2 game where one agent strictly dominates: Nash is pure.

        M = [[0.5, 0.3], [0.7, 0.5]]:
          - Agent 1 beats agent 0 with rate 0.7.
          - Agent 1 beats agent 1 with rate 0.5 (tie).
          - Agent 1 strictly dominates agent 0 → Nash = [0, 1].
        """
        M = np.array([[0.5, 0.3], [0.7, 0.5]])
        dist = compute_nash_distribution(M)
        self.assertTrue(_is_valid_distribution(dist))
        # Agent 1 (index 1) dominates → Nash should weight agent 1 more.
        self.assertGreater(dist[1], dist[0])

    def test_lp_and_regret_matching_agree(self) -> None:
        """LP and regret matching should produce similar Nash distributions."""
        M = np.array([
            [0.5, 0.9, 0.1],
            [0.1, 0.5, 0.9],
            [0.9, 0.1, 0.5],
        ])
        dist_lp = compute_nash_distribution(M, use_lp=True)
        dist_rm = compute_nash_distribution(M, use_lp=False, n_iterations=20_000)
        np.testing.assert_allclose(dist_lp, dist_rm, atol=0.1)

    def test_regret_matching_only_valid(self) -> None:
        """Regret matching without LP still produces a valid distribution."""
        M = np.array([[0.5, 0.7], [0.3, 0.5]])
        dist = compute_nash_distribution(M, use_lp=False)
        self.assertTrue(_is_valid_distribution(dist))


# ---------------------------------------------------------------------------
# nash_entropy
# ---------------------------------------------------------------------------


class TestNashEntropy(unittest.TestCase):
    """Unit tests for nash_entropy."""

    def test_uniform_distribution_max_entropy(self) -> None:
        """Uniform distribution should have entropy = log(N)."""
        for n in [2, 3, 4]:
            dist = np.ones(n) / n
            expected = math.log(n)
            self.assertAlmostEqual(nash_entropy(dist), expected, places=5)

    def test_pure_strategy_zero_entropy(self) -> None:
        """A Dirac distribution should have entropy ≈ 0."""
        dist = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(nash_entropy(dist), 0.0, places=5)

    def test_binary_half_half(self) -> None:
        """[0.5, 0.5] should have entropy = log(2)."""
        dist = np.array([0.5, 0.5])
        self.assertAlmostEqual(nash_entropy(dist), math.log(2), places=5)

    def test_entropy_non_negative(self) -> None:
        """Entropy must always be ≥ 0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.random(5)
            dist = x / x.sum()
            self.assertGreaterEqual(nash_entropy(dist), 0.0)

    def test_normalises_unnormalised_input(self) -> None:
        """Input need not be normalised; result should match normalised."""
        dist = np.array([2.0, 2.0])
        self.assertAlmostEqual(nash_entropy(dist), math.log(2), places=5)

    def test_single_element_zero_entropy(self) -> None:
        dist = np.array([1.0])
        self.assertAlmostEqual(nash_entropy(dist), 0.0, places=5)

    def test_all_zero_returns_zero(self) -> None:
        """Zero vector should return 0.0 (no information)."""
        dist = np.zeros(3)
        self.assertAlmostEqual(nash_entropy(dist), 0.0, places=5)


# ---------------------------------------------------------------------------
# build_payoff_matrix
# ---------------------------------------------------------------------------


class TestBuildPayoffMatrix(unittest.TestCase):
    """Tests for build_payoff_matrix."""

    def test_diagonal_is_0_5(self) -> None:
        agent_ids = ["a", "b", "c"]
        M = build_payoff_matrix(agent_ids, lambda a, b: None, unknown_win_rate=0.5)
        np.testing.assert_array_equal(np.diag(M), [0.5, 0.5, 0.5])

    def test_unknown_win_rate_used_for_missing(self) -> None:
        agent_ids = ["a", "b"]
        M = build_payoff_matrix(agent_ids, lambda a, b: None, unknown_win_rate=0.7)
        # Off-diagonals should be 0.7 (unknown win rate).
        self.assertAlmostEqual(M[0, 1], 0.7)
        self.assertAlmostEqual(M[1, 0], 0.7)

    def test_known_win_rates_filled_correctly(self) -> None:
        agent_ids = ["a", "b", "c"]
        win_rates = {
            ("a", "b"): 0.8,
            ("b", "c"): 0.6,
            ("c", "a"): 0.4,
        }

        def _wr(ai, aj):
            return win_rates.get((ai, aj))

        M = build_payoff_matrix(agent_ids, _wr, unknown_win_rate=0.5)
        # Check known entries.
        self.assertAlmostEqual(M[0, 1], 0.8)  # a vs b (known)
        self.assertAlmostEqual(M[1, 2], 0.6)  # b vs c (known)
        self.assertAlmostEqual(M[2, 0], 0.4)  # c vs a (known)
        # Reverse entries inferred via constant-sum: M[j, i] = 1 - M[i, j].
        self.assertAlmostEqual(M[1, 0], 0.2)  # b vs a = 1 - 0.8
        self.assertAlmostEqual(M[2, 1], 0.4)  # c vs b = 1 - 0.6
        self.assertAlmostEqual(M[0, 2], 0.6)  # a vs c = 1 - 0.4

    def test_constant_sum_inference_only_when_reverse_known(self) -> None:
        """Constant-sum inference is only applied when the reverse is known."""
        agent_ids = ["a", "b", "c"]
        # Only a→b is known; b→c and c→a are not.
        win_rates = {("a", "b"): 0.7}

        def _wr(ai, aj):
            return win_rates.get((ai, aj))

        M = build_payoff_matrix(agent_ids, _wr, unknown_win_rate=0.5)
        # a→b known directly.
        self.assertAlmostEqual(M[0, 1], 0.7)
        # b→a inferred from constant-sum.
        self.assertAlmostEqual(M[1, 0], 0.3)
        # Others are truly unknown → fall back to unknown_win_rate.
        self.assertAlmostEqual(M[1, 2], 0.5)
        self.assertAlmostEqual(M[2, 1], 0.5)
        self.assertAlmostEqual(M[0, 2], 0.5)
        self.assertAlmostEqual(M[2, 0], 0.5)

    def test_output_shape(self) -> None:
        for n in [1, 2, 3, 5]:
            ids = [str(i) for i in range(n)]
            M = build_payoff_matrix(ids, lambda a, b: None)
            self.assertEqual(M.shape, (n, n))

    def test_dtype_is_float64(self) -> None:
        M = build_payoff_matrix(["a", "b"], lambda a, b: None)
        self.assertEqual(M.dtype, np.float64)

    def test_empty_agent_list(self) -> None:
        M = build_payoff_matrix([], lambda a, b: None)
        self.assertEqual(M.shape, (0, 0))

    def test_integration_with_match_database(self) -> None:
        """build_payoff_matrix integrates with MatchDatabase.win_rate."""
        from training.league.match_database import MatchDatabase

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "matches.jsonl"
            db = MatchDatabase(db_path)
            db.record("a", "b", 1.0)
            db.record("a", "b", 1.0)
            db.record("b", "a", 0.0)

            agent_ids = ["a", "b"]
            M = build_payoff_matrix(agent_ids, db.win_rate, unknown_win_rate=0.5)
            self.assertAlmostEqual(M[0, 1], 1.0)  # a always beats b
            self.assertAlmostEqual(M[1, 0], 0.0)  # b never beats a
            self.assertAlmostEqual(M[0, 0], 0.5)  # diagonal
            self.assertAlmostEqual(M[1, 1], 0.5)  # diagonal


# ---------------------------------------------------------------------------
# Integration: Nash → matchmaker
# ---------------------------------------------------------------------------


class TestNashMatchmakerIntegration(unittest.TestCase):
    """Test that LeagueMatchmaker respects Nash weights."""

    def setUp(self) -> None:
        from training.league.agent_pool import AgentPool, AgentType
        from training.league.match_database import MatchDatabase
        from training.league.matchmaker import LeagueMatchmaker

        self._tmpdir = tempfile.TemporaryDirectory()
        manifest = Path(self._tmpdir.name) / "pool.json"
        db_path = Path(self._tmpdir.name) / "matches.jsonl"
        snap_path = Path(self._tmpdir.name) / "snap.pt"
        snap_path.write_bytes(b"")

        self.pool = AgentPool(pool_manifest=manifest, max_size=50)
        self.db = MatchDatabase(db_path)
        self.pool.add(snap_path, AgentType.MAIN_AGENT, agent_id="a")
        self.pool.add(snap_path, AgentType.MAIN_AGENT, agent_id="b")
        self.pool.add(snap_path, AgentType.MAIN_AGENT, agent_id="c")
        self.mm = LeagueMatchmaker(self.pool, self.db)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_set_nash_weights_concentrates_sampling(self) -> None:
        """Setting Nash weights that concentrate on 'b' should mostly select 'b'."""
        self.mm.set_nash_weights({"a": 0.0, "b": 1.0, "c": 0.0})
        rng = np.random.default_rng(99)
        counts = {"a": 0, "b": 0, "c": 0}
        for _ in range(200):
            rec = self.mm.select_opponent("a", rng=rng)
            if rec is not None:
                counts[rec.agent_id] += 1
        # All selections should be 'b' (or 'c' if 'b' has 0 weight and fallback kicks in).
        # Because weights for 'b' = 1.0 and others = 0.0, only 'b' selected.
        self.assertEqual(counts["b"], 200)
        self.assertEqual(counts["a"], 0)
        self.assertEqual(counts["c"], 0)

    def test_clear_nash_weights_reverts_to_pfsp(self) -> None:
        """Clearing Nash weights (None) should revert to PFSP-based sampling."""
        self.mm.set_nash_weights({"a": 0.0, "b": 1.0, "c": 0.0})
        self.mm.set_nash_weights(None)
        # Record wins so PFSP can differentiate.
        for _ in range(10):
            self.db.record("a", "b", 1.0)  # easy — weight 0
            self.db.record("a", "c", 0.0)  # hard — weight 1

        rng = np.random.default_rng(7)
        counts = {"b": 0, "c": 0}
        for _ in range(500):
            rec = self.mm.select_opponent("a", rng=rng)
            if rec and rec.agent_id in counts:
                counts[rec.agent_id] += 1
        # PFSP hard-first: 'c' (loss rate 1.0, weight=1) >> 'b' (win rate 1.0, weight=0).
        self.assertGreater(counts["c"], counts["b"])

    def test_opponent_probabilities_reflect_nash_weights(self) -> None:
        """opponent_probabilities should mirror Nash weights."""
        self.mm.set_nash_weights({"a": 0.0, "b": 0.3, "c": 0.7})
        probs = self.mm.opponent_probabilities("a")
        self.assertAlmostEqual(probs.get("b", 0.0), 0.3, delta=0.01)
        self.assertAlmostEqual(probs.get("c", 0.0), 0.7, delta=0.01)
        self.assertNotIn("a", probs)

    def test_all_zero_nash_weights_falls_back_to_uniform(self) -> None:
        """When all Nash weights are 0, fallback to uniform."""
        self.mm.set_nash_weights({"a": 0.0, "b": 0.0, "c": 0.0})
        probs = self.mm.opponent_probabilities("a")
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=6)


# ---------------------------------------------------------------------------
# Integration: Nash distribution end-to-end
# ---------------------------------------------------------------------------


class TestNashEndToEnd(unittest.TestCase):
    """End-to-end: build payoff matrix → Nash dist → entropy → matchmaker."""

    def test_full_pipeline_3_agents(self) -> None:
        from training.league.match_database import MatchDatabase

        with tempfile.TemporaryDirectory() as tmp:
            db = MatchDatabase(Path(tmp) / "matches.jsonl")
            # Record a cyclic relationship: a > b > c > a.
            for _ in range(5):
                db.record("a", "b", 1.0)
                db.record("b", "c", 1.0)
                db.record("c", "a", 1.0)

            ids = ["a", "b", "c"]
            M = build_payoff_matrix(ids, db.win_rate, unknown_win_rate=0.5)
            # Off-diagonals with known win rates.
            self.assertAlmostEqual(M[0, 1], 1.0)
            self.assertAlmostEqual(M[1, 2], 1.0)
            self.assertAlmostEqual(M[2, 0], 1.0)

            nash_dist = compute_nash_distribution(M)
            self.assertTrue(_is_valid_distribution(nash_dist))

            entropy = nash_entropy(nash_dist)
            max_entropy = math.log(3)
            # Cyclic game → should be close to max entropy.
            self.assertGreater(entropy, max_entropy * 0.5)

    def test_pipeline_with_single_dominant_agent(self) -> None:
        from training.league.match_database import MatchDatabase

        with tempfile.TemporaryDirectory() as tmp:
            db = MatchDatabase(Path(tmp) / "matches.jsonl")
            for _ in range(5):
                db.record("a", "b", 1.0)
                db.record("a", "c", 1.0)

            ids = ["a", "b", "c"]
            M = build_payoff_matrix(ids, db.win_rate, unknown_win_rate=0.5)
            nash_dist = compute_nash_distribution(M)

            # Dominant agent should have highest probability.
            self.assertEqual(np.argmax(nash_dist), 0)

            entropy = nash_entropy(nash_dist)
            max_entropy = math.log(3)
            # Low diversity — entropy should be below maximum.
            self.assertLess(entropy, max_entropy)


if __name__ == "__main__":
    unittest.main()
