# tests/test_diversity.py
"""Tests for training/league/diversity.py — strategy diversity metrics (E4.6)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.league.diversity import (
    DiversityTracker,
    TrajectoryBatch,
    diversity_score,
    embed_trajectory,
    pairwise_cosine_distances,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(
    n_steps: int = 50,
    action_dim: int = 3,
    seed: int = 0,
    action_bias: float = 0.0,
    pos_bias: float = 0.0,
) -> TrajectoryBatch:
    """Return a synthetic TrajectoryBatch for testing."""
    rng = np.random.default_rng(seed)
    actions = rng.standard_normal((n_steps, action_dim)).astype(np.float32) + action_bias
    positions = np.clip(
        rng.random((n_steps, 2)).astype(np.float32) + pos_bias, 0.0, 1.0
    )
    return TrajectoryBatch(actions=actions, positions=positions, agent_id=f"agent_{seed}")


# ---------------------------------------------------------------------------
# TrajectoryBatch
# ---------------------------------------------------------------------------


class TestTrajectoryBatch(unittest.TestCase):
    """Tests for TrajectoryBatch construction and properties."""

    def test_basic_construction(self) -> None:
        traj = _make_trajectory()
        self.assertEqual(traj.n_steps, 50)
        self.assertEqual(traj.action_dim, 3)

    def test_invalid_1d_actions_raises(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryBatch(
                actions=np.ones(10),
                positions=np.ones((10, 2)),
            )

    def test_invalid_positions_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryBatch(
                actions=np.ones((10, 3)),
                positions=np.ones((10, 3)),  # should be (T, 2)
            )

    def test_mismatched_timesteps_raises(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryBatch(
                actions=np.ones((10, 3)),
                positions=np.ones((5, 2)),
            )

    def test_zero_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryBatch(
                actions=np.ones((0, 3)),
                positions=np.ones((0, 2)),
            )

    def test_agent_id_stored(self) -> None:
        traj = TrajectoryBatch(
            actions=np.ones((5, 2)),
            positions=np.ones((5, 2)) * 0.5,
            agent_id="test_agent",
        )
        self.assertEqual(traj.agent_id, "test_agent")

    def test_repr_contains_agent_id(self) -> None:
        traj = _make_trajectory()
        self.assertIn("agent_0", repr(traj))


# ---------------------------------------------------------------------------
# embed_trajectory
# ---------------------------------------------------------------------------


class TestEmbedTrajectory(unittest.TestCase):
    """Tests for embed_trajectory."""

    def test_output_is_1d(self) -> None:
        traj = _make_trajectory(n_steps=50, action_dim=3)
        emb = embed_trajectory(traj)
        self.assertEqual(emb.ndim, 1)

    def test_output_length(self) -> None:
        """Embedding length = action_dim * n_action_bins + n_pos_bins^2 + 4."""
        n_action_bins = 8
        n_pos_bins = 4
        action_dim = 3
        traj = _make_trajectory(n_steps=50, action_dim=action_dim)
        emb = embed_trajectory(traj, n_action_bins=n_action_bins, n_pos_bins=n_pos_bins)
        expected_len = action_dim * n_action_bins + n_pos_bins ** 2 + 4
        self.assertEqual(len(emb), expected_len)

    def test_unit_norm(self) -> None:
        """Embedding should be L2-normalised to unit length."""
        traj = _make_trajectory()
        emb = embed_trajectory(traj)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

    def test_dtype_float64(self) -> None:
        traj = _make_trajectory()
        emb = embed_trajectory(traj)
        self.assertEqual(emb.dtype, np.float64)

    def test_different_behaviors_differ(self) -> None:
        """Two agents with opposite action biases should have different embeddings."""
        traj_a = _make_trajectory(action_dim=2, action_bias=5.0, seed=1)
        traj_b = _make_trajectory(action_dim=2, action_bias=-5.0, seed=2)
        emb_a = embed_trajectory(traj_a)
        emb_b = embed_trajectory(traj_b)
        # Embeddings should not be identical.
        self.assertFalse(np.allclose(emb_a, emb_b))

    def test_single_step_trajectory(self) -> None:
        """Single-step trajectory should still produce a valid embedding."""
        traj = TrajectoryBatch(
            actions=np.ones((1, 2)),
            positions=np.array([[0.5, 0.5]]),
        )
        emb = embed_trajectory(traj)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

    def test_invalid_n_action_bins_raises(self) -> None:
        traj = _make_trajectory()
        with self.assertRaises(ValueError):
            embed_trajectory(traj, n_action_bins=0)

    def test_invalid_n_pos_bins_raises(self) -> None:
        traj = _make_trajectory()
        with self.assertRaises(ValueError):
            embed_trajectory(traj, n_pos_bins=0)

    def test_custom_bin_counts(self) -> None:
        traj = _make_trajectory(action_dim=4)
        emb = embed_trajectory(traj, n_action_bins=4, n_pos_bins=2)
        expected_len = 4 * 4 + 2 ** 2 + 4
        self.assertEqual(len(emb), expected_len)

    def test_positions_outside_unit_range_clipped(self) -> None:
        """Positions outside [0,1] should be silently clipped, not cause errors."""
        traj = TrajectoryBatch(
            actions=np.ones((10, 2)),
            positions=np.full((10, 2), 2.5),  # outside [0, 1]
        )
        emb = embed_trajectory(traj)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)


# ---------------------------------------------------------------------------
# pairwise_cosine_distances
# ---------------------------------------------------------------------------


class TestPairwiseCosineDists(unittest.TestCase):
    """Tests for pairwise_cosine_distances."""

    def test_output_shape(self) -> None:
        embeddings = np.random.default_rng(0).random((5, 16))
        D = pairwise_cosine_distances(embeddings)
        self.assertEqual(D.shape, (5, 5))

    def test_diagonal_is_zero(self) -> None:
        embeddings = np.random.default_rng(1).random((4, 10))
        D = pairwise_cosine_distances(embeddings)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_symmetric(self) -> None:
        embeddings = np.random.default_rng(2).random((4, 10))
        D = pairwise_cosine_distances(embeddings)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_non_negative(self) -> None:
        embeddings = np.random.default_rng(3).random((5, 10))
        D = pairwise_cosine_distances(embeddings)
        self.assertTrue(np.all(D >= -1e-10))

    def test_identical_embeddings_zero_distance(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        embeddings = np.stack([v, v])
        D = pairwise_cosine_distances(embeddings)
        self.assertAlmostEqual(D[0, 1], 0.0, places=5)
        self.assertAlmostEqual(D[1, 0], 0.0, places=5)

    def test_orthogonal_embeddings_distance_one(self) -> None:
        """Orthogonal unit vectors have cosine distance = 1."""
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        D = pairwise_cosine_distances(np.stack([e1, e2]))
        self.assertAlmostEqual(D[0, 1], 1.0, places=5)

    def test_single_row_returns_1x1_zero(self) -> None:
        embeddings = np.ones((1, 5))
        D = pairwise_cosine_distances(embeddings)
        self.assertEqual(D.shape, (1, 1))
        self.assertAlmostEqual(D[0, 0], 0.0, places=5)

    def test_invalid_1d_raises(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_cosine_distances(np.ones(5))

    def test_invalid_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_cosine_distances(np.empty((0, 5)))

    def test_unnormalised_input_handled(self) -> None:
        """Function normalises internally; scaling should not change distances."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        D_unit = pairwise_cosine_distances(np.stack([v1, v2]))
        D_scaled = pairwise_cosine_distances(np.stack([v1 * 10, v2 * 0.1]))
        np.testing.assert_allclose(D_unit, D_scaled, atol=1e-6)


# ---------------------------------------------------------------------------
# diversity_score
# ---------------------------------------------------------------------------


class TestDiversityScore(unittest.TestCase):
    """Tests for the diversity_score function."""

    def test_identical_embeddings_zero_diversity(self) -> None:
        v = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        score = diversity_score(v)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_orthogonal_embeddings_max_diversity(self) -> None:
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        score = diversity_score(np.stack([e1, e2]))
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_single_agent_returns_zero(self) -> None:
        score = diversity_score(np.ones((1, 5)))
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_score_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            E = rng.random((5, 16))
            self.assertGreaterEqual(diversity_score(E), 0.0)

    def test_aggregation_mean(self) -> None:
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        score = diversity_score(np.stack([e1, e2]), aggregation="mean")
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_aggregation_min(self) -> None:
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        e3 = np.array([1.0, 0.0])  # same as e1
        E = np.stack([e1, e2, e3])
        score_min = diversity_score(E, aggregation="min")
        # min distance: e1 vs e3 = 0
        self.assertAlmostEqual(score_min, 0.0, places=5)

    def test_aggregation_median(self) -> None:
        rng = np.random.default_rng(7)
        E = rng.random((6, 10))
        score_mean = diversity_score(E, aggregation="mean")
        score_median = diversity_score(E, aggregation="median")
        # Both should be in [0, 2].
        self.assertGreaterEqual(score_mean, 0.0)
        self.assertGreaterEqual(score_median, 0.0)

    def test_invalid_aggregation_raises(self) -> None:
        with self.assertRaises(ValueError):
            diversity_score(np.ones((2, 4)), aggregation="max")

    def test_invalid_1d_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            diversity_score(np.ones(5))

    def test_more_diverse_pool_higher_score(self) -> None:
        """A diverse set of embeddings should score higher than a similar set."""
        rng = np.random.default_rng(0)
        # Nearly identical agents.
        base = rng.random(16)
        similar = np.stack([base + rng.random(16) * 0.01 for _ in range(4)])
        # Orthogonally diverse agents.
        diverse = np.eye(4, 16)
        self.assertGreater(diversity_score(diverse), diversity_score(similar))


# ---------------------------------------------------------------------------
# DiversityTracker
# ---------------------------------------------------------------------------


class TestDiversityTracker(unittest.TestCase):
    """Tests for DiversityTracker."""

    def _make_tracker(self) -> DiversityTracker:
        return DiversityTracker(n_action_bins=4, n_pos_bins=2)

    def test_empty_tracker_score_zero(self) -> None:
        tracker = self._make_tracker()
        self.assertAlmostEqual(tracker.diversity_score(), 0.0, places=5)

    def test_single_agent_score_zero(self) -> None:
        tracker = self._make_tracker()
        tracker.update("agent_a", _make_trajectory(seed=0))
        self.assertAlmostEqual(tracker.diversity_score(), 0.0, places=5)

    def test_two_agents_score_positive(self) -> None:
        """Two behaviourally different agents should yield diversity > 0."""
        tracker = self._make_tracker()
        tracker.update("agent_a", _make_trajectory(action_bias=5.0, seed=1))
        tracker.update("agent_b", _make_trajectory(action_bias=-5.0, seed=2))
        score = tracker.diversity_score()
        self.assertGreater(score, 0.0)

    def test_update_replaces_existing(self) -> None:
        """Updating an existing agent replaces its embedding."""
        tracker = self._make_tracker()
        traj1 = _make_trajectory(seed=0)
        traj2 = _make_trajectory(seed=99)
        tracker.update("a", traj1)
        tracker.update("a", traj2)
        self.assertEqual(tracker.pool_size, 1)

    def test_remove_agent(self) -> None:
        tracker = self._make_tracker()
        tracker.update("a", _make_trajectory(seed=0))
        tracker.update("b", _make_trajectory(seed=1))
        tracker.remove("a")
        self.assertEqual(tracker.pool_size, 1)
        self.assertNotIn("a", tracker.agent_ids)

    def test_remove_unknown_agent_noop(self) -> None:
        tracker = self._make_tracker()
        tracker.remove("nonexistent")  # should not raise

    def test_clear(self) -> None:
        tracker = self._make_tracker()
        tracker.update("a", _make_trajectory(seed=0))
        tracker.update("b", _make_trajectory(seed=1))
        tracker.clear()
        self.assertEqual(tracker.pool_size, 0)

    def test_update_embedding_direct(self) -> None:
        tracker = self._make_tracker()
        emb = np.array([1.0, 0.0, 0.0])
        tracker.update_embedding("agent_x", emb)
        self.assertIn("agent_x", tracker.agent_ids)

    def test_embeddings_matrix_shape(self) -> None:
        tracker = self._make_tracker()
        tracker.update("a", _make_trajectory(seed=0))
        tracker.update("b", _make_trajectory(seed=1))
        ids, matrix = tracker.embeddings_matrix()
        self.assertEqual(len(ids), 2)
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(matrix.shape[0], 2)

    def test_embeddings_matrix_unknown_agent_raises(self) -> None:
        tracker = self._make_tracker()
        with self.assertRaises(KeyError):
            tracker.embeddings_matrix(["nonexistent"])

    def test_pairwise_distances_shape(self) -> None:
        tracker = self._make_tracker()
        tracker.update("a", _make_trajectory(seed=0))
        tracker.update("b", _make_trajectory(seed=1))
        tracker.update("c", _make_trajectory(seed=2))
        ids, D = tracker.pairwise_distances()
        self.assertEqual(D.shape, (3, 3))
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_pairwise_distances_single_agent(self) -> None:
        tracker = self._make_tracker()
        tracker.update("a", _make_trajectory(seed=0))
        ids, D = tracker.pairwise_distances()
        self.assertEqual(D.shape, (1, 1))

    def test_repr_contains_pool_size(self) -> None:
        tracker = self._make_tracker()
        tracker.update("a", _make_trajectory(seed=0))
        self.assertIn("pool_size=1", repr(tracker))

    def test_diversity_score_increases_with_more_diverse_agents(self) -> None:
        """Adding more diverse agents should not decrease diversity."""
        tracker = self._make_tracker()
        # Two identical agents — zero diversity.
        base_traj = _make_trajectory(seed=0)
        tracker.update_embedding("a", np.array([1.0, 0.0, 0.0]))
        tracker.update_embedding("b", np.array([1.0, 0.0, 0.0]))
        score_low = tracker.diversity_score()

        # Now replace one with an orthogonal agent.
        tracker.update_embedding("b", np.array([0.0, 1.0, 0.0]))
        score_high = tracker.diversity_score()
        self.assertGreater(score_high, score_low)

    def test_subset_diversity_score(self) -> None:
        """diversity_score can be limited to a subset of agents."""
        tracker = self._make_tracker()
        tracker.update_embedding("a", np.array([1.0, 0.0]))
        tracker.update_embedding("b", np.array([0.0, 1.0]))
        tracker.update_embedding("c", np.array([1.0, 0.0]))  # same as a
        score_all = tracker.diversity_score()
        score_ab = tracker.diversity_score(["a", "b"])
        # a vs b = 1.0; mean({a,b},{a,c},{b,c}) = mean(1.0, 0.0, 1.0) = 0.667
        self.assertAlmostEqual(score_ab, 1.0, places=5)
        self.assertLess(score_all, 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# Integration: embed → track → score
# ---------------------------------------------------------------------------


class TestDiversityEndToEnd(unittest.TestCase):
    """End-to-end: generate trajectories, embed, compute diversity."""

    def test_full_pipeline_three_agents(self) -> None:
        tracker = DiversityTracker()

        # Agent A: moves a lot (large positive actions)
        tracker.update("a", _make_trajectory(action_bias=3.0, seed=10))
        # Agent B: stays still (near-zero actions)
        tracker.update("b", _make_trajectory(action_bias=0.0, seed=20))
        # Agent C: moves in opposite direction (large negative actions)
        tracker.update("c", _make_trajectory(action_bias=-3.0, seed=30))

        score = tracker.diversity_score()
        # Three behaviourally different agents should yield diversity > 0.
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 2.0)  # cosine distance upper bound

    def test_identical_agents_zero_diversity(self) -> None:
        rng = np.random.default_rng(0)
        traj = TrajectoryBatch(
            actions=rng.random((40, 2)).astype(np.float32),
            positions=rng.random((40, 2)).astype(np.float32),
        )
        tracker = DiversityTracker()
        tracker.update("a", traj)
        tracker.update("b", traj)  # same trajectory
        score = tracker.diversity_score()
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_diversity_exported_from_package(self) -> None:
        """diversity module symbols are accessible via training.league package."""
        from training.league import (
            DiversityTracker,
            TrajectoryBatch,
            diversity_score,
            embed_trajectory,
            pairwise_cosine_distances,
        )
        # Basic smoke test.
        traj = _make_trajectory()
        emb = embed_trajectory(traj)
        self.assertEqual(emb.ndim, 1)


if __name__ == "__main__":
    unittest.main()
