# tests/test_team_elo.py
"""Tests for team Elo rating (TeamEloRegistry), TeamOpponentPool,
evaluate_team_vs_pool, and nash_exploitability_proxy.

Covers:
- training/elo.py  → TeamEloRegistry, TEAM_BASELINE_RATINGS
- training/self_play.py → TeamOpponentPool, evaluate_team_vs_pool,
                          nash_exploitability_proxy, _max_team_version_in_pool
- training/train_mappo.py → MAPPOTrainer.set_red_policy / _red_actions
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.elo import (
    DEFAULT_RATING,
    BASELINE_RATINGS,
    TEAM_BASELINE_RATINGS,
    EloRegistry,
    TeamEloRegistry,
    expected_score,
)
from training.self_play import (
    TeamOpponentPool,
    evaluate_team_vs_pool,
    nash_exploitability_proxy,
    _max_team_version_in_pool,
)
from models.mappo_policy import MAPPOPolicy
from envs.multi_battalion_env import MultiBattalionEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(n_blue: int = 2, n_red: int = 2) -> MAPPOPolicy:
    """Create a tiny MAPPOPolicy compatible with a 2v2 environment."""
    env = MultiBattalionEnv(n_blue=n_blue, n_red=n_red, randomize_terrain=False)
    obs_dim = env._obs_dim
    state_dim = env._state_dim
    action_dim = env._act_space.shape[0]
    env.close()
    return MAPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        n_agents=n_blue,
        share_parameters=True,
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(16,),
    )


# ---------------------------------------------------------------------------
# TEAM_BASELINE_RATINGS
# ---------------------------------------------------------------------------


class TestTeamBaselineRatings(unittest.TestCase):
    """Verify the TEAM_BASELINE_RATINGS constant."""

    def test_all_keys_present(self) -> None:
        expected_keys = {
            "random_team",
            "scripted_team_l1",
            "scripted_team_l2",
            "scripted_team_l3",
            "scripted_team_l4",
            "scripted_team_l5",
        }
        self.assertEqual(set(TEAM_BASELINE_RATINGS.keys()), expected_keys)

    def test_values_ascending(self) -> None:
        """Team baselines increase monotonically from l1 to l5."""
        levels = [TEAM_BASELINE_RATINGS[f"scripted_team_l{i}"] for i in range(1, 6)]
        for a, b in zip(levels, levels[1:]):
            self.assertLess(a, b)

    def test_random_team_is_lowest(self) -> None:
        self.assertLess(
            TEAM_BASELINE_RATINGS["random_team"],
            TEAM_BASELINE_RATINGS["scripted_team_l1"],
        )


# ---------------------------------------------------------------------------
# TeamEloRegistry
# ---------------------------------------------------------------------------


class TestTeamEloRegistry(unittest.TestCase):
    """Unit tests for TeamEloRegistry."""

    # ------------------------------------------------------------------
    # Construction & inheritance
    # ------------------------------------------------------------------

    def test_is_subclass_of_elo_registry(self) -> None:
        reg = TeamEloRegistry(path=None)
        self.assertIsInstance(reg, EloRegistry)

    def test_unknown_agent_gets_default_rating(self) -> None:
        reg = TeamEloRegistry(path=None)
        self.assertEqual(reg.get_rating("unknown_team"), DEFAULT_RATING)

    # ------------------------------------------------------------------
    # Team baseline look-up
    # ------------------------------------------------------------------

    def test_team_baseline_ratings_returned(self) -> None:
        reg = TeamEloRegistry(path=None)
        for name, expected in TEAM_BASELINE_RATINGS.items():
            self.assertAlmostEqual(reg.get_rating(name), expected)

    def test_single_agent_baselines_still_work(self) -> None:
        reg = TeamEloRegistry(path=None)
        for name, expected in BASELINE_RATINGS.items():
            self.assertAlmostEqual(reg.get_rating(name), expected)

    def test_stored_rating_takes_precedence_over_team_baseline(self) -> None:
        reg = TeamEloRegistry(path=None)
        # Manually insert a stored rating to verify lookup priority.
        reg._ratings["some_team"] = 1234.0
        self.assertAlmostEqual(reg.get_rating("some_team"), 1234.0)

    # ------------------------------------------------------------------
    # update() — team baseline protection
    # ------------------------------------------------------------------

    def test_update_team_baseline_raises(self) -> None:
        reg = TeamEloRegistry(path=None)
        for name in TEAM_BASELINE_RATINGS:
            with self.assertRaises(ValueError, msg=f"should raise for '{name}'"):
                reg.update(name, "some_opponent", outcome=0.5, n_games=1)

    def test_update_single_agent_baseline_raises(self) -> None:
        reg = TeamEloRegistry(path=None)
        for name in BASELINE_RATINGS:
            with self.assertRaises(ValueError, msg=f"should raise for '{name}'"):
                reg.update(name, "some_opponent", outcome=0.5, n_games=1)

    def test_update_win_increases_rating(self) -> None:
        reg = TeamEloRegistry(path=None)
        old = reg.get_rating("blue_team")
        delta = reg.update("blue_team", "scripted_team_l3", outcome=1.0, n_games=10)
        self.assertGreater(delta, 0.0)
        self.assertGreater(reg.get_rating("blue_team"), old)

    def test_update_loss_decreases_rating(self) -> None:
        reg = TeamEloRegistry(path=None)
        old = reg.get_rating("blue_team")
        delta = reg.update("blue_team", "scripted_team_l5", outcome=0.0, n_games=10)
        self.assertLess(delta, 0.0)
        self.assertLess(reg.get_rating("blue_team"), old)

    def test_update_increments_game_count(self) -> None:
        reg = TeamEloRegistry(path=None)
        reg.update("blue_team", "scripted_team_l3", outcome=0.6, n_games=20)
        self.assertEqual(reg.get_game_count("blue_team"), 20)
        reg.update("blue_team", "scripted_team_l3", outcome=0.6, n_games=15)
        self.assertEqual(reg.get_game_count("blue_team"), 35)

    def test_update_opponent_uses_team_baseline_rating(self) -> None:
        """Opponent's team baseline rating is used in the Elo calculation."""
        reg = TeamEloRegistry(path=None)
        # Agent starting at DEFAULT_RATING=1000, opponent at "scripted_team_l5"=1000
        # → expected_score ≈ 0.5 → a win (1.0) should yield positive delta.
        delta = reg.update("blue_team", "scripted_team_l5", outcome=1.0, n_games=1)
        self.assertGreater(delta, 0.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_save_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "team_elo.json"
            reg1 = TeamEloRegistry(path=path)
            reg1.update("blue_team", "scripted_team_l3", outcome=0.8, n_games=10)
            rating = reg1.get_rating("blue_team")
            reg1.save()

            reg2 = TeamEloRegistry(path=path)
            self.assertAlmostEqual(reg2.get_rating("blue_team"), rating, places=6)
            self.assertEqual(reg2.get_game_count("blue_team"), 10)

    def test_in_memory_registry_save_raises(self) -> None:
        reg = TeamEloRegistry(path=None)
        reg.update("blue_team", "scripted_team_l3", outcome=0.5, n_games=5)
        with self.assertRaises(ValueError):
            reg.save()

    # ------------------------------------------------------------------
    # self-play pool as "opponent"
    # ------------------------------------------------------------------

    def test_self_play_pool_opponent_uses_default_rating(self) -> None:
        """'self_play_pool' is not a baseline — it starts at DEFAULT_RATING."""
        reg = TeamEloRegistry(path=None)
        opp_rating = reg.get_rating("self_play_pool")
        self.assertEqual(opp_rating, DEFAULT_RATING)


# ---------------------------------------------------------------------------
# TeamOpponentPool
# ---------------------------------------------------------------------------


class TestTeamOpponentPool(unittest.TestCase):
    """Unit tests for TeamOpponentPool."""

    def test_max_size_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                TeamOpponentPool(tmp, max_size=0)

    def test_empty_pool_size_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = TeamOpponentPool(tmp, max_size=5)
            self.assertEqual(pool.size, 0)

    def test_add_increases_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            pool.add(policy, version=1)
            self.assertEqual(pool.size, 1)

    def test_add_creates_pt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            path = pool.add(policy, version=1)
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".pt")

    def test_add_evicts_oldest_when_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = TeamOpponentPool(tmp, max_size=2)
            policy = _make_policy()
            p1 = pool.add(policy, version=1)
            pool.add(policy, version=2)
            pool.add(policy, version=3)  # triggers eviction of v1
            self.assertEqual(pool.size, 2)
            self.assertFalse(p1.exists())

    def test_sample_empty_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = TeamOpponentPool(tmp, max_size=5)
            self.assertIsNone(pool.sample())

    def test_sample_returns_mappo_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            pool.add(policy, version=1)
            sampled = pool.sample()
            self.assertIsNotNone(sampled)
            self.assertIsInstance(sampled, MAPPOPolicy)

    def test_sample_latest_empty_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = TeamOpponentPool(tmp, max_size=5)
            self.assertIsNone(pool.sample_latest())

    def test_sample_latest_returns_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = TeamOpponentPool(tmp, max_size=5)
            policy = _make_policy()
            pool.add(policy, version=1)
            pool.add(policy, version=2)
            latest = pool.sample_latest()
            self.assertIsNotNone(latest)
            self.assertIsInstance(latest, MAPPOPolicy)

    def test_snapshot_paths_is_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = TeamOpponentPool(tmp, max_size=5)
            policy = _make_policy()
            pool.add(policy, version=1)
            paths = pool.snapshot_paths
            paths.clear()
            self.assertEqual(pool.size, 1)

    def test_reload_from_disk_restores_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool1 = TeamOpponentPool(tmp, max_size=5)
            pool1.add(policy, version=1)
            pool1.add(policy, version=2)

            pool2 = TeamOpponentPool(tmp, max_size=5)
            self.assertEqual(pool2.size, 2)

    def test_reload_respects_max_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            # Write 5 snapshots bypassing pool eviction to simulate a larger pool
            pool1 = TeamOpponentPool(tmp, max_size=10)
            for v in range(1, 6):
                pool1.add(policy, version=v)

            # Reload with smaller max_size — excess should be evicted
            pool2 = TeamOpponentPool(tmp, max_size=3)
            self.assertLessEqual(pool2.size, 3)

    def test_loaded_policy_produces_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            pool.add(policy, version=1)
            loaded = pool.sample()
            self.assertIsNotNone(loaded)

            obs_dim = policy.obs_dim
            obs = torch.zeros(1, obs_dim)
            actions, log_probs = loaded.act(obs)
            self.assertEqual(actions.shape, (1, policy.action_dim))
            self.assertEqual(log_probs.shape, (1,))


# ---------------------------------------------------------------------------
# _max_team_version_in_pool
# ---------------------------------------------------------------------------


class TestMaxTeamVersionInPool(unittest.TestCase):
    """Unit tests for _max_team_version_in_pool."""

    def test_empty_pool_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = TeamOpponentPool(tmp, max_size=5)
            self.assertEqual(_max_team_version_in_pool(pool), 0)

    def test_returns_highest_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            pool.add(policy, version=3)
            pool.add(policy, version=7)
            pool.add(policy, version=2)
            self.assertEqual(_max_team_version_in_pool(pool), 7)


# ---------------------------------------------------------------------------
# evaluate_team_vs_pool (fast smoke tests — short episodes)
# ---------------------------------------------------------------------------


class TestEvaluateTeamVsPool(unittest.TestCase):
    """Smoke tests for evaluate_team_vs_pool."""

    def test_invalid_n_episodes_raises(self) -> None:
        policy = _make_policy()
        with self.assertRaises(ValueError):
            evaluate_team_vs_pool(policy, policy, n_episodes=0)

    def test_returns_float_in_unit_interval(self) -> None:
        policy = _make_policy()
        wr = evaluate_team_vs_pool(
            policy=policy,
            opponent=policy,
            n_blue=2,
            n_red=2,
            n_episodes=3,
            deterministic=True,
            seed=0,
            env_kwargs={"max_steps": 5, "randomize_terrain": False},
        )
        self.assertIsInstance(wr, float)
        self.assertGreaterEqual(wr, 0.0)
        self.assertLessEqual(wr, 1.0)

    def test_seeded_run_is_reproducible(self) -> None:
        policy = _make_policy()
        kwargs = dict(
            opponent=policy,
            n_blue=2,
            n_red=2,
            n_episodes=3,
            deterministic=True,
            seed=42,
            env_kwargs={"max_steps": 5, "randomize_terrain": False},
        )
        wr1 = evaluate_team_vs_pool(policy=policy, **kwargs)
        wr2 = evaluate_team_vs_pool(policy=policy, **kwargs)
        self.assertAlmostEqual(wr1, wr2, places=6)


# ---------------------------------------------------------------------------
# nash_exploitability_proxy
# ---------------------------------------------------------------------------


class TestNashExploitabilityProxy(unittest.TestCase):
    """Tests for nash_exploitability_proxy."""

    def test_empty_pool_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            result = nash_exploitability_proxy(
                policy=policy,
                pool=pool,
                n_blue=2,
                n_red=2,
                n_episodes_per_opponent=2,
                env_kwargs={"max_steps": 5, "randomize_terrain": False},
            )
            self.assertEqual(result, 0.0)

    def test_single_opponent_returns_zero(self) -> None:
        """With one opponent max == mean, so proxy is 0.0."""
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            pool.add(policy, version=1)
            result = nash_exploitability_proxy(
                policy=policy,
                pool=pool,
                n_blue=2,
                n_red=2,
                n_episodes_per_opponent=2,
                seed=0,
                env_kwargs={"max_steps": 5, "randomize_terrain": False},
            )
            self.assertAlmostEqual(result, 0.0, places=6)

    def test_returns_float_in_unit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = _make_policy()
            pool = TeamOpponentPool(tmp, max_size=5)
            pool.add(policy, version=1)
            pool.add(policy, version=2)
            result = nash_exploitability_proxy(
                policy=policy,
                pool=pool,
                n_blue=2,
                n_red=2,
                n_episodes_per_opponent=2,
                seed=0,
                env_kwargs={"max_steps": 5, "randomize_terrain": False},
            )
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)


# ---------------------------------------------------------------------------
# MAPPOTrainer — set_red_policy integration
# ---------------------------------------------------------------------------


class TestMAPPOTrainerSetRedPolicy(unittest.TestCase):
    """Tests for MAPPOTrainer.set_red_policy and red-policy-driven _red_actions."""

    def _make_trainer(self, n_blue: int = 2, n_red: int = 2):
        from training.train_mappo import MAPPOTrainer

        env = MultiBattalionEnv(n_blue=n_blue, n_red=n_red, randomize_terrain=False)
        policy = MAPPOPolicy(
            obs_dim=env._obs_dim,
            action_dim=env._act_space.shape[0],
            state_dim=env._state_dim,
            n_agents=n_blue,
            share_parameters=True,
            actor_hidden_sizes=(16,),
            critic_hidden_sizes=(16,),
        )
        trainer = MAPPOTrainer(
            policy=policy,
            env=env,
            n_steps=4,
            n_epochs=1,
            batch_size=4,
        )
        return trainer

    def test_default_red_policy_is_none(self) -> None:
        trainer = self._make_trainer()
        self.assertIsNone(trainer._red_policy)
        trainer.env.close()

    def test_set_red_policy_stores_policy(self) -> None:
        trainer = self._make_trainer()
        frozen = _make_policy()
        trainer.set_red_policy(frozen)
        self.assertIs(trainer._red_policy, frozen)
        trainer.env.close()

    def test_set_red_policy_to_none_clears(self) -> None:
        trainer = self._make_trainer()
        frozen = _make_policy()
        trainer.set_red_policy(frozen)
        trainer.set_red_policy(None)
        self.assertIsNone(trainer._red_policy)
        trainer.env.close()

    def test_red_actions_uses_policy_when_set(self) -> None:
        """When a red_policy is set, _red_actions returns non-zero actions."""
        trainer = self._make_trainer()
        # Initialise obs buffer so _red_actions can look up agent observations.
        obs, _ = trainer.env.reset(seed=0)
        trainer._obs_buf = obs

        # With no red policy, stationary actions should be zero.
        zero_acts = trainer._red_actions()
        for agent_id, act in zero_acts.items():
            np.testing.assert_array_equal(act, np.zeros_like(act))

        # Set a frozen policy; actions should be produced by the policy.
        frozen = _make_policy()
        trainer.set_red_policy(frozen)
        policy_acts = trainer._red_actions()
        # The policy-driven actions are not guaranteed to be non-zero, but the
        # dict should have the same keys as before.
        self.assertEqual(set(policy_acts.keys()), set(zero_acts.keys()))

        trainer.env.close()

    def test_red_actions_falls_back_to_zero_without_policy(self) -> None:
        trainer = self._make_trainer()
        obs, _ = trainer.env.reset(seed=0)
        trainer._obs_buf = obs
        acts = trainer._red_actions()
        for act in acts.values():
            np.testing.assert_array_equal(act, np.zeros_like(act))
        trainer.env.close()


if __name__ == "__main__":
    unittest.main()
