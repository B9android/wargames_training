# tests/test_self_play.py
"""Tests for training/self_play.py and the red_policy integration in BattalionEnv."""

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

from envs.battalion_env import BattalionEnv, RedPolicy
from training.self_play import (
    OpponentPool,
    SelfPlayCallback,
    WinRateVsPoolCallback,
    _iter_envs,
    _max_version_in_pool,
    evaluate_vs_pool,
)


# ---------------------------------------------------------------------------
# Dummy policy helpers
# ---------------------------------------------------------------------------


class _ConstantPolicy:
    """Minimal RedPolicy that always returns the same action."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = action.astype(np.float32)

    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, Any]:
        return self._action.copy(), None


_FORWARD_ACTION = np.array([1.0, 0.0, 1.0], dtype=np.float32)
_IDLE_ACTION    = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def _make_env(**kwargs) -> BattalionEnv:
    return BattalionEnv(randomize_terrain=False, **kwargs)


# ---------------------------------------------------------------------------
# RedPolicy Protocol
# ---------------------------------------------------------------------------


class TestRedPolicyProtocol(unittest.TestCase):
    """Verify the RedPolicy Protocol is correctly defined."""

    def test_constant_policy_satisfies_protocol(self) -> None:
        policy = _ConstantPolicy(_IDLE_ACTION)
        self.assertIsInstance(policy, RedPolicy)

    def test_object_without_predict_does_not_satisfy_protocol(self) -> None:
        class NoPredictObj:
            pass

        self.assertNotIsInstance(NoPredictObj(), RedPolicy)


# ---------------------------------------------------------------------------
# BattalionEnv — red_policy integration
# ---------------------------------------------------------------------------


class TestBattalionEnvRedPolicy(unittest.TestCase):
    """Test that BattalionEnv correctly uses a red_policy when supplied."""

    def test_default_red_policy_is_none(self) -> None:
        env = _make_env()
        self.assertIsNone(env.red_policy)
        env.close()

    def test_red_policy_set_at_init(self) -> None:
        policy = _ConstantPolicy(_IDLE_ACTION)
        env = _make_env(red_policy=policy)
        self.assertIs(env.red_policy, policy)
        env.close()

    def test_set_red_policy_at_runtime(self) -> None:
        env = _make_env()
        policy = _ConstantPolicy(_IDLE_ACTION)
        env.set_red_policy(policy)
        self.assertIs(env.red_policy, policy)
        env.close()

    def test_set_red_policy_to_none_reverts_to_scripted(self) -> None:
        policy = _ConstantPolicy(_IDLE_ACTION)
        env = _make_env(red_policy=policy)
        env.set_red_policy(None)
        self.assertIsNone(env.red_policy)
        env.close()

    def test_episode_runs_with_policy_red(self) -> None:
        """A full episode should complete without error when red_policy is set."""
        policy = _ConstantPolicy(_FORWARD_ACTION)
        env = _make_env(red_policy=policy, max_steps=50)
        obs, _ = env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        self.assertGreater(steps, 0)
        env.close()

    def test_obs_shape_unchanged_with_policy_red(self) -> None:
        policy = _ConstantPolicy(_IDLE_ACTION)
        env = _make_env(red_policy=policy)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape, (17,))
        env.close()

    def test_info_keys_present_with_policy_red(self) -> None:
        policy = _ConstantPolicy(_IDLE_ACTION)
        env = _make_env(red_policy=policy, max_steps=5)
        obs, _ = env.reset(seed=2)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertIn("blue_damage_dealt", info)
        self.assertIn("red_damage_dealt", info)
        env.close()

    def test_get_red_obs_shape(self) -> None:
        """_get_red_obs should return a (17,) array within observation bounds."""
        env = _make_env()
        env.reset(seed=3)
        red_obs = env._get_red_obs()
        self.assertEqual(red_obs.shape, (17,))
        self.assertTrue(np.all(red_obs >= env.observation_space.low))
        self.assertTrue(np.all(red_obs <= env.observation_space.high))
        env.close()

    def test_policy_predict_called_each_step(self) -> None:
        """red_policy.predict should be called once per step."""
        policy = _ConstantPolicy(_IDLE_ACTION)
        predict_spy = MagicMock(side_effect=policy.predict)
        policy.predict = predict_spy

        env = _make_env(red_policy=policy, max_steps=5)
        env.reset(seed=4)
        for _ in range(5):
            env.step(env.action_space.sample())
        self.assertEqual(predict_spy.call_count, 5)
        env.close()

    def test_scripted_opponent_still_works_without_policy(self) -> None:
        """Default env (no red_policy) should still run without error."""
        env = _make_env(max_steps=20)
        env.reset(seed=5)
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        env.close()


# ---------------------------------------------------------------------------
# OpponentPool
# ---------------------------------------------------------------------------


class TestOpponentPool(unittest.TestCase):
    """Unit tests for OpponentPool."""

    def test_max_size_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                OpponentPool(tmpdir, max_size=0)

    def test_empty_pool_size_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            self.assertEqual(pool.size, 0)

    def test_sample_empty_pool_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            self.assertIsNone(pool.sample())

    def test_sample_latest_empty_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            self.assertIsNone(pool.sample_latest())

    def test_add_increments_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            model = _make_dummy_model(env)
            pool = OpponentPool(tmpdir, max_size=5)
            pool.add(model, version=1)
            self.assertEqual(pool.size, 1)
            env.close()

    def test_add_saves_zip_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            model = _make_dummy_model(env)
            pool = OpponentPool(tmpdir, max_size=5)
            saved_path = pool.add(model, version=1)
            self.assertTrue(saved_path.exists())
            env.close()

    def test_eviction_when_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            pool = OpponentPool(tmpdir, max_size=2)
            for v in range(1, 5):
                model = _make_dummy_model(env)
                pool.add(model, version=v)
            self.assertEqual(pool.size, 2)
            env.close()

    def test_eviction_removes_oldest_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            pool = OpponentPool(tmpdir, max_size=2)
            paths = []
            for v in range(1, 4):
                model = _make_dummy_model(env)
                paths.append(pool.add(model, version=v))
            # First snapshot should be evicted.
            self.assertFalse(paths[0].exists())
            # Second and third should exist.
            self.assertTrue(paths[1].exists())
            self.assertTrue(paths[2].exists())
            env.close()

    def test_sample_returns_ppo_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            model = _make_dummy_model(env)
            pool = OpponentPool(tmpdir, max_size=5)
            pool.add(model, version=1)
            sampled = pool.sample(rng=np.random.default_rng(0))
            from stable_baselines3 import PPO
            self.assertIsInstance(sampled, PPO)
            env.close()

    def test_sample_latest_returns_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            pool = OpponentPool(tmpdir, max_size=5)
            for v in range(1, 4):
                model = _make_dummy_model(env)
                pool.add(model, version=v)
            latest = pool.sample_latest()
            from stable_baselines3 import PPO
            self.assertIsInstance(latest, PPO)
            env.close()

    def test_reload_from_disk(self) -> None:
        """Pool should restore snapshots saved by a previous pool instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            pool1 = OpponentPool(tmpdir, max_size=5)
            for v in range(1, 4):
                model = _make_dummy_model(env)
                pool1.add(model, version=v)

            pool2 = OpponentPool(tmpdir, max_size=5)
            self.assertEqual(pool2.size, 3)
            env.close()

    def test_reload_deletes_excess_files_from_disk(self) -> None:
        """When pool_dir has more files than max_size, excess are deleted on reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            # First pool writes 5 snapshots.
            pool1 = OpponentPool(tmpdir, max_size=5)
            paths = []
            for v in range(1, 6):
                model = _make_dummy_model(env)
                paths.append(pool1.add(model, version=v))

            # Second pool with smaller max_size=2 should delete excess files.
            pool2 = OpponentPool(tmpdir, max_size=2)
            self.assertEqual(pool2.size, 2)
            # Oldest 3 should be deleted from disk.
            for p in paths[:3]:
                self.assertFalse(p.exists(), f"{p} should have been deleted")
            # Newest 2 should still exist.
            for p in paths[3:]:
                self.assertTrue(p.exists(), f"{p} should still exist")
            env.close()

    def test_snapshot_paths_property(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            pool = OpponentPool(tmpdir, max_size=5)
            for v in range(1, 3):
                model = _make_dummy_model(env)
                pool.add(model, version=v)
            paths = pool.snapshot_paths
            self.assertEqual(len(paths), 2)
            # Should be a copy, not the internal list.
            paths.clear()
            self.assertEqual(pool.size, 2)
            env.close()


# ---------------------------------------------------------------------------
# evaluate_vs_pool
# ---------------------------------------------------------------------------


class TestEvaluateVsPool(unittest.TestCase):
    """Tests for the evaluate_vs_pool standalone function."""

    def test_invalid_n_episodes_raises(self) -> None:
        env = _make_env()
        model = _make_dummy_model(env)
        with self.assertRaises(ValueError):
            evaluate_vs_pool(model, model, n_episodes=0)
        env.close()

    def test_returns_float_in_unit_interval(self) -> None:
        env = _make_env()
        model = _make_dummy_model(env)
        win_rate = evaluate_vs_pool(model, model, n_episodes=3, seed=0)
        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)
        env.close()

    def test_seeded_deterministic_run_is_reproducible(self) -> None:
        env = _make_env()
        model = _make_dummy_model(env)
        r1 = evaluate_vs_pool(model, model, n_episodes=5, seed=42, deterministic=True)
        r2 = evaluate_vs_pool(model, model, n_episodes=5, seed=42, deterministic=True)
        self.assertAlmostEqual(r1, r2)
        env.close()


# ---------------------------------------------------------------------------
# SelfPlayCallback
# ---------------------------------------------------------------------------


class TestSelfPlayCallback(unittest.TestCase):
    """Light-weight tests for SelfPlayCallback."""

    def test_invalid_snapshot_freq_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            with self.assertRaises(ValueError):
                SelfPlayCallback(pool=pool, snapshot_freq=0)

    def test_negative_snapshot_freq_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            with self.assertRaises(ValueError):
                SelfPlayCallback(pool=pool, snapshot_freq=-1)

    def test_version_initialized_from_empty_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            cb = SelfPlayCallback(pool=pool, snapshot_freq=100)
            self.assertEqual(cb._version, 0)

    def test_version_initialized_from_existing_pool(self) -> None:
        """_version should start from the max existing snapshot version on restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            pool1 = OpponentPool(tmpdir, max_size=5)
            for v in range(1, 4):
                model = _make_dummy_model(env)
                pool1.add(model, version=v)

            pool2 = OpponentPool(tmpdir, max_size=5)
            cb = SelfPlayCallback(pool=pool2, snapshot_freq=100)
            self.assertEqual(cb._version, 3)
            env.close()

    def test_snapshot_not_taken_at_step_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            cb = SelfPlayCallback(pool=pool, snapshot_freq=100)
            cb.num_timesteps = 0
            cb.locals = {}
            cb._on_step()
            self.assertEqual(pool.size, 0)

    def test_snapshot_taken_at_freq(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            model = _make_dummy_model(env)
            pool = OpponentPool(tmpdir, max_size=5)
            # Pass None as vec_env so the callback skips policy injection.
            cb = SelfPlayCallback(pool=pool, snapshot_freq=100, vec_env=None, verbose=0)
            # Simulate the SB3 callback wiring via the internal _locals dict.
            cb.model = model
            cb.num_timesteps = 100
            cb.locals = {}
            # Call the snapshot method directly (bypasses _on_step guards).
            cb._take_snapshot_and_update()
            self.assertEqual(pool.size, 1)
            env.close()

    def test_snapshot_registered_in_manifest(self) -> None:
        """Snapshot should be indexed in the manifest immediately when written."""
        from training.artifacts import CheckpointManifest

        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            model = _make_dummy_model(env)
            pool = OpponentPool(tmpdir, max_size=5)
            manifest = CheckpointManifest(Path(tmpdir) / "manifest.jsonl")

            cb = SelfPlayCallback(
                pool=pool,
                snapshot_freq=100,
                vec_env=None,
                manifest=manifest,
                seed=7,
                curriculum_level=3,
                run_id="run-sp",
                config_hash="hash-sp",
            )
            cb.model = model
            cb.num_timesteps = 100
            cb.locals = {}
            cb._take_snapshot_and_update()

            rows = manifest._read_rows()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["type"], "self_play_snapshot")
            self.assertEqual(row["seed"], 7)
            self.assertEqual(row["curriculum_level"], 3)
            self.assertEqual(row["step"], 100)
            self.assertEqual(row["run_id"], "run-sp")
            env.close()

    def test_snapshot_no_manifest_still_works(self) -> None:
        """SelfPlayCallback with no manifest should behave identically to before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            model = _make_dummy_model(env)
            pool = OpponentPool(tmpdir, max_size=5)
            cb = SelfPlayCallback(pool=pool, snapshot_freq=50, vec_env=None)
            cb.model = model
            cb.num_timesteps = 50
            cb.locals = {}
            cb._take_snapshot_and_update()
            self.assertEqual(pool.size, 1)
            env.close()


# ---------------------------------------------------------------------------
# WinRateVsPoolCallback
# ---------------------------------------------------------------------------


class TestWinRateVsPoolCallback(unittest.TestCase):
    """Light-weight tests for WinRateVsPoolCallback."""

    def test_invalid_eval_freq_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            with self.assertRaises(ValueError):
                WinRateVsPoolCallback(pool=pool, eval_freq=0)

    def test_negative_eval_freq_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            with self.assertRaises(ValueError):
                WinRateVsPoolCallback(pool=pool, eval_freq=-5)

    def test_invalid_n_eval_episodes_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            with self.assertRaises(ValueError):
                WinRateVsPoolCallback(pool=pool, n_eval_episodes=0)

    def test_no_eval_when_pool_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            cb = WinRateVsPoolCallback(pool=pool, eval_freq=100, verbose=0)
            env = _make_env()
            model = _make_dummy_model(env)
            cb.model = model
            cb.num_timesteps = 100
            cb.locals = {}
            # Should be a no-op; verify by confirming no exception is raised and
            # no wandb logging would occur (pool is empty).
            cb._evaluate()  # Must not raise.
            env.close()

    def test_eval_runs_with_pool_snapshot(self) -> None:
        """_evaluate should call evaluate_vs_pool and produce a valid win rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            model = _make_dummy_model(env)
            pool = OpponentPool(tmpdir, max_size=5)
            pool.add(model, version=1)

            cb = WinRateVsPoolCallback(
                pool=pool, eval_freq=100, n_eval_episodes=2, verbose=0
            )

            # Mock evaluate_vs_pool to avoid a full episode run and capture result.
            with patch("training.self_play.evaluate_vs_pool", return_value=0.5) as mock_eval:
                cb.model = model
                cb.num_timesteps = 100
                # Initialise the SB3 logger.
                from stable_baselines3.common.logger import configure as sb3_configure
                model.set_logger(sb3_configure(None, ["stdout"]))
                cb._evaluate()
                mock_eval.assert_called_once()
                # Verify the call was made with a PPO model as the opponent argument.
                call_kwargs = mock_eval.call_args.kwargs
                from stable_baselines3 import PPO as _PPO
                opponent_arg = (
                    call_kwargs.get("opponent")
                    if "opponent" in call_kwargs
                    else mock_eval.call_args.args[1]
                )
                self.assertIsInstance(opponent_arg, _PPO)

            env.close()


# ---------------------------------------------------------------------------
# _iter_envs helper
# ---------------------------------------------------------------------------


class TestIterEnvs(unittest.TestCase):
    def test_yields_battalion_env_instances(self) -> None:
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(
            lambda: _make_env(), n_envs=2
        )
        envs = list(_iter_envs(vec_env))
        self.assertEqual(len(envs), 2)
        for e in envs:
            self.assertIsInstance(e, BattalionEnv)
        vec_env.close()


# ---------------------------------------------------------------------------
# _max_version_in_pool helper
# ---------------------------------------------------------------------------


class TestMaxVersionInPool(unittest.TestCase):
    def test_empty_pool_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(tmpdir, max_size=5)
            self.assertEqual(_max_version_in_pool(pool), 0)

    def test_returns_max_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env()
            pool = OpponentPool(tmpdir, max_size=5)
            for v in [1, 5, 3]:
                model = _make_dummy_model(env)
                pool.add(model, version=v)
            self.assertEqual(_max_version_in_pool(pool), 5)
            env.close()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _make_dummy_model(env: BattalionEnv):
    """Create a minimal PPO model for testing."""
    from stable_baselines3 import PPO

    return PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)


if __name__ == "__main__":
    unittest.main()
