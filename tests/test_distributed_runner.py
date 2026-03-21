# tests/test_distributed_runner.py
"""Tests for E4.7 — distributed rollout runner (Ray actor pool).

Covers:
- RemoteMultiBattalionEnv actor construction and basic interaction.
- DistributedRolloutRunner: collect_rollouts(), win_rate(), steps_per_sec(),
  mean_episode_length(), shutdown().
- Smoke test: 2-worker Ray cluster collects 4 episodes end-to-end.
- RolloutResult data container.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ray

from envs.remote_multi_battalion_env import RemoteMultiBattalionEnv, make_remote_envs
from training.league.distributed_runner import (
    DistributedRolloutRunner,
    RolloutResult,
    benchmark,
)


# ---------------------------------------------------------------------------
# Module-level Ray initialisation
# ---------------------------------------------------------------------------

def setUpModule() -> None:  # noqa: N802
    """Initialise a local Ray instance once for the whole test module."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True, log_to_driver=False)


def tearDownModule() -> None:  # noqa: N802
    """Shut down Ray after all tests in this module complete."""
    if ray.is_initialized():
        ray.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(num_workers: int = 2, **env_kwargs: Any) -> DistributedRolloutRunner:
    """Return a DistributedRolloutRunner with sensible defaults for testing."""
    defaults: Dict[str, Any] = {"n_blue": 1, "n_red": 1, "max_steps": 20}
    defaults.update(env_kwargs)
    return DistributedRolloutRunner(
        num_workers=num_workers,
        env_kwargs=defaults,
        num_cpus_per_worker=0.5,  # fractional so tests fit on 2-core CI runners
    )


def _fake_result(steps: int = 10, reward: float = 1.0, won: bool = True) -> RolloutResult:
    return RolloutResult(
        worker_id=0,
        episode_steps=steps,
        total_reward=reward,
        agent_rewards={"blue_0": reward},
        blue_won=won,
        observations={"blue_0": np.zeros(5)},
        elapsed_sec=1.0,
    )


# ---------------------------------------------------------------------------
# 1. RolloutResult
# ---------------------------------------------------------------------------


class TestRolloutResult(unittest.TestCase):
    """Test the RolloutResult data container."""

    def test_construction_defaults(self) -> None:
        r = _fake_result()
        self.assertEqual(r.worker_id, 0)
        self.assertEqual(r.episode_steps, 10)
        self.assertAlmostEqual(r.total_reward, 1.0)
        self.assertTrue(r.blue_won)
        self.assertEqual(r.metadata, {})

    def test_metadata_stored(self) -> None:
        r = RolloutResult(
            worker_id=1,
            episode_steps=5,
            total_reward=0.0,
            agent_rewards={},
            blue_won=False,
            observations={},
            elapsed_sec=0.5,
            metadata={"opponent": "snap_001"},
        )
        self.assertEqual(r.metadata["opponent"], "snap_001")

    def test_none_metadata_defaults_to_empty_dict(self) -> None:
        r = RolloutResult(0, 1, 0.0, {}, False, {}, 0.1, None)
        self.assertEqual(r.metadata, {})


# ---------------------------------------------------------------------------
# 2. RemoteMultiBattalionEnv actor
# ---------------------------------------------------------------------------


class TestRemoteMultiBattalionEnv(unittest.TestCase):
    """Test the Ray remote environment actor."""

    def setUp(self) -> None:
        self.actor = RemoteMultiBattalionEnv.options(num_cpus=0.5).remote(
            n_blue=1, n_red=1, max_steps=10
        )

    def tearDown(self) -> None:
        ray.kill(self.actor)

    def test_reset_returns_obs_and_infos(self) -> None:
        obs, infos = ray.get(self.actor.reset.remote(seed=0))
        self.assertIsInstance(obs, dict)
        self.assertIn("blue_0", obs)
        self.assertIn("red_0", obs)

    def test_step_returns_five_tuple(self) -> None:
        ray.get(self.actor.reset.remote(seed=0))
        agents = ray.get(self.actor.get_agents.remote())
        # Build zero actions for all agents
        obs_space = ray.get(self.actor.observation_space.remote("blue_0"))
        act_space = ray.get(self.actor.action_space.remote("blue_0"))
        actions = {a: act_space.sample() for a in agents}
        result = ray.get(self.actor.step.remote(actions))
        self.assertEqual(len(result), 5)  # obs, rew, term, trunc, info

    def test_get_n_blue_n_red(self) -> None:
        n_blue = ray.get(self.actor.get_n_blue.remote())
        n_red = ray.get(self.actor.get_n_red.remote())
        self.assertEqual(n_blue, 1)
        self.assertEqual(n_red, 1)

    def test_get_possible_agents(self) -> None:
        agents = ray.get(self.actor.get_possible_agents.remote())
        self.assertIn("blue_0", agents)
        self.assertIn("red_0", agents)

    def test_state_returns_array(self) -> None:
        ray.get(self.actor.reset.remote(seed=0))
        state = ray.get(self.actor.state.remote())
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.ndim, 1)

    def test_run_episode_random_policy(self) -> None:
        result = ray.get(self.actor.run_episode.remote(seed=0))
        self.assertIn("steps", result)
        self.assertIn("blue_won", result)
        self.assertGreater(result["steps"], 0)


# ---------------------------------------------------------------------------
# 3. make_remote_envs
# ---------------------------------------------------------------------------


class TestMakeRemoteEnvs(unittest.TestCase):
    """Test the make_remote_envs factory."""

    def test_creates_correct_number_of_actors(self) -> None:
        envs = make_remote_envs(num_envs=3, num_cpus_per_env=0.5, n_blue=1, n_red=1)
        self.assertEqual(len(envs), 3)
        for e in envs:
            ray.kill(e)

    def test_raises_if_ray_not_initialized(self) -> None:
        # We can't easily un-init Ray within a test; instead verify the guard
        # by patching ray.is_initialized.
        import unittest.mock as mock

        with mock.patch("ray.is_initialized", return_value=False):
            with self.assertRaises(RuntimeError):
                make_remote_envs(num_envs=1, n_blue=1, n_red=1)


# ---------------------------------------------------------------------------
# 4. DistributedRolloutRunner construction
# ---------------------------------------------------------------------------


class TestDistributedRolloutRunnerInit(unittest.TestCase):
    """Test DistributedRolloutRunner construction and validation."""

    def test_creates_correct_number_of_workers(self) -> None:
        runner = _make_runner(num_workers=2)
        self.assertEqual(runner.num_workers, 2)
        runner.shutdown()

    def test_zero_workers_raises(self) -> None:
        with self.assertRaises(ValueError):
            DistributedRolloutRunner(num_workers=0, env_kwargs={"n_blue": 1, "n_red": 1})

    def test_negative_workers_raises(self) -> None:
        with self.assertRaises(ValueError):
            DistributedRolloutRunner(num_workers=-1, env_kwargs={"n_blue": 1, "n_red": 1})

    def test_context_manager(self) -> None:
        with _make_runner(num_workers=1) as runner:
            self.assertEqual(runner.num_workers, 1)
        # No exception means shutdown succeeded.


# ---------------------------------------------------------------------------
# 5. collect_rollouts
# ---------------------------------------------------------------------------


class TestCollectRollouts(unittest.TestCase):
    """Test collect_rollouts() with a live 2-worker pool."""

    def setUp(self) -> None:
        self.runner = _make_runner(num_workers=2)

    def tearDown(self) -> None:
        self.runner.shutdown()

    def test_returns_correct_number_of_results(self) -> None:
        results = self.runner.collect_rollouts(n_episodes=4, base_seed=0)
        self.assertEqual(len(results), 4)

    def test_result_types(self) -> None:
        results = self.runner.collect_rollouts(n_episodes=2, base_seed=0)
        for r in results:
            self.assertIsInstance(r, RolloutResult)

    def test_episode_steps_positive(self) -> None:
        results = self.runner.collect_rollouts(n_episodes=2, base_seed=0)
        for r in results:
            self.assertGreater(r.episode_steps, 0)

    def test_different_seeds_give_results(self) -> None:
        results = self.runner.collect_rollouts(n_episodes=2, base_seed=100)
        self.assertEqual(len(results), 2)

    def test_zero_episodes_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.collect_rollouts(n_episodes=0)

    def test_metadata_propagated(self) -> None:
        meta = [{"opponent": f"snap_{i}"} for i in range(2)]
        results = self.runner.collect_rollouts(n_episodes=2, base_seed=0, metadata=meta)
        for r in results:
            self.assertIn("opponent", r.metadata)

    def test_metadata_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.collect_rollouts(n_episodes=3, metadata=[{"k": 1}])


# ---------------------------------------------------------------------------
# 6. collect_rollouts_async
# ---------------------------------------------------------------------------


class TestCollectRolloutsAsync(unittest.TestCase):
    """Test collect_rollouts_async() returns ObjectRefs."""

    def setUp(self) -> None:
        self.runner = _make_runner(num_workers=2)

    def tearDown(self) -> None:
        self.runner.shutdown()

    def test_returns_object_refs(self) -> None:
        refs = self.runner.collect_rollouts_async(n_episodes=2, base_seed=0)
        self.assertEqual(len(refs), 2)
        results = ray.get(refs)
        for r in results:
            self.assertIsInstance(r, RolloutResult)


# ---------------------------------------------------------------------------
# 7. Statistics helpers
# ---------------------------------------------------------------------------


class TestStatisticsHelpers(unittest.TestCase):
    """Test steps_per_second, win_rate, mean_episode_length."""

    def test_steps_per_second_basic(self) -> None:
        results = [_fake_result(steps=100)]
        sps = DistributedRolloutRunner.steps_per_second(results)
        self.assertAlmostEqual(sps, 100.0)

    def test_steps_per_second_empty(self) -> None:
        self.assertEqual(DistributedRolloutRunner.steps_per_second([]), 0.0)

    def test_win_rate_all_wins(self) -> None:
        results = [_fake_result(won=True)] * 4
        self.assertAlmostEqual(DistributedRolloutRunner.win_rate(results), 1.0)

    def test_win_rate_no_wins(self) -> None:
        results = [_fake_result(won=False)] * 4
        self.assertAlmostEqual(DistributedRolloutRunner.win_rate(results), 0.0)

    def test_win_rate_half(self) -> None:
        results = [_fake_result(won=True), _fake_result(won=False)]
        self.assertAlmostEqual(DistributedRolloutRunner.win_rate(results), 0.5)

    def test_win_rate_empty(self) -> None:
        self.assertEqual(DistributedRolloutRunner.win_rate([]), 0.0)

    def test_mean_episode_length_basic(self) -> None:
        results = [_fake_result(steps=10), _fake_result(steps=20)]
        self.assertAlmostEqual(DistributedRolloutRunner.mean_episode_length(results), 15.0)

    def test_mean_episode_length_empty(self) -> None:
        self.assertEqual(DistributedRolloutRunner.mean_episode_length([]), 0.0)

    def test_steps_per_second_zero_elapsed_returns_inf(self) -> None:
        r = RolloutResult(0, 10, 0.0, {}, True, {}, 0.0)
        sps = DistributedRolloutRunner.steps_per_second([r])
        self.assertEqual(sps, float("inf"))


# ---------------------------------------------------------------------------
# 8. Smoke test — 2-worker Ray cluster, 4 episodes (E4.7 CI acceptance)
# ---------------------------------------------------------------------------


class TestSmokeTest(unittest.TestCase):
    """Smoke test: 2-worker Ray cluster collects 4 episodes end-to-end.

    This is the test exercised by the CI workflow (ray_smoke_test.yml).
    It must complete in < 5 minutes.
    """

    def test_smoke_two_workers_four_episodes(self) -> None:
        """2-worker pool collects 4 episodes without errors."""
        runner = DistributedRolloutRunner(
            num_workers=2,
            env_kwargs={"n_blue": 1, "n_red": 1, "max_steps": 30},
            num_cpus_per_worker=0.5,
        )
        try:
            results = runner.collect_rollouts(n_episodes=4, base_seed=7)
            self.assertEqual(len(results), 4)
            total_steps = sum(r.episode_steps for r in results)
            self.assertGreater(total_steps, 0)
            # Win rate is in [0, 1]
            wr = DistributedRolloutRunner.win_rate(results)
            self.assertGreaterEqual(wr, 0.0)
            self.assertLessEqual(wr, 1.0)
        finally:
            runner.shutdown()

    def test_smoke_async_two_workers(self) -> None:
        """Async dispatch with 2 workers resolves correctly."""
        runner = DistributedRolloutRunner(
            num_workers=2,
            env_kwargs={"n_blue": 1, "n_red": 1, "max_steps": 30},
            num_cpus_per_worker=0.5,
        )
        try:
            refs = runner.collect_rollouts_async(n_episodes=2, base_seed=0)
            results = ray.get(refs)
            self.assertEqual(len(results), 2)
        finally:
            runner.shutdown()


if __name__ == "__main__":
    unittest.main()
