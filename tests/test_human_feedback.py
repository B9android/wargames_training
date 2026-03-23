# tests/test_human_feedback.py
"""Tests for training/human_feedback.py (Epic E9.3).

Coverage
--------
* HumanDemonstration — construction, immutability, field types
* DemonstrationBuffer — add / sample / save / load / capacity eviction
* HumanFeedbackRecorder — record_episode with random and policy-driven actions
* DAggerTrainer — collect_dagger_rollout, train_step, train loop
* GAILDiscriminator — forward, compute_reward, train_step, accuracy
* GAILRewardWrapper — step returns combined reward, gail_reward in info
* AARAnnotator — annotate returns valid AARReport with turning points
"""

from __future__ import annotations

import dataclasses
import math
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

from training.human_feedback import (
    AARAnnotator,
    AARReport,
    DAggerTrainer,
    DecisionAnnotation,
    DemonstrationBuffer,
    GAILDiscriminator,
    GAILRewardWrapper,
    HumanDemonstration,
    HumanFeedbackRecorder,
    TurningPoint,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

OBS_DIM = 17
ACTION_DIM = 3


def _make_demo(
    obs_dim: int = OBS_DIM,
    action_dim: int = ACTION_DIM,
    episode_id: int = 0,
    step_id: int = 0,
    reward: float = 0.5,
    terminated: bool = False,
    truncated: bool = False,
    rng: np.random.Generator | None = None,
) -> HumanDemonstration:
    if rng is None:
        rng = np.random.default_rng(0)
    return HumanDemonstration(
        observation=rng.random(obs_dim).astype(np.float32),
        action=rng.random(action_dim).astype(np.float32),
        reward=reward,
        next_observation=rng.random(obs_dim).astype(np.float32),
        terminated=terminated,
        truncated=truncated,
        episode_id=episode_id,
        step_id=step_id,
    )


def _make_buffer(
    n: int = 10,
    obs_dim: int = OBS_DIM,
    action_dim: int = ACTION_DIM,
    capacity: int = 100,
) -> DemonstrationBuffer:
    buf = DemonstrationBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=capacity)
    rng = np.random.default_rng(42)
    for i in range(n):
        buf.add(_make_demo(obs_dim, action_dim, step_id=i, rng=rng))
    return buf


class _FakeEnv:
    """Minimal gymnasium-like environment for unit tests."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, max_steps: int = 5) -> None:
        import gymnasium as gym
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self._max_steps = max_steps
        self._step_count = 0
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        self._step_count = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs_dim = self.observation_space.shape[0]
        return self._rng.random(obs_dim).astype(np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        obs_dim = self.observation_space.shape[0]
        obs = self._rng.random(obs_dim).astype(np.float32)
        reward = float(self._rng.random())
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {}

    def close(self) -> None:
        pass


class _ConstantPolicy:
    """Always returns the same fixed action."""

    def __init__(self, action_dim: int = ACTION_DIM, value: float = 0.3) -> None:
        self._action = np.full(action_dim, value, dtype=np.float32)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        return self._action.copy(), None


# ===========================================================================
# 1. HumanDemonstration
# ===========================================================================


class TestHumanDemonstration(unittest.TestCase):
    """Data class construction and field types."""

    def _make(self, **overrides) -> HumanDemonstration:
        rng = np.random.default_rng(1)
        defaults = dict(
            observation=rng.random(OBS_DIM).astype(np.float32),
            action=rng.random(ACTION_DIM).astype(np.float32),
            reward=1.0,
            next_observation=rng.random(OBS_DIM).astype(np.float32),
            terminated=False,
            truncated=False,
            episode_id=0,
            step_id=0,
        )
        defaults.update(overrides)
        return HumanDemonstration(**defaults)

    def test_construction_succeeds(self) -> None:
        demo = self._make()
        self.assertIsInstance(demo, HumanDemonstration)

    def test_observation_shape(self) -> None:
        demo = self._make()
        self.assertEqual(demo.observation.shape, (OBS_DIM,))

    def test_action_shape(self) -> None:
        demo = self._make()
        self.assertEqual(demo.action.shape, (ACTION_DIM,))

    def test_reward_is_float(self) -> None:
        demo = self._make(reward=2.5)
        self.assertIsInstance(demo.reward, float)
        self.assertAlmostEqual(demo.reward, 2.5)

    def test_episode_and_step_ids(self) -> None:
        demo = self._make(episode_id=3, step_id=7)
        self.assertEqual(demo.episode_id, 3)
        self.assertEqual(demo.step_id, 7)

    def test_terminated_truncated_flags(self) -> None:
        demo = self._make(terminated=True, truncated=False)
        self.assertTrue(demo.terminated)
        self.assertFalse(demo.truncated)

    def test_frozen_raises_on_assignment(self) -> None:
        demo = self._make()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            demo.reward = 99.0  # type: ignore[misc]

    def test_default_info_is_empty_dict(self) -> None:
        demo = self._make()
        self.assertIsInstance(demo.info, dict)
        self.assertEqual(len(demo.info), 0)


# ===========================================================================
# 2. DemonstrationBuffer
# ===========================================================================


class TestDemonstrationBufferConstruction(unittest.TestCase):
    def test_default_construction(self) -> None:
        buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        self.assertEqual(len(buf), 0)

    def test_custom_capacity(self) -> None:
        buf = DemonstrationBuffer(obs_dim=8, action_dim=2, capacity=200)
        self.assertEqual(buf.capacity, 200)

    def test_invalid_obs_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            DemonstrationBuffer(obs_dim=0, action_dim=3)

    def test_invalid_action_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            DemonstrationBuffer(obs_dim=4, action_dim=0)

    def test_invalid_capacity_raises(self) -> None:
        with self.assertRaises(ValueError):
            DemonstrationBuffer(obs_dim=4, action_dim=3, capacity=0)


class TestDemonstrationBufferAdd(unittest.TestCase):
    def setUp(self) -> None:
        self.buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM, capacity=50)

    def test_add_increments_size(self) -> None:
        demo = _make_demo()
        self.buf.add(demo)
        self.assertEqual(len(self.buf), 1)

    def test_add_multiple(self) -> None:
        rng = np.random.default_rng(7)
        for i in range(10):
            self.buf.add(_make_demo(step_id=i, rng=rng))
        self.assertEqual(len(self.buf), 10)

    def test_capacity_eviction(self) -> None:
        rng = np.random.default_rng(9)
        for i in range(60):
            self.buf.add(_make_demo(step_id=i, rng=rng))
        self.assertEqual(len(self.buf), 50)  # capped at capacity

    def test_add_batch(self) -> None:
        rng = np.random.default_rng(11)
        demos = [_make_demo(step_id=i, rng=rng) for i in range(5)]
        self.buf.add_batch(demos)
        self.assertEqual(len(self.buf), 5)

    def test_clear_resets_size(self) -> None:
        rng = np.random.default_rng(13)
        for i in range(5):
            self.buf.add(_make_demo(step_id=i, rng=rng))
        self.buf.clear()
        self.assertEqual(len(self.buf), 0)


class TestDemonstrationBufferSample(unittest.TestCase):
    def setUp(self) -> None:
        self.buf = _make_buffer(n=20)

    def test_sample_returns_correct_shapes(self) -> None:
        obs, actions, rewards = self.buf.sample(8)
        self.assertEqual(obs.shape, (8, OBS_DIM))
        self.assertEqual(actions.shape, (8, ACTION_DIM))
        self.assertEqual(rewards.shape, (8,))

    def test_sample_too_large_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.buf.sample(100)  # buffer has only 20

    def test_sample_is_float32(self) -> None:
        obs, actions, _ = self.buf.sample(5)
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(actions.dtype, np.float32)

    def test_sample_reproducible_with_rng(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        obs1, _, _ = self.buf.sample(5, rng=rng1)
        obs2, _, _ = self.buf.sample(5, rng=rng2)
        np.testing.assert_array_equal(obs1, obs2)

    def test_sample_all_returns_all(self) -> None:
        obs, actions, rewards = self.buf.sample_all()
        self.assertEqual(len(obs), len(self.buf))


class TestDemonstrationBufferPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.save_path = Path(self._tmpdir.name) / "demo_buffer"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_creates_file(self) -> None:
        buf = _make_buffer(n=5)
        saved = buf.save(self.save_path)
        # save() now always normalises to .npz and returns the resolved path.
        self.assertEqual(saved.suffix, ".npz")
        self.assertTrue(saved.exists())

    def test_save_and_load_roundtrip(self) -> None:
        buf = _make_buffer(n=10)
        obs_before, acts_before, _ = buf.sample_all()
        buf.save(self.save_path)
        loaded = DemonstrationBuffer.load(self.save_path)
        obs_after, acts_after, _ = loaded.sample_all()
        np.testing.assert_array_almost_equal(obs_before, obs_after)
        np.testing.assert_array_almost_equal(acts_before, acts_after)

    def test_load_restores_size(self) -> None:
        buf = _make_buffer(n=7)
        buf.save(self.save_path)
        loaded = DemonstrationBuffer.load(self.save_path)
        self.assertEqual(len(loaded), 7)

    def test_load_restores_dimensions(self) -> None:
        buf = _make_buffer(n=3, obs_dim=8, action_dim=2)
        buf.save(self.save_path)
        loaded = DemonstrationBuffer.load(self.save_path)
        self.assertEqual(loaded.obs_dim, 8)
        self.assertEqual(loaded.action_dim, 2)


# ===========================================================================
# 3. HumanFeedbackRecorder
# ===========================================================================


class TestHumanFeedbackRecorderConstruction(unittest.TestCase):
    def test_constructs_with_env_and_buffer(self) -> None:
        env = _FakeEnv()
        buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        rec = HumanFeedbackRecorder(env, buf)
        self.assertIsNotNone(rec)


class TestHumanFeedbackRecorderRecordEpisode(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _FakeEnv(max_steps=5)
        self.buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        self.rec = HumanFeedbackRecorder(self.env, self.buf)

    def test_record_episode_returns_list(self) -> None:
        episode = self.rec.record_episode(seed=0)
        self.assertIsInstance(episode, list)

    def test_episode_has_correct_length(self) -> None:
        episode = self.rec.record_episode(seed=0)
        self.assertEqual(len(episode), 5)

    def test_episode_elements_are_demos(self) -> None:
        episode = self.rec.record_episode(seed=0)
        for demo in episode:
            self.assertIsInstance(demo, HumanDemonstration)

    def test_buffer_populated_after_record(self) -> None:
        self.rec.record_episode(seed=0)
        self.assertGreater(len(self.buf), 0)

    def test_buffer_size_matches_episode_length(self) -> None:
        ep = self.rec.record_episode(seed=1)
        self.assertEqual(len(self.buf), len(ep))

    def test_last_episode_matches_return_value(self) -> None:
        ep = self.rec.record_episode(seed=2)
        self.assertEqual(len(self.rec.last_episode), len(ep))

    def test_record_with_policy(self) -> None:
        policy = _ConstantPolicy()
        ep = self.rec.record_episode(policy=policy, seed=3)
        self.assertEqual(len(ep), 5)
        # All actions should be the constant policy's output
        for demo in ep:
            np.testing.assert_array_almost_equal(demo.action, policy._action)

    def test_episode_ids_increment_across_calls(self) -> None:
        ep1 = self.rec.record_episode(seed=0)
        ep2 = self.rec.record_episode(seed=1)
        self.assertEqual(ep1[0].episode_id, 0)
        self.assertEqual(ep2[0].episode_id, 1)

    def test_step_ids_are_sequential(self) -> None:
        ep = self.rec.record_episode(seed=0)
        for i, demo in enumerate(ep):
            self.assertEqual(demo.step_id, i)

    def test_last_step_terminated_or_truncated(self) -> None:
        ep = self.rec.record_episode(seed=0)
        last = ep[-1]
        self.assertTrue(last.terminated or last.truncated)


# ===========================================================================
# 4. DAggerTrainer
# ===========================================================================


class TestDAggerTrainerConstruction(unittest.TestCase):
    def _make_trainer(self, **kwargs) -> DAggerTrainer:
        env = _FakeEnv(max_steps=5)
        buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        defaults = dict(
            env=env, buffer=buf, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            action_low=-1.0, action_high=1.0, device="cpu",
        )
        defaults.update(kwargs)
        return DAggerTrainer(**defaults)

    def test_construction_succeeds(self) -> None:
        trainer = self._make_trainer()
        self.assertIsNotNone(trainer)

    def test_policy_net_is_nn_module(self) -> None:
        import torch.nn as nn
        trainer = self._make_trainer()
        self.assertIsInstance(trainer.policy_net, nn.Module)

    def test_predict_returns_clipped_action(self) -> None:
        trainer = self._make_trainer()
        obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
        action, state = trainer.predict(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertIsNone(state)
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_predict_batched_obs(self) -> None:
        trainer = self._make_trainer()
        obs_batch = np.random.default_rng(1).random((4, OBS_DIM)).astype(np.float32)
        actions, _ = trainer.predict(obs_batch)
        self.assertEqual(actions.shape, (4, ACTION_DIM))

    def test_custom_hidden_sizes(self) -> None:
        trainer = self._make_trainer(hidden_sizes=[64])
        # Just check it constructs without error.
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action, _ = trainer.predict(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))


class TestDAggerTrainerCollectRollout(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _FakeEnv(max_steps=5)
        self.buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        self.trainer = DAggerTrainer(
            env=self.env, buffer=self.buf,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            device="cpu",
        )
        self.expert = _ConstantPolicy()

    def test_collect_returns_positive_count(self) -> None:
        n = self.trainer.collect_dagger_rollout(self.expert, seed=0)
        self.assertGreater(n, 0)

    def test_collect_populates_buffer(self) -> None:
        self.trainer.collect_dagger_rollout(self.expert, seed=0)
        self.assertGreater(len(self.buf), 0)

    def test_buffer_actions_are_expert_actions(self) -> None:
        self.trainer.collect_dagger_rollout(self.expert, seed=0)
        _, acts, _ = self.buf.sample_all()
        # All actions should match the constant expert policy's value.
        np.testing.assert_array_almost_equal(
            acts, np.full_like(acts, 0.3), decimal=5
        )

    def test_episode_ids_increment_across_rollouts(self) -> None:
        """Each rollout should receive a distinct episode_id."""
        self.trainer.collect_dagger_rollout(self.expert, seed=0)
        n_first = len(self.buf)
        self.trainer.collect_dagger_rollout(self.expert, seed=1)
        # Episode IDs in the second rollout should be > those in the first.
        ids_first = set(self.buf._episode_ids[:n_first])
        ids_second = set(self.buf._episode_ids[n_first:len(self.buf)])
        self.assertTrue(ids_second.isdisjoint(ids_first))


class TestDAggerTrainerTrainStep(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _FakeEnv(max_steps=5)
        self.buf = _make_buffer(n=20)
        self.trainer = DAggerTrainer(
            env=self.env, buffer=self.buf,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            batch_size=8, device="cpu",
        )

    def test_train_step_returns_float(self) -> None:
        loss = self.trainer.train_step()
        self.assertIsInstance(loss, float)
        self.assertFalse(math.isnan(loss))

    def test_train_step_increments_grad_steps(self) -> None:
        self.trainer.train_step()
        self.assertEqual(self.trainer.total_grad_steps, 1)

    def test_train_step_updates_last_bc_loss(self) -> None:
        self.trainer.train_step()
        self.assertFalse(math.isnan(self.trainer.last_bc_loss))

    def test_repeated_train_steps(self) -> None:
        for _ in range(5):
            self.trainer.train_step()
        self.assertEqual(self.trainer.total_grad_steps, 5)

    def test_train_step_on_empty_buffer_raises(self) -> None:
        """train_step() on an empty buffer should raise ValueError, not produce NaN."""
        empty_buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        trainer = DAggerTrainer(
            env=self.env, buffer=empty_buf,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu",
        )
        with self.assertRaises(ValueError):
            trainer.train_step()


class TestDAggerTrainerTrain(unittest.TestCase):
    def test_train_returns_list_of_losses(self) -> None:
        env = _FakeEnv(max_steps=3)
        buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM, capacity=1000)
        trainer = DAggerTrainer(
            env=env, buffer=buf,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            n_grad_steps=2, batch_size=4, device="cpu",
        )
        expert = _ConstantPolicy()
        losses = trainer.train(n_iterations=3, expert_policy=expert, seed=0)
        self.assertEqual(len(losses), 3)
        for l in losses:
            self.assertFalse(math.isnan(l))

    def test_train_invalid_n_iterations_raises(self) -> None:
        env = _FakeEnv()
        buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        trainer = DAggerTrainer(
            env=env, buffer=buf,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu",
        )
        with self.assertRaises(ValueError):
            trainer.train(n_iterations=0, expert_policy=_ConstantPolicy())


# ===========================================================================
# 5. GAILDiscriminator
# ===========================================================================


class TestGAILDiscriminatorConstruction(unittest.TestCase):
    def test_construction_succeeds(self) -> None:
        disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        self.assertIsNotNone(disc)

    def test_invalid_obs_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            GAILDiscriminator(obs_dim=0, action_dim=ACTION_DIM)

    def test_invalid_action_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            GAILDiscriminator(obs_dim=OBS_DIM, action_dim=0)

    def test_is_nn_module(self) -> None:
        disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        import torch.nn as nn
        self.assertIsInstance(disc, nn.Module)


class TestGAILDiscriminatorForward(unittest.TestCase):
    def setUp(self) -> None:
        self.disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)

    def test_forward_output_shape_single(self) -> None:
        obs = torch.zeros(1, OBS_DIM)
        act = torch.zeros(1, ACTION_DIM)
        out = self.disc(obs, act)
        self.assertEqual(out.shape, (1, 1))

    def test_forward_output_shape_batch(self) -> None:
        obs = torch.zeros(8, OBS_DIM)
        act = torch.zeros(8, ACTION_DIM)
        out = self.disc(obs, act)
        self.assertEqual(out.shape, (8, 1))

    def test_forward_is_finite(self) -> None:
        obs = torch.randn(4, OBS_DIM)
        act = torch.randn(4, ACTION_DIM)
        out = self.disc(obs, act)
        self.assertTrue(torch.all(torch.isfinite(out)))


class TestGAILDiscriminatorComputeReward(unittest.TestCase):
    def setUp(self) -> None:
        self.disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)

    def test_compute_reward_scalar_for_1d_input(self) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        act = np.zeros(ACTION_DIM, dtype=np.float32)
        r = self.disc.compute_reward(obs, act)
        self.assertIsInstance(r, float)

    def test_compute_reward_array_for_batch(self) -> None:
        obs = np.zeros((4, OBS_DIM), dtype=np.float32)
        act = np.zeros((4, ACTION_DIM), dtype=np.float32)
        r = self.disc.compute_reward(obs, act)
        self.assertEqual(r.shape, (4,))

    def test_compute_reward_is_finite(self) -> None:
        obs = np.random.default_rng(0).random((5, OBS_DIM)).astype(np.float32)
        act = np.random.default_rng(1).random((5, ACTION_DIM)).astype(np.float32)
        r = self.disc.compute_reward(obs, act)
        self.assertTrue(np.all(np.isfinite(r)))

    def test_compute_reward_is_non_negative_for_untrained(self) -> None:
        """GAIL reward r = −log(1 − D) = softplus(logit) ≥ 0 always."""
        obs = np.random.default_rng(2).random((10, OBS_DIM)).astype(np.float32)
        act = np.random.default_rng(3).random((10, ACTION_DIM)).astype(np.float32)
        r = self.disc.compute_reward(obs, act)
        self.assertTrue(np.all(r >= 0.0))


class TestGAILDiscriminatorTrainStep(unittest.TestCase):
    def setUp(self) -> None:
        self.disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        rng = np.random.default_rng(5)
        self.h_obs = rng.random((16, OBS_DIM)).astype(np.float32)
        self.h_act = rng.random((16, ACTION_DIM)).astype(np.float32)
        self.p_obs = rng.random((16, OBS_DIM)).astype(np.float32)
        self.p_act = rng.random((16, ACTION_DIM)).astype(np.float32)

    def test_train_step_returns_float(self) -> None:
        loss = self.disc.train_step(self.h_obs, self.h_act, self.p_obs, self.p_act)
        self.assertIsInstance(loss, float)
        self.assertFalse(math.isnan(loss))

    def test_train_step_updates_last_discriminator_loss(self) -> None:
        self.disc.train_step(self.h_obs, self.h_act, self.p_obs, self.p_act)
        self.assertFalse(math.isnan(self.disc.last_discriminator_loss))

    def test_loss_decreases_with_training(self) -> None:
        """After multiple steps on the same data, loss should generally decrease."""
        first_loss = self.disc.train_step(self.h_obs, self.h_act, self.p_obs, self.p_act)
        for _ in range(50):
            self.disc.train_step(self.h_obs, self.h_act, self.p_obs, self.p_act)
        last_loss = self.disc.last_discriminator_loss
        # Loss should not explode (it may not strictly decrease on small data).
        self.assertLess(last_loss, first_loss * 5)


class TestGAILDiscriminatorAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        self.disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        rng = np.random.default_rng(6)
        self.h_obs = rng.random((32, OBS_DIM)).astype(np.float32)
        self.h_act = rng.random((32, ACTION_DIM)).astype(np.float32)
        self.p_obs = rng.random((32, OBS_DIM)).astype(np.float32)
        self.p_act = rng.random((32, ACTION_DIM)).astype(np.float32)

    def test_accuracy_in_range(self) -> None:
        acc = self.disc.accuracy(
            self.h_obs, self.h_act, self.p_obs, self.p_act
        )
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_accuracy_after_training_exceeds_random(self) -> None:
        """After enough discriminator training, accuracy should exceed 50%."""
        for _ in range(100):
            self.disc.train_step(self.h_obs, self.h_act, self.p_obs, self.p_act)
        acc = self.disc.accuracy(
            self.h_obs, self.h_act, self.p_obs, self.p_act
        )
        self.assertGreater(acc, 0.5)


# ===========================================================================
# 6. GAILRewardWrapper
# ===========================================================================


class TestGAILRewardWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _FakeEnv(max_steps=5)
        self.disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        self.wrapper = GAILRewardWrapper(self.env, self.disc, gail_coef=1.0, env_coef=0.0)

    def test_reset_delegates_to_env(self) -> None:
        obs, info = self.wrapper.reset(seed=0)
        self.assertEqual(obs.shape, (OBS_DIM,))

    def test_step_returns_5_tuple(self) -> None:
        self.wrapper.reset(seed=0)
        action = self.env.action_space.sample()
        result = self.wrapper.step(action)
        self.assertEqual(len(result), 5)

    def test_step_reward_is_float(self) -> None:
        self.wrapper.reset(seed=0)
        action = self.env.action_space.sample()
        _, reward, *_ = self.wrapper.step(action)
        self.assertIsInstance(reward, float)

    def test_gail_reward_in_info(self) -> None:
        self.wrapper.reset(seed=0)
        action = self.env.action_space.sample()
        _, _, _, _, info = self.wrapper.step(action)
        self.assertIn("gail_reward", info)

    def test_env_reward_in_info(self) -> None:
        self.wrapper.reset(seed=0)
        action = self.env.action_space.sample()
        _, _, _, _, info = self.wrapper.step(action)
        self.assertIn("env_reward", info)

    def test_env_coef_zero_ignores_env_reward(self) -> None:
        """When env_coef=0, the combined reward equals gail_coef * gail_reward."""
        self.wrapper.reset(seed=0)
        action = self.env.action_space.sample()
        _, combined, _, _, info = self.wrapper.step(action)
        self.assertAlmostEqual(combined, info["gail_reward"], places=6)

    def test_pure_env_reward(self) -> None:
        """When gail_coef=0 and env_coef=1, combined equals env_reward."""
        wrapper = GAILRewardWrapper(self.env, self.disc, gail_coef=0.0, env_coef=1.0)
        wrapper.reset(seed=0)
        action = self.env.action_space.sample()
        _, combined, _, _, info = wrapper.step(action)
        self.assertAlmostEqual(combined, info["env_reward"], places=6)

    def test_observation_space_exposed(self) -> None:
        self.assertEqual(
            self.wrapper.observation_space.shape, self.env.observation_space.shape
        )

    def test_action_space_exposed(self) -> None:
        self.assertEqual(
            self.wrapper.action_space.shape, self.env.action_space.shape
        )

    def test_close_does_not_raise(self) -> None:
        self.wrapper.close()

    def test_reset_caches_prev_obs(self) -> None:
        """After reset(), _prev_obs should be set and match the initial obs."""
        obs, _ = self.wrapper.reset(seed=0)
        self.assertIsNotNone(self.wrapper._prev_obs)
        np.testing.assert_array_almost_equal(self.wrapper._prev_obs, obs)

    def test_gail_reward_uses_pre_action_obs(self) -> None:
        """The discriminator should be evaluated on the pre-step observation."""
        from unittest.mock import patch, MagicMock
        obs, _ = self.wrapper.reset(seed=0)
        prev = self.wrapper._prev_obs.copy()
        action = self.env.action_space.sample()

        captured: list = []

        def _fake_compute_reward(o, a):
            captured.append(np.array(o, dtype=np.float32))
            return 0.5

        with patch.object(self.disc, "compute_reward", side_effect=_fake_compute_reward):
            self.wrapper.step(action)

        self.assertEqual(len(captured), 1)
        np.testing.assert_array_almost_equal(captured[0], prev)

    def test_gail_reward_is_non_negative(self) -> None:
        """GAIL reward = softplus(logit) ≥ 0 for any input."""
        self.wrapper.reset(seed=0)
        for _ in range(5):
            action = self.env.action_space.sample()
            _, combined, terminated, truncated, info = self.wrapper.step(action)
            self.assertGreaterEqual(info["gail_reward"], 0.0)
            if terminated or truncated:
                break


# ===========================================================================
# 7. AARAnnotator
# ===========================================================================


class TestAARAnnotatorConstruction(unittest.TestCase):
    def test_constructs_with_policy(self) -> None:
        policy = _ConstantPolicy()
        ann = AARAnnotator(reference_policy=policy)
        self.assertIsNotNone(ann)

    def test_default_threshold(self) -> None:
        policy = _ConstantPolicy()
        ann = AARAnnotator(reference_policy=policy)
        self.assertAlmostEqual(ann.divergence_threshold, 0.2)

    def test_custom_threshold(self) -> None:
        ann = AARAnnotator(reference_policy=_ConstantPolicy(), divergence_threshold=0.5)
        self.assertAlmostEqual(ann.divergence_threshold, 0.5)


class TestAARAnnotatorAnnotateEmptyEpisode(unittest.TestCase):
    def test_empty_episode_returns_valid_report(self) -> None:
        ann = AARAnnotator(reference_policy=_ConstantPolicy())
        report = ann.annotate([], episode_id=0)
        self.assertIsInstance(report, AARReport)
        self.assertEqual(report.n_steps, 0)
        self.assertEqual(report.annotations, [])
        self.assertEqual(report.turning_points, [])


class TestAARAnnotatorAnnotate(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _ConstantPolicy(value=0.3)
        self.ann = AARAnnotator(
            reference_policy=self.policy,
            divergence_threshold=0.1,
            top_k_turning_points=5,
        )
        rng = np.random.default_rng(42)
        self.episode = [
            _make_demo(step_id=i, rng=rng) for i in range(10)
        ]

    def test_annotate_returns_aar_report(self) -> None:
        report = self.ann.annotate(self.episode, episode_id=3)
        self.assertIsInstance(report, AARReport)

    def test_report_episode_id(self) -> None:
        report = self.ann.annotate(self.episode, episode_id=3)
        self.assertEqual(report.episode_id, 3)

    def test_report_n_steps(self) -> None:
        report = self.ann.annotate(self.episode)
        self.assertEqual(report.n_steps, 10)

    def test_annotations_length_matches_episode(self) -> None:
        report = self.ann.annotate(self.episode)
        self.assertEqual(len(report.annotations), 10)

    def test_annotations_are_decision_annotation(self) -> None:
        report = self.ann.annotate(self.episode)
        for ann in report.annotations:
            self.assertIsInstance(ann, DecisionAnnotation)

    def test_annotation_step_ids_sequential(self) -> None:
        report = self.ann.annotate(self.episode)
        for i, ann in enumerate(report.annotations):
            self.assertEqual(ann.step_id, i)

    def test_quality_scores_in_range(self) -> None:
        report = self.ann.annotate(self.episode)
        for ann in report.annotations:
            self.assertGreaterEqual(ann.quality_score, 0.0)
            self.assertLessEqual(ann.quality_score, 1.0)

    def test_mean_quality_score_in_range(self) -> None:
        report = self.ann.annotate(self.episode)
        self.assertGreaterEqual(report.mean_quality_score, 0.0)
        self.assertLessEqual(report.mean_quality_score, 1.0)

    def test_agreement_rate_in_range(self) -> None:
        report = self.ann.annotate(self.episode)
        self.assertGreaterEqual(report.agreement_rate, 0.0)
        self.assertLessEqual(report.agreement_rate, 1.0)

    def test_turning_points_are_turning_point_objects(self) -> None:
        report = self.ann.annotate(self.episode)
        for tp in report.turning_points:
            self.assertIsInstance(tp, TurningPoint)

    def test_turning_points_capped_at_top_k(self) -> None:
        report = self.ann.annotate(self.episode)
        self.assertLessEqual(len(report.turning_points), 5)

    def test_turning_points_sorted_descending(self) -> None:
        report = self.ann.annotate(self.episode)
        scores = [tp.divergence_score for tp in report.turning_points]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_turning_points_above_threshold(self) -> None:
        report = self.ann.annotate(self.episode)
        for tp in report.turning_points:
            self.assertGreaterEqual(tp.divergence_score, self.ann.divergence_threshold)

    def test_ai_action_matches_policy_output(self) -> None:
        """All AI-recommended actions should equal the constant policy value."""
        report = self.ann.annotate(self.episode)
        expected = self.policy._action
        for ann in report.annotations:
            np.testing.assert_array_almost_equal(ann.ai_action, expected)


class TestAARAnnotatorPerfectAgreement(unittest.TestCase):
    """When human actions match the AI exactly, agreement_rate should be 1.0."""

    def test_agreement_rate_one_when_actions_match(self) -> None:
        policy = _ConstantPolicy(value=0.3)
        ann = AARAnnotator(reference_policy=policy, divergence_threshold=0.01)
        # Construct episode where human always takes the AI's action.
        rng = np.random.default_rng(0)
        episode = []
        for i in range(5):
            episode.append(
                HumanDemonstration(
                    observation=rng.random(OBS_DIM).astype(np.float32),
                    action=policy._action.copy(),  # exact match
                    reward=0.0,
                    next_observation=rng.random(OBS_DIM).astype(np.float32),
                    terminated=False,
                    truncated=False,
                    episode_id=0,
                    step_id=i,
                )
            )
        report = ann.annotate(episode)
        self.assertAlmostEqual(report.agreement_rate, 1.0, places=5)
        self.assertEqual(len(report.turning_points), 0)


class TestAARAnnotatorWithValuePolicy(unittest.TestCase):
    """AARAnnotator correctly uses predict_values when available."""

    def test_quality_scores_differ_when_values_differ(self) -> None:
        """If value estimates differ across steps, quality scores should differ."""

        class _ValuePolicy:
            """Policy that returns increasing values per call."""

            def __init__(self) -> None:
                self._call_count = 0

            def predict(self, obs, deterministic=True):
                return np.zeros(ACTION_DIM, dtype=np.float32), None

            def predict_values(self, obs_t):
                self._call_count += 1
                # Return a different value each call.
                val = torch.tensor([[float(self._call_count)]])
                return val

        policy = _ValuePolicy()
        ann = AARAnnotator(reference_policy=policy, divergence_threshold=0.5)
        rng = np.random.default_rng(7)
        episode = [_make_demo(step_id=i, rng=rng) for i in range(5)]
        report = ann.annotate(episode)
        scores = [a.quality_score for a in report.annotations]
        # With increasing values, the highest step should have quality_score=1.0.
        self.assertAlmostEqual(max(scores), 1.0, places=5)
        self.assertAlmostEqual(min(scores), 0.0, places=5)


# ===========================================================================
# 8. Integration — record + annotate pipeline
# ===========================================================================


class TestEndToEndPipeline(unittest.TestCase):
    """Smoke test: record an episode and annotate it."""

    def test_record_then_annotate(self) -> None:
        env = _FakeEnv(max_steps=8)
        buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        recorder = HumanFeedbackRecorder(env, buf)
        policy = _ConstantPolicy()
        episode = recorder.record_episode(policy=policy, seed=0)

        annotator = AARAnnotator(reference_policy=policy, divergence_threshold=0.5)
        report = annotator.annotate(episode, episode_id=0)

        self.assertEqual(report.n_steps, len(episode))
        self.assertIsInstance(report.mean_quality_score, float)
        self.assertGreaterEqual(report.agreement_rate, 0.0)

    def test_dagger_then_gail_reward(self) -> None:
        """DAgger trains policy; GAIL wrapper computes intrinsic reward."""
        env = _FakeEnv(max_steps=3)
        buf = DemonstrationBuffer(obs_dim=OBS_DIM, action_dim=ACTION_DIM, capacity=500)
        trainer = DAggerTrainer(
            env=env, buffer=buf,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            n_grad_steps=2, batch_size=4, device="cpu",
        )
        expert = _ConstantPolicy()
        trainer.train(n_iterations=2, expert_policy=expert, seed=0)

        disc = GAILDiscriminator(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        wrapped = GAILRewardWrapper(env, disc)
        wrapped.reset(seed=0)
        action = env.action_space.sample()
        _, reward, *_ = wrapped.step(action)
        self.assertIsInstance(reward, float)
        self.assertFalse(math.isnan(reward))


# ===========================================================================
# 9. __all__ export list
# ===========================================================================


class TestPublicAPI(unittest.TestCase):
    def test_all_symbols_importable(self) -> None:
        from training.human_feedback import __all__ as exported
        import training.human_feedback as mod
        for name in exported:
            self.assertTrue(hasattr(mod, name), f"Missing export: {name}")


if __name__ == "__main__":
    unittest.main()
