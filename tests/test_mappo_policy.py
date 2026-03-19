# tests/test_mappo_policy.py
"""Tests for models/mappo_policy.py and training/train_mappo.py.

Coverage
--------
* MAPPOActor  — forward pass shapes, distribution, evaluate_actions
* MAPPOCritic — forward pass shapes
* MAPPOPolicy — shared/separate parameter modes, act(), get_value(), evaluate_actions()
* MAPPORolloutBuffer — add/fill, GAE computation, minibatch iteration
* MAPPOTrainer — collect_rollout, update_policy (smoke test on 2v2 env)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.multi_battalion_env import MultiBattalionEnv
from models.mappo_policy import MAPPOActor, MAPPOCritic, MAPPOPolicy
from training.train_mappo import MAPPORolloutBuffer, MAPPOTrainer

# ---------------------------------------------------------------------------
# Constants matching a 2v2 environment (computed identically to the env)
# ---------------------------------------------------------------------------
N_BLUE = 2
N_RED = 2
N_TOTAL = N_BLUE + N_RED
OBS_DIM = 6 + 5 * (N_TOTAL - 1) + 1   # = 22
ACTION_DIM = 3
STATE_DIM = 6 * N_TOTAL + 1            # = 25
BATCH = 4
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# MAPPOActor tests
# ---------------------------------------------------------------------------


class TestMAPPOActor(unittest.TestCase):

    def setUp(self):
        self.actor = MAPPOActor(OBS_DIM, ACTION_DIM, hidden_sizes=(64, 32))

    def test_forward_shape(self):
        obs = torch.zeros(BATCH, OBS_DIM)
        mean, std = self.actor(obs)
        self.assertEqual(mean.shape, (BATCH, ACTION_DIM))
        self.assertEqual(std.shape, (BATCH, ACTION_DIM))

    def test_std_positive(self):
        obs = torch.zeros(BATCH, OBS_DIM)
        _, std = self.actor(obs)
        self.assertTrue((std > 0).all().item())

    def test_get_distribution_type(self):
        obs = torch.zeros(BATCH, OBS_DIM)
        dist = self.actor.get_distribution(obs)
        from torch.distributions import Normal
        self.assertIsInstance(dist, Normal)

    def test_evaluate_actions_shapes(self):
        obs = torch.zeros(BATCH, OBS_DIM)
        actions = torch.zeros(BATCH, ACTION_DIM)
        log_probs, entropy = self.actor.evaluate_actions(obs, actions)
        self.assertEqual(log_probs.shape, (BATCH,))
        self.assertEqual(entropy.shape, (BATCH,))

    def test_single_obs_forward(self):
        """Actor should handle a single observation (no batch dim)."""
        obs = torch.zeros(OBS_DIM)
        mean, std = self.actor(obs)
        self.assertEqual(mean.shape, (ACTION_DIM,))

    def test_no_nan(self):
        obs = torch.randn(BATCH, OBS_DIM)
        mean, std = self.actor(obs)
        self.assertFalse(torch.isnan(mean).any().item())
        self.assertFalse(torch.isnan(std).any().item())


# ---------------------------------------------------------------------------
# MAPPOCritic tests
# ---------------------------------------------------------------------------


class TestMAPPOCritic(unittest.TestCase):

    def setUp(self):
        self.critic = MAPPOCritic(STATE_DIM, hidden_sizes=(64, 32))

    def test_forward_shape_batched(self):
        state = torch.zeros(BATCH, STATE_DIM)
        values = self.critic(state)
        self.assertEqual(values.shape, (BATCH,))

    def test_forward_shape_single(self):
        state = torch.zeros(1, STATE_DIM)
        values = self.critic(state)
        self.assertEqual(values.shape, (1,))

    def test_no_nan(self):
        state = torch.randn(BATCH, STATE_DIM)
        values = self.critic(state)
        self.assertFalse(torch.isnan(values).any().item())


# ---------------------------------------------------------------------------
# MAPPOPolicy — shared parameters
# ---------------------------------------------------------------------------


class TestMAPPOPolicyShared(unittest.TestCase):

    def setUp(self):
        self.policy = MAPPOPolicy(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            n_agents=N_BLUE,
            share_parameters=True,
            actor_hidden_sizes=(64, 32),
            critic_hidden_sizes=(64, 32),
        )

    def test_has_single_actor(self):
        self.assertTrue(hasattr(self.policy, "actor"))
        self.assertFalse(hasattr(self.policy, "actors"))

    def test_get_actor_returns_same(self):
        a0 = self.policy.get_actor(0)
        a1 = self.policy.get_actor(1)
        self.assertIs(a0, a1)

    def test_act_shapes(self):
        obs = torch.zeros(N_BLUE, OBS_DIM)
        actions, log_probs = self.policy.act(obs)
        self.assertEqual(actions.shape, (N_BLUE, ACTION_DIM))
        self.assertEqual(log_probs.shape, (N_BLUE,))

    def test_get_value_shape_batched(self):
        state = torch.zeros(BATCH, STATE_DIM)
        v = self.policy.get_value(state)
        self.assertEqual(v.shape, (BATCH,))

    def test_evaluate_actions_shapes(self):
        obs = torch.zeros(BATCH, OBS_DIM)
        actions = torch.zeros(BATCH, ACTION_DIM)
        state = torch.zeros(BATCH, STATE_DIM)
        lp, ent, vals = self.policy.evaluate_actions(obs, actions, state)
        self.assertEqual(lp.shape, (BATCH,))
        self.assertEqual(ent.shape, (BATCH,))
        self.assertEqual(vals.shape, (BATCH,))

    def test_parameter_count_returns_dict(self):
        counts = self.policy.parameter_count()
        self.assertIn("actor", counts)
        self.assertIn("critic", counts)
        self.assertIn("total", counts)
        self.assertEqual(counts["total"], counts["actor"] + counts["critic"])
        self.assertGreater(counts["total"], 0)

    def test_deterministic_act(self):
        obs = torch.zeros(N_BLUE, OBS_DIM)
        a1, _ = self.policy.act(obs, deterministic=True)
        a2, _ = self.policy.act(obs, deterministic=True)
        self.assertTrue(torch.allclose(a1, a2))

    def test_no_nan_act(self):
        obs = torch.randn(N_BLUE, OBS_DIM)
        actions, log_probs = self.policy.act(obs)
        self.assertFalse(torch.isnan(actions).any().item())
        self.assertFalse(torch.isnan(log_probs).any().item())


# ---------------------------------------------------------------------------
# MAPPOPolicy — separate parameters (ablation)
# ---------------------------------------------------------------------------


class TestMAPPOPolicySeparate(unittest.TestCase):

    def setUp(self):
        self.policy = MAPPOPolicy(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            n_agents=N_BLUE,
            share_parameters=False,
            actor_hidden_sizes=(64, 32),
            critic_hidden_sizes=(64, 32),
        )

    def test_has_actors_list(self):
        self.assertFalse(hasattr(self.policy, "actor"))
        self.assertTrue(hasattr(self.policy, "actors"))
        self.assertEqual(len(self.policy.actors), N_BLUE)

    def test_get_actor_returns_different(self):
        a0 = self.policy.get_actor(0)
        a1 = self.policy.get_actor(1)
        self.assertIsNot(a0, a1)

    def test_separate_params_more_params_than_shared(self):
        separate_count = self.policy.parameter_count()["actor"]
        shared_policy = MAPPOPolicy(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            n_agents=N_BLUE,
            share_parameters=True,
            actor_hidden_sizes=(64, 32),
            critic_hidden_sizes=(64, 32),
        )
        shared_count = shared_policy.parameter_count()["actor"]
        self.assertEqual(separate_count, N_BLUE * shared_count)

    def test_act_per_agent(self):
        obs = torch.zeros(1, OBS_DIM)
        a0, _ = self.policy.act(obs, agent_idx=0)
        self.assertEqual(a0.shape, (1, ACTION_DIM))


# ---------------------------------------------------------------------------
# MAPPORolloutBuffer tests
# ---------------------------------------------------------------------------


class TestMAPPORolloutBuffer(unittest.TestCase):

    N_STEPS = 16
    N_AGENTS = 2
    GAMMA = 0.99
    GAE_LAMBDA = 0.95

    def _make_buffer(self) -> MAPPORolloutBuffer:
        return MAPPORolloutBuffer(
            n_steps=self.N_STEPS,
            n_agents=self.N_AGENTS,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            gamma=self.GAMMA,
            gae_lambda=self.GAE_LAMBDA,
        )

    def _fill_buffer(self, buf: MAPPORolloutBuffer) -> None:
        rng = np.random.default_rng(0)
        for _ in range(self.N_STEPS):
            buf.add(
                obs=rng.random((self.N_AGENTS, OBS_DIM)).astype(np.float32),
                actions=rng.random((self.N_AGENTS, ACTION_DIM)).astype(np.float32),
                log_probs=rng.random(self.N_AGENTS).astype(np.float32),
                rewards=rng.random(self.N_AGENTS).astype(np.float32),
                done=False,
                value=float(rng.random()),
                global_state=rng.random(STATE_DIM).astype(np.float32),
            )

    def test_add_and_full(self):
        buf = self._make_buffer()
        self.assertFalse(buf.full)
        self._fill_buffer(buf)
        self.assertTrue(buf.full)

    def test_add_overflow_raises(self):
        buf = self._make_buffer()
        self._fill_buffer(buf)
        with self.assertRaises(RuntimeError):
            buf.add(
                obs=np.zeros((self.N_AGENTS, OBS_DIM), dtype=np.float32),
                actions=np.zeros((self.N_AGENTS, ACTION_DIM), dtype=np.float32),
                log_probs=np.zeros(self.N_AGENTS, dtype=np.float32),
                rewards=np.zeros(self.N_AGENTS, dtype=np.float32),
                done=False,
                value=0.0,
                global_state=np.zeros(STATE_DIM, dtype=np.float32),
            )

    def test_reset_clears_buffer(self):
        buf = self._make_buffer()
        self._fill_buffer(buf)
        self.assertTrue(buf.full)
        buf.reset()
        self.assertFalse(buf.full)
        self.assertEqual(buf._ptr, 0)

    def test_compute_returns_and_advantages_shapes(self):
        buf = self._make_buffer()
        self._fill_buffer(buf)
        buf.compute_returns_and_advantages(last_value=0.0)
        self.assertEqual(buf.advantages.shape, (self.N_STEPS, self.N_AGENTS))
        self.assertEqual(buf.returns.shape, (self.N_STEPS, self.N_AGENTS))

    def test_advantages_no_nan(self):
        buf = self._make_buffer()
        self._fill_buffer(buf)
        buf.compute_returns_and_advantages(last_value=0.5)
        self.assertFalse(np.isnan(buf.advantages).any())
        self.assertFalse(np.isnan(buf.returns).any())

    def test_get_batches_coverage(self):
        """All samples should appear in at least one batch."""
        buf = self._make_buffer()
        self._fill_buffer(buf)
        buf.compute_returns_and_advantages(last_value=0.0)
        batches = buf.get_batches(batch_size=8, device=DEVICE)
        total = sum(b["obs"].shape[0] for b in batches)
        self.assertEqual(total, self.N_STEPS * self.N_AGENTS)

    def test_get_batches_tensor_shapes(self):
        buf = self._make_buffer()
        self._fill_buffer(buf)
        buf.compute_returns_and_advantages(last_value=0.0)
        batches = buf.get_batches(batch_size=8, device=DEVICE)
        for b in batches:
            bsz = b["obs"].shape[0]
            self.assertEqual(b["obs"].shape, (bsz, OBS_DIM))
            self.assertEqual(b["actions"].shape, (bsz, ACTION_DIM))
            self.assertEqual(b["old_log_probs"].shape, (bsz,))
            self.assertEqual(b["advantages"].shape, (bsz,))
            self.assertEqual(b["returns"].shape, (bsz,))
            self.assertEqual(b["global_states"].shape, (bsz, STATE_DIM))


# ---------------------------------------------------------------------------
# MAPPOTrainer smoke tests
# ---------------------------------------------------------------------------


class TestMAPPOTrainer(unittest.TestCase):
    """Smoke tests: verify that the training loop runs without errors."""

    def _make_trainer(self, share_parameters: bool = True) -> MAPPOTrainer:
        env = MultiBattalionEnv(n_blue=N_BLUE, n_red=N_RED, max_steps=50, randomize_terrain=False)
        policy = MAPPOPolicy(
            obs_dim=env._obs_dim,
            action_dim=env._act_space.shape[0],
            state_dim=env._state_dim,
            n_agents=env.n_blue,
            share_parameters=share_parameters,
            actor_hidden_sizes=(32, 16),
            critic_hidden_sizes=(32, 16),
        )
        trainer = MAPPOTrainer(
            policy=policy,
            env=env,
            n_steps=16,
            n_epochs=2,
            batch_size=8,
            lr=1e-3,
            seed=0,
        )
        return trainer

    def test_collect_rollout_fills_buffer(self):
        trainer = self._make_trainer()
        trainer._reset_env(seed=0)
        trainer.collect_rollout()
        self.assertTrue(trainer.buffer.full)

    def test_update_policy_returns_loss_dict(self):
        trainer = self._make_trainer()
        trainer._reset_env(seed=0)
        trainer.collect_rollout()
        losses = trainer.update_policy()
        self.assertIn("policy_loss", losses)
        self.assertIn("value_loss", losses)
        self.assertIn("entropy", losses)
        self.assertIn("total_loss", losses)

    def test_no_nan_losses(self):
        trainer = self._make_trainer()
        trainer._reset_env(seed=0)
        trainer.collect_rollout()
        losses = trainer.update_policy()
        for k, v in losses.items():
            self.assertFalse(
                np.isnan(v),
                msg=f"Loss '{k}' is NaN after one update step",
            )

    def test_total_steps_increments(self):
        trainer = self._make_trainer()
        trainer._reset_env(seed=0)
        trainer.collect_rollout()
        self.assertEqual(trainer._total_steps, 16)

    def test_separate_parameters_mode(self):
        """Training should work with separate (non-shared) actor parameters."""
        trainer = self._make_trainer(share_parameters=False)
        trainer._reset_env(seed=0)
        trainer.collect_rollout()
        losses = trainer.update_policy()
        self.assertFalse(np.isnan(losses["total_loss"]))

    def test_learn_short_run(self):
        """learn() should complete a very short run without errors."""
        trainer = self._make_trainer()
        # Run for exactly 2 rollouts worth of steps (no W&B logging needed)
        import unittest.mock as mock

        with mock.patch("wandb.log"):  # suppress W&B calls
            trainer.learn(total_timesteps=32, log_interval=1_000_000)

        self.assertGreaterEqual(trainer._total_steps, 32)

    def test_env_dimensions_match_policy(self):
        """Policy dims must match what the env reports."""
        env = MultiBattalionEnv(n_blue=2, n_red=2, max_steps=50, randomize_terrain=False)
        policy = MAPPOPolicy(
            obs_dim=env._obs_dim,
            action_dim=env._act_space.shape[0],
            state_dim=env._state_dim,
            n_agents=env.n_blue,
            share_parameters=True,
        )
        obs_arr = np.zeros((2, env._obs_dim), dtype=np.float32)
        state_arr = np.zeros((1, env._state_dim), dtype=np.float32)
        obs_t = torch.as_tensor(obs_arr)
        state_t = torch.as_tensor(state_arr)
        actions, lp = policy.act(obs_t)
        self.assertEqual(actions.shape, (2, env._act_space.shape[0]))
        values = policy.get_value(state_t)
        self.assertEqual(values.shape, (1,))
        env.close()


if __name__ == "__main__":
    unittest.main()
