# tests/test_train.py
"""Tests for training/train.py and models/mlp_policy.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.battalion_env import BattalionEnv
from models.mlp_policy import BattalionMlpPolicy


# ---------------------------------------------------------------------------
# models/mlp_policy.py
# ---------------------------------------------------------------------------


class TestBattalionMlpPolicy(unittest.TestCase):
    """Unit tests for BattalionMlpPolicy."""

    def setUp(self) -> None:
        self.env = BattalionEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_policy_imported(self) -> None:
        """BattalionMlpPolicy can be imported from models.mlp_policy."""
        from models.mlp_policy import BattalionMlpPolicy as P  # noqa: F401

    def test_ppo_creates_with_default_arch(self) -> None:
        """PPO model instantiates with BattalionMlpPolicy and default [128, 128] arch."""
        model = PPO(BattalionMlpPolicy, self.env, verbose=0)
        mlp = model.policy.mlp_extractor
        # Both policy_net and value_net should have 4 modules: L(12→128), Tanh, L(128→128), Tanh
        self.assertEqual(len(list(mlp.policy_net.children())), 4)
        self.assertEqual(len(list(mlp.value_net.children())), 4)

    def test_ppo_creates_with_custom_arch(self) -> None:
        """PPO honours net_arch override via policy_kwargs."""
        model = PPO(
            BattalionMlpPolicy,
            self.env,
            policy_kwargs={"net_arch": [64, 64, 64]},
            verbose=0,
        )
        mlp = model.policy.mlp_extractor
        # 3 hidden layers → 6 child modules per net (L, Act, L, Act, L, Act)
        self.assertEqual(len(list(mlp.policy_net.children())), 6)

    def test_policy_predict(self) -> None:
        """Policy produces actions of the correct shape."""
        model = PPO(BattalionMlpPolicy, self.env, verbose=0)
        obs, _ = self.env.reset(seed=0)
        action, _state = model.predict(obs, deterministic=True)
        self.assertEqual(action.shape, self.env.action_space.shape)
        self.assertTrue(self.env.action_space.contains(action))

    def test_action_space_bounds(self) -> None:
        """Predicted actions lie within the action space bounds."""
        model = PPO(BattalionMlpPolicy, self.env, verbose=0)
        obs, _ = self.env.reset(seed=1)
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=False)
            self.assertTrue(self.env.action_space.contains(action))
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, _ = self.env.reset()


# ---------------------------------------------------------------------------
# training/train.py — WandbCallback
# ---------------------------------------------------------------------------


class TestWandbCallback(unittest.TestCase):
    """Unit tests for the WandbCallback helper in training/train.py."""

    def test_callback_instantiates(self) -> None:
        from training.train import WandbCallback

        cb = WandbCallback(log_freq=500)
        self.assertEqual(cb.log_freq, 500)

    def test_on_step_logs_when_buffer_full(self) -> None:
        """_on_step logs to wandb when buffer has data and log_freq is met."""
        from training import train as train_mod
        from training.train import WandbCallback

        with patch.object(train_mod, "wandb") as mock_wandb:
            cb = WandbCallback(log_freq=1)
            model_mock = MagicMock()
            model_mock.ep_info_buffer = [{"r": 5.0, "l": 100}, {"r": 3.0, "l": 90}]
            cb.model = model_mock
            cb.num_timesteps = 100
            cb.n_calls = 1  # 1 % 1 == 0 → should log

            result = cb._on_step()

            self.assertTrue(result)
            mock_wandb.log.assert_called_once()
            logged = mock_wandb.log.call_args[0][0]
            self.assertIn("rollout/ep_rew_mean", logged)
            self.assertAlmostEqual(logged["rollout/ep_rew_mean"], 4.0)

    def test_on_step_skips_when_buffer_empty(self) -> None:
        """_on_step does not log when ep_info_buffer is empty."""
        from training import train as train_mod
        from training.train import WandbCallback

        with patch.object(train_mod, "wandb") as mock_wandb:
            cb = WandbCallback(log_freq=1)
            model_mock = MagicMock()
            model_mock.ep_info_buffer = []
            cb.model = model_mock
            cb.num_timesteps = 10
            cb.n_calls = 1

            cb._on_step()
            mock_wandb.log.assert_not_called()

    def test_on_rollout_end_logs_losses(self) -> None:
        """_on_rollout_end forwards SB3 logger values to wandb."""
        from training import train as train_mod
        from training.train import WandbCallback

        with patch.object(train_mod, "wandb") as mock_wandb:
            cb = WandbCallback(log_freq=1000)
            model_mock = MagicMock()
            model_mock.logger.name_to_value = {"loss/policy": 0.1, "loss/value": 0.5}
            cb.model = model_mock
            cb.num_timesteps = 2048

            cb._on_rollout_end()

            mock_wandb.log.assert_called_once()
            logged = mock_wandb.log.call_args[0][0]
            self.assertIn("train/loss/policy", logged)
            self.assertIn("train/loss/value", logged)


# ---------------------------------------------------------------------------
# training/train.py — short integration run (no W&B, no Hydra)
# ---------------------------------------------------------------------------


class TestShortTrainingRun(unittest.TestCase):
    """Smoke-tests: run PPO for a handful of timesteps without W&B."""

    def test_ppo_learn_short_run(self) -> None:
        """PPO can complete a very short training loop without errors."""
        env = make_vec_env(BattalionEnv, n_envs=2, seed=0)
        try:
            model = PPO(
                BattalionMlpPolicy,
                env,
                n_steps=64,
                batch_size=32,
                n_epochs=2,
                verbose=0,
            )
            model.learn(total_timesteps=128)
        finally:
            env.close()

    def test_checkpoint_saving(self) -> None:
        """PPO model can be saved and reloaded from disk."""
        import tempfile

        env = make_vec_env(BattalionEnv, n_envs=1, seed=42)
        try:
            model = PPO(BattalionMlpPolicy, env, n_steps=32, batch_size=16, verbose=0)
            model.learn(total_timesteps=32)
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "test_checkpoint"
                model.save(str(save_path))
                self.assertTrue((save_path.with_suffix(".zip")).exists())

                # Reload and verify action shape
                loaded = PPO.load(str(save_path), env=env)
                obs = env.reset()
                action, _ = loaded.predict(obs, deterministic=True)
                self.assertEqual(action.shape, (1, 3))
        finally:
            env.close()

    def test_eval_callback_runs(self) -> None:
        """EvalCallback executes without errors during a short training run."""
        from stable_baselines3.common.callbacks import EvalCallback

        import tempfile

        train_env = make_vec_env(BattalionEnv, n_envs=2, seed=0)
        eval_env = make_vec_env(BattalionEnv, n_envs=1, seed=999)
        try:
            model = PPO(
                BattalionMlpPolicy,
                train_env,
                n_steps=64,
                batch_size=32,
                verbose=0,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                eval_cb = EvalCallback(
                    eval_env,
                    eval_freq=64,
                    n_eval_episodes=2,
                    best_model_save_path=tmpdir,
                    deterministic=True,
                    verbose=0,
                )
                model.learn(total_timesteps=128, callback=eval_cb)
        finally:
            train_env.close()
            eval_env.close()


# ---------------------------------------------------------------------------
# training/train.py — W&B init (mocked)
# ---------------------------------------------------------------------------


class TestTrainWandbInit(unittest.TestCase):
    """Verify W&B initialisation is called with the right arguments."""

    def test_wandb_init_called(self) -> None:
        """main() calls wandb.init with project, config, and tags."""
        import tempfile

        from omegaconf import OmegaConf
        from training import train as train_mod
        from training.train import main

        cfg = OmegaConf.create(
            {
                "wandb": {
                    "project": "test_project",
                    "entity": None,
                    "tags": ["v1", "ppo"],
                    "log_freq": 1000,
                },
                "env": {
                    "map_width": 1000.0,
                    "map_height": 1000.0,
                    "max_steps": 50,
                    "num_envs": 1,
                    "randomize_terrain": True,
                    "hill_speed_factor": 0.5,
                    "curriculum_level": 5,
                },
                "reward": {
                    "delta_enemy_strength": 5.0,
                    "delta_own_strength": 5.0,
                    "survival_bonus": 0.0,
                    "win_bonus": 10.0,
                    "loss_penalty": -10.0,
                    "time_penalty": -0.01,
                },
                "training": {
                    "total_timesteps": 64,
                    "learning_rate": 3e-4,
                    "n_steps": 32,
                    "batch_size": 16,
                    "n_epochs": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "seed": 0,
                },
                "eval": {
                    "n_eval_episodes": 2,
                    "eval_freq": 64,
                    "checkpoint_freq": 64,
                    "checkpoint_dir": "checkpoints/",
                },
                "logging": {"level": "WARNING", "log_dir": "logs/"},
            }
        )

        fake_env = MagicMock()
        fake_model = MagicMock()
        fake_model.logger.name_to_value = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.eval.checkpoint_dir = tmpdir + "/checkpoints"
            cfg.logging.log_dir = tmpdir + "/logs"

            with (
                patch.object(train_mod, "wandb") as mock_wandb,
                patch.object(train_mod, "make_vec_env", return_value=fake_env),
                patch.object(train_mod, "PPO", return_value=fake_model),
                patch.object(train_mod, "CallbackList"),
                patch.object(train_mod, "EvalCallback"),
                patch.object(train_mod, "CheckpointCallback"),
            ):
                mock_wandb.init.return_value = MagicMock(url="https://wandb.ai/test")
                mock_wandb.Artifact = MagicMock()
                main.__wrapped__(cfg)  # __wrapped__ bypasses @hydra.main

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        self.assertEqual(call_kwargs["project"], "test_project")
        self.assertIn("config", call_kwargs)
        self.assertIn("v1", call_kwargs["tags"])


if __name__ == "__main__":
    unittest.main()
