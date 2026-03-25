# SPDX-License-Identifier: MIT
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
# Module-level reference to training.train (the module, not the function).
# training/__init__.py exports `train` as a *function*, which shadows the
# submodule name for `import training.train as x` (IMPORT_FROM bytecode).
# Use sys.modules to get the actual module reliably.
# ---------------------------------------------------------------------------
import training  # noqa: E402 — ensures training.train is loaded into sys.modules
_TRAIN_MODULE = sys.modules["training.train"]


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
        train_mod = _TRAIN_MODULE
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
        train_mod = _TRAIN_MODULE
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
        train_mod = _TRAIN_MODULE
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
# training/train.py — RewardBreakdownCallback
# ---------------------------------------------------------------------------


class TestRewardBreakdownCallback(unittest.TestCase):
    """Unit tests for RewardBreakdownCallback in training/train.py."""

    def _make_cb(self, log_freq: int = 1000):
        """Create a callback with pre-initialised step accumulators for 1 env."""
        from training.train import RewardBreakdownCallback
        cb = RewardBreakdownCallback(log_freq=log_freq)
        cb.num_timesteps = 0
        # Manually initialise step accumulators (normally done in _on_training_start).
        cb._step_sums = [{k: 0.0 for k in cb._COMPONENT_KEYS}]
        return cb

    def _make_cb_n(self, n_envs: int, log_freq: int = 1000):
        """Create a callback with pre-initialised step accumulators for n envs."""
        from training.train import RewardBreakdownCallback
        cb = RewardBreakdownCallback(log_freq=log_freq)
        cb.num_timesteps = 0
        cb._step_sums = [{k: 0.0 for k in cb._COMPONENT_KEYS} for _ in range(n_envs)]
        return cb

    def _make_info(self, **overrides) -> dict:
        base = {
            "reward/delta_enemy_strength": 0.5,
            "reward/delta_own_strength": -0.2,
            "reward/survival_bonus": 0.0,
            "reward/win_bonus": 0.0,
            "reward/loss_penalty": 0.0,
            "reward/time_penalty": -0.01,
            "reward/total": 0.29,
        }
        base.update(overrides)
        return base

    def test_instantiates(self) -> None:
        from training.train import RewardBreakdownCallback
        cb = RewardBreakdownCallback(log_freq=500)
        self.assertEqual(cb.log_freq, 500)
        self.assertEqual(cb._ep_count, 0)

    def test_accumulates_per_step_not_just_terminal(self) -> None:
        """Components should be summed across all steps, not only the terminal one."""
        train_mod = _TRAIN_MODULE

        with patch.object(train_mod, "wandb"):
            cb = self._make_cb(log_freq=1000)

            # Step 1 — not done
            cb.num_timesteps = 1
            cb.locals = {
                "infos": [self._make_info(**{"reward/delta_enemy_strength": 1.0})],
                "dones": np.array([False]),
            }
            cb._on_step()

            # Step 2 — done (episode ends)
            cb.num_timesteps = 2
            cb.locals = {
                "infos": [self._make_info(**{"reward/delta_enemy_strength": 2.0})],
                "dones": np.array([True]),
            }
            cb._on_step()

            # Episode sum should include BOTH steps' delta_enemy_strength.
            self.assertEqual(cb._ep_count, 1)
            self.assertAlmostEqual(
                cb._ep_sums["reward/delta_enemy_strength"], 3.0
            )

    def test_logs_episode_means_at_log_freq(self) -> None:
        """Flush should occur when num_timesteps % log_freq == 0 and ep_count > 0."""
        train_mod = _TRAIN_MODULE

        with patch.object(train_mod, "wandb") as mock_wandb:
            cb = self._make_cb(log_freq=10)

            # Simulate one complete episode — done=True at t=10.
            cb.num_timesteps = 10
            cb.locals = {
                "infos": [self._make_info(**{"reward/total": 5.0})],
                "dones": np.array([True]),
            }
            cb._on_step()

            mock_wandb.log.assert_called_once()
            logged = mock_wandb.log.call_args[0][0]
            self.assertIn("reward_breakdown/total", logged)
            self.assertAlmostEqual(logged["reward_breakdown/total"], 5.0)

    def test_no_log_when_no_episodes_completed(self) -> None:
        """No W&B log when log_freq is met but no episodes have finished."""
        train_mod = _TRAIN_MODULE

        with patch.object(train_mod, "wandb") as mock_wandb:
            cb = self._make_cb(log_freq=10)
            cb.num_timesteps = 10
            # done=False — no episode completed.
            cb.locals = {
                "infos": [self._make_info()],
                "dones": np.array([False]),
            }
            cb._on_step()
            mock_wandb.log.assert_not_called()

    def test_accumulators_reset_after_flush(self) -> None:
        """Episode accumulators must be zeroed after logging."""
        train_mod = _TRAIN_MODULE

        with patch.object(train_mod, "wandb"):
            cb = self._make_cb(log_freq=10)
            cb.num_timesteps = 10
            cb.locals = {
                "infos": [self._make_info()],
                "dones": np.array([True]),
            }
            cb._on_step()

            self.assertEqual(cb._ep_count, 0)
            for v in cb._ep_sums.values():
                self.assertAlmostEqual(v, 0.0)

    def test_on_training_end_flushes_remaining_episodes(self) -> None:
        """_on_training_end must log any episodes accumulated since the last flush."""
        train_mod = _TRAIN_MODULE

        with patch.object(train_mod, "wandb") as mock_wandb:
            cb = self._make_cb(log_freq=10000)  # large → won't auto-flush
            cb.num_timesteps = 50

            # Complete one episode.
            cb.locals = {
                "infos": [self._make_info(**{"reward/total": 7.0})],
                "dones": np.array([True]),
            }
            cb._on_step()

            # Should NOT have logged yet (50 % 10000 ≠ 0).
            mock_wandb.log.assert_not_called()

            # End of training — should flush.
            cb._on_training_end()
            mock_wandb.log.assert_called_once()
            logged = mock_wandb.log.call_args[0][0]
            self.assertAlmostEqual(logged["reward_breakdown/total"], 7.0)

    def test_multi_env_episode_boundaries(self) -> None:
        """Multiple parallel envs finishing simultaneously are each counted."""
        train_mod = _TRAIN_MODULE

        with patch.object(train_mod, "wandb") as mock_wandb:
            cb = self._make_cb_n(n_envs=2, log_freq=10)
            cb.num_timesteps = 10

            # Two envs both finish at the same step.
            cb.locals = {
                "infos": [
                    self._make_info(**{"reward/total": 4.0}),
                    self._make_info(**{"reward/total": 6.0}),
                ],
                "dones": np.array([True, True]),
            }
            cb._on_step()

            # 2 episodes; mean total should be (4+6)/2 = 5.
            mock_wandb.log.assert_called_once()
            logged = mock_wandb.log.call_args[0][0]
            self.assertAlmostEqual(logged["reward_breakdown/total"], 5.0)


# ---------------------------------------------------------------------------
# training/train.py — manifest-aware callbacks
# ---------------------------------------------------------------------------


class TestManifestCallbacks(unittest.TestCase):
    """Unit tests for manifest-aware checkpoint registration callbacks."""

    def test_periodic_checkpoint_registered_when_created(self) -> None:
        import tempfile

        train_mod = _TRAIN_MODULE
        from training.artifacts import CheckpointManifest
        from training.train import ManifestCheckpointCallback

        with tempfile.TemporaryDirectory() as tmp:
            manifest = CheckpointManifest(Path(tmp) / "manifest.jsonl")
            callback = ManifestCheckpointCallback(
                save_freq=1,
                save_path=tmp,
                name_prefix="ppo_battalion_s1_c5",
                manifest=manifest,
                seed=1,
                curriculum_level=5,
                run_id="run-1",
                config_hash="hash-1",
            )

            def _fake_checkpoint_step(self):
                target = Path(self._checkpoint_path(extension="zip"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("checkpoint", encoding="utf-8")
                return True

            callback.model = MagicMock()
            callback.n_calls = 1
            callback.num_timesteps = 100

            with patch.object(
                train_mod.CheckpointCallback,
                "_on_step",
                autospec=True,
                side_effect=_fake_checkpoint_step,
            ):
                callback._on_step()

            row = manifest.latest_entry_for_path(Path(tmp) / "ppo_battalion_s1_c5_100_steps.zip")
            self.assertIsNotNone(row)
            self.assertEqual(row["type"], "periodic")
            self.assertEqual(row["step"], 100)

    def test_best_checkpoint_registered_when_created(self) -> None:
        import tempfile

        train_mod = _TRAIN_MODULE
        from training.artifacts import CheckpointManifest
        from training.train import ManifestEvalCallback
        from envs.battalion_env import BattalionEnv

        with tempfile.TemporaryDirectory() as tmp:
            manifest = CheckpointManifest(Path(tmp) / "manifest.jsonl")
            eval_env = BattalionEnv(randomize_terrain=False)
            try:
                callback = ManifestEvalCallback(
                    eval_env,
                    eval_freq=1,
                    n_eval_episodes=1,
                    best_model_save_path=tmp,
                    log_path=tmp,
                    deterministic=True,
                    manifest=manifest,
                    seed=3,
                    curriculum_level=4,
                    run_id="run-3",
                    config_hash="hash-3",
                    enable_naming_v2=True,
                )

                def _fake_eval_step(self):
                    target = Path(self.best_model_save_path) / "best_model.zip"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("best", encoding="utf-8")
                    self.best_mean_reward = 1.5
                    return True

                callback.model = MagicMock()
                callback.n_calls = 1
                callback.num_timesteps = 250
                callback.best_mean_reward = float("-inf")

                with patch.object(
                    train_mod.EvalCallback,
                    "_on_step",
                    autospec=True,
                    side_effect=_fake_eval_step,
                ):
                    callback._on_step()

                alias_row = manifest.latest_entry_for_path(Path(tmp) / "best_model.zip")
                canonical_path = Path(tmp) / "ppo_battalion_s3_c4_best.zip"
                canonical_row = manifest.latest_entry_for_path(canonical_path)
                self.assertIsNotNone(alias_row)
                self.assertIsNotNone(canonical_row)
                self.assertEqual(alias_row["step"], 250)
                self.assertEqual(canonical_row["type"], "best")
                self.assertTrue(canonical_path.exists())
            finally:
                eval_env.close()


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
        train_mod = _TRAIN_MODULE
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
                patch.object(train_mod, "ManifestEvalCallback"),
                patch.object(train_mod, "ManifestCheckpointCallback"),
            ):
                mock_wandb.init.return_value = MagicMock(url="https://wandb.ai/test")
                mock_wandb.Artifact = MagicMock()
                main.__wrapped__(cfg)  # __wrapped__ bypasses @hydra.main

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        self.assertEqual(call_kwargs["project"], "test_project")
        self.assertIn("config", call_kwargs)
        self.assertIn("v1", call_kwargs["tags"])


# ---------------------------------------------------------------------------
# training/train.py — EloEvalCallback
# ---------------------------------------------------------------------------


class TestEloEvalCallback(unittest.TestCase):
    """Unit tests for EloEvalCallback in training/train.py."""

    def _make_fake_result(self, wins: int = 1, losses: int = 1, n: int = 2):
        """Build a fake EvaluationResult (wins + losses = n)."""
        from training.evaluate import EvaluationResult
        draws = n - wins - losses
        return EvaluationResult(
            wins=wins,
            draws=draws,
            losses=losses,
            n_episodes=n,
            win_rate=wins / n,
            draw_rate=draws / n,
            loss_rate=losses / n,
        )

    def test_instantiates(self) -> None:
        """EloEvalCallback initialises with the expected attributes."""
        from training.elo import EloRegistry
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        cb = EloEvalCallback(
            opponents=["scripted_l3"],
            n_eval_episodes=5,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
            seed=42,
        )
        self.assertEqual(cb.eval_freq, 100)
        self.assertEqual(cb.agent_name, "test_agent")
        self.assertEqual(cb.opponents, ["scripted_l3"])
        self.assertEqual(cb.n_eval_episodes, 5)
        self.assertEqual(cb.seed, 42)

    def test_env_kwargs_stored(self) -> None:
        """env_kwargs passed to the constructor are stored on the callback."""
        from training.elo import EloRegistry
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        kw = {"map_width": 500.0, "max_steps": 100}
        cb = EloEvalCallback(
            opponents=["scripted_l3"],
            n_eval_episodes=2,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
            env_kwargs=kw,
        )
        self.assertEqual(cb.env_kwargs["map_width"], 500.0)
        self.assertEqual(cb.env_kwargs["max_steps"], 100)

    def test_on_step_does_not_trigger_before_freq(self) -> None:
        """_on_step does not run evaluation before eval_freq steps."""
        train_mod = _TRAIN_MODULE
        from training.elo import EloRegistry
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        cb = EloEvalCallback(
            opponents=["scripted_l3"],
            n_eval_episodes=2,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 50  # less than eval_freq

        with patch.object(train_mod, "run_episodes_with_model") as mock_run:
            cb._on_step()
            mock_run.assert_not_called()

    def test_on_step_triggers_at_eval_freq(self) -> None:
        """_on_step evaluates and logs to W&B when eval_freq is reached."""
        train_mod = _TRAIN_MODULE
        from training.elo import EloRegistry
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        cb = EloEvalCallback(
            opponents=["scripted_l3"],
            n_eval_episodes=2,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 100

        fake_result = self._make_fake_result(wins=1, losses=1, n=2)
        with (
            patch.object(train_mod, "run_episodes_with_model", return_value=fake_result),
            patch.object(train_mod, "wandb") as mock_wandb,
        ):
            cb._on_step()
            mock_wandb.log.assert_called_once()
            logged = mock_wandb.log.call_args[0][0]
            self.assertIn("elo/rating_vs_scripted_l3", logged)
            self.assertIn("elo/win_rate_vs_scripted_l3", logged)
            self.assertIn("elo/delta_vs_scripted_l3", logged)

    def test_on_step_does_not_trigger_twice_at_same_step(self) -> None:
        """_on_step does not re-evaluate if called twice at the same timestep."""
        train_mod = _TRAIN_MODULE
        from training.elo import EloRegistry
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        cb = EloEvalCallback(
            opponents=["scripted_l3"],
            n_eval_episodes=2,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 100

        fake_result = self._make_fake_result()
        with (
            patch.object(train_mod, "run_episodes_with_model", return_value=fake_result),
            patch.object(train_mod, "wandb"),
        ):
            cb._on_step()  # first call — should trigger
            cb._on_step()  # second call at same timestep — should NOT trigger

        # run_episodes_with_model called only once (first trigger)
        with patch.object(train_mod, "run_episodes_with_model") as mock_run:
            cb.num_timesteps = 100  # same step
            cb._on_step()
            mock_run.assert_not_called()

    def test_elo_registry_updated_on_win(self) -> None:
        """Rating increases when the agent wins all evaluation episodes."""
        train_mod = _TRAIN_MODULE
        from training.elo import EloRegistry, DEFAULT_RATING
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        cb = EloEvalCallback(
            opponents=["scripted_l5"],
            n_eval_episodes=10,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 100

        # All wins → rating should exceed default
        fake_result = self._make_fake_result(wins=10, losses=0, n=10)
        with (
            patch.object(train_mod, "run_episodes_with_model", return_value=fake_result),
            patch.object(train_mod, "wandb"),
        ):
            cb._on_step()

        self.assertGreater(registry.get_rating("test_agent"), DEFAULT_RATING)

    def test_elo_registry_updated_on_loss(self) -> None:
        """Rating decreases when the agent loses all evaluation episodes."""
        train_mod = _TRAIN_MODULE
        from training.elo import EloRegistry, DEFAULT_RATING
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        cb = EloEvalCallback(
            opponents=["scripted_l5"],
            n_eval_episodes=10,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 100

        # All losses → rating should fall below default
        fake_result = self._make_fake_result(wins=0, losses=10, n=10)
        with (
            patch.object(train_mod, "run_episodes_with_model", return_value=fake_result),
            patch.object(train_mod, "wandb"),
        ):
            cb._on_step()

        self.assertLess(registry.get_rating("test_agent"), DEFAULT_RATING)

    def test_registry_saved_to_disk(self) -> None:
        """_run_elo_eval persists the registry when it has a file path."""
        import tempfile
        train_mod = _TRAIN_MODULE
        from training.elo import EloRegistry
        from training.train import EloEvalCallback

        with tempfile.TemporaryDirectory() as tmp:
            reg_path = Path(tmp) / "elo.json"
            registry = EloRegistry(path=reg_path)
            cb = EloEvalCallback(
                opponents=["scripted_l3"],
                n_eval_episodes=2,
                registry=registry,
                agent_name="test_agent",
                eval_freq=10,
            )
            cb.model = MagicMock()
            cb.num_timesteps = 10

            fake_result = self._make_fake_result()
            with (
                patch.object(train_mod, "run_episodes_with_model", return_value=fake_result),
                patch.object(train_mod, "wandb"),
            ):
                cb._on_step()

            self.assertTrue(reg_path.exists())

    def test_env_kwargs_forwarded_to_run_episodes(self) -> None:
        """env_kwargs stored on callback are forwarded to run_episodes_with_model."""
        train_mod = _TRAIN_MODULE
        from training.elo import EloRegistry
        from training.train import EloEvalCallback

        registry = EloRegistry(path=None)
        kw = {"map_width": 500.0}
        cb = EloEvalCallback(
            opponents=["scripted_l3"],
            n_eval_episodes=2,
            registry=registry,
            agent_name="test_agent",
            eval_freq=100,
            env_kwargs=kw,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 100

        fake_result = self._make_fake_result()
        with (
            patch.object(train_mod, "run_episodes_with_model", return_value=fake_result) as mock_run,
            patch.object(train_mod, "wandb"),
        ):
            cb._on_step()
            call_kwargs = mock_run.call_args[1]
            self.assertIn("env_kwargs", call_kwargs)
            self.assertEqual(call_kwargs["env_kwargs"]["map_width"], 500.0)


if __name__ == "__main__":
    unittest.main()
