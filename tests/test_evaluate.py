# tests/test_evaluate.py
"""Tests for training/evaluate.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.battalion_env import BattalionEnv
from models.mlp_policy import BattalionMlpPolicy
from training.evaluate import evaluate, main


class TestEvaluateFunction(unittest.TestCase):
    """Unit tests for the evaluate() function."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train a tiny model and save it for use across tests."""
        cls.tmpdir = tempfile.TemporaryDirectory()
        env = make_vec_env(BattalionEnv, n_envs=1, seed=0)
        model = PPO(BattalionMlpPolicy, env, n_steps=32, batch_size=16, verbose=0)
        model.learn(total_timesteps=32)
        cls.checkpoint_path = Path(cls.tmpdir.name) / "test_model"
        model.save(str(cls.checkpoint_path))
        env.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_evaluate_returns_float(self) -> None:
        """evaluate() returns a float win rate in [0, 1]."""
        win_rate = evaluate(str(self.checkpoint_path), n_episodes=3, seed=42)
        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_n_episodes(self) -> None:
        """Win rate numerator is consistent with n_episodes denominator."""
        n = 5
        win_rate = evaluate(str(self.checkpoint_path), n_episodes=n, seed=0)
        # wins must be a whole number of n
        wins = win_rate * n
        self.assertAlmostEqual(wins, round(wins), places=5)

    def test_evaluate_deterministic_reproducible(self) -> None:
        """Two deterministic runs with the same seed produce the same win rate."""
        r1 = evaluate(str(self.checkpoint_path), n_episodes=3, deterministic=True, seed=7)
        r2 = evaluate(str(self.checkpoint_path), n_episodes=3, deterministic=True, seed=7)
        self.assertAlmostEqual(r1, r2)

    def test_evaluate_no_seed(self) -> None:
        """evaluate() runs without error when seed is None."""
        win_rate = evaluate(str(self.checkpoint_path), n_episodes=2, seed=None)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)


class TestEvaluateMain(unittest.TestCase):
    """Tests for the CLI entry point main()."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        env = make_vec_env(BattalionEnv, n_envs=1, seed=1)
        model = PPO(BattalionMlpPolicy, env, n_steps=32, batch_size=16, verbose=0)
        model.learn(total_timesteps=32)
        cls.checkpoint_path = Path(cls.tmpdir.name) / "cli_model"
        model.save(str(cls.checkpoint_path))
        env.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_main_prints_win_rate(self) -> None:
        """main() prints a win-rate line to stdout."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--checkpoint", str(self.checkpoint_path), "--n-episodes", "3", "--seed", "0"])

        output = buf.getvalue()
        self.assertIn("Win rate:", output)
        self.assertIn("/3", output)

    def test_main_requires_checkpoint_arg(self) -> None:
        """main() exits with an error when --checkpoint is missing."""
        with self.assertRaises(SystemExit):
            main([])

    def test_main_stochastic_flag(self) -> None:
        """--stochastic flag is accepted without error."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "--checkpoint", str(self.checkpoint_path),
                "--n-episodes", "2",
                "--stochastic",
                "--seed", "1",
            ])
        self.assertIn("Win rate:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
