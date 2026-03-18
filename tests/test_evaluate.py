# tests/test_evaluate.py
"""Tests for training/evaluate.py."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.battalion_env import BattalionEnv
from models.mlp_policy import BattalionMlpPolicy
from training.evaluate import evaluate, main


# ---------------------------------------------------------------------------
# Shared checkpoint — trained once for the entire module
# ---------------------------------------------------------------------------

_SHARED_TMPDIR: tempfile.TemporaryDirectory | None = None
_SHARED_CHECKPOINT: str = ""


def setUpModule() -> None:  # noqa: N802
    global _SHARED_TMPDIR, _SHARED_CHECKPOINT
    _SHARED_TMPDIR = tempfile.TemporaryDirectory()
    env = make_vec_env(BattalionEnv, n_envs=1, seed=0)
    model = PPO(BattalionMlpPolicy, env, n_steps=32, batch_size=16, verbose=0)
    model.learn(total_timesteps=32)
    ckpt = Path(_SHARED_TMPDIR.name) / "shared_model"
    model.save(str(ckpt))
    env.close()
    _SHARED_CHECKPOINT = str(ckpt)


def tearDownModule() -> None:  # noqa: N802
    global _SHARED_TMPDIR
    if _SHARED_TMPDIR is not None:
        _SHARED_TMPDIR.cleanup()
        _SHARED_TMPDIR = None


class TestEvaluateFunction(unittest.TestCase):
    """Unit tests for the evaluate() function."""

    def _ckpt(self) -> str:
        return _SHARED_CHECKPOINT

    def test_evaluate_returns_float(self) -> None:
        """evaluate() returns a float win rate in [0, 1]."""
        win_rate = evaluate(self._ckpt(), n_episodes=3, seed=42)
        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_n_episodes(self) -> None:
        """Win rate numerator is consistent with n_episodes denominator."""
        n = 5
        win_rate = evaluate(self._ckpt(), n_episodes=n, seed=0)
        # wins must be a whole number of n
        wins = win_rate * n
        self.assertAlmostEqual(wins, round(wins), places=5)

    def test_evaluate_deterministic_reproducible(self) -> None:
        """Two deterministic runs with the same seed produce the same win rate."""
        r1 = evaluate(self._ckpt(), n_episodes=3, deterministic=True, seed=7)
        r2 = evaluate(self._ckpt(), n_episodes=3, deterministic=True, seed=7)
        self.assertAlmostEqual(r1, r2)

    def test_evaluate_no_seed(self) -> None:
        """evaluate() runs without error when seed is None."""
        win_rate = evaluate(self._ckpt(), n_episodes=2, seed=None)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_zero_episodes_raises(self) -> None:
        """evaluate() raises ValueError when n_episodes < 1."""
        with self.assertRaises(ValueError):
            evaluate(self._ckpt(), n_episodes=0)


class TestEvaluateMain(unittest.TestCase):
    """Tests for the CLI entry point main()."""

    def _ckpt(self) -> str:
        return _SHARED_CHECKPOINT

    def test_main_prints_win_rate(self) -> None:
        """main() prints a win-rate line to stdout."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--checkpoint", self._ckpt(), "--n-episodes", "3", "--seed", "0"])

        output = buf.getvalue()
        self.assertIn("Win rate:", output)
        self.assertIn("/3", output)

    def test_main_requires_checkpoint_arg(self) -> None:
        """main() exits with an error when --checkpoint is missing."""
        with self.assertRaises(SystemExit):
            main([])

    def test_main_stochastic_flag(self) -> None:
        """--stochastic flag is accepted without error."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "--checkpoint", self._ckpt(),
                "--n-episodes", "2",
                "--stochastic",
                "--seed", "1",
            ])
        self.assertIn("Win rate:", buf.getvalue())

    def test_main_invalid_n_episodes_exits(self) -> None:
        """--n-episodes 0 causes main() to exit with an error."""
        with self.assertRaises(SystemExit):
            main(["--checkpoint", self._ckpt(), "--n-episodes", "0"])


if __name__ == "__main__":
    unittest.main()
