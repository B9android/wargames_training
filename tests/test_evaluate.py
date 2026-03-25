# SPDX-License-Identifier: MIT
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
from training.evaluate import (
    EvaluationResult,
    evaluate,
    evaluate_detailed,
    main,
    run_episodes_with_model,
)


# ---------------------------------------------------------------------------
# Shared checkpoint — trained once for the entire module
# ---------------------------------------------------------------------------

_SHARED_TMPDIR: tempfile.TemporaryDirectory | None = None
_SHARED_CHECKPOINT: str = ""
_SHARED_MODEL: PPO | None = None


def setUpModule() -> None:  # noqa: N802
    global _SHARED_TMPDIR, _SHARED_CHECKPOINT, _SHARED_MODEL
    _SHARED_TMPDIR = tempfile.TemporaryDirectory()
    env = make_vec_env(BattalionEnv, n_envs=1, seed=0)
    model = PPO(BattalionMlpPolicy, env, n_steps=32, batch_size=16, verbose=0)
    model.learn(total_timesteps=32)
    ckpt = Path(_SHARED_TMPDIR.name) / "shared_model"
    model.save(str(ckpt))
    env.close()
    _SHARED_CHECKPOINT = str(ckpt)
    _SHARED_MODEL = model


def tearDownModule() -> None:  # noqa: N802
    global _SHARED_TMPDIR
    if _SHARED_TMPDIR is not None:
        _SHARED_TMPDIR.cleanup()
        _SHARED_TMPDIR = None


# ---------------------------------------------------------------------------
# TestEvaluateFunction
# ---------------------------------------------------------------------------


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

    def test_evaluate_with_scripted_l1_opponent(self) -> None:
        """evaluate() accepts 'scripted_l1' as opponent."""
        win_rate = evaluate(self._ckpt(), n_episodes=2, seed=1, opponent="scripted_l1")
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_with_scripted_l3_opponent(self) -> None:
        """evaluate() accepts 'scripted_l3' as opponent."""
        win_rate = evaluate(self._ckpt(), n_episodes=2, seed=1, opponent="scripted_l3")
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_with_random_opponent(self) -> None:
        """evaluate() accepts 'random' as opponent."""
        win_rate = evaluate(self._ckpt(), n_episodes=2, seed=2, opponent="random")
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_default_opponent_is_scripted_l5(self) -> None:
        """evaluate() default opponent (scripted_l5) matches explicit scripted_l5."""
        r_default = evaluate(self._ckpt(), n_episodes=3, seed=10)
        r_explicit = evaluate(self._ckpt(), n_episodes=3, seed=10, opponent="scripted_l5")
        self.assertAlmostEqual(r_default, r_explicit)


# ---------------------------------------------------------------------------
# TestEvaluateDetailed
# ---------------------------------------------------------------------------


class TestEvaluateDetailed(unittest.TestCase):
    """Unit tests for evaluate_detailed() and EvaluationResult."""

    def _ckpt(self) -> str:
        return _SHARED_CHECKPOINT

    def test_returns_evaluation_result(self) -> None:
        """evaluate_detailed() returns an EvaluationResult instance."""
        result = evaluate_detailed(self._ckpt(), n_episodes=3, seed=42)
        self.assertIsInstance(result, EvaluationResult)

    def test_counts_sum_to_n_episodes(self) -> None:
        """wins + draws + losses == n_episodes."""
        n = 5
        result = evaluate_detailed(self._ckpt(), n_episodes=n, seed=0)
        self.assertEqual(result.wins + result.draws + result.losses, n)
        self.assertEqual(result.n_episodes, n)

    def test_rates_sum_to_one(self) -> None:
        """win_rate + draw_rate + loss_rate ≈ 1.0."""
        result = evaluate_detailed(self._ckpt(), n_episodes=4, seed=1)
        self.assertAlmostEqual(
            result.win_rate + result.draw_rate + result.loss_rate, 1.0, places=5
        )

    def test_win_rate_consistent_with_wins(self) -> None:
        """win_rate == wins / n_episodes."""
        result = evaluate_detailed(self._ckpt(), n_episodes=4, seed=3)
        self.assertAlmostEqual(result.win_rate, result.wins / result.n_episodes, places=5)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed → same result."""
        r1 = evaluate_detailed(self._ckpt(), n_episodes=3, seed=99)
        r2 = evaluate_detailed(self._ckpt(), n_episodes=3, seed=99)
        self.assertEqual(r1, r2)

    def test_scripted_opponent_levels(self) -> None:
        """evaluate_detailed() works for every scripted curriculum level."""
        for level in range(1, 6):
            result = evaluate_detailed(
                self._ckpt(), n_episodes=2, seed=0, opponent=f"scripted_l{level}"
            )
            self.assertEqual(result.n_episodes, 2)

    def test_zero_episodes_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_detailed(self._ckpt(), n_episodes=0)


# ---------------------------------------------------------------------------
# TestRunEpisodesWithModel
# ---------------------------------------------------------------------------


class TestRunEpisodesWithModel(unittest.TestCase):
    """Tests for run_episodes_with_model() helper."""

    def test_returns_evaluation_result(self) -> None:
        result = run_episodes_with_model(
            _SHARED_MODEL, opponent="scripted_l3", n_episodes=3, seed=0
        )
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.n_episodes, 3)

    def test_counts_add_up(self) -> None:
        result = run_episodes_with_model(
            _SHARED_MODEL, n_episodes=5, seed=1
        )
        self.assertEqual(result.wins + result.draws + result.losses, 5)

    def test_random_opponent(self) -> None:
        result = run_episodes_with_model(
            _SHARED_MODEL, opponent="random", n_episodes=2, seed=42
        )
        self.assertIsInstance(result, EvaluationResult)

    def test_invalid_n_episodes(self) -> None:
        with self.assertRaises(ValueError):
            run_episodes_with_model(_SHARED_MODEL, n_episodes=0)


# ---------------------------------------------------------------------------
# TestEvaluateMain
# ---------------------------------------------------------------------------


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
        self.assertIn("in 3 episodes", output)

    def test_main_prints_opponent_and_elo(self) -> None:
        """main() prints opponent, win rate, and Elo lines."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "--checkpoint", self._ckpt(),
                "--n-episodes", "3",
                "--seed", "0",
                "--opponent", "scripted_l3",
            ])
        output = buf.getvalue()
        self.assertIn("Opponent:", output)
        self.assertIn("scripted_l3", output)
        self.assertIn("Win rate:", output)
        self.assertIn("Elo:", output)

    def test_main_elo_persists_to_registry(self) -> None:
        """--elo-registry causes the JSON file to be created."""
        with tempfile.TemporaryDirectory() as tmp:
            reg_path = str(Path(tmp) / "test_registry.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                main([
                    "--checkpoint", self._ckpt(),
                    "--n-episodes", "2",
                    "--seed", "0",
                    "--opponent", "scripted_l1",
                    "--elo-registry", reg_path,
                    "--agent-name", "test_agent",
                ])
            self.assertTrue(Path(reg_path).exists())
            output = buf.getvalue()
            self.assertIn("Registry:", output)

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

    def test_main_scripted_l5_default_output(self) -> None:
        """Default opponent (scripted_l5) is shown in output."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--checkpoint", self._ckpt(), "--n-episodes", "2", "--seed", "5"])
        output = buf.getvalue()
        self.assertIn("scripted_l5", output)

    def test_main_random_opponent(self) -> None:
        """--opponent random is accepted without error."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "--checkpoint", self._ckpt(),
                "--n-episodes", "2",
                "--opponent", "random",
                "--seed", "3",
            ])
        self.assertIn("Win rate:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

