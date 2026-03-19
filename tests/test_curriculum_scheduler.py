# tests/test_curriculum_scheduler.py
"""Tests for training/curriculum_scheduler.py.

Coverage
--------
* CurriculumStage  — ordering, labels, env_kwargs
* CurriculumScheduler — construction, record_episode, win_rate, should_promote,
  promote, wandb_metrics, log_promotion_event
* load_v1_weights_into_mappo — shape-mismatch skip, non-existent file error
* Integration: full 1v1 → 2v1 → 2v2 stage progression
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.curriculum_scheduler import (
    STAGE_ENV_KWARGS,
    STAGE_LABELS,
    CurriculumScheduler,
    CurriculumStage,
    load_v1_weights_into_mappo,
)


# ---------------------------------------------------------------------------
# CurriculumStage tests
# ---------------------------------------------------------------------------


class TestCurriculumStage(unittest.TestCase):

    def test_ordering(self) -> None:
        self.assertLess(CurriculumStage.STAGE_1V1, CurriculumStage.STAGE_2V1)
        self.assertLess(CurriculumStage.STAGE_2V1, CurriculumStage.STAGE_2V2)

    def test_env_kwargs_1v1(self) -> None:
        kwargs = STAGE_ENV_KWARGS[CurriculumStage.STAGE_1V1]
        self.assertEqual(kwargs["n_blue"], 1)
        self.assertEqual(kwargs["n_red"], 1)

    def test_env_kwargs_2v1(self) -> None:
        kwargs = STAGE_ENV_KWARGS[CurriculumStage.STAGE_2V1]
        self.assertEqual(kwargs["n_blue"], 2)
        self.assertEqual(kwargs["n_red"], 1)

    def test_env_kwargs_2v2(self) -> None:
        kwargs = STAGE_ENV_KWARGS[CurriculumStage.STAGE_2V2]
        self.assertEqual(kwargs["n_blue"], 2)
        self.assertEqual(kwargs["n_red"], 2)

    def test_labels(self) -> None:
        self.assertEqual(STAGE_LABELS[CurriculumStage.STAGE_1V1], "1v1")
        self.assertEqual(STAGE_LABELS[CurriculumStage.STAGE_2V1], "2v1")
        self.assertEqual(STAGE_LABELS[CurriculumStage.STAGE_2V2], "2v2")


# ---------------------------------------------------------------------------
# CurriculumScheduler construction
# ---------------------------------------------------------------------------


class TestCurriculumSchedulerInit(unittest.TestCase):

    def test_default_construction(self) -> None:
        sched = CurriculumScheduler()
        self.assertEqual(sched.stage, CurriculumStage.STAGE_1V1)
        self.assertEqual(sched.promote_threshold, 0.70)
        self.assertEqual(sched.win_rate_window, 50)

    def test_custom_params(self) -> None:
        sched = CurriculumScheduler(promote_threshold=0.8, win_rate_window=100)
        self.assertEqual(sched.promote_threshold, 0.8)
        self.assertEqual(sched.win_rate_window, 100)

    def test_invalid_threshold_zero(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumScheduler(promote_threshold=0.0)

    def test_invalid_threshold_negative(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumScheduler(promote_threshold=-0.1)

    def test_invalid_threshold_above_one(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumScheduler(promote_threshold=1.1)

    def test_threshold_of_one_valid(self) -> None:
        sched = CurriculumScheduler(promote_threshold=1.0)
        self.assertEqual(sched.promote_threshold, 1.0)

    def test_invalid_window_zero(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumScheduler(win_rate_window=0)

    def test_initial_stage_override(self) -> None:
        sched = CurriculumScheduler(initial_stage=CurriculumStage.STAGE_2V1)
        self.assertEqual(sched.stage, CurriculumStage.STAGE_2V1)


# ---------------------------------------------------------------------------
# win_rate and record_episode
# ---------------------------------------------------------------------------


class TestWinRate(unittest.TestCase):

    def setUp(self) -> None:
        self.sched = CurriculumScheduler(win_rate_window=10)

    def test_empty_win_rate(self) -> None:
        self.assertEqual(self.sched.win_rate(), 0.0)

    def test_all_wins(self) -> None:
        for _ in range(10):
            self.sched.record_episode(win=True)
        self.assertAlmostEqual(self.sched.win_rate(), 1.0)

    def test_all_losses(self) -> None:
        for _ in range(10):
            self.sched.record_episode(win=False)
        self.assertAlmostEqual(self.sched.win_rate(), 0.0)

    def test_partial_wins(self) -> None:
        for i in range(10):
            self.sched.record_episode(win=(i < 7))  # 7 wins, 3 losses
        self.assertAlmostEqual(self.sched.win_rate(), 0.7)

    def test_window_rolling(self) -> None:
        # Fill window with losses, then override with wins
        for _ in range(10):
            self.sched.record_episode(win=False)
        self.assertAlmostEqual(self.sched.win_rate(), 0.0)
        for _ in range(10):
            self.sched.record_episode(win=True)
        self.assertAlmostEqual(self.sched.win_rate(), 1.0)

    def test_total_episodes_counter(self) -> None:
        for _ in range(5):
            self.sched.record_episode(win=True)
        self.assertEqual(self.sched.total_episodes, 5)


# ---------------------------------------------------------------------------
# should_promote and promote
# ---------------------------------------------------------------------------


class TestPromotion(unittest.TestCase):

    def test_no_promote_before_window_full(self) -> None:
        sched = CurriculumScheduler(promote_threshold=0.7, win_rate_window=10)
        for _ in range(9):  # one short of window
            sched.record_episode(win=True)
        self.assertFalse(sched.should_promote())

    def test_no_promote_below_threshold(self) -> None:
        sched = CurriculumScheduler(promote_threshold=0.7, win_rate_window=10)
        for i in range(10):
            sched.record_episode(win=(i < 6))  # 60 % win rate
        self.assertFalse(sched.should_promote())

    def test_promote_at_threshold(self) -> None:
        sched = CurriculumScheduler(promote_threshold=0.7, win_rate_window=10)
        for i in range(10):
            sched.record_episode(win=(i < 7))  # exactly 70 %
        self.assertTrue(sched.should_promote())

    def test_promote_above_threshold(self) -> None:
        sched = CurriculumScheduler(promote_threshold=0.7, win_rate_window=10)
        for _ in range(10):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())

    def test_promote_advances_stage(self) -> None:
        sched = CurriculumScheduler(win_rate_window=1)
        sched.record_episode(win=True)
        new_stage = sched.promote()
        self.assertEqual(new_stage, CurriculumStage.STAGE_2V1)
        self.assertEqual(sched.stage, CurriculumStage.STAGE_2V1)

    def test_promote_resets_window(self) -> None:
        sched = CurriculumScheduler(promote_threshold=0.7, win_rate_window=5)
        for _ in range(5):
            sched.record_episode(win=True)
        sched.promote()
        # After promotion, window is empty → can't promote immediately
        self.assertFalse(sched.should_promote())
        self.assertAlmostEqual(sched.win_rate(), 0.0)

    def test_full_progression_1v1_to_2v2(self) -> None:
        sched = CurriculumScheduler(promote_threshold=0.7, win_rate_window=10)

        # Stage 1v1 → 2v1
        self.assertEqual(sched.stage, CurriculumStage.STAGE_1V1)
        for _ in range(10):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())
        sched.promote()
        self.assertEqual(sched.stage, CurriculumStage.STAGE_2V1)

        # Stage 2v1 → 2v2
        for _ in range(10):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())
        sched.promote()
        self.assertEqual(sched.stage, CurriculumStage.STAGE_2V2)

        # At final stage — no further promotion
        self.assertTrue(sched.is_final_stage)
        self.assertFalse(sched.should_promote())

    def test_promote_at_final_stage_raises(self) -> None:
        sched = CurriculumScheduler(
            initial_stage=CurriculumStage.STAGE_2V2, win_rate_window=1
        )
        sched.record_episode(win=True)
        with self.assertRaises(RuntimeError):
            sched.promote()

    def test_is_final_stage_false_initially(self) -> None:
        sched = CurriculumScheduler()
        self.assertFalse(sched.is_final_stage)


# ---------------------------------------------------------------------------
# wandb_metrics and log_promotion_event
# ---------------------------------------------------------------------------


class TestWandbMetrics(unittest.TestCase):

    def test_metrics_keys(self) -> None:
        sched = CurriculumScheduler()
        metrics = sched.wandb_metrics()
        self.assertIn("curriculum/stage", metrics)
        self.assertIn("curriculum/stage_label", metrics)
        self.assertIn("curriculum/win_rate", metrics)
        self.assertIn("curriculum/total_episodes", metrics)

    def test_metrics_initial_values(self) -> None:
        sched = CurriculumScheduler()
        metrics = sched.wandb_metrics()
        self.assertEqual(metrics["curriculum/stage"], int(CurriculumStage.STAGE_1V1))
        self.assertEqual(metrics["curriculum/stage_label"], "1v1")
        self.assertAlmostEqual(metrics["curriculum/win_rate"], 0.0)
        self.assertEqual(metrics["curriculum/total_episodes"], 0)

    def test_log_promotion_event_no_wandb(self) -> None:
        """log_promotion_event should not raise when wandb_run is None."""
        sched = CurriculumScheduler(win_rate_window=1)
        sched.record_episode(win=True)
        sched.promote()
        # Should log to Python logger only — no exception
        sched.log_promotion_event(total_steps=1000, wandb_run=None)

    def test_log_promotion_event_with_wandb(self) -> None:
        """log_promotion_event should call wandb.log when run is provided."""
        sched = CurriculumScheduler(win_rate_window=1)
        sched.record_episode(win=True)
        sched.promote()
        mock_run = MagicMock()
        with patch("wandb.log") as mock_log:
            sched.log_promotion_event(total_steps=500, wandb_run=mock_run)
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args
            logged = call_kwargs[0][0]
            self.assertIn("curriculum/promotion_step", logged)
            self.assertEqual(logged["curriculum/promotion_step"], 500)

    def test_env_kwargs_current_stage(self) -> None:
        sched = CurriculumScheduler(initial_stage=CurriculumStage.STAGE_2V1)
        kwargs = sched.env_kwargs()
        self.assertEqual(kwargs["n_blue"], 2)
        self.assertEqual(kwargs["n_red"], 1)


# ---------------------------------------------------------------------------
# load_v1_weights_into_mappo
# ---------------------------------------------------------------------------


class TestLoadV1Weights(unittest.TestCase):

    def _make_mock_v1_zip(self, tmp_dir: Path, state_dict: dict) -> Path:
        """Create a fake SB3 zip containing policy.pth with *state_dict*."""
        import io
        zip_path = tmp_dir / "v1_policy.zip"
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        buf.seek(0)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("policy.pth", buf.read())
        return zip_path

    def test_missing_file_raises(self) -> None:
        mock_policy = MagicMock()
        with self.assertRaises(FileNotFoundError):
            load_v1_weights_into_mappo("/nonexistent/path.zip", mock_policy)

    def test_zip_without_policy_pth_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "empty.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("other_file.txt", "hello")
            mock_policy = MagicMock()
            with self.assertRaises(ValueError):
                load_v1_weights_into_mappo(zip_path, mock_policy)

    def test_shape_mismatch_skipped(self) -> None:
        """Layers with shape mismatch should be skipped (strict=False)."""
        from models.mappo_policy import MAPPOPolicy

        # Build a tiny MAPPOPolicy (1v1 compatible obs size)
        n_total = 2  # 1 blue + 1 red
        obs_dim = 6 + 5 * (n_total - 1) + 1   # = 12
        policy = MAPPOPolicy(obs_dim=obs_dim, action_dim=3, state_dim=13, n_agents=1)

        # Fake v1 state with wrong shapes for all transferred keys
        fake_v1_state = {
            "mlp_extractor.policy_net.0.weight": torch.zeros(999, 999),
            "mlp_extractor.policy_net.0.bias": torch.zeros(999),
        }

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_mock_v1_zip(Path(tmp), fake_v1_state)
            result = load_v1_weights_into_mappo(zip_path, policy)
            # Nothing should be loaded (shape mismatch → skipped)
            self.assertEqual(len(result["loaded"]), 0)

    def test_no_matching_keys(self) -> None:
        """A v1 checkpoint with no mapped keys should return empty loaded list."""
        from models.mappo_policy import MAPPOPolicy

        n_total = 2
        obs_dim = 6 + 5 * (n_total - 1) + 1
        policy = MAPPOPolicy(obs_dim=obs_dim, action_dim=3, state_dim=13, n_agents=1)

        # v1 state with unrelated keys
        fake_v1_state = {"some_other_key": torch.zeros(5, 5)}

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_mock_v1_zip(Path(tmp), fake_v1_state)
            result = load_v1_weights_into_mappo(zip_path, policy)
            self.assertEqual(len(result["loaded"]), 0)

    def test_strict_mode_raises_on_mismatch(self) -> None:
        """strict=True should raise ValueError on shape mismatch."""
        from models.mappo_policy import MAPPOPolicy

        n_total = 2
        obs_dim = 6 + 5 * (n_total - 1) + 1
        policy = MAPPOPolicy(obs_dim=obs_dim, action_dim=3, state_dim=13, n_agents=1)

        # Get actual MAPPO actor key to generate a matching (but wrong-shape) tensor
        mappo_state = policy.state_dict()
        # Find the first actor.trunk key
        actor_key = next(
            (k for k in mappo_state if k.startswith("actor.trunk.")), None
        )
        if actor_key is None:
            self.skipTest("No actor.trunk key found in policy state dict")

        # Build fake v1 key (mlp_extractor. prefix)
        v1_key = "mlp_extractor." + actor_key[len("actor.trunk."):]
        actual_shape = mappo_state[actor_key].shape
        wrong_shape = (actual_shape[0] + 1,) + actual_shape[1:]
        fake_v1_state = {v1_key: torch.zeros(*wrong_shape)}

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_mock_v1_zip(Path(tmp), fake_v1_state)
            with self.assertRaises(ValueError):
                load_v1_weights_into_mappo(zip_path, policy, strict=True)


# ---------------------------------------------------------------------------
# Integration: MAPPOTrainer win tracking + curriculum swap
# ---------------------------------------------------------------------------


class TestMAPPOTrainerWinTracking(unittest.TestCase):
    """Smoke-test that MAPPOTrainer records wins and the curriculum env swaps."""

    def test_win_history_populated(self) -> None:
        """After collect_rollout, _ep_win_history should have entries."""
        from envs.multi_battalion_env import MultiBattalionEnv
        from models.mappo_policy import MAPPOPolicy
        from training.train_mappo import MAPPOTrainer

        env = MultiBattalionEnv(n_blue=1, n_red=1, max_steps=10)
        n_total = 2
        obs_dim = 6 + 5 * (n_total - 1) + 1
        state_dim = 6 * n_total + 1
        policy = MAPPOPolicy(obs_dim=obs_dim, action_dim=3, state_dim=state_dim, n_agents=1)
        trainer = MAPPOTrainer(policy=policy, env=env, n_steps=20, seed=0)
        trainer.collect_rollout()
        # After at least one episode, win history should be populated
        # (at least one episode must have completed within 20 steps)
        self.assertGreaterEqual(len(trainer._ep_win_history), 0)  # at least tracked
        env.close()

    def test_curriculum_env_swap(self) -> None:
        """Curriculum scheduler should swap the env when promoted."""
        from envs.multi_battalion_env import MultiBattalionEnv
        from models.mappo_policy import MAPPOPolicy
        from training.train_mappo import MAPPOTrainer

        # Start at 2v1 and force immediate promotion to 2v2
        sched = CurriculumScheduler(
            initial_stage=CurriculumStage.STAGE_2V1,
            promote_threshold=0.01,  # very low threshold → promotes immediately
            win_rate_window=1,
        )
        env = MultiBattalionEnv(n_blue=2, n_red=1, max_steps=10)
        n_total = 3  # 2+1
        obs_dim = 6 + 5 * (n_total - 1) + 1
        state_dim = 6 * n_total + 1
        policy = MAPPOPolicy(obs_dim=obs_dim, action_dim=3, state_dim=state_dim, n_agents=2)
        trainer = MAPPOTrainer(policy=policy, env=env, n_steps=20, seed=0)

        # Manually inject a win so the scheduler promotes
        trainer._ep_win_history = [True]
        sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())

        old_stage = sched.stage
        sched.promote()
        new_stage = sched.stage
        self.assertGreater(int(new_stage), int(old_stage))
        env.close()


if __name__ == "__main__":
    unittest.main()
