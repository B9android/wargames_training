# tests/test_hrl_curriculum.py
"""Integration tests for E3.4 — Hierarchical Curriculum (bottom-up training).

Coverage
--------
1. :class:`~training.hrl_curriculum.HRLCurriculumScheduler` — unit tests for
   phase tracking, win-rate promotion, Elo criterion, and dual promotion.
2. :mod:`~training.utils.freeze_policy` — freeze helpers for MAPPO and SB3.
3. Full three-phase curriculum smoke test on a small scenario (~512 steps/phase).
   Each phase verifies that frozen-level weights are not modified during
   subsequent training.

Acceptance criteria (from E3.4 epic)
-------------------------------------
* Each phase trains in isolation without modifying frozen-level weights.
* Promotion criteria documented and enforced programmatically.
* Integration test passes on CI (small scenario, ~512 steps per phase).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.hrl_curriculum import (
    HRLPhase,
    HRLCurriculumScheduler,
    PHASE_LABELS,
    PHASE_DESCRIPTIONS,
)
from training.utils.freeze_policy import (
    freeze_mappo_policy,
    freeze_sb3_policy,
    assert_frozen,
    load_and_freeze_mappo,
)
from models.mappo_policy import MAPPOPolicy
from training.elo import EloRegistry
from envs.brigade_env import BrigadeEnv
from envs.division_env import DivisionEnv


# ---------------------------------------------------------------------------
# 1. HRLCurriculumScheduler unit tests
# ---------------------------------------------------------------------------


class TestHRLPhaseEnum(unittest.TestCase):
    """Sanity-check the HRLPhase enum."""

    def test_phase_values(self) -> None:
        self.assertEqual(int(HRLPhase.PHASE_1_BATTALION), 1)
        self.assertEqual(int(HRLPhase.PHASE_2_BRIGADE), 2)
        self.assertEqual(int(HRLPhase.PHASE_3_DIVISION), 3)

    def test_all_phases_have_labels(self) -> None:
        for phase in HRLPhase:
            self.assertIn(phase, PHASE_LABELS)
            self.assertIn(phase, PHASE_DESCRIPTIONS)
            self.assertIsInstance(PHASE_LABELS[phase], str)


class TestHRLCurriculumSchedulerInit(unittest.TestCase):
    """Verify scheduler initialises correctly."""

    def test_default_init(self) -> None:
        sched = HRLCurriculumScheduler()
        self.assertEqual(sched.phase, HRLPhase.PHASE_1_BATTALION)
        self.assertAlmostEqual(sched.win_rate_threshold, 0.70)
        self.assertEqual(sched.win_rate_window, 50)
        self.assertAlmostEqual(sched.elo_threshold, 800.0)

    def test_custom_init(self) -> None:
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.60,
            win_rate_window=20,
            elo_threshold=None,
            initial_phase=HRLPhase.PHASE_2_BRIGADE,
        )
        self.assertEqual(sched.phase, HRLPhase.PHASE_2_BRIGADE)
        self.assertIsNone(sched.elo_threshold)

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            HRLCurriculumScheduler(win_rate_threshold=0.0)
        with self.assertRaises(ValueError):
            HRLCurriculumScheduler(win_rate_threshold=1.5)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            HRLCurriculumScheduler(win_rate_window=0)


class TestHRLCurriculumSchedulerWinRate(unittest.TestCase):
    """Test win-rate tracking and promotion logic."""

    def test_initial_win_rate_zero(self) -> None:
        sched = HRLCurriculumScheduler()
        self.assertAlmostEqual(sched.win_rate(), 0.0)

    def test_win_rate_after_all_wins(self) -> None:
        sched = HRLCurriculumScheduler(win_rate_window=10)
        for _ in range(10):
            sched.record_episode(win=True)
        self.assertAlmostEqual(sched.win_rate(), 1.0)

    def test_win_rate_rolling_window(self) -> None:
        sched = HRLCurriculumScheduler(win_rate_window=4)
        # Push 4 losses then 4 wins
        for _ in range(4):
            sched.record_episode(win=False)
        for _ in range(4):
            sched.record_episode(win=True)
        # Window now contains 4 wins only (the 4 losses fell out)
        self.assertAlmostEqual(sched.win_rate(), 1.0)

    def test_should_promote_false_before_window_full(self) -> None:
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70, win_rate_window=10, elo_threshold=None
        )
        for _ in range(9):  # one short of full window
            sched.record_episode(win=True)
        self.assertFalse(sched.should_promote())

    def test_should_promote_true_win_rate_only(self) -> None:
        """When elo_threshold is None, win-rate alone should gate promotion."""
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70, win_rate_window=10, elo_threshold=None
        )
        for _ in range(10):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())

    def test_should_not_promote_at_final_phase(self) -> None:
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.50,
            win_rate_window=2,
            elo_threshold=None,
            initial_phase=HRLPhase.PHASE_3_DIVISION,
        )
        sched.record_episode(win=True)
        sched.record_episode(win=True)
        self.assertFalse(sched.should_promote())
        self.assertTrue(sched.is_final_phase)


class TestHRLCurriculumSchedulerEloCriterion(unittest.TestCase):
    """Test Elo-gated promotion."""

    def _make_scheduler(self, **kwargs) -> HRLCurriculumScheduler:
        return HRLCurriculumScheduler(
            win_rate_threshold=0.70,
            win_rate_window=5,
            elo_threshold=800.0,
            **kwargs,
        )

    def test_elo_blocks_promotion_when_not_set(self) -> None:
        sched = self._make_scheduler()
        for _ in range(5):
            sched.record_episode(win=True)
        # Win-rate met but Elo not yet recorded
        self.assertFalse(sched.should_promote())

    def test_elo_blocks_promotion_when_below_threshold(self) -> None:
        sched = self._make_scheduler()
        for _ in range(5):
            sched.record_episode(win=True)
        sched.set_elo(750.0)
        self.assertFalse(sched.should_promote())

    def test_both_criteria_met_promotes(self) -> None:
        sched = self._make_scheduler()
        for _ in range(5):
            sched.record_episode(win=True)
        sched.set_elo(820.0)
        self.assertTrue(sched.should_promote())

    def test_update_elo_via_registry(self) -> None:
        """update_elo() caches the rating and raises it after wins."""
        sched = self._make_scheduler()
        registry = EloRegistry(path=None)

        # Seed win-rate window
        for _ in range(5):
            sched.record_episode(win=True)

        # First update: agent starts at DEFAULT_RATING=1000; vs scripted_l3=800
        # so a win should push rating above 1000.
        rating_before = registry.get_rating("test_agent")  # DEFAULT_RATING
        sched.update_elo(
            registry,
            agent_name="test_agent",
            opponent="scripted_l3",
            win_rate=1.0,
            n_episodes=10,
        )
        rating_after = registry.get_rating("test_agent")

        # Rating must have increased from the default after a strong win
        self.assertGreater(rating_after, rating_before)
        # Cached Elo must be set
        self.assertIsNotNone(sched._current_elo)
        self.assertAlmostEqual(sched._current_elo, rating_after)


class TestHRLCurriculumSchedulerPromote(unittest.TestCase):
    """Test phase promotion transitions."""

    def test_promote_advances_phase(self) -> None:
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70, win_rate_window=5, elo_threshold=None
        )
        for _ in range(5):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())
        new_phase = sched.promote()
        self.assertEqual(new_phase, HRLPhase.PHASE_2_BRIGADE)
        self.assertEqual(sched.phase, HRLPhase.PHASE_2_BRIGADE)

    def test_promote_resets_win_rate_window(self) -> None:
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70, win_rate_window=5, elo_threshold=None
        )
        for _ in range(5):
            sched.record_episode(win=True)
        sched.promote()
        # Window should be empty after promotion
        self.assertAlmostEqual(sched.win_rate(), 0.0)
        self.assertEqual(sched.phase_episodes, 0)

    def test_promote_resets_elo(self) -> None:
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70, win_rate_window=5, elo_threshold=None
        )
        sched.set_elo(850.0)
        for _ in range(5):
            sched.record_episode(win=True)
        sched.promote()
        self.assertIsNone(sched._current_elo)

    def test_promote_past_final_raises(self) -> None:
        sched = HRLCurriculumScheduler(
            initial_phase=HRLPhase.PHASE_3_DIVISION, elo_threshold=None
        )
        with self.assertRaises(RuntimeError):
            sched.promote()

    def test_full_curriculum_three_phases(self) -> None:
        """Walk through all three phases programmatically."""
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70, win_rate_window=5, elo_threshold=None
        )
        self.assertEqual(sched.phase, HRLPhase.PHASE_1_BATTALION)

        for _ in range(5):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())
        sched.promote()
        self.assertEqual(sched.phase, HRLPhase.PHASE_2_BRIGADE)

        for _ in range(5):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())
        sched.promote()
        self.assertEqual(sched.phase, HRLPhase.PHASE_3_DIVISION)

        # Final phase — should not promote further
        for _ in range(5):
            sched.record_episode(win=True)
        self.assertFalse(sched.should_promote())
        self.assertTrue(sched.is_final_phase)


class TestHRLSchedulerPromotionStatus(unittest.TestCase):
    """Test promotion_status() dict contents."""

    def test_status_keys(self) -> None:
        sched = HRLCurriculumScheduler(elo_threshold=None)
        status = sched.promotion_status()
        expected_keys = {
            "phase", "phase_label", "win_rate", "win_rate_met",
            "elo", "elo_met", "should_promote", "phase_episodes",
            "total_episodes",
        }
        self.assertEqual(set(status.keys()), expected_keys)

    def test_wandb_metrics_keys(self) -> None:
        sched = HRLCurriculumScheduler()
        metrics = sched.wandb_metrics()
        for key in (
            "hrl_curriculum/phase",
            "hrl_curriculum/phase_label",
            "hrl_curriculum/win_rate",
            "hrl_curriculum/elo",
            "hrl_curriculum/phase_episodes",
            "hrl_curriculum/total_episodes",
        ):
            self.assertIn(key, metrics)


# ---------------------------------------------------------------------------
# 2. Freeze policy utility tests
# ---------------------------------------------------------------------------


class TestFreezeMAPPOPolicy(unittest.TestCase):
    """Verify freeze_mappo_policy and assert_frozen for MAPPOPolicy."""

    def _make_policy(self) -> MAPPOPolicy:
        return MAPPOPolicy(
            obs_dim=22, action_dim=3, state_dim=25, n_agents=2
        )

    def test_freeze_sets_no_grad(self) -> None:
        policy = self._make_policy()
        # Before freezing, at least some params should require grad
        trainable_before = sum(
            p.numel() for p in policy.parameters() if p.requires_grad
        )
        self.assertGreater(trainable_before, 0)

        freeze_mappo_policy(policy)

        trainable_after = sum(
            p.numel() for p in policy.parameters() if p.requires_grad
        )
        self.assertEqual(trainable_after, 0)

    def test_assert_frozen_passes_after_freeze(self) -> None:
        policy = self._make_policy()
        freeze_mappo_policy(policy)
        assert_frozen(policy)  # must not raise

    def test_assert_frozen_raises_if_not_frozen(self) -> None:
        policy = self._make_policy()
        with self.assertRaises(RuntimeError):
            assert_frozen(policy)

    def test_freeze_sets_eval_mode(self) -> None:
        policy = self._make_policy()
        policy.train()
        freeze_mappo_policy(policy)
        self.assertFalse(policy.training)

    def test_frozen_policy_weights_unchanged_by_forward(self) -> None:
        """Forward pass on a frozen policy must not change weights."""
        policy = self._make_policy()
        freeze_mappo_policy(policy)

        # Record initial weights
        initial_weights = {
            k: v.clone() for k, v in policy.state_dict().items()
        }

        # Run a forward pass
        obs = torch.zeros(2, 22)
        with torch.no_grad():
            policy.act(obs)

        for k, v in policy.state_dict().items():
            self.assertTrue(
                torch.allclose(v, initial_weights[k]),
                f"Weight {k} changed after forward pass on frozen policy",
            )


class TestFreezeSB3Policy(unittest.TestCase):
    """Verify freeze_sb3_policy for SB3 PPO."""

    def _make_sb3_model(self):
        """Build a tiny SB3 PPO model for testing."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym

        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = PPO("MlpPolicy", env, verbose=0)
        env.close()
        return model

    def test_freeze_sb3_sets_no_grad(self) -> None:
        model = self._make_sb3_model()
        freeze_sb3_policy(model)
        trainable = sum(
            p.numel() for p in model.policy.parameters() if p.requires_grad
        )
        self.assertEqual(trainable, 0)

    def test_assert_frozen_passes_after_sb3_freeze(self) -> None:
        model = self._make_sb3_model()
        freeze_sb3_policy(model)
        assert_frozen(model.policy)

    def test_freeze_sb3_invalid_object_raises(self) -> None:
        with self.assertRaises(AttributeError):
            freeze_sb3_policy(object())

    def test_frozen_sb3_weights_unchanged_after_brigade_train(self) -> None:
        """Frozen battalion policy weights must not change during brigade training."""
        # Build a fresh (random) MAPPOPolicy and freeze it
        policy = MAPPOPolicy(obs_dim=22, action_dim=3, state_dim=25, n_agents=2)
        freeze_mappo_policy(policy)

        # Record initial weights
        initial_state = {k: v.clone() for k, v in policy.state_dict().items()}

        # Run a brief BrigadeEnv episode using the frozen policy
        env = BrigadeEnv(
            n_blue=2, n_red=2,
            max_steps=20,
            map_width=300.0, map_height=300.0,
            battalion_policy=policy,
        )
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            action = env.action_space.sample()
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        env.close()

        # Weights must be unchanged
        for k, v in policy.state_dict().items():
            self.assertTrue(
                torch.allclose(v, initial_state[k]),
                f"Frozen battalion weight '{k}' was modified during brigade episode.",
            )


# ---------------------------------------------------------------------------
# 3. Integration test — three-phase curriculum (small scenario, 10k steps)
# ---------------------------------------------------------------------------


class TestHRLCurriculumIntegration(unittest.TestCase):
    """Full three-phase curriculum smoke test.

    Trains each phase for a very small number of steps (enough to verify no
    crashes, correct API usage, and frozen-weight integrity).  The step count
    is deliberately tiny so the CI passes quickly.
    """

    # Maximum wall-time per phase: ~30 seconds on CI
    PHASE1_STEPS = 512   # MAPPO battalion
    PHASE2_STEPS = 512   # brigade PPO
    PHASE3_STEPS = 512   # division PPO
    N_EVAL_EPISODES = 5

    @classmethod
    def _make_mappo_policy(cls) -> MAPPOPolicy:
        """Create a small random MAPPOPolicy (simulates a Phase 1 checkpoint)."""
        policy = MAPPOPolicy(obs_dim=22, action_dim=3, state_dim=25, n_agents=2)
        return policy

    def test_phase1_produces_trainable_policy(self) -> None:
        """Phase 1: build a MAPPOPolicy and verify it has trainable params."""
        policy = self._make_mappo_policy()
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.assertGreater(trainable, 0, "Phase 1 policy must have trainable params")

    def test_phase1_to_phase2_freeze_and_brigade_train(self) -> None:
        """Phase 1→2: freeze battalion policy, run brief brigade training."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Simulate end of Phase 1: build and freeze battalion policy
        battalion_policy = self._make_mappo_policy()
        freeze_mappo_policy(battalion_policy)
        assert_frozen(battalion_policy)

        # Record pre-training weights
        pre_weights = {k: v.clone() for k, v in battalion_policy.state_dict().items()}

        # Phase 2: train brigade PPO with frozen battalion policy
        def _make_env():
            return BrigadeEnv(
                n_blue=2, n_red=2,
                max_steps=30,
                map_width=300.0, map_height=300.0,
                battalion_policy=battalion_policy,
            )

        train_env = DummyVecEnv([_make_env])
        model = PPO(
            "MlpPolicy", train_env,
            n_steps=64, batch_size=32, n_epochs=1,
            verbose=0, seed=0,
        )
        model.learn(total_timesteps=self.PHASE2_STEPS)
        train_env.close()

        # Frozen battalion weights must not have changed
        for k, v in battalion_policy.state_dict().items():
            self.assertTrue(
                torch.allclose(v, pre_weights[k]),
                f"Frozen battalion weight '{k}' changed during Phase 2 brigade training.",
            )

    def test_phase2_to_phase3_freeze_and_division_train(self) -> None:
        """Phase 2→3: freeze brigade policy, run brief division training.

        Note: the brigade model must be trained with ``n_blue`` equal to the
        total number of Red battalions in DivisionEnv (``n_red_brigades *
        n_red_per_brigade``).  Here we use 1 Red brigade × 2 battalions = 2,
        so the brigade policy is trained with ``n_blue=2``.
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        # --- Phase 2: train brigade model with n_blue=2 ---
        # The Red side of DivisionEnv will be 1 brigade × 2 battalions = 2
        # total Red battalions, so the brigade policy obs has shape
        # 3 + 7*2 + 1 = 18, matching a BrigadeEnv(n_blue=2).
        def _make_brigade_env():
            return BrigadeEnv(
                n_blue=2, n_red=2,
                max_steps=30,
                map_width=300.0, map_height=300.0,
            )

        brigade_train_env = DummyVecEnv([_make_brigade_env])
        brigade_model = PPO(
            "MlpPolicy", brigade_train_env,
            n_steps=64, batch_size=32, n_epochs=1,
            verbose=0, seed=0,
        )
        brigade_model.learn(total_timesteps=self.PHASE2_STEPS)
        brigade_train_env.close()

        # Freeze the brigade model
        freeze_sb3_policy(brigade_model)
        assert_frozen(brigade_model.policy)

        # Record pre-Phase-3 weights
        pre_weights = {
            k: v.clone() for k, v in brigade_model.policy.state_dict().items()
        }

        # --- Phase 3: train division PPO with frozen brigade policy ---
        # Use 1 Red brigade with 2 Red battalions per brigade so that the
        # Red observation fed to brigade_model has obs_dim = 3 + 7*2 + 1 = 18,
        # matching the brigade model's training observation space.
        def _make_division_env():
            return DivisionEnv(
                n_brigades=2, n_blue_per_brigade=2,
                n_red_brigades=1, n_red_per_brigade=2,
                max_steps=30,
                map_width=300.0, map_height=300.0,
                brigade_policy=brigade_model,
            )

        div_train_env = DummyVecEnv([_make_division_env])
        div_model = PPO(
            "MlpPolicy", div_train_env,
            n_steps=64, batch_size=32, n_epochs=1,
            verbose=0, seed=0,
        )
        div_model.learn(total_timesteps=self.PHASE3_STEPS)
        div_train_env.close()

        # Frozen brigade weights must not have changed
        for k, v in brigade_model.policy.state_dict().items():
            self.assertTrue(
                torch.allclose(v, pre_weights[k]),
                f"Frozen brigade weight '{k}' changed during Phase 3 division training.",
            )

    def test_scheduler_promotion_criteria_in_curriculum(self) -> None:
        """Verify the scheduler correctly gates phase transitions."""
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70,
            win_rate_window=5,
            elo_threshold=None,  # skip Elo for simplicity
        )
        self.assertEqual(sched.phase, HRLPhase.PHASE_1_BATTALION)

        # Simulate Phase 1 — feed 5 wins
        for _ in range(5):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())
        phase = sched.promote()
        self.assertEqual(phase, HRLPhase.PHASE_2_BRIGADE)

        # Simulate Phase 2 — feed 5 wins
        for _ in range(5):
            sched.record_episode(win=True)
        self.assertTrue(sched.should_promote())
        phase = sched.promote()
        self.assertEqual(phase, HRLPhase.PHASE_3_DIVISION)

        # Phase 3 — final, no more promotions
        for _ in range(5):
            sched.record_episode(win=True)
        self.assertFalse(sched.should_promote())
        self.assertTrue(sched.is_final_phase)

    def test_scheduler_dual_criteria_enforced(self) -> None:
        """Both win-rate AND Elo must be met to promote."""
        sched = HRLCurriculumScheduler(
            win_rate_threshold=0.70,
            win_rate_window=5,
            elo_threshold=800.0,
        )

        # Meet win-rate but not Elo
        for _ in range(5):
            sched.record_episode(win=True)
        sched.set_elo(750.0)
        self.assertFalse(sched.should_promote(), "Should not promote without Elo")

        # Meet Elo but not win-rate (reset with losses)
        sched2 = HRLCurriculumScheduler(
            win_rate_threshold=0.70,
            win_rate_window=5,
            elo_threshold=800.0,
        )
        for _ in range(5):
            sched2.record_episode(win=False)
        sched2.set_elo(900.0)
        self.assertFalse(sched2.should_promote(), "Should not promote without win-rate")

        # Meet both
        sched3 = HRLCurriculumScheduler(
            win_rate_threshold=0.70,
            win_rate_window=5,
            elo_threshold=800.0,
        )
        for _ in range(5):
            sched3.record_episode(win=True)
        sched3.set_elo(850.0)
        self.assertTrue(sched3.should_promote(), "Should promote with both criteria met")

    def test_load_and_freeze_mappo_missing_file_raises(self) -> None:
        """load_and_freeze_mappo raises FileNotFoundError for missing path."""
        with self.assertRaises(FileNotFoundError):
            load_and_freeze_mappo(
                checkpoint_path="/tmp/nonexistent_mappo.pt",
                obs_dim=22, action_dim=3, state_dim=25, n_agents=2,
            )

    def test_load_and_freeze_mappo_from_saved_checkpoint(self) -> None:
        """Save a MAPPOPolicy checkpoint, reload via load_and_freeze_mappo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_mappo.pt"

            # Save a tiny policy
            policy = MAPPOPolicy(obs_dim=22, action_dim=3, state_dim=25, n_agents=2)
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "total_steps": 1000,
                },
                ckpt_path,
            )

            # Reload and freeze
            loaded = load_and_freeze_mappo(
                checkpoint_path=ckpt_path,
                obs_dim=22, action_dim=3, state_dim=25, n_agents=2,
            )
            assert_frozen(loaded)

            # Weights must match
            for k in policy.state_dict():
                self.assertTrue(
                    torch.allclose(
                        policy.state_dict()[k],
                        loaded.state_dict()[k],
                    ),
                    f"Loaded weight '{k}' does not match original.",
                )


if __name__ == "__main__":
    unittest.main()
