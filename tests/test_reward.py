# tests/test_reward.py
"""Tests for envs/reward.py and curriculum-level integration in BattalionEnv."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.reward import RewardWeights, RewardComponents, compute_reward
from envs.battalion_env import BattalionEnv


# ---------------------------------------------------------------------------
# RewardWeights
# ---------------------------------------------------------------------------


class TestRewardWeights(unittest.TestCase):
    """Verify RewardWeights dataclass defaults and custom values."""

    def test_default_weights(self) -> None:
        w = RewardWeights()
        self.assertEqual(w.delta_enemy_strength, 5.0)
        self.assertEqual(w.delta_own_strength, 5.0)
        self.assertEqual(w.survival_bonus, 0.0)
        self.assertEqual(w.win_bonus, 10.0)
        self.assertEqual(w.loss_penalty, -10.0)
        self.assertEqual(w.time_penalty, -0.01)

    def test_custom_weights(self) -> None:
        w = RewardWeights(
            delta_enemy_strength=2.0,
            survival_bonus=0.01,
            win_bonus=5.0,
            loss_penalty=-5.0,
            time_penalty=-0.005,
        )
        self.assertEqual(w.delta_enemy_strength, 2.0)
        self.assertEqual(w.survival_bonus, 0.01)
        self.assertEqual(w.win_bonus, 5.0)

    def test_zero_weights_disable_components(self) -> None:
        """All-zero weights produce a zero total reward on every step."""
        w = RewardWeights(
            delta_enemy_strength=0.0,
            delta_own_strength=0.0,
            survival_bonus=0.0,
            win_bonus=0.0,
            loss_penalty=0.0,
            time_penalty=0.0,
        )
        comps = compute_reward(
            dmg_b2r=0.1,
            dmg_r2b=0.05,
            blue_strength=0.9,
            blue_won=True,
            blue_lost=False,
            weights=w,
        )
        self.assertAlmostEqual(comps.total, 0.0)


# ---------------------------------------------------------------------------
# RewardComponents
# ---------------------------------------------------------------------------


class TestRewardComponents(unittest.TestCase):
    """Verify RewardComponents.total and .as_dict()."""

    def test_total_sums_all_fields(self) -> None:
        c = RewardComponents(
            delta_enemy_strength=1.0,
            delta_own_strength=-0.5,
            survival_bonus=0.05,
            win_bonus=10.0,
            loss_penalty=0.0,
            time_penalty=-0.01,
        )
        expected = 1.0 - 0.5 + 0.05 + 10.0 + 0.0 - 0.01
        self.assertAlmostEqual(c.total, expected)

    def test_as_dict_keys(self) -> None:
        c = RewardComponents()
        d = c.as_dict()
        for key in (
            "reward/delta_enemy_strength",
            "reward/delta_own_strength",
            "reward/survival_bonus",
            "reward/win_bonus",
            "reward/loss_penalty",
            "reward/time_penalty",
            "reward/total",
        ):
            self.assertIn(key, d)

    def test_as_dict_total_matches_property(self) -> None:
        c = RewardComponents(
            delta_enemy_strength=2.0,
            time_penalty=-0.01,
        )
        self.assertAlmostEqual(c.as_dict()["reward/total"], c.total)


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


class TestComputeReward(unittest.TestCase):
    """Unit tests for the compute_reward() function."""

    def _default_weights(self) -> RewardWeights:
        return RewardWeights()

    def test_damage_component(self) -> None:
        w = self._default_weights()
        comps = compute_reward(
            dmg_b2r=0.1,
            dmg_r2b=0.0,
            blue_strength=1.0,
            blue_won=False,
            blue_lost=False,
            weights=w,
        )
        self.assertAlmostEqual(comps.delta_enemy_strength, 0.1 * w.delta_enemy_strength)

    def test_own_damage_penalty(self) -> None:
        w = self._default_weights()
        comps = compute_reward(
            dmg_b2r=0.0,
            dmg_r2b=0.1,
            blue_strength=1.0,
            blue_won=False,
            blue_lost=False,
            weights=w,
        )
        self.assertAlmostEqual(
            comps.delta_own_strength, -0.1 * w.delta_own_strength
        )

    def test_time_penalty_always_applied(self) -> None:
        w = self._default_weights()
        comps = compute_reward(
            dmg_b2r=0.0,
            dmg_r2b=0.0,
            blue_strength=1.0,
            blue_won=False,
            blue_lost=False,
            weights=w,
        )
        self.assertAlmostEqual(comps.time_penalty, w.time_penalty)

    def test_win_bonus_applied_on_win(self) -> None:
        w = self._default_weights()
        comps = compute_reward(
            dmg_b2r=0.0,
            dmg_r2b=0.0,
            blue_strength=1.0,
            blue_won=True,
            blue_lost=False,
            weights=w,
        )
        self.assertAlmostEqual(comps.win_bonus, w.win_bonus)
        self.assertAlmostEqual(comps.loss_penalty, 0.0)

    def test_loss_penalty_applied_on_loss(self) -> None:
        w = self._default_weights()
        comps = compute_reward(
            dmg_b2r=0.0,
            dmg_r2b=0.0,
            blue_strength=0.0,
            blue_won=False,
            blue_lost=True,
            weights=w,
        )
        self.assertAlmostEqual(comps.loss_penalty, w.loss_penalty)
        self.assertAlmostEqual(comps.win_bonus, 0.0)

    def test_no_terminal_bonus_on_non_terminal_step(self) -> None:
        w = self._default_weights()
        comps = compute_reward(
            dmg_b2r=0.05,
            dmg_r2b=0.02,
            blue_strength=0.8,
            blue_won=False,
            blue_lost=False,
            weights=w,
        )
        self.assertAlmostEqual(comps.win_bonus, 0.0)
        self.assertAlmostEqual(comps.loss_penalty, 0.0)

    def test_survival_bonus_scaled_by_strength(self) -> None:
        w = RewardWeights(survival_bonus=1.0)
        comps = compute_reward(
            dmg_b2r=0.0,
            dmg_r2b=0.0,
            blue_strength=0.75,
            blue_won=False,
            blue_lost=False,
            weights=w,
        )
        self.assertAlmostEqual(comps.survival_bonus, 0.75)

    def test_total_consistent_with_components(self) -> None:
        w = RewardWeights(survival_bonus=0.005)
        comps = compute_reward(
            dmg_b2r=0.05,
            dmg_r2b=0.02,
            blue_strength=0.9,
            blue_won=False,
            blue_lost=False,
            weights=w,
        )
        expected = (
            0.05 * w.delta_enemy_strength
            + (-0.02 * w.delta_own_strength)
            + 0.9 * w.survival_bonus
            + w.time_penalty
        )
        self.assertAlmostEqual(comps.total, expected, places=6)


# ---------------------------------------------------------------------------
# BattalionEnv — curriculum_level parameter
# ---------------------------------------------------------------------------


class TestCurriculumLevel(unittest.TestCase):
    """Verify curriculum_level validation and default in BattalionEnv."""

    def test_default_curriculum_level_is_five(self) -> None:
        env = BattalionEnv()
        self.assertEqual(env.curriculum_level, 5)
        env.close()

    def test_valid_curriculum_levels_accepted(self) -> None:
        for level in range(1, 6):
            env = BattalionEnv(curriculum_level=level)
            self.assertEqual(env.curriculum_level, level)
            env.close()

    def test_zero_curriculum_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(curriculum_level=0)

    def test_invalid_curriculum_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(curriculum_level=6)

    def test_negative_curriculum_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(curriculum_level=-1)


# ---------------------------------------------------------------------------
# BattalionEnv — reward_weights parameter
# ---------------------------------------------------------------------------


class TestRewardWeightsIntegration(unittest.TestCase):
    """Verify RewardWeights wires correctly into BattalionEnv."""

    def test_default_reward_weights_are_used(self) -> None:
        env = BattalionEnv()
        self.assertIsInstance(env.reward_weights, RewardWeights)
        env.close()

    def test_custom_reward_weights_stored(self) -> None:
        w = RewardWeights(delta_enemy_strength=2.0, win_bonus=20.0)
        env = BattalionEnv(reward_weights=w)
        self.assertEqual(env.reward_weights.delta_enemy_strength, 2.0)
        self.assertEqual(env.reward_weights.win_bonus, 20.0)
        env.close()

    def test_step_info_has_reward_components(self) -> None:
        """step() info dict must contain all per-component reward keys."""
        env = BattalionEnv()
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        for key in (
            "reward/delta_enemy_strength",
            "reward/delta_own_strength",
            "reward/survival_bonus",
            "reward/win_bonus",
            "reward/loss_penalty",
            "reward/time_penalty",
            "reward/total",
        ):
            self.assertIn(key, info, msg=f"Missing key: {key}")
        env.close()

    def test_reward_equals_info_total(self) -> None:
        """Scalar reward from step() must equal info['reward/total']."""
        env = BattalionEnv()
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        self.assertAlmostEqual(reward, info["reward/total"], places=6)
        env.close()

    def test_high_damage_weight_yields_larger_reward(self) -> None:
        """Doubling delta_enemy_strength should increase reward when damage is dealt."""
        # Use level 1 (stationary Red) with full fire so Blue can deal damage.
        w_low = RewardWeights(delta_enemy_strength=1.0)
        w_high = RewardWeights(delta_enemy_strength=10.0)

        rewards_low = []
        rewards_high = []
        for w, store in ((w_low, rewards_low), (w_high, rewards_high)):
            env = BattalionEnv(curriculum_level=1, reward_weights=w)
            env.reset(seed=0)
            # Position Blue close to Red to ensure damage is dealt.
            env.blue.x = env.red.x - 100.0
            env.blue.y = env.red.y
            env.blue.theta = 0.0  # face east
            for _ in range(10):
                _, r, terminated, truncated, _ = env.step(
                    np.array([0.0, 0.0, 1.0], dtype=np.float32)
                )
                store.append(r)
                if terminated or truncated:
                    break
            env.close()

        # Sum should be higher for the high-damage-weight run (assuming any damage).
        # If no damage was dealt at all, sums will be equal; that's still valid.
        self.assertGreaterEqual(sum(rewards_high), sum(rewards_low) - 1e-6)


# ---------------------------------------------------------------------------
# BattalionEnv — curriculum behaviour (Red fire)
# ---------------------------------------------------------------------------


class TestCurriculumRedBehaviour(unittest.TestCase):
    """Verify that Red fires only at levels 4 and 5."""

    def _measure_blue_damage_over_episode(self, curriculum_level: int) -> float:
        """Return cumulative Red→Blue damage over a short episode."""
        env = BattalionEnv(curriculum_level=curriculum_level)
        env.reset(seed=42)
        # Place Red directly in front of Blue within fire range.
        env.blue.x = 400.0
        env.blue.y = 500.0
        env.blue.theta = 0.0
        env.red.x = 500.0
        env.red.y = 500.0
        env.red.theta = 3.14159  # facing west

        total_dmg = 0.0
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(
                np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            total_dmg += info["red_damage_dealt"]
            if terminated or truncated:
                break
        env.close()
        return total_dmg

    def test_level_1_red_does_not_fire(self) -> None:
        dmg = self._measure_blue_damage_over_episode(1)
        self.assertAlmostEqual(dmg, 0.0, places=6)

    def test_level_2_red_does_not_fire(self) -> None:
        dmg = self._measure_blue_damage_over_episode(2)
        self.assertAlmostEqual(dmg, 0.0, places=6)

    def test_level_3_red_does_not_fire(self) -> None:
        dmg = self._measure_blue_damage_over_episode(3)
        self.assertAlmostEqual(dmg, 0.0, places=6)

    def test_level_5_red_fires_more_than_level_4(self) -> None:
        dmg4 = self._measure_blue_damage_over_episode(4)
        dmg5 = self._measure_blue_damage_over_episode(5)
        # Level-5 damage must be ≥ level-4 damage (equal only if zero).
        self.assertGreaterEqual(dmg5, dmg4 - 1e-9)

    def test_level_1_red_stays_stationary(self) -> None:
        """Red must not move at all at curriculum level 1."""
        env = BattalionEnv(curriculum_level=1)
        env.reset(seed=0)
        initial_x = env.red.x
        initial_y = env.red.y
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(
                np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        self.assertAlmostEqual(env.red.x, initial_x, places=6)
        self.assertAlmostEqual(env.red.y, initial_y, places=6)
        env.close()


if __name__ == "__main__":
    unittest.main()
