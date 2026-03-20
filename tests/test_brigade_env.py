"""Tests for envs/brigade_env.py — Brigade Commander Environment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium import spaces

from envs.brigade_env import BrigadeEnv, BRIGADE_OBS_DIM, N_SECTORS
from envs.options import MacroAction, make_default_options
from models.mappo_policy import MAPPOPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(n_blue: int = 2, n_red: int = 2, **kwargs) -> BrigadeEnv:
    """Return a freshly constructed 2v2 BrigadeEnv."""
    return BrigadeEnv(n_blue=n_blue, n_red=n_red, **kwargs)


def run_episode(env: BrigadeEnv, seed: int = 0, max_steps: int = 200) -> dict:
    """Run a full macro-episode and return a summary."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    macro_steps = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        macro_steps += 1
        if terminated or truncated:
            break
    return {
        "obs": obs,
        "total_reward": total_reward,
        "macro_steps": macro_steps,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


# ---------------------------------------------------------------------------
# 1. Construction and spaces
# ---------------------------------------------------------------------------


class TestBrigadeEnvConstruction(unittest.TestCase):
    """Verify BrigadeEnv can be constructed with various arguments."""

    def test_default_construction(self) -> None:
        env = make_env()
        self.assertIsInstance(env.action_space, spaces.MultiDiscrete)
        self.assertIsInstance(env.observation_space, spaces.Box)
        env.close()

    def test_obs_dim_formula_2v2(self) -> None:
        """obs_dim = 3 + 7 * n_blue + 1 = 18 for n_blue=2."""
        env = make_env(n_blue=2)
        expected = N_SECTORS + 7 * 2 + 1
        self.assertEqual(env._obs_dim, expected)
        self.assertEqual(env.observation_space.shape[0], expected)
        env.close()

    def test_obs_dim_formula_3v3(self) -> None:
        """obs_dim = 3 + 7 * 3 + 1 = 25 for n_blue=3."""
        env = make_env(n_blue=3, n_red=3)
        expected = N_SECTORS + 7 * 3 + 1
        self.assertEqual(env._obs_dim, expected)
        env.close()

    def test_action_space_shape(self) -> None:
        """Action space should be MultiDiscrete with n_blue elements."""
        env = make_env(n_blue=2)
        self.assertEqual(len(env.action_space.nvec), 2)
        self.assertTrue(all(v == env.n_options for v in env.action_space.nvec))
        env.close()

    def test_action_space_1v1(self) -> None:
        env = make_env(n_blue=1, n_red=1)
        self.assertEqual(len(env.action_space.nvec), 1)
        env.close()

    def test_obs_space_bounds(self) -> None:
        """Observation space bounds must be consistent."""
        env = make_env()
        self.assertTrue(np.all(env.observation_space.high >= env.observation_space.low))
        env.close()

    def test_invalid_n_blue_raises(self) -> None:
        with self.assertRaises(ValueError):
            BrigadeEnv(n_blue=0)

    def test_invalid_n_red_raises(self) -> None:
        with self.assertRaises(ValueError):
            BrigadeEnv(n_red=0)

    def test_empty_options_raises(self) -> None:
        with self.assertRaises(ValueError):
            BrigadeEnv(options=[])

    def test_module_constant_brigade_obs_dim(self) -> None:
        """Module-level BRIGADE_OBS_DIM should equal obs_dim for n_blue=2."""
        env = make_env(n_blue=2)
        self.assertEqual(env._obs_dim, BRIGADE_OBS_DIM)
        env.close()


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------


class TestBrigadeEnvReset(unittest.TestCase):
    """Verify reset() returns valid observations."""

    def test_reset_returns_obs_array(self) -> None:
        env = make_env()
        obs, info = env.reset(seed=0)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (env._obs_dim,))
        self.assertIsInstance(info, dict)
        env.close()

    def test_reset_obs_within_bounds(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=42)
        lo = env.observation_space.low
        hi = env.observation_space.high
        self.assertTrue(
            np.all(obs >= lo - 1e-5) and np.all(obs <= hi + 1e-5),
            msg=f"Obs out of bounds: min={obs.min():.4f} max={obs.max():.4f}",
        )
        env.close()

    def test_reset_seed_determinism(self) -> None:
        env = make_env()
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    def test_reset_different_seeds_differ(self) -> None:
        env = make_env()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=999)
        # With different seeds, observations should usually differ
        # (not guaranteed, but very likely for battalion positions)
        self.assertFalse(
            np.allclose(obs1, obs2),
            "Expected different observations for different seeds",
        )
        env.close()

    def test_reset_step_count_zero(self) -> None:
        """After reset, step progress feature should be 0."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        step_progress = float(obs[-1])
        self.assertAlmostEqual(step_progress, 0.0, places=5)
        env.close()

    def test_reset_clears_macro_steps(self) -> None:
        env = make_env()
        env.reset(seed=0)
        env.step(env.action_space.sample())
        self.assertGreater(env._macro_steps, 0)
        env.reset(seed=0)
        self.assertEqual(env._macro_steps, 0)
        env.close()


# ---------------------------------------------------------------------------
# 3. Step
# ---------------------------------------------------------------------------


class TestBrigadeEnvStep(unittest.TestCase):
    """Verify step() returns valid outputs."""

    def test_step_returns_correct_types(self) -> None:
        env = make_env()
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(bool(terminated), bool)
        self.assertIsInstance(bool(truncated), bool)
        self.assertIsInstance(info, dict)
        env.close()

    def test_step_obs_within_bounds(self) -> None:
        env = make_env()
        env.reset(seed=1)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        lo = env.observation_space.low
        hi = env.observation_space.high
        self.assertTrue(
            np.all(obs >= lo - 1e-5) and np.all(obs <= hi + 1e-5),
            msg=f"Obs out of bounds after step",
        )
        env.close()

    def test_step_obs_shape(self) -> None:
        env = make_env()
        env.reset(seed=2)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        self.assertEqual(obs.shape, (env._obs_dim,))
        env.close()

    def test_step_info_keys(self) -> None:
        env = make_env()
        env.reset(seed=3)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertIn("macro_steps", info)
        self.assertIn("primitive_steps", info)
        self.assertIn("option_names", info)
        self.assertIn("option_steps", info)
        self.assertIn("blue_rewards", info)
        env.close()

    def test_step_primitive_steps_positive(self) -> None:
        env = make_env()
        env.reset(seed=4)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertGreater(info["primitive_steps"], 0)
        env.close()

    def test_step_all_six_options(self) -> None:
        """All six macro-action indices can be passed without error."""
        env = make_env()
        env.reset(seed=5)
        for opt_idx in range(6):
            if not env._inner.agents:
                env.reset(seed=opt_idx)
            action = np.array([opt_idx] * env.n_blue, dtype=np.int64)
            env.step(action)
        env.close()

    def test_step_macro_steps_increment(self) -> None:
        env = make_env()
        env.reset(seed=0)
        for i in range(3):
            if env._inner.agents:
                env.step(env.action_space.sample())
        self.assertGreaterEqual(env._macro_steps, 1)
        env.close()


# ---------------------------------------------------------------------------
# 4. Episode lifecycle
# ---------------------------------------------------------------------------


class TestBrigadeEpisode(unittest.TestCase):
    """Verify episodes run to completion."""

    def test_1v1_episode_terminates(self) -> None:
        env = make_env(n_blue=1, n_red=1, max_steps=200)
        result = run_episode(env, seed=42, max_steps=100)
        self.assertGreater(result["macro_steps"], 0)
        env.close()

    def test_2v2_episode_terminates(self) -> None:
        env = make_env(n_blue=2, n_red=2, max_steps=200)
        result = run_episode(env, seed=0, max_steps=100)
        self.assertGreater(result["macro_steps"], 0)
        env.close()

    def test_obs_is_finite_after_episode(self) -> None:
        env = make_env(max_steps=100)
        result = run_episode(env, seed=7, max_steps=50)
        self.assertTrue(
            np.all(np.isfinite(result["obs"])),
            "Observation contains non-finite values after episode",
        )
        env.close()

    def test_primitive_steps_exceed_macro_steps(self) -> None:
        """Primitive steps should exceed macro steps (temporal abstraction)."""
        env = make_env()
        result = run_episode(env, seed=10, max_steps=20)
        if result["macro_steps"] > 0 and result["info"].get("primitive_steps", 0) > 0:
            self.assertGreater(
                result["info"]["primitive_steps"], result["macro_steps"]
            )
        env.close()

    def test_reward_is_finite(self) -> None:
        env = make_env()
        result = run_episode(env, seed=2, max_steps=30)
        self.assertTrue(np.isfinite(result["total_reward"]))
        env.close()


# ---------------------------------------------------------------------------
# 5. Sector control observation
# ---------------------------------------------------------------------------


class TestSectorControl(unittest.TestCase):
    """Verify sector control features are in [0, 1]."""

    def test_sector_control_range(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        sector_obs = obs[:N_SECTORS]
        self.assertTrue(
            np.all(sector_obs >= 0.0) and np.all(sector_obs <= 1.0),
            msg=f"Sector control out of [0,1]: {sector_obs}",
        )
        env.close()

    def test_sector_control_after_step(self) -> None:
        env = make_env()
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        sector_obs = obs[:N_SECTORS]
        self.assertTrue(np.all(sector_obs >= 0.0) and np.all(sector_obs <= 1.0))
        env.close()


# ---------------------------------------------------------------------------
# 6. Battalion state observation
# ---------------------------------------------------------------------------


class TestBattalionStateObs(unittest.TestCase):
    """Verify per-battalion strength/morale features are in [0, 1]."""

    def test_battalion_strength_morale_range(self) -> None:
        env = make_env(n_blue=2)
        obs, _ = env.reset(seed=0)
        # strength/morale slice: [3 : 3+2*n_blue]
        bt_obs = obs[N_SECTORS: N_SECTORS + 2 * env.n_blue]
        self.assertTrue(
            np.all(bt_obs >= 0.0) and np.all(bt_obs <= 1.0),
            msg=f"Battalion state obs out of [0,1]: {bt_obs}",
        )
        env.close()

    def test_battalion_state_initial_values(self) -> None:
        """After reset, battalions should start at full strength and morale."""
        env = make_env(n_blue=2)
        obs, _ = env.reset(seed=0)
        bt_obs = obs[N_SECTORS: N_SECTORS + 2 * env.n_blue]
        # strength[0] and morale[0] and strength[1] and morale[1] should be 1.0
        for v in bt_obs:
            self.assertAlmostEqual(float(v), 1.0, places=4)
        env.close()


# ---------------------------------------------------------------------------
# 7. Enemy threat vector
# ---------------------------------------------------------------------------


class TestEnemyThreatVector(unittest.TestCase):
    """Verify enemy threat vector features are in their declared ranges."""

    def test_threat_dist_in_range(self) -> None:
        env = make_env(n_blue=2)
        obs, _ = env.reset(seed=0)
        threat_start = N_SECTORS + 2 * env.n_blue
        for i in range(env.n_blue):
            block = obs[threat_start + 5 * i: threat_start + 5 * i + 5]
            dist = float(block[0])
            cos_b = float(block[1])
            sin_b = float(block[2])
            e_str = float(block[3])
            e_mor = float(block[4])
            self.assertGreaterEqual(dist, 0.0)
            self.assertLessEqual(dist, 1.0)
            self.assertGreaterEqual(cos_b, -1.0)
            self.assertLessEqual(cos_b, 1.0)
            self.assertGreaterEqual(sin_b, -1.0)
            self.assertLessEqual(sin_b, 1.0)
            self.assertGreaterEqual(e_str, 0.0)
            self.assertLessEqual(e_str, 1.0)
            self.assertGreaterEqual(e_mor, 0.0)
            self.assertLessEqual(e_mor, 1.0)
        env.close()


# ---------------------------------------------------------------------------
# 8. Frozen battalion policy
# ---------------------------------------------------------------------------


class TestFrozenBattalionPolicy(unittest.TestCase):
    """Verify frozen policy integration."""

    def _make_mock_policy(self, n_blue: int = 2) -> MAPPOPolicy:
        """Create a small MAPPOPolicy for testing."""
        from envs.multi_battalion_env import MultiBattalionEnv
        inner = MultiBattalionEnv(n_blue=n_blue, n_red=n_blue)
        obs_dim = inner._obs_dim
        action_dim = inner._act_space.shape[0]
        state_dim = inner._state_dim
        inner.close()
        return MAPPOPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_blue,
            share_parameters=True,
        )

    def test_set_battalion_policy_freezes_params(self) -> None:
        """After set_battalion_policy(), all params must have requires_grad=False."""
        env = make_env(n_blue=2, n_red=2)
        policy = self._make_mock_policy(n_blue=2)
        env.set_battalion_policy(policy)
        for name, param in policy.named_parameters():
            self.assertFalse(
                param.requires_grad,
                msg=f"Parameter {name!r} should be frozen but requires_grad=True",
            )
        env.close()

    def test_set_battalion_policy_eval_mode(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        policy = self._make_mock_policy(n_blue=2)
        env.set_battalion_policy(policy)
        self.assertFalse(
            policy.training,
            msg="Frozen policy should be in eval mode",
        )
        env.close()

    def test_set_battalion_policy_none_clears(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        policy = self._make_mock_policy(n_blue=2)
        env.set_battalion_policy(policy)
        env.set_battalion_policy(None)
        self.assertIsNone(env._battalion_policy)
        env.close()

    def test_constructor_battalion_policy_frozen(self) -> None:
        """Policy passed via constructor should also be frozen."""
        policy = self._make_mock_policy(n_blue=2)
        env = BrigadeEnv(n_blue=2, n_red=2, battalion_policy=policy)
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.assertEqual(trainable, 0)
        env.close()

    def test_episode_runs_with_battalion_policy(self) -> None:
        """Episodes should complete when a frozen battalion policy drives Red."""
        policy = self._make_mock_policy(n_blue=2)
        env = BrigadeEnv(n_blue=2, n_red=2, battalion_policy=policy, max_steps=100)
        result = run_episode(env, seed=99, max_steps=50)
        self.assertGreater(result["macro_steps"], 0)
        env.close()


# ---------------------------------------------------------------------------
# 9. Red random mode
# ---------------------------------------------------------------------------


class TestRedRandomMode(unittest.TestCase):
    """Verify red_random mode runs without errors."""

    def test_episode_with_red_random(self) -> None:
        env = BrigadeEnv(n_blue=2, n_red=2, red_random=True, max_steps=100)
        result = run_episode(env, seed=0, max_steps=30)
        self.assertGreater(result["macro_steps"], 0)
        env.close()


# ---------------------------------------------------------------------------
# 10. Custom options
# ---------------------------------------------------------------------------


class TestCustomOptions(unittest.TestCase):
    """Verify custom option vocabularies work."""

    def test_custom_option_count(self) -> None:
        options = make_default_options()[:3]  # only 3 options
        env = BrigadeEnv(n_blue=2, n_red=2, options=options)
        self.assertEqual(env.n_options, 3)
        self.assertTrue(all(v == 3 for v in env.action_space.nvec))
        env.close()

    def test_custom_options_episode_runs(self) -> None:
        options = make_default_options()[:2]
        env = BrigadeEnv(n_blue=2, n_red=2, options=options, max_steps=100)
        result = run_episode(env, seed=0, max_steps=20)
        self.assertGreater(result["macro_steps"], 0)
        env.close()


# ---------------------------------------------------------------------------
# 11. load_frozen_battalion_policy (unit test with a saved mock)
# ---------------------------------------------------------------------------


class TestLoadFrozenBattalionPolicy(unittest.TestCase):
    """Verify loading a frozen battalion policy from a checkpoint."""

    def test_load_and_freeze(self) -> None:
        import tempfile
        import os
        from training.train_brigade import load_frozen_battalion_policy
        from envs.multi_battalion_env import MultiBattalionEnv

        inner = MultiBattalionEnv(n_blue=2, n_red=2)
        obs_dim = inner._obs_dim
        action_dim = inner._act_space.shape[0]
        state_dim = inner._state_dim
        inner.close()

        # Save a mock checkpoint
        policy = MAPPOPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "mock_policy.pt"
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": {},
                    "total_steps": 123,
                    "episodes": 10,
                },
                ckpt_path,
            )

            frozen = load_frozen_battalion_policy(
                checkpoint_path=ckpt_path,
                obs_dim=obs_dim,
                action_dim=action_dim,
                state_dim=state_dim,
                n_agents=2,
            )

        # All parameters must be frozen
        for name, param in frozen.named_parameters():
            self.assertFalse(
                param.requires_grad,
                msg=f"Parameter {name!r} still has requires_grad=True",
            )
        # Must be in eval mode
        self.assertFalse(frozen.training)


# ---------------------------------------------------------------------------
# 12. Gymnasium API compliance
# ---------------------------------------------------------------------------


class TestGymnasiumCompliance(unittest.TestCase):
    """Basic Gymnasium API checks."""

    def test_reset_returns_tuple(self) -> None:
        env = make_env()
        result = env.reset(seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        env.close()

    def test_step_returns_5_tuple(self) -> None:
        env = make_env()
        env.reset(seed=0)
        result = env.step(env.action_space.sample())
        self.assertEqual(len(result), 5)
        env.close()

    def test_action_space_sample_valid(self) -> None:
        env = make_env()
        env.reset(seed=0)
        for _ in range(10):
            action = env.action_space.sample()
            self.assertTrue(env.action_space.contains(action))
        env.close()

    def test_obs_in_observation_space(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        self.assertTrue(
            env.observation_space.contains(obs.astype(np.float32)),
            msg="Initial obs not in observation_space",
        )
        env.close()


if __name__ == "__main__":
    unittest.main()
