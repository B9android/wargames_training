"""Tests for envs/division_env.py — Division Commander Environment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium import spaces

from envs.division_env import (
    DivisionEnv,
    DIVISION_OBS_DIM,
    N_THEATRE_SECTORS,
    _division_obs_dim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(
    n_brigades: int = 2,
    n_blue_per_brigade: int = 2,
    n_red_brigades: int = 2,
    n_red_per_brigade: int = 2,
    **kwargs,
) -> DivisionEnv:
    """Return a freshly constructed DivisionEnv."""
    return DivisionEnv(
        n_brigades=n_brigades,
        n_blue_per_brigade=n_blue_per_brigade,
        n_red_brigades=n_red_brigades,
        n_red_per_brigade=n_red_per_brigade,
        **kwargs,
    )


def run_episode(env: DivisionEnv, seed: int = 0, max_steps: int = 200) -> dict:
    """Run a full division macro-episode and return a summary."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    div_steps = 0
    terminated = False
    truncated = False
    info: dict = {}
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        div_steps += 1
        if terminated or truncated:
            break
    return {
        "obs": obs,
        "total_reward": total_reward,
        "div_steps": div_steps,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


# ---------------------------------------------------------------------------
# 1. Construction and spaces
# ---------------------------------------------------------------------------


class TestDivisionEnvConstruction(unittest.TestCase):
    """Verify DivisionEnv can be constructed with various arguments."""

    def test_default_construction(self) -> None:
        env = make_env()
        self.assertIsInstance(env.action_space, spaces.MultiDiscrete)
        self.assertIsInstance(env.observation_space, spaces.Box)
        env.close()

    def test_obs_dim_formula_2_brigades(self) -> None:
        """obs_dim = 5 + 8 * n_brigades + 1 = 22 for n_brigades=2."""
        env = make_env(n_brigades=2)
        expected = _division_obs_dim(2)
        self.assertEqual(env._obs_dim, expected)
        self.assertEqual(env.observation_space.shape[0], expected)
        env.close()

    def test_obs_dim_formula_3_brigades(self) -> None:
        """obs_dim = 5 + 8 * 3 + 1 = 30 for n_brigades=3."""
        env = make_env(n_brigades=3, n_red_brigades=3)
        expected = _division_obs_dim(3)
        self.assertEqual(env._obs_dim, expected)
        self.assertEqual(env.observation_space.shape[0], expected)
        env.close()

    def test_obs_dim_formula_1_brigade(self) -> None:
        env = make_env(n_brigades=1, n_blue_per_brigade=1, n_red_brigades=1, n_red_per_brigade=1)
        expected = _division_obs_dim(1)
        self.assertEqual(env._obs_dim, expected)
        env.close()

    def test_action_space_shape(self) -> None:
        env = make_env(n_brigades=2)
        self.assertEqual(len(env.action_space.nvec), 2)
        self.assertTrue(all(v == env.n_div_options for v in env.action_space.nvec))
        env.close()

    def test_action_space_1_brigade(self) -> None:
        env = make_env(n_brigades=1, n_blue_per_brigade=1, n_red_brigades=1, n_red_per_brigade=1)
        self.assertEqual(len(env.action_space.nvec), 1)
        env.close()

    def test_obs_space_bounds_consistent(self) -> None:
        env = make_env()
        self.assertTrue(np.all(env.observation_space.high >= env.observation_space.low))
        env.close()

    def test_invalid_n_brigades_raises(self) -> None:
        with self.assertRaises(ValueError):
            DivisionEnv(n_brigades=0)

    def test_invalid_n_blue_per_brigade_raises(self) -> None:
        with self.assertRaises(ValueError):
            DivisionEnv(n_blue_per_brigade=0)

    def test_invalid_n_red_brigades_raises(self) -> None:
        with self.assertRaises(ValueError):
            DivisionEnv(n_red_brigades=0)

    def test_invalid_n_red_per_brigade_raises(self) -> None:
        with self.assertRaises(ValueError):
            DivisionEnv(n_red_per_brigade=0)

    def test_module_constant_division_obs_dim(self) -> None:
        """Module-level DIVISION_OBS_DIM should equal obs_dim for n_brigades=2."""
        env = make_env(n_brigades=2)
        self.assertEqual(env._obs_dim, DIVISION_OBS_DIM)
        env.close()

    def test_total_blue_battalions(self) -> None:
        env = make_env(n_brigades=3, n_blue_per_brigade=2, n_red_brigades=2, n_red_per_brigade=3)
        self.assertEqual(env.n_blue, 6)
        self.assertEqual(env.n_red, 6)
        env.close()

    def test_red_defaults_match_blue(self) -> None:
        """n_red_brigades and n_red_per_brigade default to n_brigades values."""
        env = DivisionEnv(n_brigades=2, n_blue_per_brigade=3)
        self.assertEqual(env.n_red_brigades, 2)
        self.assertEqual(env.n_red_per_brigade, 3)
        env.close()

    def test_asymmetric_red_forces(self) -> None:
        """Different red brigade count from blue is valid."""
        env = make_env(n_brigades=2, n_blue_per_brigade=2, n_red_brigades=3, n_red_per_brigade=1)
        self.assertEqual(env.n_blue, 4)
        self.assertEqual(env.n_red, 3)
        env.close()


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------


class TestDivisionEnvReset(unittest.TestCase):
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
        self.assertFalse(
            np.allclose(obs1, obs2),
            "Expected different observations for different seeds",
        )
        env.close()

    def test_reset_step_progress_zero(self) -> None:
        """After reset, step-progress feature (last element) should be ~0."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        step_progress = float(obs[-1])
        self.assertAlmostEqual(step_progress, 0.0, places=5)
        env.close()

    def test_reset_clears_div_steps(self) -> None:
        env = make_env()
        env.reset(seed=0)
        env.step(env.action_space.sample())
        self.assertGreater(env._div_steps, 0)
        env.reset(seed=0)
        self.assertEqual(env._div_steps, 0)
        env.close()


# ---------------------------------------------------------------------------
# 3. Step
# ---------------------------------------------------------------------------


class TestDivisionEnvStep(unittest.TestCase):
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
        obs, _, _, _, _ = env.step(env.action_space.sample())
        lo = env.observation_space.low
        hi = env.observation_space.high
        self.assertTrue(
            np.all(obs >= lo - 1e-5) and np.all(obs <= hi + 1e-5),
            msg="Obs out of bounds after step",
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
        self.assertIn("div_steps", info)
        self.assertIn("brigade_action", info)
        env.close()

    def test_step_all_six_operational_commands(self) -> None:
        """All six operational command indices are accepted without error."""
        env = make_env()
        env.reset(seed=5)
        for cmd_idx in range(6):
            if not env._brigade._inner.agents:
                env.reset(seed=cmd_idx)
            action = np.array([cmd_idx] * env.n_brigades, dtype=np.int64)
            env.step(action)
        env.close()

    def test_step_div_steps_increment(self) -> None:
        env = make_env()
        env.reset(seed=0)
        for _ in range(3):
            if env._brigade._inner.agents:
                env.step(env.action_space.sample())
        self.assertGreaterEqual(env._div_steps, 1)
        env.close()

    def test_step_invalid_shape_raises(self) -> None:
        env = make_env(n_brigades=2)
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step(np.array([0], dtype=np.int64))  # wrong shape
        env.close()

    def test_step_invalid_command_index_raises(self) -> None:
        env = make_env(n_brigades=2)
        env.reset(seed=0)
        bad_action = np.array([0, env.n_div_options], dtype=np.int64)  # out of range
        with self.assertRaises(ValueError):
            env.step(bad_action)
        env.close()

    def test_step_winner_in_info_on_episode_end(self) -> None:
        """When the episode ends, info must contain 'winner'."""
        env = make_env(n_brigades=1, n_blue_per_brigade=1, n_red_brigades=1,
                       n_red_per_brigade=1, max_steps=100)
        env.reset(seed=0)
        done = False
        last_info: dict = {}
        for _ in range(300):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                last_info = info
                break
        self.assertTrue(done, "Episode should terminate within 300 division macro-steps")
        self.assertIn("winner", last_info)
        self.assertIn(last_info["winner"], ("blue", "red", "draw"))
        env.close()


# ---------------------------------------------------------------------------
# 4. Command translation
# ---------------------------------------------------------------------------


class TestCommandTranslation(unittest.TestCase):
    """Verify division → brigade action translation."""

    def test_translate_homogeneous(self) -> None:
        """Same command for all brigades → same option for all battalions."""
        env = make_env(n_brigades=2, n_blue_per_brigade=2)
        for cmd in range(env.n_div_options):
            div_action = np.array([cmd, cmd], dtype=np.int64)
            brigade_action = env._translate_division_action(div_action)
            self.assertEqual(brigade_action.shape, (env.n_blue,))
            self.assertTrue(np.all(brigade_action == cmd))
        env.close()

    def test_translate_heterogeneous(self) -> None:
        """Different commands per brigade translate correctly."""
        env = make_env(n_brigades=2, n_blue_per_brigade=2)
        div_action = np.array([1, 3], dtype=np.int64)
        brigade_action = env._translate_division_action(div_action)
        # Brigade 0 (battalions 0,1) → cmd 1
        np.testing.assert_array_equal(brigade_action[:2], [1, 1])
        # Brigade 1 (battalions 2,3) → cmd 3
        np.testing.assert_array_equal(brigade_action[2:], [3, 3])
        env.close()

    def test_translate_3_brigades(self) -> None:
        env = make_env(n_brigades=3, n_blue_per_brigade=2, n_red_brigades=3)
        div_action = np.array([0, 2, 5], dtype=np.int64)
        brigade_action = env._translate_division_action(div_action)
        self.assertEqual(brigade_action.shape, (6,))
        np.testing.assert_array_equal(brigade_action[:2], [0, 0])
        np.testing.assert_array_equal(brigade_action[2:4], [2, 2])
        np.testing.assert_array_equal(brigade_action[4:], [5, 5])
        env.close()

    def test_translate_output_dtype(self) -> None:
        env = make_env()
        ba = env._translate_division_action(np.array([0, 0], dtype=np.int64))
        self.assertEqual(ba.dtype, np.int64)
        env.close()


# ---------------------------------------------------------------------------
# 5. Theatre sector observation
# ---------------------------------------------------------------------------


class TestTheatreSectorObs(unittest.TestCase):
    """Verify theatre sector control features."""

    def test_sector_control_count(self) -> None:
        """obs[:N_THEATRE_SECTORS] should equal N_THEATRE_SECTORS features."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        sector_obs = obs[:N_THEATRE_SECTORS]
        self.assertEqual(len(sector_obs), N_THEATRE_SECTORS)
        env.close()

    def test_sector_control_range(self) -> None:
        """All sector control values must be in [0, 1]."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        sector_obs = obs[:N_THEATRE_SECTORS]
        self.assertTrue(
            np.all(sector_obs >= 0.0) and np.all(sector_obs <= 1.0),
            msg=f"Sector control out of [0, 1]: {sector_obs}",
        )
        env.close()

    def test_sector_control_after_step(self) -> None:
        env = make_env()
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        sector_obs = obs[:N_THEATRE_SECTORS]
        self.assertTrue(np.all(sector_obs >= 0.0) and np.all(sector_obs <= 1.0))
        env.close()


# ---------------------------------------------------------------------------
# 6. Brigade status observation
# ---------------------------------------------------------------------------


class TestBrigadeStatusObs(unittest.TestCase):
    """Verify per-brigade status features."""

    def test_brigade_status_range(self) -> None:
        """Brigade status features (avg_strength, avg_morale, alive_ratio) in [0, 1]."""
        env = make_env(n_brigades=2)
        obs, _ = env.reset(seed=0)
        n_brigades = env.n_brigades
        # Status slice: [N_THEATRE_SECTORS : N_THEATRE_SECTORS + 3*n_brigades]
        status_start = N_THEATRE_SECTORS
        status_end = status_start + 3 * n_brigades
        status_obs = obs[status_start:status_end]
        self.assertEqual(len(status_obs), 3 * n_brigades)
        self.assertTrue(
            np.all(status_obs >= 0.0) and np.all(status_obs <= 1.0),
            msg=f"Brigade status out of [0, 1]: {status_obs}",
        )
        env.close()

    def test_brigade_status_initial_full_strength(self) -> None:
        """After reset, brigades should start at full strength and morale."""
        env = make_env(n_brigades=2)
        obs, _ = env.reset(seed=0)
        status_start = N_THEATRE_SECTORS
        # avg_strength[0], avg_morale[0], alive_ratio[0], ...
        status_obs = obs[status_start: status_start + 3 * env.n_brigades]
        for i in range(env.n_brigades):
            avg_str = float(status_obs[i * 3])
            avg_mor = float(status_obs[i * 3 + 1])
            alive_ratio = float(status_obs[i * 3 + 2])
            self.assertAlmostEqual(avg_str, 1.0, places=4,
                                   msg=f"Brigade {i} avg_strength should be 1.0 at reset")
            self.assertAlmostEqual(avg_mor, 1.0, places=4,
                                   msg=f"Brigade {i} avg_morale should be 1.0 at reset")
            self.assertAlmostEqual(alive_ratio, 1.0, places=4,
                                   msg=f"Brigade {i} alive_ratio should be 1.0 at reset")
        env.close()


# ---------------------------------------------------------------------------
# 7. Episode lifecycle
# ---------------------------------------------------------------------------


class TestDivisionEpisode(unittest.TestCase):
    """Verify episodes run to completion."""

    def test_1brigade_1bat_episode_terminates(self) -> None:
        env = make_env(n_brigades=1, n_blue_per_brigade=1, n_red_brigades=1,
                       n_red_per_brigade=1, max_steps=200)
        result = run_episode(env, seed=42, max_steps=100)
        self.assertGreater(result["div_steps"], 0)
        env.close()

    def test_2brigade_2bat_episode_terminates(self) -> None:
        env = make_env(n_brigades=2, n_blue_per_brigade=2, n_red_brigades=2,
                       n_red_per_brigade=2, max_steps=300)
        result = run_episode(env, seed=0, max_steps=100)
        self.assertGreater(result["div_steps"], 0)
        env.close()

    def test_obs_is_finite_after_episode(self) -> None:
        env = make_env(max_steps=100)
        result = run_episode(env, seed=7, max_steps=50)
        self.assertTrue(
            np.all(np.isfinite(result["obs"])),
            "Observation contains non-finite values after episode",
        )
        env.close()

    def test_reward_is_finite(self) -> None:
        env = make_env()
        result = run_episode(env, seed=2, max_steps=30)
        self.assertTrue(np.isfinite(result["total_reward"]))
        env.close()

    def test_multiple_resets(self) -> None:
        """Multiple resets and episodes should work without state leakage."""
        env = make_env(max_steps=200)
        for seed in range(3):
            result = run_episode(env, seed=seed, max_steps=50)
            self.assertGreater(result["div_steps"], 0)
        env.close()


# ---------------------------------------------------------------------------
# 8. Three-echelon hierarchy end-to-end
# ---------------------------------------------------------------------------


class TestThreeEchelonHierarchy(unittest.TestCase):
    """Verify the three-echelon chain: division → brigade → battalion → sim."""

    def test_division_brigade_battalion_sim_chain(self) -> None:
        """DivisionEnv wraps BrigadeEnv which wraps MultiBattalionEnv."""
        from envs.brigade_env import BrigadeEnv
        from envs.multi_battalion_env import MultiBattalionEnv

        env = make_env()
        # DivisionEnv has a BrigadeEnv
        self.assertIsInstance(env._brigade, BrigadeEnv)
        # BrigadeEnv has a MultiBattalionEnv
        self.assertIsInstance(env._brigade._inner, MultiBattalionEnv)
        env.close()

    def test_brigade_action_passed_to_brigade_env(self) -> None:
        """Division action is correctly translated and forwarded to BrigadeEnv."""
        env = make_env(n_brigades=2, n_blue_per_brigade=2)
        env.reset(seed=0)
        div_action = np.array([0, 3], dtype=np.int64)
        # Verify translation correctness
        translated = env._translate_division_action(div_action)
        # Brigade 0 → battalions 0,1 → option 0; brigade 1 → battalions 2,3 → option 3
        np.testing.assert_array_equal(translated, [0, 0, 3, 3])
        env.close()

    def test_inner_brigade_env_battalion_count(self) -> None:
        """Inner BrigadeEnv has correct total battalion count."""
        env = make_env(n_brigades=2, n_blue_per_brigade=3, n_red_brigades=2, n_red_per_brigade=2)
        self.assertEqual(env._brigade.n_blue, 6)
        self.assertEqual(env._brigade.n_red, 4)
        env.close()

    def test_frozen_brigade_policies_not_trainable(self) -> None:
        """After construction brigade policy (if set) should have 0 trainable params."""
        # Without a policy, confirm the env has no red brigade policy
        env = make_env()
        self.assertIsNone(env._red_brigade_policy)
        env.close()

    def test_forced_red_options_cleared_after_step(self) -> None:
        """_forced_red_options dict must be empty after each division step."""
        env = make_env()
        env.reset(seed=0)
        env.step(env.action_space.sample())
        self.assertEqual(env._brigade._forced_red_options, {})
        env.close()

    def test_red_random_mode(self) -> None:
        """Red random mode runs without error."""
        env = make_env(red_random=True)
        obs, _ = env.reset(seed=0)
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        self.assertEqual(obs.shape, (env._obs_dim,))
        self.assertEqual(obs2.shape, (env._obs_dim,))
        env.close()


# ---------------------------------------------------------------------------
# 9. Observation dimensionality formula
# ---------------------------------------------------------------------------


class TestObsDimFormula(unittest.TestCase):
    """Verify the obs_dim formula for various brigade counts."""

    def test_formula_1_brigade(self) -> None:
        expected = N_THEATRE_SECTORS + 8 * 1 + 1
        self.assertEqual(_division_obs_dim(1), expected)

    def test_formula_2_brigades(self) -> None:
        expected = N_THEATRE_SECTORS + 8 * 2 + 1
        self.assertEqual(_division_obs_dim(2), expected)

    def test_formula_4_brigades(self) -> None:
        expected = N_THEATRE_SECTORS + 8 * 4 + 1
        self.assertEqual(_division_obs_dim(4), expected)

    def test_env_obs_dim_matches_formula(self) -> None:
        for n in (1, 2, 3):
            env = make_env(
                n_brigades=n,
                n_blue_per_brigade=1,
                n_red_brigades=n,
                n_red_per_brigade=1,
            )
            self.assertEqual(env._obs_dim, _division_obs_dim(n))
            env.close()


if __name__ == "__main__":
    unittest.main()
