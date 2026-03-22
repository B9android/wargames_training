# tests/test_battalion_env.py
"""Tests for envs/battalion_env.py — BattalionEnv Gymnasium environment."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from envs.battalion_env import (
    BattalionEnv,
    DESTROYED_THRESHOLD,
    MAX_STEPS,
    MAP_WIDTH,
    MAP_HEIGHT,
)
from envs.sim.terrain import TerrainMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(**kwargs) -> BattalionEnv:
    return BattalionEnv(**kwargs)


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestInit(unittest.TestCase):
    """Verify argument validation in BattalionEnv.__init__."""

    def test_zero_map_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(map_width=0)

    def test_negative_map_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(map_width=-100)

    def test_zero_map_height_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(map_height=0)

    def test_negative_map_height_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(map_height=-1)

    def test_zero_max_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(max_steps=0)

    def test_negative_max_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(max_steps=-5)

    def test_unsupported_render_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(render_mode="rgb_array")

    def test_none_render_mode_accepted(self) -> None:
        env = BattalionEnv(render_mode=None)
        env.close()

    def test_valid_positive_dims_accepted(self) -> None:
        env = BattalionEnv(map_width=500.0, map_height=500.0, max_steps=100)
        env.close()


# ---------------------------------------------------------------------------
# Observation & action spaces
# ---------------------------------------------------------------------------


class TestSpaces(unittest.TestCase):
    """Verify observation and action space shapes and dtypes."""

    def setUp(self) -> None:
        self.env = make_env()
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_observation_space_shape(self) -> None:
        self.assertEqual(self.env.observation_space.shape, (17,))

    def test_observation_space_dtype(self) -> None:
        self.assertEqual(self.env.observation_space.dtype, np.float32)

    def test_action_space_shape(self) -> None:
        self.assertEqual(self.env.action_space.shape, (3,))

    def test_action_space_dtype(self) -> None:
        self.assertEqual(self.env.action_space.dtype, np.float32)

    def test_action_space_bounds(self) -> None:
        np.testing.assert_array_equal(
            self.env.action_space.low, [-1.0, -1.0, 0.0]
        )
        np.testing.assert_array_equal(
            self.env.action_space.high, [1.0, 1.0, 1.0]
        )

    def test_observation_space_lower_bounds(self) -> None:
        # Angles (indices 2, 3, 7, 8) may be -1; all terrain features are >= 0
        expected_low = np.array(
            [0, 0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(self.env.observation_space.low, expected_low)

    def test_observation_space_upper_bounds(self) -> None:
        np.testing.assert_array_almost_equal(
            self.env.observation_space.high,
            np.ones(17, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset(unittest.TestCase):
    """Verify reset() behaviour."""

    def test_reset_returns_obs_and_dict(self) -> None:
        env = make_env()
        result = env.reset(seed=1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        obs, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        env.close()

    def test_reset_obs_in_observation_space(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=2)
        self.assertTrue(
            env.observation_space.contains(obs),
            msg=f"Observation {obs} outside observation space.",
        )
        env.close()

    def test_reset_obs_dtype(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=3)
        self.assertEqual(obs.dtype, np.float32)
        env.close()

    def test_reset_initialises_full_strength(self) -> None:
        """Both battalions start at full strength (obs[4] and obs[9] == 1.0)."""
        env = make_env()
        obs, _ = env.reset(seed=4)
        self.assertAlmostEqual(float(obs[4]), 1.0, places=5)  # blue strength
        self.assertAlmostEqual(float(obs[9]), 1.0, places=5)  # red strength
        env.close()

    def test_reset_step_norm_is_zero(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=5)
        self.assertAlmostEqual(float(obs[11]), 0.0, places=5)
        env.close()

    def test_reset_angle_encoded_as_cos_sin(self) -> None:
        """Angle components (cos²+sin² ≈ 1)."""
        env = make_env()
        obs, _ = env.reset(seed=6)
        unit_circle = float(obs[2]) ** 2 + float(obs[3]) ** 2
        self.assertAlmostEqual(unit_circle, 1.0, places=5)
        env.close()

    def test_reset_seeded_reproducible(self) -> None:
        """Same seed produces the same initial observation."""
        env = make_env()
        obs_a, _ = env.reset(seed=42)
        obs_b, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs_a, obs_b)
        env.close()

    def test_reset_different_seeds_produce_different_obs(self) -> None:
        env = make_env()
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        self.assertFalse(
            np.allclose(obs_a, obs_b),
            "Different seeds produced identical observations.",
        )
        env.close()


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestStep(unittest.TestCase):
    """Verify step() return values and semantics."""

    def setUp(self) -> None:
        self.env = make_env()
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def _zero_action(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def test_step_returns_five_tuple(self) -> None:
        result = self.env.step(self._zero_action())
        self.assertEqual(len(result), 5)

    def test_step_obs_in_observation_space(self) -> None:
        obs, *_ = self.env.step(self._zero_action())
        self.assertTrue(
            self.env.observation_space.contains(obs),
            msg=f"Step observation {obs} outside observation space.",
        )

    def test_step_obs_dtype_float32(self) -> None:
        obs, *_ = self.env.step(self._zero_action())
        self.assertEqual(obs.dtype, np.float32)

    def test_step_reward_is_float(self) -> None:
        _, reward, *_ = self.env.step(self._zero_action())
        self.assertIsInstance(reward, float)

    def test_step_terminated_and_truncated_are_bool(self) -> None:
        _, _, terminated, truncated, _ = self.env.step(self._zero_action())
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_step_info_is_dict(self) -> None:
        *_, info = self.env.step(self._zero_action())
        self.assertIsInstance(info, dict)

    def test_step_info_keys(self) -> None:
        *_, info = self.env.step(self._zero_action())
        for key in (
            "blue_damage_dealt",
            "red_damage_dealt",
            "blue_routed",
            "red_routed",
            "step_count",
        ):
            self.assertIn(key, info)

    def test_step_increments_step_norm(self) -> None:
        """Observation index 11 (step norm) increases with steps."""
        obs0, _ = self.env.reset(seed=0)
        obs1, *_ = self.env.step(self._zero_action())
        self.assertGreater(float(obs1[11]), float(obs0[11]))

    def test_terminated_and_truncated_mutually_exclusive(self) -> None:
        env = make_env(max_steps=200)
        env.reset(seed=0)
        for _ in range(300):
            obs, reward, terminated, truncated, info = env.step(
                np.array([1.0, 0.0, 1.0], dtype=np.float32)
            )
            self.assertFalse(
                terminated and truncated,
                "terminated and truncated both True in the same step.",
            )
            if terminated or truncated:
                break
        env.close()

    def test_step_before_reset_raises(self) -> None:
        env = make_env()
        with self.assertRaises((RuntimeError, AssertionError)):
            env.step(self._zero_action())
        env.close()


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


class TestTermination(unittest.TestCase):
    """Verify episodes terminate deterministically."""

    def test_episode_terminates_within_max_steps(self) -> None:
        """Episode must end by or at max_steps — no infinite loops."""
        env = make_env(max_steps=500)
        env.reset(seed=99)
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(
                np.array([1.0, 0.0, 1.0], dtype=np.float32)
            )
            done = terminated or truncated
            steps += 1
            self.assertLessEqual(steps, 500 + 1, "Episode exceeded max_steps.")
        env.close()

    def test_truncated_when_max_steps_reached(self) -> None:
        """If no combat termination, truncated=True at max_steps."""
        # Use a small map so we can run to timeout quickly.
        env = make_env(max_steps=10)
        # Place blue and red far apart so no fire can occur
        env.reset(seed=0)
        # Force both battalions far apart so no combat
        env.blue.x = 0.0
        env.blue.y = 500.0
        env.red.x = 1000.0
        env.red.y = 500.0
        truncated = False
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(
                np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        # Should have reached a truncation or termination within 15 steps
        self.assertTrue(terminated or truncated)
        env.close()

    def test_episode_length_bounded(self) -> None:
        """step_count never exceeds max_steps."""
        env = make_env(max_steps=50)
        env.reset(seed=7)
        for _ in range(60):
            obs, _, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            self.assertLessEqual(info["step_count"], env.max_steps)
            if terminated or truncated:
                break
        env.close()


# ---------------------------------------------------------------------------
# Seeding and reproducibility
# ---------------------------------------------------------------------------


class TestSeeding(unittest.TestCase):
    """Verify that seeding produces reproducible trajectories."""

    def _rollout(self, seed: int, steps: int = 20) -> list[np.ndarray]:
        env = make_env()
        obs_list = []
        obs, _ = env.reset(seed=seed)
        obs_list.append(obs.copy())
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            move   = rng.uniform(-1.0, 1.0)
            rotate = rng.uniform(-1.0, 1.0)
            fire   = rng.uniform(0.0, 1.0)
            action = np.array([move, rotate, fire], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            obs_list.append(obs.copy())
            if terminated or truncated:
                break
        env.close()
        return obs_list

    def test_same_seed_same_trajectory(self) -> None:
        traj_a = self._rollout(seed=123)
        traj_b = self._rollout(seed=123)
        self.assertEqual(len(traj_a), len(traj_b))
        for a, b in zip(traj_a, traj_b):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_trajectories(self) -> None:
        traj_a = self._rollout(seed=1)
        traj_b = self._rollout(seed=2)
        # At least the first observations (positions) should differ
        self.assertFalse(
            np.allclose(traj_a[0], traj_b[0]),
            "Different seeds produced identical initial observations.",
        )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalisation(unittest.TestCase):
    """Verify all observation components stay within declared bounds."""

    def test_obs_within_bounds_throughout_episode(self) -> None:
        env = make_env(max_steps=200)
        obs, _ = env.reset(seed=0)
        for step_i in range(250):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            self.assertTrue(
                env.observation_space.contains(obs),
                msg=f"Out-of-bounds observation at step {step_i}: {obs}",
            )
            if terminated or truncated:
                break
        env.close()

    def test_angles_encoded_as_cos_sin(self) -> None:
        """obs[2]² + obs[3]² ≈ 1 (blue heading)."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        for _ in range(30):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            unit = float(obs[2]) ** 2 + float(obs[3]) ** 2
            self.assertAlmostEqual(unit, 1.0, places=5)
            if terminated or truncated:
                break
        env.close()


# ---------------------------------------------------------------------------
# Gymnasium env_checker
# ---------------------------------------------------------------------------


class TestEnvChecker(unittest.TestCase):
    """Run the official gymnasium env_checker."""

    def test_check_env_passes(self) -> None:
        env = make_env()
        # check_env raises on any violation; success means no exception.
        check_env(env, warn=True, skip_render_check=True)
        env.close()


# ---------------------------------------------------------------------------
# Terrain randomization
# ---------------------------------------------------------------------------


class TestTerrainRandomization(unittest.TestCase):
    """Verify terrain-randomization behaviour in BattalionEnv."""

    def test_default_env_has_randomize_terrain_enabled(self) -> None:
        env = BattalionEnv()
        self.assertTrue(env.randomize_terrain)
        env.close()

    def test_fixed_terrain_disables_randomization(self) -> None:
        fixed = TerrainMap.flat(1000.0, 1000.0)
        env = BattalionEnv(terrain=fixed)
        self.assertFalse(env.randomize_terrain)
        env.close()

    def test_randomize_terrain_false_disables_randomization(self) -> None:
        env = BattalionEnv(randomize_terrain=False)
        self.assertFalse(env.randomize_terrain)
        env.close()

    def test_same_seed_produces_same_terrain(self) -> None:
        """reset(seed=N) must yield identical terrain on repeated calls."""
        env = BattalionEnv()
        env.reset(seed=77)
        elev_a = env.terrain.elevation.copy()
        cov_a = env.terrain.cover.copy()
        env.reset(seed=77)
        elev_b = env.terrain.elevation.copy()
        cov_b = env.terrain.cover.copy()
        np.testing.assert_array_equal(elev_a, elev_b)
        np.testing.assert_array_equal(cov_a, cov_b)
        env.close()

    def test_different_seeds_produce_different_terrain(self) -> None:
        """Different seeds must generate different terrain layouts."""
        env = BattalionEnv()
        env.reset(seed=1)
        elev_a = env.terrain.elevation.copy()
        env.reset(seed=2)
        elev_b = env.terrain.elevation.copy()
        self.assertFalse(
            np.array_equal(elev_a, elev_b),
            "Different seeds produced identical terrain.",
        )
        env.close()

    def test_terrain_changes_each_episode_without_seed(self) -> None:
        """Consecutive resets without a fixed seed generate different terrain.

        Two independent resets with no seed override both derive from the
        internal counter-based RNG state, which advances between calls.
        A collision of the full 20×20 elevation grid is astronomically
        unlikely, so we treat differing arrays as the expected outcome.
        """
        env = BattalionEnv()
        env.reset()
        elev_a = env.terrain.elevation.copy()
        env.reset()
        elev_b = env.terrain.elevation.copy()
        self.assertFalse(
            np.array_equal(elev_a, elev_b),
            "Consecutive resets without a seed produced identical terrain.",
        )
        env.close()

    def test_invalid_hill_speed_factor_raises(self) -> None:
        with self.assertRaises(ValueError):
            BattalionEnv(hill_speed_factor=0.0)
        with self.assertRaises(ValueError):
            BattalionEnv(hill_speed_factor=-0.1)
        with self.assertRaises(ValueError):
            BattalionEnv(hill_speed_factor=1.1)

    def test_hill_speed_factor_one_accepted(self) -> None:
        env = BattalionEnv(hill_speed_factor=1.0)
        env.close()

    def test_hills_slow_blue_movement(self) -> None:
        """On hills (hill_speed_factor < 1), Blue moves a shorter distance."""
        import math

        # Build a uniform high-elevation terrain so every cell is a hill
        elev = np.ones((4, 4), dtype=np.float32)
        cov = np.zeros((4, 4), dtype=np.float32)
        hill_terrain = TerrainMap.from_arrays(1000.0, 1000.0, elev, cov)

        env_hill = BattalionEnv(terrain=hill_terrain, hill_speed_factor=0.5)
        env_flat = BattalionEnv(terrain=TerrainMap.flat(1000.0, 1000.0), hill_speed_factor=0.5)

        for env in (env_hill, env_flat):
            env.reset(seed=0)
            # Force Blue to a fixed position facing east
            env.blue.x = 200.0
            env.blue.y = 500.0
            env.blue.theta = 0.0  # facing east

        x_before_hill = env_hill.blue.x
        x_before_flat = env_flat.blue.x

        # Take one step with full forward movement and no fire
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        env_hill.step(action)
        env_flat.step(action)

        dx_hill = env_hill.blue.x - x_before_hill
        dx_flat = env_flat.blue.x - x_before_flat

        # Hill should produce less or equal displacement than flat
        self.assertLessEqual(dx_hill, dx_flat + 1e-6)
        # Hill displacement must be meaningfully less
        self.assertLess(dx_hill, dx_flat - 1e-6)

        env_hill.close()
        env_flat.close()

    def test_env_checker_passes_with_terrain_randomization(self) -> None:
        env = BattalionEnv(randomize_terrain=True)
        check_env(env, warn=True, skip_render_check=True)
        env.close()


if __name__ == "__main__":
    unittest.main()
