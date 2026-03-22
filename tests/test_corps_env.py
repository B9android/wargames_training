"""Tests for envs/corps_env.py — Corps Commander Environment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium import spaces

from envs.corps_env import (
    CorpsEnv,
    CORPS_OBS_DIM,
    CORPS_MAP_WIDTH,
    CORPS_MAP_HEIGHT,
    N_CORPS_SECTORS,
    N_OBJECTIVES,
    N_ROAD_FEATURES,
    ObjectiveType,
    OperationalObjective,
    _corps_obs_dim,
)
from envs.sim.road_network import RoadNetwork, RoadSegment, ROAD_SPEED_BONUS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(
    n_divisions: int = 2,
    n_brigades_per_division: int = 2,
    n_blue_per_brigade: int = 2,
    **kwargs,
) -> CorpsEnv:
    """Return a freshly constructed CorpsEnv with small defaults for speed."""
    return CorpsEnv(
        n_divisions=n_divisions,
        n_brigades_per_division=n_brigades_per_division,
        n_blue_per_brigade=n_blue_per_brigade,
        **kwargs,
    )


def run_episode(env: CorpsEnv, seed: int = 0, max_steps: int = 100) -> dict:
    """Run a short episode and return a summary dict."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    corps_steps = 0
    terminated = False
    truncated = False
    info: dict = {}
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        corps_steps += 1
        if terminated or truncated:
            break
    return {
        "obs": obs,
        "total_reward": total_reward,
        "corps_steps": corps_steps,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


# ---------------------------------------------------------------------------
# 1. Road Network
# ---------------------------------------------------------------------------


class TestRoadNetwork(unittest.TestCase):
    """Verify RoadNetwork and RoadSegment behaviour."""

    def test_road_segment_distance_on_segment(self) -> None:
        seg = RoadSegment(x0=0.0, y0=0.0, x1=100.0, y1=0.0)
        self.assertAlmostEqual(seg.distance_to_point(50.0, 0.0), 0.0, places=6)

    def test_road_segment_distance_perpendicular(self) -> None:
        seg = RoadSegment(x0=0.0, y0=0.0, x1=100.0, y1=0.0)
        self.assertAlmostEqual(seg.distance_to_point(50.0, 30.0), 30.0, places=5)

    def test_road_segment_distance_beyond_endpoint(self) -> None:
        seg = RoadSegment(x0=0.0, y0=0.0, x1=100.0, y1=0.0)
        # Point beyond the end — distance to endpoint (100, 0) from (200, 0)
        self.assertAlmostEqual(seg.distance_to_point(200.0, 0.0), 100.0, places=5)

    def test_road_segment_degenerate(self) -> None:
        # Zero-length segment — treated as a point
        seg = RoadSegment(x0=50.0, y0=50.0, x1=50.0, y1=50.0)
        self.assertAlmostEqual(seg.distance_to_point(53.0, 54.0), 5.0, places=4)

    def test_is_on_road_true(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 500.0, 1000.0, 500.0)],
            road_half_width=30.0,
        )
        self.assertTrue(net.is_on_road(500.0, 500.0))
        self.assertTrue(net.is_on_road(500.0, 520.0))  # within half-width

    def test_is_on_road_false(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 500.0, 1000.0, 500.0)],
            road_half_width=30.0,
        )
        self.assertFalse(net.is_on_road(500.0, 600.0))  # 100 m away

    def test_speed_modifier_on_road(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 500.0, 1000.0, 500.0)],
            road_half_width=30.0,
        )
        self.assertAlmostEqual(net.get_speed_modifier(500.0, 500.0), ROAD_SPEED_BONUS)

    def test_speed_modifier_off_road(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 500.0, 1000.0, 500.0)],
            road_half_width=30.0,
        )
        self.assertAlmostEqual(net.get_speed_modifier(500.0, 700.0), 1.0)

    def test_road_speed_bonus_value(self) -> None:
        self.assertAlmostEqual(ROAD_SPEED_BONUS, 1.5)

    def test_fraction_on_road_empty(self) -> None:
        net = RoadNetwork.generate_default(1000.0, 1000.0)
        self.assertAlmostEqual(net.fraction_on_road([]), 0.0)

    def test_fraction_on_road_all_on(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 500.0, 1000.0, 500.0)],
            road_half_width=50.0,
        )
        positions = [(100.0, 500.0), (500.0, 500.0), (800.0, 510.0)]
        frac = net.fraction_on_road(positions)
        self.assertAlmostEqual(frac, 1.0)

    def test_fraction_on_road_partial(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 500.0, 1000.0, 500.0)],
            road_half_width=30.0,
        )
        positions = [(100.0, 500.0), (100.0, 700.0)]
        frac = net.fraction_on_road(positions)
        self.assertAlmostEqual(frac, 0.5)

    def test_generate_default_has_segments(self) -> None:
        net = RoadNetwork.generate_default(10_000.0, 5_000.0)
        self.assertGreater(len(net), 0)

    def test_generate_default_n_roads(self) -> None:
        net = RoadNetwork.generate_default(1000.0, 1000.0, n_roads=2)
        # 2 horizontal + 1 central north-south = 3
        self.assertEqual(len(net), 3)

    def test_road_network_len(self) -> None:
        net = RoadNetwork(segments=[RoadSegment(0.0, 0.0, 1.0, 0.0)])
        self.assertEqual(len(net), 1)


# ---------------------------------------------------------------------------
# 2. Operational Objectives
# ---------------------------------------------------------------------------


class TestOperationalObjective(unittest.TestCase):
    """Verify OperationalObjective state and control logic."""

    def _make_inner_mock(self, positions: dict[str, tuple[float, float]]):
        """Create a minimal mock for the inner MultiBattalionEnv."""
        from envs.sim.battalion import Battalion

        class _Mock:
            _battalions: dict = {}
            _alive: set = set()

        mock = _Mock()
        for agent_id, (x, y) in positions.items():
            b = Battalion(x=x, y=y, theta=0.0, strength=1.0, team=0)
            mock._battalions[agent_id] = b
            mock._alive.add(agent_id)
        return mock

    def test_initial_control_value_neutral(self) -> None:
        obj = OperationalObjective(x=500.0, y=500.0, radius=100.0,
                                   obj_type=ObjectiveType.CAPTURE_HEX)
        self.assertAlmostEqual(obj.control_value, 0.5)

    def test_update_blue_control(self) -> None:
        obj = OperationalObjective(x=500.0, y=500.0, radius=100.0,
                                   obj_type=ObjectiveType.CAPTURE_HEX)
        inner = self._make_inner_mock({
            "blue_0": (500.0, 500.0),
            "blue_1": (510.0, 490.0),
        })
        obj.update(inner)
        self.assertTrue(obj.is_blue_controlled)
        self.assertFalse(obj.is_red_controlled)

    def test_update_red_control(self) -> None:
        obj = OperationalObjective(x=500.0, y=500.0, radius=100.0,
                                   obj_type=ObjectiveType.CAPTURE_HEX)
        inner = self._make_inner_mock({
            "red_0": (500.0, 500.0),
            "red_1": (510.0, 490.0),
        })
        obj.update(inner)
        self.assertFalse(obj.is_blue_controlled)
        self.assertTrue(obj.is_red_controlled)

    def test_update_contested(self) -> None:
        obj = OperationalObjective(x=500.0, y=500.0, radius=100.0,
                                   obj_type=ObjectiveType.CAPTURE_HEX)
        inner = self._make_inner_mock({
            "blue_0": (500.0, 500.0),
            "red_0": (510.0, 490.0),
        })
        obj.update(inner)
        self.assertAlmostEqual(obj.control_value, 0.5)
        self.assertFalse(obj.is_blue_controlled)
        self.assertFalse(obj.is_red_controlled)

    def test_update_outside_radius(self) -> None:
        obj = OperationalObjective(x=500.0, y=500.0, radius=50.0,
                                   obj_type=ObjectiveType.CAPTURE_HEX)
        inner = self._make_inner_mock({"blue_0": (700.0, 700.0)})
        obj.update(inner)
        # Nobody within radius
        self.assertAlmostEqual(obj.control_value, 0.5)

    def test_reset_clears_control(self) -> None:
        obj = OperationalObjective(x=500.0, y=500.0, radius=100.0,
                                   obj_type=ObjectiveType.CAPTURE_HEX)
        inner = self._make_inner_mock({"blue_0": (500.0, 500.0)})
        obj.update(inner)
        self.assertTrue(obj.is_blue_controlled)
        obj.reset()
        self.assertAlmostEqual(obj.control_value, 0.5)

    def test_objective_types_exist(self) -> None:
        self.assertEqual(ObjectiveType.CAPTURE_HEX, 0)
        self.assertEqual(ObjectiveType.CUT_SUPPLY_LINE, 1)
        self.assertEqual(ObjectiveType.FIX_AND_FLANK, 2)


# ---------------------------------------------------------------------------
# 3. Construction and spaces
# ---------------------------------------------------------------------------


class TestCorpsEnvConstruction(unittest.TestCase):
    """Verify CorpsEnv can be constructed with various arguments."""

    def test_default_construction(self) -> None:
        env = CorpsEnv()
        self.assertIsInstance(env.action_space, spaces.MultiDiscrete)
        self.assertIsInstance(env.observation_space, spaces.Box)
        env.close()

    def test_obs_dim_formula_2_divisions(self) -> None:
        env = make_env(n_divisions=2)
        expected = _corps_obs_dim(2)
        self.assertEqual(env._obs_dim, expected)
        self.assertEqual(env.observation_space.shape[0], expected)
        env.close()

    def test_obs_dim_formula_3_divisions(self) -> None:
        env = CorpsEnv(n_divisions=3)
        expected = _corps_obs_dim(3)
        self.assertEqual(env._obs_dim, expected)
        self.assertEqual(env.observation_space.shape[0], expected)
        env.close()

    def test_obs_dim_module_constant(self) -> None:
        env = CorpsEnv(n_divisions=3)
        self.assertEqual(env._obs_dim, CORPS_OBS_DIM)
        env.close()

    def test_action_space_shape(self) -> None:
        env = make_env(n_divisions=3)
        self.assertEqual(len(env.action_space.nvec), 3)
        self.assertTrue(all(v == env.n_corps_options for v in env.action_space.nvec))
        env.close()

    def test_obs_space_bounds_consistent(self) -> None:
        env = make_env()
        self.assertTrue(np.all(env.observation_space.high >= env.observation_space.low))
        env.close()

    def test_invalid_n_divisions_raises(self) -> None:
        with self.assertRaises(ValueError):
            CorpsEnv(n_divisions=0)

    def test_invalid_n_brigades_per_division_raises(self) -> None:
        with self.assertRaises(ValueError):
            CorpsEnv(n_brigades_per_division=0)

    def test_invalid_n_blue_per_brigade_raises(self) -> None:
        with self.assertRaises(ValueError):
            CorpsEnv(n_blue_per_brigade=0)

    def test_invalid_n_red_divisions_raises(self) -> None:
        with self.assertRaises(ValueError):
            CorpsEnv(n_red_divisions=0)

    def test_invalid_n_red_brigades_per_division_raises(self) -> None:
        with self.assertRaises(ValueError):
            CorpsEnv(n_red_brigades_per_division=0)

    def test_invalid_n_red_per_brigade_raises(self) -> None:
        with self.assertRaises(ValueError):
            CorpsEnv(n_red_per_brigade=0)

    def test_total_blue_count(self) -> None:
        env = make_env(n_divisions=2, n_brigades_per_division=3, n_blue_per_brigade=2)
        self.assertEqual(env.n_blue, 12)  # 2 * 3 * 2
        env.close()

    def test_total_red_defaults_match_blue(self) -> None:
        env = make_env(n_divisions=2, n_brigades_per_division=3, n_blue_per_brigade=2)
        self.assertEqual(env.n_red_divisions, 2)
        self.assertEqual(env.n_red_brigades_per_division, 3)
        self.assertEqual(env.n_red_per_brigade, 2)
        env.close()

    def test_asymmetric_red(self) -> None:
        env = CorpsEnv(
            n_divisions=2, n_brigades_per_division=2, n_blue_per_brigade=2,
            n_red_divisions=3, n_red_brigades_per_division=2, n_red_per_brigade=2,
        )
        self.assertEqual(env.n_blue_brigades, 4)
        self.assertEqual(env.n_red_brigades, 6)
        env.close()

    def test_road_network_generated_by_default(self) -> None:
        env = make_env()
        self.assertIsInstance(env.road_network, RoadNetwork)
        self.assertGreater(len(env.road_network), 0)
        env.close()

    def test_custom_road_network_accepted(self) -> None:
        net = RoadNetwork(segments=[RoadSegment(0.0, 100.0, 1000.0, 100.0)])
        env = make_env(road_network=net)
        self.assertIs(env.road_network, net)
        env.close()

    def test_inner_road_network_set(self) -> None:
        env = make_env()
        inner = env._division._brigade._inner
        self.assertIs(inner.road_network, env.road_network)
        env.close()

    def test_default_objectives_created(self) -> None:
        env = make_env()
        self.assertEqual(len(env.objectives), N_OBJECTIVES)
        types = {obj.obj_type for obj in env.objectives}
        self.assertIn(ObjectiveType.CAPTURE_HEX, types)
        self.assertIn(ObjectiveType.CUT_SUPPLY_LINE, types)
        self.assertIn(ObjectiveType.FIX_AND_FLANK, types)
        env.close()

    def test_custom_objectives_accepted(self) -> None:
        custom = [
            OperationalObjective(x=100.0, y=100.0, radius=50.0,
                                 obj_type=ObjectiveType.CAPTURE_HEX),
        ]
        env = make_env(objectives=custom)
        self.assertEqual(len(env.objectives), 1)
        env.close()

    def test_map_size_defaults(self) -> None:
        env = CorpsEnv()
        self.assertAlmostEqual(env.map_width, CORPS_MAP_WIDTH)
        self.assertAlmostEqual(env.map_height, CORPS_MAP_HEIGHT)
        env.close()

    def test_large_scale_5_divisions(self) -> None:
        """CorpsEnv must support 5 divisions per side."""
        env = CorpsEnv(n_divisions=5, n_brigades_per_division=2,
                       n_blue_per_brigade=2, n_red_divisions=5)
        self.assertEqual(env.n_divisions, 5)
        self.assertGreaterEqual(env.n_blue, 20)
        env.close()


# ---------------------------------------------------------------------------
# 4. Reset
# ---------------------------------------------------------------------------


class TestCorpsEnvReset(unittest.TestCase):
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
        self.assertFalse(np.allclose(obs1, obs2))
        env.close()

    def test_reset_step_progress_zero(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        step_progress = float(obs[-1])
        self.assertAlmostEqual(step_progress, 0.0, places=5)
        env.close()

    def test_reset_clears_corps_steps(self) -> None:
        env = make_env()
        env.reset(seed=0)
        env.step(env.action_space.sample())
        self.assertGreater(env._corps_steps, 0)
        env.reset(seed=0)
        self.assertEqual(env._corps_steps, 0)
        env.close()

    def test_reset_reattaches_road_network(self) -> None:
        env = make_env()
        env.reset(seed=0)
        inner = env._division._brigade._inner
        self.assertIs(inner.road_network, env.road_network)
        env.close()

    def test_reset_clears_objective_state(self) -> None:
        env = make_env()
        env.reset(seed=0)
        # Force an objective into a controlled state
        for obj in env.objectives:
            if obj.obj_type != ObjectiveType.FIX_AND_FLANK:
                obj._blue_units = 5
        # After reset, objectives should be neutral
        env.reset(seed=0)
        for obj in env.objectives:
            if obj.obj_type != ObjectiveType.FIX_AND_FLANK:
                self.assertAlmostEqual(obj.control_value, 0.5)
        env.close()


# ---------------------------------------------------------------------------
# 5. Step
# ---------------------------------------------------------------------------


class TestCorpsEnvStep(unittest.TestCase):
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

    def test_step_obs_shape(self) -> None:
        env = make_env()
        env.reset(seed=1)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        self.assertEqual(obs.shape, (env._obs_dim,))
        env.close()

    def test_step_obs_within_bounds(self) -> None:
        env = make_env()
        env.reset(seed=2)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        lo = env.observation_space.low
        hi = env.observation_space.high
        self.assertTrue(
            np.all(obs >= lo - 1e-5) and np.all(obs <= hi + 1e-5),
            msg="Obs out of bounds after step",
        )
        env.close()

    def test_step_info_keys(self) -> None:
        env = make_env()
        env.reset(seed=3)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertIn("corps_steps", info)
        self.assertIn("division_action", info)
        self.assertIn("objective_rewards", info)
        env.close()

    def test_step_objective_rewards_keys(self) -> None:
        env = make_env()
        env.reset(seed=4)
        _, _, _, _, info = env.step(env.action_space.sample())
        obj_r = info["objective_rewards"]
        self.assertIn("capture_hex", obj_r)
        self.assertIn("cut_supply_line", obj_r)
        self.assertIn("fix_and_flank", obj_r)
        env.close()

    def test_step_corps_steps_increment(self) -> None:
        env = make_env()
        env.reset(seed=0)
        env.step(env.action_space.sample())
        self.assertEqual(env._corps_steps, 1)
        env.step(env.action_space.sample())
        self.assertEqual(env._corps_steps, 2)
        env.close()

    def test_step_invalid_shape_raises(self) -> None:
        env = make_env(n_divisions=2)
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step(np.array([0], dtype=np.int64))  # wrong shape
        env.close()

    def test_step_invalid_command_index_raises(self) -> None:
        env = make_env(n_divisions=2)
        env.reset(seed=0)
        bad_action = np.array([0, env.n_corps_options], dtype=np.int64)
        with self.assertRaises(ValueError):
            env.step(bad_action)
        env.close()

    def test_step_all_six_commands(self) -> None:
        """All six corps command indices are accepted without error."""
        env = make_env()
        env.reset(seed=5)
        for cmd_idx in range(6):
            if not env._division._brigade._inner.agents:
                env.reset(seed=cmd_idx)
            action = np.array([cmd_idx] * env.n_divisions, dtype=np.int64)
            env.step(action)
        env.close()

    def test_step_reward_is_finite(self) -> None:
        env = make_env()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        self.assertTrue(np.isfinite(reward))
        env.close()


# ---------------------------------------------------------------------------
# 6. Corps-level observation features
# ---------------------------------------------------------------------------


class TestCorpsObsFeatures(unittest.TestCase):
    """Verify specific observation slices."""

    def test_sector_control_count(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        sector_obs = obs[:N_CORPS_SECTORS]
        self.assertEqual(len(sector_obs), N_CORPS_SECTORS)
        env.close()

    def test_sector_control_range(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        sector_obs = obs[:N_CORPS_SECTORS]
        self.assertTrue(np.all(sector_obs >= 0.0) and np.all(sector_obs <= 1.0))
        env.close()

    def test_division_status_initial_full_strength(self) -> None:
        env = make_env(n_divisions=2)
        obs, _ = env.reset(seed=0)
        status_start = N_CORPS_SECTORS
        n_div = env.n_divisions
        for i in range(n_div):
            avg_str = float(obs[status_start + i * 3])
            avg_mor = float(obs[status_start + i * 3 + 1])
            alive_ratio = float(obs[status_start + i * 3 + 2])
            self.assertAlmostEqual(avg_str, 1.0, places=4,
                                   msg=f"Division {i} avg_strength should be 1.0")
            self.assertAlmostEqual(avg_mor, 1.0, places=4,
                                   msg=f"Division {i} avg_morale should be 1.0")
            self.assertAlmostEqual(alive_ratio, 1.0, places=4,
                                   msg=f"Division {i} alive_ratio should be 1.0")
        env.close()

    def test_road_usage_in_bounds(self) -> None:
        env = make_env(n_divisions=2)
        obs, _ = env.reset(seed=0)
        road_start = N_CORPS_SECTORS + 8 * env.n_divisions
        road_obs = obs[road_start: road_start + N_ROAD_FEATURES]
        self.assertEqual(len(road_obs), N_ROAD_FEATURES)
        self.assertTrue(np.all(road_obs >= 0.0) and np.all(road_obs <= 1.0))
        env.close()

    def test_objective_obs_in_bounds(self) -> None:
        env = make_env(n_divisions=2)
        obs, _ = env.reset(seed=0)
        obj_start = N_CORPS_SECTORS + 8 * env.n_divisions + N_ROAD_FEATURES
        obj_obs = obs[obj_start: obj_start + N_OBJECTIVES]
        self.assertEqual(len(obj_obs), N_OBJECTIVES)
        self.assertTrue(np.all(obj_obs >= 0.0) and np.all(obj_obs <= 1.0))
        env.close()

    def test_step_progress_at_reset(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        self.assertAlmostEqual(float(obs[-1]), 0.0, places=5)
        env.close()

    def test_obs_length_matches_formula(self) -> None:
        for n_div in [1, 2, 3, 4]:
            env = make_env(n_divisions=n_div)
            obs, _ = env.reset(seed=0)
            expected = _corps_obs_dim(n_div)
            self.assertEqual(obs.shape[0], expected,
                             msg=f"Mismatch for n_divisions={n_div}")
            env.close()


# ---------------------------------------------------------------------------
# 7. Road network integration
# ---------------------------------------------------------------------------


class TestRoadNetworkIntegration(unittest.TestCase):
    """Verify road network is wired into the inner simulation."""

    def test_inner_road_network_matches_corps(self) -> None:
        env = make_env()
        env.reset(seed=0)
        inner = env._division._brigade._inner
        self.assertIs(inner.road_network, env.road_network)
        env.close()

    def test_speed_modifier_1_5_on_road(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 100.0, 10000.0, 100.0)],
            road_half_width=50.0,
        )
        self.assertAlmostEqual(net.get_speed_modifier(5000.0, 100.0), 1.5)

    def test_speed_modifier_1_0_off_road(self) -> None:
        net = RoadNetwork(
            segments=[RoadSegment(0.0, 100.0, 10000.0, 100.0)],
            road_half_width=50.0,
        )
        self.assertAlmostEqual(net.get_speed_modifier(5000.0, 300.0), 1.0)


# ---------------------------------------------------------------------------
# 8. Command translation
# ---------------------------------------------------------------------------


class TestCommandTranslation(unittest.TestCase):
    """Verify corps → division action expansion."""

    def test_translate_homogeneous(self) -> None:
        env = make_env(n_divisions=2, n_brigades_per_division=2)
        for cmd in range(env.n_corps_options):
            ca = np.array([cmd, cmd], dtype=np.int64)
            da = env._translate_corps_action(ca)
            self.assertEqual(da.shape, (env.n_blue_brigades,))
            self.assertTrue(np.all(da == cmd))
        env.close()

    def test_translate_heterogeneous(self) -> None:
        env = make_env(n_divisions=2, n_brigades_per_division=2)
        ca = np.array([1, 3], dtype=np.int64)
        da = env._translate_corps_action(ca)
        np.testing.assert_array_equal(da[:2], [1, 1])
        np.testing.assert_array_equal(da[2:], [3, 3])
        env.close()

    def test_translate_output_dtype(self) -> None:
        env = make_env()
        da = env._translate_corps_action(
            np.array([0] * env.n_divisions, dtype=np.int64)
        )
        self.assertEqual(da.dtype, np.int64)
        env.close()


# ---------------------------------------------------------------------------
# 9. Inter-division communication radius
# ---------------------------------------------------------------------------


class TestCommRadius(unittest.TestCase):
    """Verify comm_radius gates threat information."""

    def test_comm_radius_stored(self) -> None:
        env = make_env(comm_radius=2000.0)
        self.assertAlmostEqual(env.comm_radius, 2000.0)
        env.close()

    def test_zero_comm_radius_gives_sentinels(self) -> None:
        """With comm_radius=0, all threat vectors should be sentinels."""
        env = make_env(n_divisions=2, comm_radius=0.0)
        obs, _ = env.reset(seed=0)
        threat_start = N_CORPS_SECTORS + 3 * env.n_divisions
        threat_end = threat_start + 5 * env.n_divisions
        threat_obs = obs[threat_start:threat_end]
        # Each 5-element block should be [1, 0, 0, 0, 0] sentinel
        for i in range(env.n_divisions):
            block = threat_obs[i * 5: (i + 1) * 5]
            self.assertAlmostEqual(float(block[0]), 1.0, places=5,
                                   msg=f"Division {i} dist should be sentinel 1.0")
        env.close()

    def test_large_comm_radius_gives_threat_data(self) -> None:
        """With a very large comm_radius, we should not always get sentinels."""
        env = make_env(n_divisions=2, comm_radius=1e9)
        obs, _ = env.reset(seed=42)
        threat_start = N_CORPS_SECTORS + 3 * env.n_divisions
        threat_obs = obs[threat_start: threat_start + 5 * env.n_divisions]
        # Not ALL divisions should have the sentinel [1, 0, 0, 0, 0]
        # (at least one should have actual threat data)
        all_sentinel = all(
            abs(float(threat_obs[i * 5]) - 1.0) < 1e-4 and
            abs(float(threat_obs[i * 5 + 1])) < 1e-4
            for i in range(env.n_divisions)
        )
        self.assertFalse(all_sentinel,
                         "Expected at least one non-sentinel threat with large comm_radius")
        env.close()


# ---------------------------------------------------------------------------
# 10. Episode lifecycle
# ---------------------------------------------------------------------------


class TestCorpsEpisode(unittest.TestCase):
    """Verify episodes run to completion."""

    def test_1division_episode_runs(self) -> None:
        env = CorpsEnv(n_divisions=1, n_brigades_per_division=1,
                       n_blue_per_brigade=1, n_red_divisions=1,
                       n_red_brigades_per_division=1, n_red_per_brigade=1,
                       max_steps=200)
        result = run_episode(env, seed=42, max_steps=100)
        self.assertGreater(result["corps_steps"], 0)
        env.close()

    def test_obs_is_finite_during_episode(self) -> None:
        env = make_env(max_steps=100)
        result = run_episode(env, seed=7, max_steps=50)
        self.assertTrue(np.all(np.isfinite(result["obs"])))
        env.close()

    def test_reward_is_finite(self) -> None:
        env = make_env()
        result = run_episode(env, seed=2, max_steps=30)
        self.assertTrue(np.isfinite(result["total_reward"]))
        env.close()

    def test_multiple_resets(self) -> None:
        env = make_env(max_steps=200)
        for seed in range(3):
            result = run_episode(env, seed=seed, max_steps=50)
            self.assertGreater(result["corps_steps"], 0)
        env.close()

    def test_info_winner_on_episode_end(self) -> None:
        env = CorpsEnv(n_divisions=1, n_brigades_per_division=1,
                       n_blue_per_brigade=1, n_red_divisions=1,
                       n_red_brigades_per_division=1, n_red_per_brigade=1,
                       max_steps=100)
        result = run_episode(env, seed=0, max_steps=300)
        if result["terminated"] or result["truncated"]:
            self.assertIn("winner", result["info"])
        env.close()

    def test_configurable_episode_length(self) -> None:
        """max_steps can be configured to any positive value."""
        for max_steps in [50, 500, 3_600]:  # 50, 500, and 1-hour analogue
            env = make_env(max_steps=max_steps)
            self.assertEqual(env.max_steps, max_steps)
            env.close()


if __name__ == "__main__":
    unittest.main()
