# tests/test_multi_battalion_env.py
"""Tests for envs/multi_battalion_env.py — MultiBattalionEnv PettingZoo env."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pettingzoo.test import parallel_api_test

from envs.multi_battalion_env import (
    MAP_HEIGHT,
    MAP_WIDTH,
    MAX_STEPS,
    VISIBILITY_RADIUS,
    MultiBattalionEnv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(**kwargs) -> MultiBattalionEnv:
    return MultiBattalionEnv(**kwargs)


def run_episode(env: MultiBattalionEnv, seed: int = 0, max_steps: int = 200) -> dict:
    """Run a full episode and return a summary dict."""
    obs, _ = env.reset(seed=seed)
    total_steps = 0
    for _ in range(max_steps):
        if not env.agents:
            break
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, terminated, truncated, _ = env.step(actions)
        total_steps += 1
        if all(terminated.get(a, False) or truncated.get(a, False) for a in terminated):
            break
    return {"steps": total_steps, "obs": obs}


# ---------------------------------------------------------------------------
# 1. Construction / argument validation
# ---------------------------------------------------------------------------


class TestInit(unittest.TestCase):
    """Verify argument validation in MultiBattalionEnv.__init__."""

    def test_default_construction(self) -> None:
        env = make_env()
        self.assertEqual(env.n_blue, 2)
        self.assertEqual(env.n_red, 2)
        env.close()

    def test_asymmetric_teams(self) -> None:
        env = make_env(n_blue=3, n_red=1)
        self.assertEqual(len(env.possible_agents), 4)
        env.close()

    def test_1v1_construction(self) -> None:
        env = make_env(n_blue=1, n_red=1)
        self.assertEqual(len(env.possible_agents), 2)
        env.close()

    def test_zero_n_blue_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(n_blue=0)

    def test_zero_n_red_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(n_red=0)

    def test_negative_n_blue_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(n_blue=-1)

    def test_zero_map_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(map_width=0)

    def test_negative_map_height_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(map_height=-100)

    def test_zero_max_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(max_steps=0)

    def test_invalid_hill_speed_factor_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(hill_speed_factor=0.0)
        with self.assertRaises(ValueError):
            make_env(hill_speed_factor=1.5)

    def test_zero_visibility_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_env(visibility_radius=0.0)


# ---------------------------------------------------------------------------
# 2. possible_agents / agent IDs
# ---------------------------------------------------------------------------


class TestAgentIDs(unittest.TestCase):
    def test_possible_agents_naming_2v2(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        self.assertEqual(
            env.possible_agents,
            ["blue_0", "blue_1", "red_0", "red_1"],
        )
        env.close()

    def test_possible_agents_naming_3v1(self) -> None:
        env = make_env(n_blue=3, n_red=1)
        self.assertEqual(
            env.possible_agents,
            ["blue_0", "blue_1", "blue_2", "red_0"],
        )
        env.close()

    def test_agents_equals_possible_after_reset(self) -> None:
        env = make_env()
        env.reset(seed=0)
        self.assertEqual(set(env.agents), set(env.possible_agents))
        env.close()


# ---------------------------------------------------------------------------
# 3. Observation space
# ---------------------------------------------------------------------------


class TestObservationSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env(n_blue=2, n_red=2)
        self.env.reset(seed=7)

    def tearDown(self) -> None:
        self.env.close()

    def test_obs_space_same_object_per_agent(self) -> None:
        """observation_space must return the exact same object for an agent."""
        for agent in self.env.possible_agents:
            self.assertIs(
                self.env.observation_space(agent),
                self.env.observation_space(agent),
            )

    def test_obs_space_correct_dim(self) -> None:
        # obs_dim = 6 + 5 * (n_blue + n_red - 1) + 1 = 6 + 5*3 + 1 = 22
        expected_dim = 6 + 5 * (2 + 2 - 1) + 1
        obs_space = self.env.observation_space("blue_0")
        self.assertEqual(obs_space.shape, (expected_dim,))

    def test_obs_within_space_bounds(self) -> None:
        obs, _ = self.env.reset(seed=42)
        for agent_id, ob in obs.items():
            space = self.env.observation_space(agent_id)
            self.assertTrue(
                np.all(ob >= space.low - 1e-6) and np.all(ob <= space.high + 1e-6),
                f"obs for {agent_id} out of bounds",
            )

    def test_obs_dtype_float32(self) -> None:
        obs, _ = self.env.reset(seed=1)
        for ob in obs.values():
            self.assertEqual(ob.dtype, np.float32)

    def test_angles_are_cos_sin_not_raw_radians(self) -> None:
        """Verify that angle features (cos, sin) stay in [-1, 1]."""
        obs, _ = self.env.reset(seed=3)
        for ob in obs.values():
            # Indices 2 and 3 are cos(θ) and sin(θ)
            self.assertGreaterEqual(float(ob[2]), -1.0 - 1e-6)
            self.assertLessEqual(float(ob[2]), 1.0 + 1e-6)
            self.assertGreaterEqual(float(ob[3]), -1.0 - 1e-6)
            self.assertLessEqual(float(ob[3]), 1.0 + 1e-6)

    def test_position_normalised_to_01(self) -> None:
        obs, _ = self.env.reset(seed=5)
        for ob in obs.values():
            self.assertGreaterEqual(float(ob[0]), 0.0 - 1e-6)
            self.assertLessEqual(float(ob[0]), 1.0 + 1e-6)
            self.assertGreaterEqual(float(ob[1]), 0.0 - 1e-6)
            self.assertLessEqual(float(ob[1]), 1.0 + 1e-6)

    def test_obs_all_blue_agents_different(self) -> None:
        """Each agent should have a different observation (different positions)."""
        obs, _ = self.env.reset(seed=99)
        b0 = obs["blue_0"]
        b1 = obs["blue_1"]
        self.assertFalse(np.allclose(b0, b1), "blue_0 and blue_1 should have different obs")

    def test_obs_step_norm_zero_after_reset(self) -> None:
        obs, _ = self.env.reset(seed=0)
        for ob in obs.values():
            self.assertAlmostEqual(float(ob[-1]), 0.0, places=5)


# ---------------------------------------------------------------------------
# 4. Action space
# ---------------------------------------------------------------------------


class TestActionSpace(unittest.TestCase):
    def test_action_space_same_object_per_agent(self) -> None:
        env = make_env()
        for agent in env.possible_agents:
            self.assertIs(
                env.action_space(agent),
                env.action_space(agent),
            )
        env.close()

    def test_action_space_shape(self) -> None:
        env = make_env()
        act_space = env.action_space("blue_0")
        self.assertEqual(act_space.shape, (3,))
        env.close()

    def test_action_space_bounds(self) -> None:
        env = make_env()
        act_space = env.action_space("blue_0")
        np.testing.assert_array_equal(act_space.low, [-1.0, -1.0, 0.0])
        np.testing.assert_array_equal(act_space.high, [1.0, 1.0, 1.0])
        env.close()


# ---------------------------------------------------------------------------
# 5. Reset
# ---------------------------------------------------------------------------


class TestReset(unittest.TestCase):
    def test_reset_returns_obs_and_infos_dicts(self) -> None:
        env = make_env()
        result = env.reset(seed=0)
        self.assertIsInstance(result, tuple)
        obs, infos = result
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(infos, dict)
        env.close()

    def test_reset_obs_keys_match_agents(self) -> None:
        env = make_env()
        obs, infos = env.reset(seed=1)
        self.assertEqual(set(obs.keys()), set(env.agents))
        self.assertEqual(set(infos.keys()), set(env.agents))
        env.close()

    def test_reset_with_seed_is_deterministic(self) -> None:
        env = make_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        for agent in obs1:
            np.testing.assert_array_equal(obs1[agent], obs2[agent])
        env.close()

    def test_reset_different_seeds_different_positions(self) -> None:
        env = make_env()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=999)
        # With different seeds the positions should differ
        different = any(
            not np.allclose(obs1[a], obs2[a]) for a in obs1
        )
        self.assertTrue(different, "Different seeds should produce different observations")
        env.close()

    def test_reset_options_accepted(self) -> None:
        """reset() should accept options kwarg without raising."""
        env = make_env()
        env.reset(seed=0, options={"dummy": True})
        env.close()

    def test_reset_restores_all_agents(self) -> None:
        """After a partial episode, reset() should restore all agents."""
        env = make_env()
        env.reset(seed=0)
        # Run a few steps to potentially kill some agents
        for _ in range(10):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        # Now reset
        env.reset(seed=0)
        self.assertEqual(set(env.agents), set(env.possible_agents))
        env.close()


# ---------------------------------------------------------------------------
# 6. Step
# ---------------------------------------------------------------------------


class TestStep(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env(n_blue=2, n_red=2)
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_step_returns_five_dicts(self) -> None:
        actions = {a: self.env.action_space(a).sample() for a in self.env.agents}
        result = self.env.step(actions)
        self.assertEqual(len(result), 5)
        obs, rew, term, trunc, info = result
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(rew, dict)
        self.assertIsInstance(term, dict)
        self.assertIsInstance(trunc, dict)
        self.assertIsInstance(info, dict)

    def test_step_keys_match_pre_step_agents(self) -> None:
        agents_before = list(self.env.agents)
        actions = {a: self.env.action_space(a).sample() for a in agents_before}
        obs, rew, term, trunc, info = self.env.step(actions)
        self.assertEqual(set(obs.keys()), set(agents_before))
        self.assertEqual(set(rew.keys()), set(agents_before))
        self.assertEqual(set(term.keys()), set(agents_before))
        self.assertEqual(set(trunc.keys()), set(agents_before))

    def test_step_obs_within_bounds(self) -> None:
        actions = {a: self.env.action_space(a).sample() for a in self.env.agents}
        obs, _, _, _, _ = self.env.step(actions)
        for agent_id, ob in obs.items():
            space = self.env.observation_space(agent_id)
            self.assertTrue(np.all(ob >= space.low - 1e-6))
            self.assertTrue(np.all(ob <= space.high + 1e-6))

    def test_empty_step_after_episode_ends(self) -> None:
        """step() on a finished env (no agents) should return empty dicts."""
        self.env.reset(seed=0)
        self.env.agents = []
        result = self.env.step({})
        for d in result:
            self.assertEqual(d, {})

    def test_step_count_increments(self) -> None:
        self.env.reset(seed=0)
        for i in range(1, 4):
            if not self.env.agents:
                break
            actions = {a: self.env.action_space(a).sample() for a in self.env.agents}
            self.env.step(actions)
            self.assertEqual(self.env._step_count, i)

    def test_terminated_agents_removed_from_agents_list(self) -> None:
        """Agents that are terminated should not appear in self.agents next call."""
        env = make_env(n_blue=1, n_red=1, max_steps=2000)
        env.reset(seed=42)
        visited_agents: set[str] = set()
        for _ in range(2000):
            if not env.agents:
                break
            visited_agents.update(env.agents)
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, _, term, trunc, _ = env.step(actions)
            done = {a for a in term if term[a] or trunc.get(a, False)}
            for a in done:
                self.assertNotIn(a, env.agents)
        env.close()


# ---------------------------------------------------------------------------
# 7. Global state
# ---------------------------------------------------------------------------


class TestState(unittest.TestCase):
    def test_state_shape_2v2(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        env.reset(seed=0)
        state = env.state()
        expected_dim = 6 * 4 + 1
        self.assertEqual(state.shape, (expected_dim,))
        env.close()

    def test_state_shape_3v1(self) -> None:
        env = make_env(n_blue=3, n_red=1)
        env.reset(seed=0)
        state = env.state()
        expected_dim = 6 * 4 + 1
        self.assertEqual(state.shape, (expected_dim,))
        env.close()

    def test_state_dtype_float32(self) -> None:
        env = make_env()
        env.reset(seed=0)
        self.assertEqual(env.state().dtype, np.float32)
        env.close()

    def test_state_step_norm_at_zero_after_reset(self) -> None:
        env = make_env()
        env.reset(seed=0)
        self.assertAlmostEqual(float(env.state()[-1]), 0.0, places=5)
        env.close()

    def test_state_step_norm_increases(self) -> None:
        env = make_env()
        env.reset(seed=0)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)
        step_norm = float(env.state()[-1])
        self.assertGreater(step_norm, 0.0)
        env.close()

    def test_state_positions_normalised(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        env.reset(seed=5)
        state = env.state()
        # First 6*4 elements are [x/w, y/h, cos, sin, str, mor] per agent
        for i in range(4):
            base = i * 6
            x_norm = state[base]
            y_norm = state[base + 1]
            self.assertGreaterEqual(float(x_norm), -1e-6)
            self.assertLessEqual(float(x_norm), 1.0 + 1e-6)
            self.assertGreaterEqual(float(y_norm), -1e-6)
            self.assertLessEqual(float(y_norm), 1.0 + 1e-6)
        env.close()


# ---------------------------------------------------------------------------
# 8. Fog of war
# ---------------------------------------------------------------------------


class TestFogOfWar(unittest.TestCase):
    def test_far_enemy_features_hidden(self) -> None:
        """Enemy beyond visibility_radius should have state fields zeroed."""
        env = make_env(n_blue=1, n_red=1, visibility_radius=1.0)
        obs, _ = env.reset(seed=0)
        # With visibility_radius=1m (almost zero), enemy should be hidden
        # blue_0 obs: [self(6)] + [red_0 block (5)] + [step(1)]
        # red_0 block should be [1.0, 0.0, 0.0, 0.0, 0.0]
        b0_obs = obs["blue_0"]
        red0_block = b0_obs[6:11]
        self.assertAlmostEqual(float(red0_block[0]), 1.0, places=4)  # dist=1.0 (far)
        self.assertAlmostEqual(float(red0_block[1]), 0.0, places=4)  # hidden
        self.assertAlmostEqual(float(red0_block[2]), 0.0, places=4)  # hidden
        self.assertAlmostEqual(float(red0_block[3]), 0.0, places=4)  # strength hidden
        self.assertAlmostEqual(float(red0_block[4]), 0.0, places=4)  # morale hidden
        env.close()

    def test_nearby_enemy_features_visible(self) -> None:
        """Enemy within visibility_radius should expose full state."""
        env = make_env(n_blue=1, n_red=1, visibility_radius=1e9)
        obs, _ = env.reset(seed=0)
        b0_obs = obs["blue_0"]
        red0_block = b0_obs[6:11]
        # Strength should be 1.0 (full strength at start) and morale 1.0
        self.assertAlmostEqual(float(red0_block[3]), 1.0, places=4)  # strength
        self.assertAlmostEqual(float(red0_block[4]), 1.0, places=4)  # morale
        env.close()


# ---------------------------------------------------------------------------
# 9. Seeding / reproducibility
# ---------------------------------------------------------------------------


class TestSeeding(unittest.TestCase):
    def test_same_seed_same_episode(self) -> None:
        """Two episodes with the same seed must be identical."""
        env = make_env(n_blue=2, n_red=2)
        rewards_a: list[float] = []
        rewards_b: list[float] = []

        for reward_list in [rewards_a, rewards_b]:
            env.reset(seed=123)
            rng = np.random.default_rng(123)
            for _ in range(30):
                if not env.agents:
                    break
                # Use fixed random actions keyed by agent_id
                fixed_actions = {
                    agent_id: np.array(
                        [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1)],
                        dtype=np.float32,
                    )
                    for agent_id in list(env.possible_agents)
                    if agent_id in env.agents
                }
                _, rew, _, _, _ = env.step(fixed_actions)
                reward_list.extend(rew.values())

        self.assertEqual(rewards_a, rewards_b)
        env.close()

    def test_max_cycles_attribute_accepted(self) -> None:
        """parallel_api_test sets max_cycles on the env — must not raise."""
        env = make_env()
        env.max_cycles = 100
        env.reset(seed=0)
        env.close()


# ---------------------------------------------------------------------------
# 10. PettingZoo parallel_api_test
# ---------------------------------------------------------------------------


class TestPettingZooAPICompliance(unittest.TestCase):
    """Verify full PettingZoo parallel API compliance."""

    def test_api_test_2v2(self) -> None:
        env = make_env(n_blue=2, n_red=2, max_steps=200)
        parallel_api_test(env, num_cycles=200)

    def test_api_test_1v1(self) -> None:
        env = make_env(n_blue=1, n_red=1, max_steps=200)
        parallel_api_test(env, num_cycles=200)

    def test_api_test_3v2(self) -> None:
        env = make_env(n_blue=3, n_red=2, max_steps=200)
        parallel_api_test(env, num_cycles=200)


# ---------------------------------------------------------------------------
# 11. Full-episode smoke tests
# ---------------------------------------------------------------------------


class TestFullEpisode(unittest.TestCase):
    def test_episode_terminates(self) -> None:
        env = make_env(n_blue=2, n_red=2, max_steps=500)
        result = run_episode(env, seed=0, max_steps=600)
        # Episode must have ended (either agents=[] or steps reached max)
        self.assertGreaterEqual(result["steps"], 1)
        env.close()

    def test_agents_list_monotone_decreasing(self) -> None:
        """Once an agent leaves env.agents, it must never return."""
        env = make_env(n_blue=2, n_red=2)
        env.reset(seed=7)
        ever_seen: set[str] = set(env.agents)
        removed: set[str] = set()
        for _ in range(500):
            if not env.agents:
                break
            current = set(env.agents)
            revived = current & removed
            self.assertEqual(revived, set(), f"Agents revived after removal: {revived}")
            removed |= ever_seen - current
            ever_seen = current
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        env.close()

    def test_rewards_are_finite(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        env.reset(seed=1)
        for _ in range(100):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, rew, _, _, _ = env.step(actions)
            for agent_id, r in rew.items():
                self.assertTrue(
                    np.isfinite(r), f"Non-finite reward for {agent_id}: {r}"
                )
        env.close()

    def test_num_agents_property(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        env.reset(seed=0)
        self.assertEqual(env.num_agents, 4)
        self.assertEqual(env.max_num_agents, 4)
        env.close()


if __name__ == "__main__":
    unittest.main()
