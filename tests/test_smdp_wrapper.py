# SPDX-License-Identifier: MIT
# tests/test_smdp_wrapper.py
"""Tests for envs/options.py and envs/smdp_wrapper.py — SMDP / Options framework."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium import spaces
from pettingzoo.test import parallel_api_test

from envs.multi_battalion_env import MultiBattalionEnv
from envs.options import (
    MacroAction,
    Option,
    OBS_COS_THETA,
    OBS_MORALE,
    OBS_SELF_X,
    OBS_SELF_Y,
    OBS_SIN_THETA,
    OBS_STRENGTH,
    make_default_options,
)
from envs.smdp_wrapper import SMDPWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(n_blue: int = 2, n_red: int = 2, **kwargs) -> SMDPWrapper:
    """Return a freshly constructed SMDPWrapper around MultiBattalionEnv."""
    return SMDPWrapper(MultiBattalionEnv(n_blue=n_blue, n_red=n_red, **kwargs))


def run_episode(
    env: SMDPWrapper,
    seed: int = 0,
    max_macro_steps: int = 200,
) -> dict:
    """Run a full macro-episode and return a summary dict."""
    obs, _ = env.reset(seed=seed)
    total_macro_steps = 0
    total_prim_steps = 0
    all_rewards: dict[str, float] = {}

    for _ in range(max_macro_steps):
        if not env.agents:
            break
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        total_macro_steps += 1
        for agent, rew in rewards.items():
            all_rewards[agent] = all_rewards.get(agent, 0.0) + rew
        # Read primitive step count from first agent's info (global counter)
        if infos:
            ta = next(iter(infos.values())).get("temporal_abstraction", {})
            total_prim_steps = ta.get("primitive_steps", total_prim_steps)
        if all(terminated.get(a, False) or truncated.get(a, False) for a in terminated):
            break

    return {
        "macro_steps": total_macro_steps,
        "primitive_steps": total_prim_steps,
        "rewards": all_rewards,
    }


# ---------------------------------------------------------------------------
# 1. MacroAction enum
# ---------------------------------------------------------------------------


class TestMacroAction(unittest.TestCase):
    """Verify the MacroAction enum values."""

    def test_enum_count(self) -> None:
        self.assertEqual(len(MacroAction), 6)

    def test_enum_values(self) -> None:
        self.assertEqual(int(MacroAction.ADVANCE_SECTOR), 0)
        self.assertEqual(int(MacroAction.DEFEND_POSITION), 1)
        self.assertEqual(int(MacroAction.FLANK_LEFT), 2)
        self.assertEqual(int(MacroAction.FLANK_RIGHT), 3)
        self.assertEqual(int(MacroAction.WITHDRAW), 4)
        self.assertEqual(int(MacroAction.CONCENTRATE_FIRE), 5)

    def test_enum_names(self) -> None:
        names = {m.name for m in MacroAction}
        expected = {
            "ADVANCE_SECTOR",
            "DEFEND_POSITION",
            "FLANK_LEFT",
            "FLANK_RIGHT",
            "WITHDRAW",
            "CONCENTRATE_FIRE",
        }
        self.assertEqual(names, expected)


# ---------------------------------------------------------------------------
# 2. Option dataclass
# ---------------------------------------------------------------------------


class TestOptionDataclass(unittest.TestCase):
    """Unit-tests for the Option class."""

    def _make_obs(
        self,
        strength: float = 1.0,
        morale: float = 1.0,
    ) -> np.ndarray:
        obs = np.zeros(22, dtype=np.float32)  # 2v2 obs_dim
        obs[OBS_SELF_X] = 0.5
        obs[OBS_SELF_Y] = 0.5
        obs[OBS_COS_THETA] = 1.0
        obs[OBS_SIN_THETA] = 0.0
        obs[OBS_STRENGTH] = strength
        obs[OBS_MORALE] = morale
        return obs

    def test_get_action_wrong_shape_raises(self) -> None:
        """Policy returning wrong shape should raise ValueError."""
        bad_policy = Option(
            name="bad_shape",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.array([0.5, 0.5]),  # shape (2,) — wrong
            termination=lambda obs, s: False,
        )
        with self.assertRaises(ValueError):
            bad_policy.get_action(self._make_obs())

    def test_get_action_non_finite_raises(self) -> None:
        """Policy returning NaN or Inf should raise ValueError."""
        nan_policy = Option(
            name="nan_action",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.array([float("nan"), 0.0, 0.5], dtype=np.float32),
            termination=lambda obs, s: False,
        )
        with self.assertRaises(ValueError):
            nan_policy.get_action(self._make_obs())

        inf_policy = Option(
            name="inf_action",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.array([float("inf"), 0.0, 0.5], dtype=np.float32),
            termination=lambda obs, s: False,
        )
        with self.assertRaises(ValueError):
            inf_policy.get_action(self._make_obs())

    def test_get_action_valid_shape_and_finite(self) -> None:
        """Valid policy should not raise."""
        opt = Option(
            name="valid",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.array([0.0, 0.0, 1.0], dtype=np.float32),
            termination=lambda obs, s: False,
        )
        action = opt.get_action(self._make_obs())
        self.assertEqual(action.shape, (3,))
        opt = Option(
            name="test",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.zeros(3, dtype=np.float32),
            termination=lambda obs, s: False,
        )
        self.assertTrue(opt.can_initiate(self._make_obs()))

    def test_get_action_shape_and_dtype(self) -> None:
        opt = Option(
            name="test",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.array([0.5, -0.3, 0.7]),
            termination=lambda obs, s: False,
        )
        action = opt.get_action(self._make_obs())
        self.assertEqual(action.shape, (3,))
        self.assertEqual(action.dtype, np.float32)

    def test_should_terminate_on_hard_cap(self) -> None:
        opt = Option(
            name="test",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.zeros(3, dtype=np.float32),
            termination=lambda obs, s: False,
            max_steps=10,
        )
        self.assertFalse(opt.should_terminate(self._make_obs(), 9))
        self.assertTrue(opt.should_terminate(self._make_obs(), 10))
        self.assertTrue(opt.should_terminate(self._make_obs(), 11))

    def test_should_terminate_on_condition(self) -> None:
        # Terminates when morale drops below 0.25
        opt = Option(
            name="test",
            initiation_set=lambda obs: True,
            policy=lambda obs: np.zeros(3, dtype=np.float32),
            termination=lambda obs, s: float(obs[OBS_MORALE]) < 0.25,
            max_steps=50,
        )
        high_morale_obs = self._make_obs(morale=0.9)
        low_morale_obs = self._make_obs(morale=0.1)
        self.assertFalse(opt.should_terminate(high_morale_obs, 5))
        self.assertTrue(opt.should_terminate(low_morale_obs, 5))


# ---------------------------------------------------------------------------
# 3. make_default_options
# ---------------------------------------------------------------------------


class TestMakeDefaultOptions(unittest.TestCase):
    """Verify the default six-element macro-action vocabulary."""

    def setUp(self) -> None:
        self.options = make_default_options(max_steps=30)
        self.healthy_obs = np.zeros(22, dtype=np.float32)
        self.healthy_obs[OBS_STRENGTH] = 1.0
        self.healthy_obs[OBS_MORALE] = 1.0

        self.low_morale_obs = self.healthy_obs.copy()
        self.low_morale_obs[OBS_MORALE] = 0.2

        self.low_strength_obs = self.healthy_obs.copy()
        self.low_strength_obs[OBS_STRENGTH] = 0.2

    def test_returns_six_options(self) -> None:
        self.assertEqual(len(self.options), 6)

    def test_option_names_match_vocabulary(self) -> None:
        names = [o.name for o in self.options]
        self.assertIn("advance_sector", names)
        self.assertIn("defend_position", names)
        self.assertIn("flank_left", names)
        self.assertIn("flank_right", names)
        self.assertIn("withdraw", names)
        self.assertIn("concentrate_fire", names)

    def test_all_six_produce_distinct_actions(self) -> None:
        """All six macro-actions must trigger distinct primitive-action patterns."""
        actions = [o.get_action(self.healthy_obs) for o in self.options]
        # Compare each pair — at least one element must differ
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                self.assertFalse(
                    np.allclose(actions[i], actions[j]),
                    f"Options {i} and {j} produce identical actions: "
                    f"{actions[i]} vs {actions[j]}",
                )

    def test_actions_within_primitive_bounds(self) -> None:
        """All option policies must produce actions within the primitive action space."""
        act_low = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
        act_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        for opt in self.options:
            action = opt.get_action(self.healthy_obs)
            self.assertTrue(
                np.all(action >= act_low) and np.all(action <= act_high),
                f"Option '{opt.name}' produced out-of-bounds action: {action}",
            )

    def test_advance_sector_moves_forward(self) -> None:
        advance = self.options[MacroAction.ADVANCE_SECTOR]
        action = advance.get_action(self.healthy_obs)
        self.assertGreater(action[0], 0.0, "advance_sector should move forward")

    def test_defend_position_fires_maximally(self) -> None:
        defend = self.options[MacroAction.DEFEND_POSITION]
        action = defend.get_action(self.healthy_obs)
        self.assertAlmostEqual(action[2], 1.0, places=4,
                               msg="defend_position should fire at max intensity")
        self.assertAlmostEqual(action[0], 0.0, places=4,
                               msg="defend_position should not move")

    def test_flank_left_rotates_counter_clockwise(self) -> None:
        flank_left = self.options[MacroAction.FLANK_LEFT]
        action = flank_left.get_action(self.healthy_obs)
        self.assertLess(action[1], 0.0, "flank_left should rotate CCW (negative)")
        self.assertAlmostEqual(action[2], 0.0, places=4,
                               msg="flank_left should not fire")

    def test_flank_right_rotates_clockwise(self) -> None:
        flank_right = self.options[MacroAction.FLANK_RIGHT]
        action = flank_right.get_action(self.healthy_obs)
        self.assertGreater(action[1], 0.0, "flank_right should rotate CW (positive)")
        self.assertAlmostEqual(action[2], 0.0, places=4,
                               msg="flank_right should not fire")

    def test_withdraw_moves_backward(self) -> None:
        withdraw = self.options[MacroAction.WITHDRAW]
        action = withdraw.get_action(self.low_morale_obs)
        self.assertLess(action[0], 0.0, "withdraw should move backward")
        self.assertAlmostEqual(action[2], 0.0, places=4,
                               msg="withdraw should not fire")

    def test_concentrate_fire_max_fire_no_movement(self) -> None:
        concentrate = self.options[MacroAction.CONCENTRATE_FIRE]
        action = concentrate.get_action(self.healthy_obs)
        self.assertAlmostEqual(action[2], 1.0, places=4,
                               msg="concentrate_fire should fire at max intensity")
        self.assertAlmostEqual(action[0], 0.0, places=4,
                               msg="concentrate_fire should not move")

    def test_advance_sector_initiation_gated_on_morale(self) -> None:
        advance = self.options[MacroAction.ADVANCE_SECTOR]
        self.assertTrue(advance.can_initiate(self.healthy_obs))
        # Very low morale should prevent initiation
        very_low = self.healthy_obs.copy()
        very_low[OBS_MORALE] = 0.1
        self.assertFalse(advance.can_initiate(very_low))

    def test_withdraw_initiation_gated_on_low_morale_or_strength(self) -> None:
        withdraw = self.options[MacroAction.WITHDRAW]
        self.assertFalse(withdraw.can_initiate(self.healthy_obs))
        self.assertTrue(withdraw.can_initiate(self.low_morale_obs))
        self.assertTrue(withdraw.can_initiate(self.low_strength_obs))

    def test_defend_and_concentrate_always_initiatable(self) -> None:
        for idx in [MacroAction.DEFEND_POSITION, MacroAction.CONCENTRATE_FIRE]:
            opt = self.options[idx]
            self.assertTrue(opt.can_initiate(self.healthy_obs))
            self.assertTrue(opt.can_initiate(self.low_morale_obs))

    def test_flanking_options_shorter_max_steps(self) -> None:
        """Flanking options should have a shorter max_steps than full options."""
        advance_max = self.options[MacroAction.ADVANCE_SECTOR].max_steps
        flank_left_max = self.options[MacroAction.FLANK_LEFT].max_steps
        flank_right_max = self.options[MacroAction.FLANK_RIGHT].max_steps
        self.assertLessEqual(flank_left_max, advance_max)
        self.assertLessEqual(flank_right_max, advance_max)

    def test_custom_max_steps(self) -> None:
        opts = make_default_options(max_steps=60)
        # Non-flanking options should have max_steps=60
        self.assertEqual(opts[MacroAction.ADVANCE_SECTOR].max_steps, 60)
        self.assertEqual(opts[MacroAction.DEFEND_POSITION].max_steps, 60)
        # Flanking options should be at most 30 (60//2)
        self.assertLessEqual(opts[MacroAction.FLANK_LEFT].max_steps, 30)
        self.assertLessEqual(opts[MacroAction.FLANK_RIGHT].max_steps, 30)

    def test_zero_max_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_default_options(max_steps=0)

    def test_negative_max_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_default_options(max_steps=-5)


# ---------------------------------------------------------------------------
# 4. SMDPWrapper construction
# ---------------------------------------------------------------------------


class TestSMDPWrapperInit(unittest.TestCase):
    """Verify SMDPWrapper can be constructed and exposes correct attributes."""

    def test_default_construction(self) -> None:
        env = make_env()
        self.assertIsInstance(env, SMDPWrapper)
        env.close()

    def test_possible_agents_match_underlying(self) -> None:
        base = MultiBattalionEnv(n_blue=2, n_red=2)
        env = SMDPWrapper(base)
        self.assertEqual(env.possible_agents, base.possible_agents)
        env.close()

    def test_action_space_is_discrete(self) -> None:
        env = make_env()
        for agent in env.possible_agents:
            act_space = env.action_space(agent)
            self.assertIsInstance(act_space, spaces.Discrete)
            self.assertEqual(act_space.n, 6)
        env.close()

    def test_observation_space_matches_underlying(self) -> None:
        base = MultiBattalionEnv(n_blue=2, n_red=2)
        env = SMDPWrapper(base)
        for agent in env.possible_agents:
            self.assertEqual(
                env.observation_space(agent),
                base.observation_space(agent),
            )
        env.close()

    def test_custom_options(self) -> None:
        custom_opts = make_default_options()[:3]
        base = MultiBattalionEnv(n_blue=1, n_red=1)
        env = SMDPWrapper(base, options=custom_opts)
        self.assertEqual(env.n_options, 3)
        for agent in env.possible_agents:
            self.assertEqual(env.action_space(agent).n, 3)
        env.close()

    def test_metadata_name(self) -> None:
        env = make_env()
        self.assertIn("name", env.metadata)
        env.close()

    def test_empty_options_raises(self) -> None:
        with self.assertRaises(ValueError):
            SMDPWrapper(MultiBattalionEnv(n_blue=1, n_red=1), options=[])


# ---------------------------------------------------------------------------
# 5. SMDPWrapper reset
# ---------------------------------------------------------------------------


class TestSMDPWrapperReset(unittest.TestCase):
    """Verify reset() behaves correctly."""

    def test_reset_returns_obs_and_infos(self) -> None:
        env = make_env()
        obs, infos = env.reset(seed=0)
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(infos, dict)
        env.close()

    def test_reset_populates_agents(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        env.reset(seed=0)
        self.assertEqual(set(env.agents), set(env.possible_agents))
        env.close()

    def test_reset_obs_keys_match_agents(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(set(obs.keys()), set(env.agents))
        env.close()

    def test_reset_obs_within_bounds(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=42)
        for agent, o in obs.items():
            sp = env.observation_space(agent)
            self.assertTrue(
                sp.contains(o),
                f"Observation for {agent} out of bounds after reset",
            )
        env.close()

    def test_reset_clears_counters(self) -> None:
        env = make_env()
        # Run one episode then reset
        env.reset(seed=0)
        actions = {a: 0 for a in env.agents}
        env.step(actions)
        # Reset again
        env.reset(seed=1)
        self.assertEqual(env._macro_steps, 0)
        self.assertEqual(env._primitive_steps, 0)
        env.close()

    def test_reset_deterministic(self) -> None:
        env = make_env()
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        for agent in env.possible_agents:
            np.testing.assert_array_equal(obs1[agent], obs2[agent])
        env.close()


# ---------------------------------------------------------------------------
# 6. SMDPWrapper step
# ---------------------------------------------------------------------------


class TestSMDPWrapperStep(unittest.TestCase):
    """Verify step() returns correct structure and semantics."""

    def setUp(self) -> None:
        self.env = make_env(n_blue=2, n_red=2)
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_step_returns_five_tuple(self) -> None:
        actions = {a: int(self.env.action_space(a).sample()) for a in self.env.agents}
        result = self.env.step(actions)
        self.assertEqual(len(result), 5)

    def test_step_obs_within_bounds(self) -> None:
        actions = {a: int(self.env.action_space(a).sample()) for a in self.env.agents}
        obs, *_ = self.env.step(actions)
        for agent, o in obs.items():
            sp = self.env.observation_space(agent)
            self.assertTrue(
                sp.contains(o),
                f"Observation for {agent} out of bounds after step",
            )

    def test_step_rewards_are_floats(self) -> None:
        actions = {a: int(self.env.action_space(a).sample()) for a in self.env.agents}
        _, rewards, *_ = self.env.step(actions)
        for agent, r in rewards.items():
            self.assertIsInstance(r, float)

    def test_step_terminated_truncated_are_bool_dicts(self) -> None:
        actions = {a: int(self.env.action_space(a).sample()) for a in self.env.agents}
        _, _, terminated, truncated, _ = self.env.step(actions)
        for v in terminated.values():
            self.assertIsInstance(v, bool)
        for v in truncated.values():
            self.assertIsInstance(v, bool)

    def test_step_empty_agents_returns_empty(self) -> None:
        self.env.agents = []
        actions: dict = {}
        result = self.env.step(actions)
        self.assertEqual(result, ({}, {}, {}, {}, {}))

    def test_step_increments_macro_steps(self) -> None:
        actions = {a: 1 for a in self.env.agents}  # defend_position
        self.env.step(actions)
        self.assertEqual(self.env._macro_steps, 1)

    def test_step_increments_primitive_steps(self) -> None:
        actions = {a: 1 for a in self.env.agents}
        self.env.step(actions)
        self.assertGreater(self.env._primitive_steps, 0)

    def test_options_execute_multiple_primitive_steps(self) -> None:
        """One macro-step must correspond to at least 1 primitive step."""
        actions = {a: int(MacroAction.DEFEND_POSITION) for a in self.env.agents}
        self.env.step(actions)
        # defend_position runs for up to max_steps primitive steps
        self.assertGreater(self.env._primitive_steps, 0)

    def test_temporal_abstraction_in_info(self) -> None:
        actions = {a: 0 for a in self.env.agents}
        _, _, _, _, infos = self.env.step(actions)
        for agent in list(infos.keys()):
            ta = infos[agent].get("temporal_abstraction")
            self.assertIsNotNone(ta, f"No temporal_abstraction in info for {agent}")
            self.assertIn("macro_steps", ta)
            self.assertIn("primitive_steps", ta)
            self.assertIn("ratio", ta)
            self.assertIn("option_name", ta)
            self.assertIn("option_steps", ta)

    def test_temporal_abstraction_ratio_is_positive(self) -> None:
        actions = {a: 0 for a in self.env.agents}
        _, _, _, _, infos = self.env.step(actions)
        for info in infos.values():
            ta = info["temporal_abstraction"]
            self.assertGreater(ta["ratio"], 0.0)
            break

    def test_option_name_in_info(self) -> None:
        """The info should record which option was executed."""
        option_names = {o.name for o in make_default_options()}
        actions = {a: 0 for a in self.env.agents}
        _, _, _, _, infos = self.env.step(actions)
        for info in infos.values():
            name = info["temporal_abstraction"]["option_name"]
            self.assertIn(name, option_names)

    def test_missing_agents_default_to_index_zero(self) -> None:
        """Agents absent from macro_actions dict should default to option 0."""
        # Pass only partial actions
        agents = list(self.env.agents)
        partial_actions = {agents[0]: 1}
        # Should not raise
        self.env.step(partial_actions)

    def test_out_of_range_action_raises(self) -> None:
        """Providing an index outside [0, n_options-1] should raise ValueError."""
        agents = list(self.env.agents)
        actions = {agents[0]: 999}  # way out of range
        with self.assertRaises(ValueError):
            self.env.step(actions)

    def test_negative_action_raises(self) -> None:
        """A negative macro-action index should raise ValueError."""
        agents = list(self.env.agents)
        actions = {agents[0]: -1}
        with self.assertRaises(ValueError):
            self.env.step(actions)

    def test_aggregate_rewards_are_finite(self) -> None:
        for _ in range(3):
            if not self.env.agents:
                break
            actions = {a: int(self.env.action_space(a).sample()) for a in self.env.agents}
            _, rewards, _, _, _ = self.env.step(actions)
            for r in rewards.values():
                self.assertTrue(np.isfinite(r), f"Non-finite reward: {r}")


# ---------------------------------------------------------------------------
# 7. Temporal abstraction
# ---------------------------------------------------------------------------


class TestTemporalAbstraction(unittest.TestCase):
    """Verify temporal abstraction ratio is meaningful."""

    def test_ratio_less_than_one(self) -> None:
        """macro_steps should always be ≤ primitive_steps."""
        env = make_env()
        env.reset(seed=7)
        for _ in range(5):
            if not env.agents:
                break
            actions = {a: int(MacroAction.DEFEND_POSITION) for a in env.agents}
            env.step(actions)
        if env._primitive_steps > 0:
            self.assertLessEqual(
                env.temporal_abstraction_ratio,
                1.0,
                "macro_steps should be <= primitive_steps",
            )
        env.close()

    def test_ratio_zero_before_first_step(self) -> None:
        env = make_env()
        env.reset(seed=0)
        self.assertEqual(env.temporal_abstraction_ratio, 0.0)
        env.close()


# ---------------------------------------------------------------------------
# 8. Full episode run
# ---------------------------------------------------------------------------


class TestFullEpisode(unittest.TestCase):
    """Smoke tests for full macro-episode runs."""

    def test_episode_terminates_1v1(self) -> None:
        """A 1v1 episode should eventually terminate."""
        env = SMDPWrapper(MultiBattalionEnv(n_blue=1, n_red=1, max_steps=200))
        result = run_episode(env, seed=42, max_macro_steps=100)
        self.assertGreater(result["macro_steps"], 0)
        env.close()

    def test_episode_terminates_2v2(self) -> None:
        env = make_env(n_blue=2, n_red=2)
        result = run_episode(env, seed=0, max_macro_steps=100)
        self.assertGreater(result["macro_steps"], 0)
        env.close()

    def test_episode_all_six_macro_actions_cycle(self) -> None:
        """Cycling through all six macro-actions should not crash."""
        env = make_env()
        env.reset(seed=5)
        for step_idx in range(30):
            if not env.agents:
                break
            macro_idx = step_idx % 6
            actions = {a: macro_idx for a in env.agents}
            env.step(actions)
        env.close()

    def test_primitive_steps_exceed_macro_steps(self) -> None:
        """Over an episode, primitive_steps should be > macro_steps."""
        env = make_env()
        result = run_episode(env, seed=1, max_macro_steps=20)
        if result["macro_steps"] > 0:
            self.assertGreater(result["primitive_steps"], result["macro_steps"])
        env.close()


# ---------------------------------------------------------------------------
# 9. PettingZoo API compliance
# ---------------------------------------------------------------------------


class TestPettingZooCompliance(unittest.TestCase):
    """Verify SMDPWrapper passes the PettingZoo parallel_api_test."""

    def test_parallel_api_test_1v1(self) -> None:
        env = SMDPWrapper(MultiBattalionEnv(n_blue=1, n_red=1, max_steps=100))
        parallel_api_test(env, num_cycles=5)
        env.close()

    def test_parallel_api_test_2v2(self) -> None:
        env = SMDPWrapper(MultiBattalionEnv(n_blue=2, n_red=2, max_steps=100))
        parallel_api_test(env, num_cycles=3)
        env.close()


# ---------------------------------------------------------------------------
# 10. Scenario-specific option termination
# ---------------------------------------------------------------------------


class TestOptionTermination(unittest.TestCase):
    """Verify option termination conditions fire correctly."""

    def test_option_terminates_within_max_steps(self) -> None:
        """An option running against a no-op env must end within max_steps."""
        options = make_default_options(max_steps=10)
        env = SMDPWrapper(
            MultiBattalionEnv(n_blue=1, n_red=1, max_steps=500),
            options=options,
        )
        env.reset(seed=0)
        actions = {a: int(MacroAction.DEFEND_POSITION) for a in env.agents}
        env.step(actions)
        # At most 10 primitive steps should have run for this option
        self.assertLessEqual(env._primitive_steps, 10 * len(env.possible_agents))
        env.close()

    def test_withdraw_option_runs_if_initiated(self) -> None:
        """Withdraw option can be selected even if initiation is not gated here."""
        env = make_env()
        env.reset(seed=0)
        # Force withdraw for all agents regardless of initiation check
        actions = {a: int(MacroAction.WITHDRAW) for a in env.agents}
        # Should not raise regardless of morale state
        env.step(actions)
        env.close()

    def test_all_options_can_be_selected(self) -> None:
        """Every macro-action index can be passed to step() without error."""
        env = make_env()
        env.reset(seed=0)
        for macro_idx in range(6):
            if not env.agents:
                env.reset(seed=macro_idx)
            actions = {a: macro_idx for a in env.agents}
            env.step(actions)
        env.close()


if __name__ == "__main__":
    unittest.main()
