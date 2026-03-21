# tests/test_human_env.py
"""Tests for envs/human_env.py and envs/rendering/web_renderer.py."""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Headless pygame — must be set before any pygame import.
# ---------------------------------------------------------------------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from envs.human_env import SCENARIOS, HumanEnv
from envs.rendering.web_renderer import WebRenderer
from envs.sim.battalion import Battalion
from envs.sim.terrain import TerrainMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(scenario: str = "open_field") -> HumanEnv:
    return HumanEnv(scenario=scenario)


def _make_blue() -> Battalion:
    return Battalion(x=300.0, y=400.0, theta=math.pi / 4, strength=0.8, team=0)


def _make_red() -> Battalion:
    return Battalion(x=700.0, y=600.0, theta=math.pi, strength=0.6, team=1)


# ---------------------------------------------------------------------------
# SCENARIOS dict
# ---------------------------------------------------------------------------


class TestScenariosDict(unittest.TestCase):
    """Validate the built-in SCENARIOS registry."""

    def test_at_least_three_scenarios(self) -> None:
        self.assertGreaterEqual(len(SCENARIOS), 3)

    def test_open_field_present(self) -> None:
        self.assertIn("open_field", SCENARIOS)

    def test_mountain_pass_present(self) -> None:
        self.assertIn("mountain_pass", SCENARIOS)

    def test_last_stand_present(self) -> None:
        self.assertIn("last_stand", SCENARIOS)

    def test_all_have_description(self) -> None:
        for name, cfg in SCENARIOS.items():
            self.assertIn(
                "description", cfg, f"Scenario {name!r} missing 'description'"
            )
            self.assertIsInstance(cfg["description"], str)
            self.assertGreater(len(cfg["description"]), 0)

    def test_all_have_valid_curriculum_level(self) -> None:
        for name, cfg in SCENARIOS.items():
            lvl = cfg.get("curriculum_level")
            self.assertIsNotNone(lvl, f"Scenario {name!r} missing 'curriculum_level'")
            self.assertIn(
                int(lvl),
                range(1, 6),
                f"Scenario {name!r}: curriculum_level={lvl} out of range",
            )

    def test_last_stand_has_reduced_strength(self) -> None:
        strength = SCENARIOS["last_stand"].get("initial_blue_strength", 1.0)
        self.assertLess(float(strength), 1.0)

    def test_all_descriptions_nonempty(self) -> None:
        for name, cfg in SCENARIOS.items():
            self.assertTrue(
                len(cfg.get("description", "")) > 0,
                f"Scenario {name!r} has empty description",
            )


# ---------------------------------------------------------------------------
# HumanEnv construction
# ---------------------------------------------------------------------------


class TestHumanEnvInit(unittest.TestCase):
    """Construction and validation of HumanEnv."""

    def test_default_construction(self) -> None:
        env = HumanEnv()
        self.assertIsNotNone(env)
        env.close()

    def test_all_built_in_scenarios_construct(self) -> None:
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                env = HumanEnv(scenario=name)
                self.assertEqual(env.scenario_name, name)
                env.close()

    def test_invalid_scenario_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            HumanEnv(scenario="nonexistent_scenario")

    def test_from_scenario_factory(self) -> None:
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                env = HumanEnv.from_scenario(name)
                self.assertEqual(env.scenario_name, name)
                env.close()

    def test_from_scenario_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            HumanEnv.from_scenario("bogus")

    def test_difficulty_overrides_curriculum_level(self) -> None:
        env = HumanEnv(scenario="open_field", difficulty=1)
        self.assertEqual(env._env.curriculum_level, 1)
        env.close()

    def test_difficulty_5_default(self) -> None:
        env = HumanEnv(scenario="open_field")
        # open_field has curriculum_level 5 in SCENARIOS
        self.assertEqual(env._env.curriculum_level, 5)
        env.close()

    def test_spaces_match_battalion_env(self) -> None:
        from envs.battalion_env import BattalionEnv

        human = HumanEnv()
        ref = BattalionEnv()
        self.assertEqual(human.observation_space.shape, ref.observation_space.shape)
        self.assertEqual(human.action_space.shape, ref.action_space.shape)
        human.close()
        ref.close()

    def test_render_mode_is_human(self) -> None:
        env = HumanEnv()
        self.assertEqual(env.render_mode, "human")
        env.close()


# ---------------------------------------------------------------------------
# Gymnasium API compliance
# ---------------------------------------------------------------------------


class TestHumanEnvGymnasiumAPI(unittest.TestCase):
    """Verify the Gymnasium API contract."""

    def setUp(self) -> None:
        self.env = HumanEnv(scenario="open_field")

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_returns_two_tuple(self) -> None:
        result = self.env.reset(seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_reset_obs_is_ndarray(self) -> None:
        obs, _ = self.env.reset(seed=0)
        self.assertIsInstance(obs, np.ndarray)

    def test_reset_obs_in_observation_space(self) -> None:
        obs, _ = self.env.reset(seed=0)
        self.assertTrue(
            self.env.observation_space.contains(obs.astype(np.float32)),
            "Initial observation is outside observation_space bounds",
        )

    def test_reset_info_contains_scenario_key(self) -> None:
        _, info = self.env.reset(seed=0)
        self.assertIn("scenario", info)
        self.assertEqual(info["scenario"], "open_field")

    def test_step_returns_five_tuple(self) -> None:
        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        result = self.env.step(action)
        self.assertEqual(len(result), 5)

    def test_step_obs_in_observation_space(self) -> None:
        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        obs, *_ = self.env.step(action)
        self.assertTrue(self.env.observation_space.contains(obs.astype(np.float32)))

    def test_action_space_sample_valid(self) -> None:
        for _ in range(10):
            action = self.env.action_space.sample()
            self.assertTrue(self.env.action_space.contains(action))

    def test_multiple_resets_reproducible(self) -> None:
        obs1, _ = self.env.reset(seed=123)
        obs2, _ = self.env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Scenario-specific behaviour
# ---------------------------------------------------------------------------


class TestLastStandScenario(unittest.TestCase):
    """The last_stand scenario should start Blue at reduced strength."""

    def setUp(self) -> None:
        self.env = HumanEnv(scenario="last_stand")
        self.env.reset(seed=42)

    def tearDown(self) -> None:
        self.env.close()

    def test_blue_starts_at_reduced_strength(self) -> None:
        expected = float(SCENARIOS["last_stand"]["initial_blue_strength"])
        self.assertAlmostEqual(self.env._env.blue.strength, expected, places=5)

    def test_obs_strength_index_matches(self) -> None:
        # After reset, obs[4] is Blue's normalised strength.
        obs, _ = self.env.reset(seed=42)
        expected = float(SCENARIOS["last_stand"]["initial_blue_strength"])
        self.assertAlmostEqual(float(obs[4]), expected, places=5)


class TestOpenFieldAndMountainPass(unittest.TestCase):
    """Smoke-test the other two built-in scenarios."""

    def _run_two_steps(self, scenario: str) -> None:
        env = HumanEnv(scenario=scenario)
        obs, info = env.reset(seed=0)
        self.assertEqual(info["scenario"], scenario)
        for _ in range(2):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(obs.astype(np.float32)))
            if terminated or truncated:
                break
        env.close()

    def test_open_field_two_steps(self) -> None:
        self._run_two_steps("open_field")

    def test_mountain_pass_two_steps(self) -> None:
        self._run_two_steps("mountain_pass")


# ---------------------------------------------------------------------------
# poll_action
# ---------------------------------------------------------------------------


class TestPollAction(unittest.TestCase):
    """Tests for HumanEnv.poll_action() in headless SDL mode."""

    def setUp(self) -> None:
        self.env = HumanEnv(scenario="open_field")
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_returns_ndarray_and_bool(self) -> None:
        action, quit_req = self.env.poll_action()
        self.assertIsInstance(action, np.ndarray)
        self.assertIsInstance(quit_req, bool)

    def test_action_shape(self) -> None:
        action, _ = self.env.poll_action()
        self.assertEqual(action.shape, (3,))

    def test_action_in_action_space(self) -> None:
        action, _ = self.env.poll_action()
        self.assertTrue(self.env.action_space.contains(action))

    def test_quit_not_requested_by_default(self) -> None:
        _, quit_req = self.env.poll_action()
        self.assertFalse(quit_req)

    def test_quit_flag_propagates(self) -> None:
        """Pre-setting _quit_requested must cause poll_action to return True."""
        self.env._quit_requested = True
        _, quit_req = self.env.poll_action()
        self.assertTrue(quit_req)


# ---------------------------------------------------------------------------
# Convenience properties
# ---------------------------------------------------------------------------


class TestHumanEnvProperties(unittest.TestCase):
    """Verify convenience properties are exposed correctly."""

    def setUp(self) -> None:
        self.env = HumanEnv(scenario="open_field")
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_step_count_starts_at_zero(self) -> None:
        self.assertEqual(self.env.step_count, 0)

    def test_step_count_increments(self) -> None:
        self.env.step(self.env.action_space.sample())
        self.assertEqual(self.env.step_count, 1)
        self.env.step(self.env.action_space.sample())
        self.assertEqual(self.env.step_count, 2)

    def test_map_dimensions(self) -> None:
        self.assertEqual(self.env.map_width, SCENARIOS["open_field"]["map_width"])
        self.assertEqual(self.env.map_height, SCENARIOS["open_field"]["map_height"])

    def test_scenario_name(self) -> None:
        self.assertEqual(self.env.scenario_name, "open_field")

    def test_scenario_description_nonempty(self) -> None:
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                env = HumanEnv(scenario=name)
                self.assertGreater(len(env.scenario_description), 0)
                env.close()


# ---------------------------------------------------------------------------
# WebRenderer
# ---------------------------------------------------------------------------


class TestWebRenderer(unittest.TestCase):
    """Tests for envs/rendering/web_renderer.py."""

    def setUp(self) -> None:
        self.renderer = WebRenderer(map_width=1000.0, map_height=1000.0)
        self.blue = _make_blue()
        self.red = _make_red()
        self.terrain = TerrainMap.flat(1000.0, 1000.0)

    # ------------------------------------------------------------------
    # Basic frame structure
    # ------------------------------------------------------------------

    def test_render_frame_returns_dict(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red, step=5)
        self.assertIsInstance(frame, dict)

    def test_frame_has_required_keys(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red)
        for key in ("step", "map", "blue", "red", "terrain_summary", "info"):
            self.assertIn(key, frame, f"Missing key {key!r}")

    def test_step_value(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red, step=42)
        self.assertEqual(frame["step"], 42)

    def test_map_dimensions(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red)
        self.assertEqual(frame["map"]["width"], 1000.0)
        self.assertEqual(frame["map"]["height"], 1000.0)

    # ------------------------------------------------------------------
    # Battalion dict
    # ------------------------------------------------------------------

    def test_blue_battalion_keys(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red)
        for key in (
            "x", "y", "x_norm", "y_norm",
            "theta", "cos_theta", "sin_theta",
            "strength", "morale", "routed", "team",
        ):
            self.assertIn(key, frame["blue"], f"Missing key {key!r} in blue dict")

    def test_battalion_normalised_positions_in_range(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red)
        for side in ("blue", "red"):
            self.assertGreaterEqual(frame[side]["x_norm"], 0.0)
            self.assertLessEqual(frame[side]["x_norm"], 1.0)
            self.assertGreaterEqual(frame[side]["y_norm"], 0.0)
            self.assertLessEqual(frame[side]["y_norm"], 1.0)

    def test_battalion_cos_sin_consistent(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red)
        theta = frame["blue"]["theta"]
        self.assertAlmostEqual(frame["blue"]["cos_theta"], math.cos(theta), places=6)
        self.assertAlmostEqual(frame["blue"]["sin_theta"], math.sin(theta), places=6)

    def test_battalion_team_values(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red)
        self.assertEqual(frame["blue"]["team"], 0)
        self.assertEqual(frame["red"]["team"], 1)

    # ------------------------------------------------------------------
    # Terrain summary
    # ------------------------------------------------------------------

    def test_terrain_summary_none_when_no_terrain(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red, terrain=None)
        self.assertIsNone(frame["terrain_summary"])

    def test_terrain_summary_with_flat_terrain(self) -> None:
        # Flat terrain has all-zero elevation → max=0 → no grid returned.
        frame = self.renderer.render_frame(self.blue, self.red, terrain=self.terrain)
        # Flat terrain has zero elevation everywhere; summary should be None.
        self.assertIsNone(frame["terrain_summary"])

    def test_terrain_summary_with_random_terrain(self) -> None:
        rng = np.random.default_rng(0)
        terrain = TerrainMap.generate_random(rng=rng, width=1000.0, height=1000.0)
        frame = self.renderer.render_frame(self.blue, self.red, terrain=terrain)
        summary = frame["terrain_summary"]
        if summary is not None:  # non-flat terrain produces a grid
            self.assertIsInstance(summary, list)
            self.assertIsInstance(summary[0], list)
            # Values should be in [0, 1]
            for row in summary:
                for val in row:
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0 + 1e-9)

    def test_terrain_grid_size_respected(self) -> None:
        rng = np.random.default_rng(1)
        terrain = TerrainMap.generate_random(rng=rng, width=1000.0, height=1000.0)
        renderer = WebRenderer(1000.0, 1000.0, terrain_grid_size=5)
        frame = renderer.render_frame(self.blue, self.red, terrain=terrain)
        if frame["terrain_summary"] is not None:
            self.assertLessEqual(len(frame["terrain_summary"]), 5)
            for row in frame["terrain_summary"]:
                self.assertLessEqual(len(row), 5)

    # ------------------------------------------------------------------
    # JSON serialisability
    # ------------------------------------------------------------------

    def test_frame_is_json_serialisable_no_terrain(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red, step=3)
        # Should not raise.
        serialised = json.dumps(frame)
        self.assertIsInstance(serialised, str)

    def test_frame_is_json_serialisable_with_terrain(self) -> None:
        rng = np.random.default_rng(2)
        terrain = TerrainMap.generate_random(rng=rng, width=1000.0, height=1000.0)
        frame = self.renderer.render_frame(
            self.blue, self.red, terrain=terrain, step=1, info={"foo": 1}
        )
        serialised = json.dumps(frame)
        self.assertIsInstance(serialised, str)

    # ------------------------------------------------------------------
    # Info passthrough
    # ------------------------------------------------------------------

    def test_info_passed_through(self) -> None:
        info = {"winner": "blue", "step_count": 10}
        frame = self.renderer.render_frame(self.blue, self.red, info=info)
        self.assertEqual(frame["info"]["winner"], "blue")
        self.assertEqual(frame["info"]["step_count"], 10)

    def test_info_defaults_to_empty_dict(self) -> None:
        frame = self.renderer.render_frame(self.blue, self.red)
        self.assertEqual(frame["info"], {})


# ---------------------------------------------------------------------------
# Renderer skip_events parameter
# ---------------------------------------------------------------------------


class TestRendererSkipEvents(unittest.TestCase):
    """The new skip_events parameter on BattalionRenderer.render_frame()."""

    def setUp(self) -> None:
        from envs.rendering.renderer import BattalionRenderer

        self.renderer = BattalionRenderer(map_width=1000.0, map_height=1000.0, fps=0)
        self.blue = _make_blue()
        self.red = _make_red()

    def tearDown(self) -> None:
        self.renderer.close()

    def test_skip_events_false_still_renders(self) -> None:
        alive = self.renderer.render_frame(
            self.blue, self.red, skip_events=False
        )
        self.assertTrue(alive)

    def test_skip_events_true_returns_true(self) -> None:
        alive = self.renderer.render_frame(
            self.blue, self.red, skip_events=True
        )
        self.assertTrue(alive)

    def test_default_skip_events_is_false(self) -> None:
        # Calling without skip_events should work (default False).
        alive = self.renderer.render_frame(self.blue, self.red)
        self.assertTrue(alive)


if __name__ == "__main__":
    unittest.main()
