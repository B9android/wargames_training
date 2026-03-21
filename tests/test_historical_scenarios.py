"""Tests for envs/scenarios/historical.py — E5.4 historical scenario validation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Allow absolute imports when running from project root or tests/ directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.scenarios.historical import (
    ComparisonResult,
    HistoricalOutcome,
    HistoricalScenario,
    OutcomeComparator,
    ScenarioLoader,
    ScenarioUnit,
    TerrainConfig,
    load_scenario,
)
from envs.sim.battalion import Battalion
from envs.sim.engine import EpisodeResult, SimEngine
from envs.sim.terrain import TerrainMap

# ---------------------------------------------------------------------------
# Paths to bundled scenario files
# ---------------------------------------------------------------------------

SCENARIOS_DIR = PROJECT_ROOT / "configs" / "scenarios" / "historical"
WATERLOO_YAML = SCENARIOS_DIR / "waterloo.yaml"
AUSTERLITZ_YAML = SCENARIOS_DIR / "austerlitz.yaml"
BORODINO_YAML = SCENARIOS_DIR / "borodino.yaml"

ALL_YAML_FILES = [WATERLOO_YAML, AUSTERLITZ_YAML, BORODINO_YAML]


# ---------------------------------------------------------------------------
# ScenarioUnit
# ---------------------------------------------------------------------------


class TestScenarioUnit(unittest.TestCase):
    def _make_unit(self, **kwargs) -> ScenarioUnit:
        defaults = dict(unit_id="test", x=100.0, y=200.0, theta=0.0, strength=0.9, team=0)
        defaults.update(kwargs)
        return ScenarioUnit(**defaults)

    def test_to_battalion_copies_position(self) -> None:
        unit = self._make_unit(x=123.4, y=567.8)
        bat = unit.to_battalion()
        self.assertAlmostEqual(bat.x, 123.4)
        self.assertAlmostEqual(bat.y, 567.8)

    def test_to_battalion_copies_team(self) -> None:
        for team in (0, 1):
            with self.subTest(team=team):
                bat = self._make_unit(team=team).to_battalion()
                self.assertEqual(bat.team, team)

    def test_to_battalion_clamps_strength_high(self) -> None:
        bat = self._make_unit(strength=1.5).to_battalion()
        self.assertLessEqual(bat.strength, 1.0)

    def test_to_battalion_clamps_strength_low(self) -> None:
        bat = self._make_unit(strength=-0.1).to_battalion()
        self.assertGreaterEqual(bat.strength, 0.0)

    def test_to_battalion_returns_new_instance_each_call(self) -> None:
        unit = self._make_unit()
        b1 = unit.to_battalion()
        b2 = unit.to_battalion()
        self.assertIsNot(b1, b2)


# ---------------------------------------------------------------------------
# ScenarioLoader — unit-level parsing
# ---------------------------------------------------------------------------


class TestScenarioLoaderParsing(unittest.TestCase):
    """Verify that ScenarioLoader handles edge-cases in field parsing."""

    def _loader(self, raw: dict) -> HistoricalScenario:
        loader = ScenarioLoader.__new__(ScenarioLoader)
        loader.path = Path("/fake/path.yaml")
        return loader._parse(raw)

    def test_named_theta_east(self) -> None:
        raw = {
            "units": {"blue": [{"id": "u", "x": 0, "y": 0, "theta": "east", "strength": 1.0}]}
        }
        scenario = self._loader(raw)
        self.assertAlmostEqual(scenario.blue_units[0].theta, 0.0)

    def test_named_theta_west(self) -> None:
        raw = {
            "units": {"blue": [{"id": "u", "x": 0, "y": 0, "theta": "west", "strength": 1.0}]}
        }
        scenario = self._loader(raw)
        self.assertAlmostEqual(scenario.blue_units[0].theta, math.pi)

    def test_named_theta_north(self) -> None:
        raw = {
            "units": {"blue": [{"id": "u", "x": 0, "y": 0, "theta": "north", "strength": 1.0}]}
        }
        scenario = self._loader(raw)
        self.assertAlmostEqual(scenario.blue_units[0].theta, math.pi / 2)

    def test_numeric_theta_preserved(self) -> None:
        raw = {
            "units": {"blue": [{"id": "u", "x": 0, "y": 0, "theta": 1.23, "strength": 1.0}]}
        }
        scenario = self._loader(raw)
        self.assertAlmostEqual(scenario.blue_units[0].theta, 1.23)

    def test_null_winner_parsed_as_none(self) -> None:
        raw = {
            "historical_outcome": {
                "winner": None,
                "blue_casualties": 0.3,
                "red_casualties": 0.4,
                "duration_steps": 400,
            }
        }
        scenario = self._loader(raw)
        self.assertIsNone(scenario.historical_outcome.winner)

    def test_draw_winner_string_parsed_as_none(self) -> None:
        raw = {"historical_outcome": {"winner": "draw"}}
        scenario = self._loader(raw)
        self.assertIsNone(scenario.historical_outcome.winner)

    def test_integer_winner_parsed(self) -> None:
        for w in (0, 1):
            with self.subTest(winner=w):
                raw = {"historical_outcome": {"winner": w}}
                scenario = self._loader(raw)
                self.assertEqual(scenario.historical_outcome.winner, w)

    def test_defaults_applied_for_missing_fields(self) -> None:
        scenario = self._loader({})
        self.assertEqual(scenario.name, "Unknown Scenario")
        self.assertEqual(scenario.terrain_config.terrain_type, "flat")
        self.assertIsNone(scenario.historical_outcome.winner)
        self.assertEqual(scenario.blue_units, [])
        self.assertEqual(scenario.red_units, [])


# ---------------------------------------------------------------------------
# ScenarioLoader — file-based loading (bundled YAML files)
# ---------------------------------------------------------------------------


class TestScenarioLoaderFiles(unittest.TestCase):
    def test_waterloo_loads_without_error(self) -> None:
        scenario = ScenarioLoader(WATERLOO_YAML).load()
        self.assertIn("Waterloo", scenario.name)

    def test_austerlitz_loads_without_error(self) -> None:
        scenario = ScenarioLoader(AUSTERLITZ_YAML).load()
        self.assertIn("Austerlitz", scenario.name)

    def test_borodino_loads_without_error(self) -> None:
        scenario = ScenarioLoader(BORODINO_YAML).load()
        self.assertIn("Borodino", scenario.name)

    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ScenarioLoader("/nonexistent/path/scenario.yaml").load()

    def test_source_path_is_set(self) -> None:
        scenario = ScenarioLoader(WATERLOO_YAML).load()
        self.assertEqual(scenario.source_path, WATERLOO_YAML)


# ---------------------------------------------------------------------------
# HistoricalScenario properties
# ---------------------------------------------------------------------------


class TestHistoricalScenarioProperties(unittest.TestCase):
    def _load(self, path: Path) -> HistoricalScenario:
        return ScenarioLoader(path).load()

    def test_n_blue_matches_unit_list(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                self.assertEqual(s.n_blue, len(s.blue_units))

    def test_n_red_matches_unit_list(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                self.assertEqual(s.n_red, len(s.red_units))

    def test_all_scenarios_have_at_least_one_blue_unit(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                self.assertGreater(s.n_blue, 0)

    def test_all_scenarios_have_at_least_one_red_unit(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                self.assertGreater(s.n_red, 0)

    def test_unit_strengths_in_range(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                for u in s.blue_units + s.red_units:
                    self.assertGreaterEqual(u.strength, 0.0)
                    self.assertLessEqual(u.strength, 1.0)

    def test_unit_teams_consistent(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                for u in s.blue_units:
                    self.assertEqual(u.team, 0)
                for u in s.red_units:
                    self.assertEqual(u.team, 1)

    def test_historical_casualties_in_range(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                ho = s.historical_outcome
                self.assertGreaterEqual(ho.blue_casualties, 0.0)
                self.assertLessEqual(ho.blue_casualties, 1.0)
                self.assertGreaterEqual(ho.red_casualties, 0.0)
                self.assertLessEqual(ho.red_casualties, 1.0)

    def test_waterloo_has_correct_historical_winner(self) -> None:
        s = self._load(WATERLOO_YAML)
        self.assertEqual(s.historical_outcome.winner, 0)

    def test_austerlitz_has_correct_historical_winner(self) -> None:
        s = self._load(AUSTERLITZ_YAML)
        self.assertEqual(s.historical_outcome.winner, 0)

    def test_borodino_has_draw_as_historical_winner(self) -> None:
        s = self._load(BORODINO_YAML)
        self.assertIsNone(s.historical_outcome.winner)

    def test_build_battalions_returns_fresh_instances(self) -> None:
        s = self._load(WATERLOO_YAML)
        blue1, red1 = s.build_battalions()
        blue2, red2 = s.build_battalions()
        # Different object identities
        self.assertIsNot(blue1[0], blue2[0])

    def test_build_terrain_returns_terrain_map(self) -> None:
        for yaml_path in ALL_YAML_FILES:
            with self.subTest(yaml=yaml_path.name):
                s = self._load(yaml_path)
                terrain = s.build_terrain()
                self.assertIsInstance(terrain, TerrainMap)


# ---------------------------------------------------------------------------
# load_scenario convenience function
# ---------------------------------------------------------------------------


class TestLoadScenario(unittest.TestCase):
    def test_returns_four_tuple(self) -> None:
        result = load_scenario(WATERLOO_YAML)
        self.assertEqual(len(result), 4)

    def test_second_element_is_list_of_battalions(self) -> None:
        _, blue, _, _ = load_scenario(WATERLOO_YAML)
        for b in blue:
            self.assertIsInstance(b, Battalion)

    def test_third_element_is_list_of_battalions(self) -> None:
        _, _, red, _ = load_scenario(WATERLOO_YAML)
        for r in red:
            self.assertIsInstance(r, Battalion)

    def test_fourth_element_is_terrain_map(self) -> None:
        _, _, _, terrain = load_scenario(WATERLOO_YAML)
        self.assertIsInstance(terrain, TerrainMap)


# ---------------------------------------------------------------------------
# All three scenarios run to completion without errors
# ---------------------------------------------------------------------------


class TestScenarioRunToCompletion(unittest.TestCase):
    """Acceptance criterion: all three scenarios complete without exceptions."""

    def _run_1v1(self, yaml_path: Path) -> EpisodeResult:
        scenario, blue, red, terrain = load_scenario(yaml_path)
        rng = np.random.default_rng(42)
        result = SimEngine(blue[0], red[0], terrain=terrain, rng=rng).run()
        return result

    def test_waterloo_runs_to_completion(self) -> None:
        result = self._run_1v1(WATERLOO_YAML)
        self._assert_valid_result(result)

    def test_austerlitz_runs_to_completion(self) -> None:
        result = self._run_1v1(AUSTERLITZ_YAML)
        self._assert_valid_result(result)

    def test_borodino_runs_to_completion(self) -> None:
        result = self._run_1v1(BORODINO_YAML)
        self._assert_valid_result(result)

    def _assert_valid_result(self, result: EpisodeResult) -> None:
        self.assertGreater(result.steps, 0)
        self.assertGreaterEqual(result.blue_strength, 0.0)
        self.assertLessEqual(result.blue_strength, 1.0)
        self.assertGreaterEqual(result.red_strength, 0.0)
        self.assertLessEqual(result.red_strength, 1.0)
        self.assertGreaterEqual(result.blue_morale, 0.0)
        self.assertLessEqual(result.blue_morale, 1.0)
        self.assertTrue(math.isfinite(result.blue_strength))
        self.assertTrue(math.isfinite(result.red_strength))


# ---------------------------------------------------------------------------
# OutcomeComparator
# ---------------------------------------------------------------------------


def _make_episode_result(**kwargs) -> EpisodeResult:
    defaults = dict(
        winner=0,
        steps=400,
        blue_strength=0.70,
        red_strength=0.30,
        blue_morale=0.80,
        red_morale=0.20,
        blue_routed=False,
        red_routed=True,
    )
    defaults.update(kwargs)
    return EpisodeResult(**defaults)


def _make_historical_outcome(**kwargs) -> HistoricalOutcome:
    defaults = dict(
        winner=0,
        blue_casualties=0.30,
        red_casualties=0.70,
        duration_steps=400,
        description="Test outcome",
    )
    defaults.update(kwargs)
    return HistoricalOutcome(**defaults)


class TestOutcomeComparator(unittest.TestCase):
    def _compare(
        self, outcome_kwargs: dict | None = None, result_kwargs: dict | None = None
    ) -> ComparisonResult:
        ho = _make_historical_outcome(**(outcome_kwargs or {}))
        result = _make_episode_result(**(result_kwargs or {}))
        return OutcomeComparator(ho).compare(result)

    # winner_matches ---------------------------------------------------------

    def test_winner_matches_when_equal(self) -> None:
        cmp = self._compare({"winner": 0}, {"winner": 0})
        self.assertTrue(cmp.winner_matches)

    def test_winner_mismatches_when_different(self) -> None:
        cmp = self._compare({"winner": 0}, {"winner": 1})
        self.assertFalse(cmp.winner_matches)

    def test_both_none_winner_counts_as_match(self) -> None:
        cmp = self._compare({"winner": None}, {"winner": None})
        self.assertTrue(cmp.winner_matches)

    def test_none_vs_int_winner_is_mismatch(self) -> None:
        cmp = self._compare({"winner": None}, {"winner": 0})
        self.assertFalse(cmp.winner_matches)

    # casualty deltas --------------------------------------------------------

    def test_casualty_delta_blue_calculation(self) -> None:
        # historical blue_cas = 0.30, sim blue_strength = 0.65 → sim blue_cas = 0.35
        cmp = self._compare(
            {"blue_casualties": 0.30},
            {"blue_strength": 0.65},
        )
        self.assertAlmostEqual(cmp.simulated_blue_casualties, 0.35, places=5)
        self.assertAlmostEqual(cmp.casualty_delta_blue, 0.05, places=5)

    def test_casualty_delta_red_calculation(self) -> None:
        cmp = self._compare(
            {"red_casualties": 0.70},
            {"red_strength": 0.20},
        )
        self.assertAlmostEqual(cmp.simulated_red_casualties, 0.80, places=5)
        self.assertAlmostEqual(cmp.casualty_delta_red, 0.10, places=5)

    # duration delta ---------------------------------------------------------

    def test_duration_delta_positive(self) -> None:
        cmp = self._compare({"duration_steps": 300}, {"steps": 450})
        self.assertEqual(cmp.duration_delta, 150)

    def test_duration_delta_negative(self) -> None:
        cmp = self._compare({"duration_steps": 500}, {"steps": 300})
        self.assertEqual(cmp.duration_delta, -200)

    # fidelity score ---------------------------------------------------------

    def test_fidelity_score_in_range(self) -> None:
        cmp = self._compare()
        self.assertGreaterEqual(cmp.fidelity_score, 0.0)
        self.assertLessEqual(cmp.fidelity_score, 1.0)

    def test_perfect_fidelity_when_exact_match(self) -> None:
        # winner=0 matches, casualties match exactly, duration matches
        cmp = self._compare(
            {
                "winner": 0,
                "blue_casualties": 0.30,
                "red_casualties": 0.70,
                "duration_steps": 400,
            },
            {
                "winner": 0,
                "blue_strength": 0.70,   # → blue_cas = 0.30 ✓
                "red_strength": 0.30,    # → red_cas  = 0.70 ✓
                "steps": 400,
            },
        )
        self.assertAlmostEqual(cmp.fidelity_score, 1.0, places=5)

    def test_fidelity_higher_for_better_match(self) -> None:
        good = self._compare(
            {"winner": 0, "blue_casualties": 0.30, "red_casualties": 0.60, "duration_steps": 400},
            {"winner": 0, "blue_strength": 0.72, "red_strength": 0.41, "steps": 410},
        )
        bad = self._compare(
            {"winner": 0, "blue_casualties": 0.30, "red_casualties": 0.60, "duration_steps": 400},
            {"winner": 1, "blue_strength": 0.10, "red_strength": 0.90, "steps": 100},
        )
        self.assertGreater(good.fidelity_score, bad.fidelity_score)

    def test_winner_mismatch_lowers_fidelity(self) -> None:
        match_cmp = self._compare({"winner": 0}, {"winner": 0})
        mismatch_cmp = self._compare({"winner": 0}, {"winner": 1})
        self.assertGreater(match_cmp.fidelity_score, mismatch_cmp.fidelity_score)

    # passthrough fields -----------------------------------------------------

    def test_result_fields_exposed(self) -> None:
        cmp = self._compare({}, {"winner": 1, "steps": 250})
        self.assertEqual(cmp.simulated_winner, 1)
        self.assertEqual(cmp.simulated_steps, 250)

    def test_historical_outcome_exposed(self) -> None:
        ho = _make_historical_outcome(winner=0, duration_steps=300)
        result = _make_episode_result()
        cmp = OutcomeComparator(ho).compare(result)
        self.assertIs(cmp.historical_outcome, ho)


# ---------------------------------------------------------------------------
# End-to-end: load → run → compare for each bundled scenario
# ---------------------------------------------------------------------------


class TestEndToEndScenarios(unittest.TestCase):
    def _end_to_end(self, yaml_path: Path) -> ComparisonResult:
        scenario, blue, red, terrain = load_scenario(yaml_path)
        rng = np.random.default_rng(0)
        result = SimEngine(blue[0], red[0], terrain=terrain, rng=rng).run()
        return OutcomeComparator(scenario.historical_outcome).compare(result)

    def test_waterloo_end_to_end(self) -> None:
        cmp = self._end_to_end(WATERLOO_YAML)
        self.assertIsInstance(cmp, ComparisonResult)
        self.assertGreaterEqual(cmp.fidelity_score, 0.0)

    def test_austerlitz_end_to_end(self) -> None:
        cmp = self._end_to_end(AUSTERLITZ_YAML)
        self.assertIsInstance(cmp, ComparisonResult)

    def test_borodino_end_to_end(self) -> None:
        cmp = self._end_to_end(BORODINO_YAML)
        self.assertIsInstance(cmp, ComparisonResult)

    def test_reproducible_with_same_seed(self) -> None:
        """Two runs with the same seed should produce identical fidelity scores."""

        def run_once() -> float:
            scenario, blue, red, terrain = load_scenario(AUSTERLITZ_YAML)
            rng = np.random.default_rng(999)
            result = SimEngine(blue[0], red[0], terrain=terrain, rng=rng).run()
            return OutcomeComparator(scenario.historical_outcome).compare(result).fidelity_score

        self.assertAlmostEqual(run_once(), run_once(), places=10)


if __name__ == "__main__":
    unittest.main()
