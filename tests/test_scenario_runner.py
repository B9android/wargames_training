import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scenario_runner as sr
from envs.sim.battalion import Battalion


class BuildScenarioTests(unittest.TestCase):
    def test_battalions_are_within_fire_range(self) -> None:
        blue, red = sr.build_scenario()
        self.assertTrue(blue.can_fire_at(red), "Blue should be able to fire at red initially")
        self.assertTrue(red.can_fire_at(blue), "Red should be able to fire at blue initially")

    def test_battalions_start_at_full_strength(self) -> None:
        blue, red = sr.build_scenario()
        self.assertAlmostEqual(blue.strength, 1.0)
        self.assertAlmostEqual(red.strength, 1.0)

    def test_returns_two_battalions_on_opposite_teams(self) -> None:
        blue, red = sr.build_scenario()
        self.assertEqual(blue.team, 0)
        self.assertEqual(red.team, 1)


class StepTests(unittest.TestCase):
    def test_step_deals_positive_damage(self) -> None:
        blue, red = sr.build_scenario()
        rng = np.random.default_rng(0)
        damage = sr.step(blue, red, rng=rng)
        self.assertGreater(damage, 0.0)

    def test_step_accepts_none_rng(self) -> None:
        blue, red = sr.build_scenario()
        damage = sr.step(blue, red, rng=None)
        self.assertGreater(damage, 0.0)

    def test_step_reduces_both_strengths(self) -> None:
        blue, red = sr.build_scenario()
        sr.step(blue, red)
        self.assertLess(blue.strength, 1.0)
        self.assertLess(red.strength, 1.0)

    def test_step_clamps_strength_to_zero(self) -> None:
        blue = Battalion(x=300.0, y=500.0, theta=0.0, strength=0.0001, team=0)
        red = Battalion(x=450.0, y=500.0, theta=3.14159, strength=1.0, team=1)
        sr.step(blue, red)
        self.assertGreaterEqual(blue.strength, 0.0)


class RunScenarioTests(unittest.TestCase):
    def test_returns_expected_keys(self) -> None:
        results = sr.run_scenario(steps=10, seed=0)
        for key in (
            "steps",
            "seed",
            "blue_strength_initial",
            "blue_strength_final",
            "red_strength_initial",
            "red_strength_final",
            "total_damage",
            "blue",
            "red",
        ):
            self.assertIn(key, results, f"Missing key: {key}")

    def test_steps_and_seed_are_recorded(self) -> None:
        results = sr.run_scenario(steps=5, seed=99)
        self.assertEqual(results["steps"], 5)
        self.assertEqual(results["seed"], 99)

    def test_total_damage_is_greater_than_zero(self) -> None:
        results = sr.run_scenario(steps=50, seed=42)
        self.assertGreater(results["total_damage"], 0.0)

    def test_final_strengths_are_in_valid_range(self) -> None:
        results = sr.run_scenario(steps=200, seed=42)
        self.assertGreaterEqual(results["blue_strength_final"], 0.0)
        self.assertLessEqual(results["blue_strength_final"], 1.0)
        self.assertGreaterEqual(results["red_strength_final"], 0.0)
        self.assertLessEqual(results["red_strength_final"], 1.0)

    def test_both_sides_take_damage(self) -> None:
        results = sr.run_scenario(steps=200, seed=42)
        self.assertLess(results["blue_strength_final"], results["blue_strength_initial"])
        self.assertLess(results["red_strength_final"], results["red_strength_initial"])

    def test_same_seed_produces_identical_results(self) -> None:
        r1 = sr.run_scenario(steps=50, seed=7)
        r2 = sr.run_scenario(steps=50, seed=7)
        self.assertAlmostEqual(r1["total_damage"], r2["total_damage"])
        self.assertAlmostEqual(r1["blue_strength_final"], r2["blue_strength_final"])
        self.assertAlmostEqual(r1["red_strength_final"], r2["red_strength_final"])


class CheckResultsTests(unittest.TestCase):
    def _good_results(self) -> dict:
        blue, red = sr.build_scenario()
        # Run a single step so strengths drop
        sr.step(blue, red)
        return {
            "steps": 1,
            "seed": 42,
            "blue_strength_initial": 1.0,
            "blue_strength_final": blue.strength,
            "red_strength_initial": 1.0,
            "red_strength_final": red.strength,
            "total_damage": 0.1,
            "blue": blue,
            "red": red,
        }

    def test_no_failures_for_valid_scenario(self) -> None:
        results = sr.run_scenario(steps=200, seed=42)
        failures = sr.check_results(results)
        self.assertEqual(failures, [], f"Unexpected failures: {failures}")

    def test_detects_out_of_range_strength(self) -> None:
        results = self._good_results()
        results["blue_strength_final"] = -0.1
        failures = sr.check_results(results)
        self.assertTrue(any("Blue strength" in f for f in failures))

    def test_detects_no_damage_to_blue(self) -> None:
        results = self._good_results()
        results["blue_strength_final"] = results["blue_strength_initial"]
        failures = sr.check_results(results)
        self.assertTrue(any("Blue took no damage" in f for f in failures))

    def test_detects_no_damage_to_red(self) -> None:
        results = self._good_results()
        results["red_strength_final"] = results["red_strength_initial"]
        failures = sr.check_results(results)
        self.assertTrue(any("Red took no damage" in f for f in failures))

    def test_detects_zero_total_damage(self) -> None:
        results = self._good_results()
        results["total_damage"] = 0.0
        failures = sr.check_results(results)
        self.assertTrue(any("No damage" in f for f in failures))

    def test_detects_non_finite_position(self) -> None:
        results = self._good_results()
        results["blue"].x = float("nan")
        failures = sr.check_results(results)
        self.assertTrue(any("non-finite" in f for f in failures))


class MainTests(unittest.TestCase):
    def test_main_returns_zero_on_success(self) -> None:
        exit_code = sr.main(["--steps", "100", "--seed", "42"])
        self.assertEqual(exit_code, 0)

    def test_main_accepts_custom_steps_and_seed(self) -> None:
        exit_code = sr.main(["--steps", "50", "--seed", "7"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
