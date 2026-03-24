"""Tests for WargamesBench — E12.2 Open Research Platform & Public Benchmark.

Coverage
--------
* BENCH_SCENARIOS — exactly 20 entries, unique names, required fields
* BenchScenario — from_dict() construction, field defaults
* BenchConfig — default field values, win_rate_tolerance
* BenchResult — field types and defaults
* BenchSummary — mean_win_rate, std_win_rate, total_episodes,
                  total_elapsed_seconds, is_reproducible, to_leaderboard_row
* BenchSummary.write_markdown — writes a file, contains key text
* WargamesBench — dry-run with scripted baseline (2 scenarios × 3 episodes)
* WargamesBench — custom callable policy
* WargamesBench — SB3-style policy with .predict()
* WargamesBench._SyntheticEnv — reset / step / close contract
* main() — CLI dry-run returns 0
* Public re-exports from benchmarks.__init__
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.wargames_bench import (
    BENCH_SCENARIOS,
    BenchScenario,
    BenchConfig,
    BenchResult,
    BenchSummary,
    WargamesBench,
    _SyntheticEnv,
    _scripted_action,
    _aggregate_episodes,
    main,
)
import benchmarks  # test public re-exports


# ---------------------------------------------------------------------------
# BENCH_SCENARIOS registry
# ---------------------------------------------------------------------------


class TestBenchScenariosRegistry(unittest.TestCase):

    def test_exactly_20_scenarios(self) -> None:
        self.assertEqual(len(BENCH_SCENARIOS), 20)

    def test_unique_names(self) -> None:
        names = [s["name"] for s in BENCH_SCENARIOS]
        self.assertEqual(len(names), len(set(names)), "Duplicate scenario names found")

    def test_required_fields_present(self) -> None:
        required = {"name", "n_blue", "n_red", "weather", "terrain_seed", "seed"}
        for entry in BENCH_SCENARIOS:
            missing = required - entry.keys()
            self.assertFalse(missing, f"Scenario {entry.get('name')} missing {missing}")

    def test_weather_values_valid(self) -> None:
        valid = {"clear", "rain", "fog", "snow"}
        for entry in BENCH_SCENARIOS:
            self.assertIn(entry["weather"], valid)

    def test_positive_unit_counts(self) -> None:
        for entry in BENCH_SCENARIOS:
            self.assertGreater(entry["n_blue"], 0)
            self.assertGreater(entry["n_red"], 0)

    def test_seeds_are_integers(self) -> None:
        for entry in BENCH_SCENARIOS:
            self.assertIsInstance(entry["seed"], int)
            self.assertIsInstance(entry["terrain_seed"], int)


# ---------------------------------------------------------------------------
# BenchScenario
# ---------------------------------------------------------------------------


class TestBenchScenario(unittest.TestCase):

    def _make(self, **kwargs) -> BenchScenario:
        defaults = BENCH_SCENARIOS[0].copy()
        defaults.update(kwargs)
        return BenchScenario.from_dict(defaults)

    def test_from_dict_name(self) -> None:
        s = self._make(name="test_scenario")
        self.assertEqual(s.name, "test_scenario")

    def test_from_dict_unit_counts(self) -> None:
        s = self._make(n_blue=5, n_red=7)
        self.assertEqual(s.n_blue, 5)
        self.assertEqual(s.n_red, 7)

    def test_from_dict_weather(self) -> None:
        s = self._make(weather="fog")
        self.assertEqual(s.weather, "fog")

    def test_from_dict_seeds(self) -> None:
        s = self._make(terrain_seed=12345, seed=99999)
        self.assertEqual(s.terrain_seed, 12345)
        self.assertEqual(s.seed, 99999)

    def test_default_n_hills_n_forests(self) -> None:
        d = {k: v for k, v in BENCH_SCENARIOS[0].items()}
        d.pop("n_hills", None)
        d.pop("n_forests", None)
        s = BenchScenario.from_dict(d)
        self.assertIsInstance(s.n_hills, int)
        self.assertIsInstance(s.n_forests, int)

    def test_map_dimensions_positive(self) -> None:
        s = BenchScenario.from_dict(BENCH_SCENARIOS[0])
        self.assertGreater(s.map_width, 0)
        self.assertGreater(s.map_height, 0)


# ---------------------------------------------------------------------------
# BenchConfig
# ---------------------------------------------------------------------------


class TestBenchConfig(unittest.TestCase):

    def test_default_n_eval_episodes(self) -> None:
        self.assertEqual(BenchConfig().n_eval_episodes, 100)

    def test_default_n_scenarios(self) -> None:
        self.assertEqual(BenchConfig().n_scenarios, 20)

    def test_default_max_steps(self) -> None:
        self.assertGreater(BenchConfig().max_steps_per_episode, 0)

    def test_default_win_rate_tolerance(self) -> None:
        self.assertAlmostEqual(BenchConfig().win_rate_tolerance, 0.02)

    def test_custom_label(self) -> None:
        cfg = BenchConfig(baseline_label="my_agent")
        self.assertEqual(cfg.baseline_label, "my_agent")

    def test_report_path_default_none(self) -> None:
        self.assertIsNone(BenchConfig().report_path)


# ---------------------------------------------------------------------------
# BenchResult
# ---------------------------------------------------------------------------


class TestBenchResult(unittest.TestCase):

    def _make_result(self, win_rate: float = 0.5) -> BenchResult:
        return BenchResult(
            scenario_name="test",
            policy_label="scripted",
            win_rate=win_rate,
            mean_steps=200.0,
            std_steps=50.0,
            n_episodes=10,
        )

    def test_win_rate_stored(self) -> None:
        r = self._make_result(win_rate=0.75)
        self.assertAlmostEqual(r.win_rate, 0.75)

    def test_elapsed_seconds_default_zero(self) -> None:
        r = self._make_result()
        self.assertEqual(r.elapsed_seconds, 0.0)


# ---------------------------------------------------------------------------
# BenchSummary
# ---------------------------------------------------------------------------


def _make_summary(
    win_rates: List[float],
    n_episodes: int = 10,
    tolerance: float = 0.02,
) -> BenchSummary:
    results = [
        BenchResult(
            scenario_name=f"s{i}",
            policy_label="test",
            win_rate=wr,
            mean_steps=100.0,
            std_steps=10.0,
            n_episodes=n_episodes,
            elapsed_seconds=1.0,
        )
        for i, wr in enumerate(win_rates)
    ]
    cfg = BenchConfig(
        n_eval_episodes=n_episodes,
        n_scenarios=len(win_rates),
        win_rate_tolerance=tolerance,
    )
    return BenchSummary(results=results, config=cfg)


class TestBenchSummary(unittest.TestCase):

    def test_mean_win_rate(self) -> None:
        s = _make_summary([0.4, 0.6])
        self.assertAlmostEqual(s.mean_win_rate, 0.5)

    def test_std_win_rate(self) -> None:
        s = _make_summary([0.4, 0.6])
        self.assertAlmostEqual(s.std_win_rate, 0.1)

    def test_total_episodes(self) -> None:
        s = _make_summary([0.5, 0.5, 0.5], n_episodes=20)
        self.assertEqual(s.total_episodes, 60)

    def test_total_elapsed_seconds(self) -> None:
        s = _make_summary([0.5, 0.5])
        self.assertAlmostEqual(s.total_elapsed_seconds, 2.0)

    def test_mean_win_rate_empty(self) -> None:
        s = BenchSummary(results=[], config=BenchConfig())
        self.assertEqual(s.mean_win_rate, 0.0)

    def test_std_win_rate_empty(self) -> None:
        s = BenchSummary(results=[], config=BenchConfig())
        self.assertEqual(s.std_win_rate, 0.0)

    def test_is_reproducible_same_results(self) -> None:
        s1 = _make_summary([0.5, 0.5, 0.5], tolerance=0.02)
        s2 = _make_summary([0.5, 0.5, 0.5], tolerance=0.02)
        self.assertTrue(s1.is_reproducible(s2))

    def test_is_reproducible_within_tolerance(self) -> None:
        s1 = _make_summary([0.50, 0.50], tolerance=0.02)
        s2 = _make_summary([0.51, 0.51], tolerance=0.02)
        self.assertTrue(s1.is_reproducible(s2))

    def test_is_reproducible_outside_tolerance(self) -> None:
        s1 = _make_summary([0.50, 0.50], tolerance=0.02)
        s2 = _make_summary([0.53, 0.50], tolerance=0.02)
        self.assertFalse(s1.is_reproducible(s2))

    def test_is_reproducible_different_length(self) -> None:
        s1 = _make_summary([0.5, 0.5], tolerance=0.02)
        s2 = _make_summary([0.5], tolerance=0.02)
        self.assertFalse(s1.is_reproducible(s2))

    def test_is_reproducible_different_names(self) -> None:
        s1 = _make_summary([0.5], tolerance=0.02)
        s2 = _make_summary([0.5], tolerance=0.02)
        s2.results[0] = BenchResult(
            scenario_name="other",
            policy_label="test",
            win_rate=0.5,
            mean_steps=100.0,
            std_steps=10.0,
            n_episodes=10,
        )
        self.assertFalse(s1.is_reproducible(s2))

    def test_to_leaderboard_row_keys(self) -> None:
        s = _make_summary([0.5, 0.6])
        row = s.to_leaderboard_row()
        for key in ("policy", "mean_win_rate", "std_win_rate", "n_scenarios", "total_episodes"):
            self.assertIn(key, row)

    def test_str_contains_mean_win_rate(self) -> None:
        s = _make_summary([0.5, 0.5])
        text = str(s)
        self.assertIn("win rate", text.lower())

    def test_write_markdown_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            s = _make_summary([0.5, 0.6])
            path = Path(tmpdir) / "test_leaderboard.md"
            out = s.write_markdown(path)
            self.assertTrue(out.exists())
            content = out.read_text()
            self.assertIn("WargamesBench", content)

    def test_write_markdown_contains_scenario_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            s = _make_summary([0.5, 0.7])
            path = Path(tmpdir) / "lb.md"
            s.write_markdown(path)
            content = path.read_text()
            for r in s.results:
                self.assertIn(r.scenario_name, content)


# ---------------------------------------------------------------------------
# _SyntheticEnv
# ---------------------------------------------------------------------------


class TestSyntheticEnv(unittest.TestCase):

    def test_reset_returns_obs_and_info(self) -> None:
        env = _SyntheticEnv(n_entities=4, seed=0, ep_length=5)
        obs, info = env.reset(seed=1)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)

    def test_obs_shape(self) -> None:
        env = _SyntheticEnv(n_entities=8, seed=0, ep_length=5)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 8)

    def test_step_returns_five_tuple(self) -> None:
        env = _SyntheticEnv(n_entities=4, seed=0, ep_length=5)
        env.reset(seed=0)
        result = env.step(np.zeros(3, dtype=np.float32))
        self.assertEqual(len(result), 5)

    def test_terminates_after_ep_length(self) -> None:
        env = _SyntheticEnv(n_entities=2, seed=0, ep_length=3)
        env.reset(seed=0)
        terminated = False
        for _ in range(3):
            _, _, terminated, _, _ = env.step(np.zeros(3))
        self.assertTrue(terminated)

    def test_close_is_safe(self) -> None:
        env = _SyntheticEnv()
        env.close()  # should not raise

    def test_reset_with_seed_deterministic(self) -> None:
        env1 = _SyntheticEnv(n_entities=4, seed=0)
        env2 = _SyntheticEnv(n_entities=4, seed=0)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_winner_info_possible(self) -> None:
        """At least one of red_routed / blue_routed should appear eventually."""
        env = _SyntheticEnv(n_entities=2, seed=7, ep_length=1)
        found_outcome = False
        for ep in range(30):
            env.reset(seed=ep)
            _, _, terminated, _, info = env.step(np.zeros(3))
            if "red_routed" in info or "blue_routed" in info:
                found_outcome = True
                break
        self.assertTrue(found_outcome)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers(unittest.TestCase):

    def test_scripted_action_shape(self) -> None:
        action = _scripted_action(np.zeros(10))
        self.assertEqual(action.shape, (3,))

    def test_scripted_action_dtype(self) -> None:
        self.assertEqual(_scripted_action(None).dtype, np.float32)

    def test_aggregate_empty(self) -> None:
        stats = _aggregate_episodes([])
        self.assertEqual(stats["win_rate"], 0.0)
        self.assertEqual(stats["n_episodes"], 0)

    def test_aggregate_all_wins(self) -> None:
        episodes = [{"won": True, "steps": 100}] * 10
        stats = _aggregate_episodes(episodes)
        self.assertAlmostEqual(stats["win_rate"], 1.0)

    def test_aggregate_all_losses(self) -> None:
        episodes = [{"won": False, "steps": 200}] * 5
        stats = _aggregate_episodes(episodes)
        self.assertAlmostEqual(stats["win_rate"], 0.0)

    def test_aggregate_no_outcome(self) -> None:
        episodes = [{"won": None, "steps": 500}] * 5
        stats = _aggregate_episodes(episodes)
        self.assertEqual(stats["win_rate"], 0.0)

    def test_aggregate_mean_steps(self) -> None:
        episodes = [{"won": True, "steps": s} for s in [100, 200, 300]]
        stats = _aggregate_episodes(episodes)
        self.assertAlmostEqual(stats["mean_steps"], 200.0)


# ---------------------------------------------------------------------------
# WargamesBench runner
# ---------------------------------------------------------------------------


class TestWargamesBench(unittest.TestCase):

    def _make_bench(self, n_scenarios: int = 2, n_episodes: int = 3) -> WargamesBench:
        cfg = BenchConfig(
            n_eval_episodes=n_episodes,
            n_scenarios=n_scenarios,
            max_steps_per_episode=10,
        )
        return WargamesBench(cfg)

    def test_run_returns_summary(self) -> None:
        bench = self._make_bench()
        summary = bench.run(policy=None)
        self.assertIsInstance(summary, BenchSummary)

    def test_run_correct_number_of_results(self) -> None:
        bench = self._make_bench(n_scenarios=3)
        summary = bench.run()
        self.assertEqual(len(summary.results), 3)

    def test_run_n_episodes_per_scenario(self) -> None:
        bench = self._make_bench(n_scenarios=2, n_episodes=5)
        summary = bench.run()
        for r in summary.results:
            self.assertEqual(r.n_episodes, 5)

    def test_run_result_names_match_registry(self) -> None:
        bench = self._make_bench(n_scenarios=4)
        summary = bench.run()
        registry_names = [s["name"] for s in BENCH_SCENARIOS[:4]]
        result_names = [r.scenario_name for r in summary.results]
        self.assertEqual(result_names, registry_names)

    def test_run_win_rate_in_range(self) -> None:
        bench = self._make_bench(n_scenarios=2, n_episodes=10)
        summary = bench.run()
        for r in summary.results:
            self.assertGreaterEqual(r.win_rate, 0.0)
            self.assertLessEqual(r.win_rate, 1.0)

    def test_run_with_callable_policy(self) -> None:
        def my_policy(obs):
            return np.zeros(3, dtype=np.float32)

        bench = self._make_bench(n_scenarios=1, n_episodes=3)
        summary = bench.run(policy=my_policy)
        self.assertEqual(len(summary.results), 1)

    def test_run_with_sb3_style_policy(self) -> None:
        mock_policy = MagicMock()
        mock_policy.predict.return_value = (np.zeros(3, dtype=np.float32), None)

        bench = self._make_bench(n_scenarios=1, n_episodes=3)
        summary = bench.run(policy=mock_policy)
        self.assertEqual(len(summary.results), 1)

    def test_run_with_custom_label(self) -> None:
        bench = self._make_bench(n_scenarios=1, n_episodes=2)
        summary = bench.run(label="custom_agent")
        self.assertEqual(summary.results[0].policy_label, "custom_agent")

    def test_run_label_propagates_to_summary_config(self) -> None:
        """Label kwarg must be reflected in BenchSummary.config.baseline_label."""
        bench = self._make_bench(n_scenarios=1, n_episodes=2)
        summary = bench.run(label="override_label")
        self.assertEqual(summary.config.baseline_label, "override_label")

    def test_run_label_propagates_to_leaderboard_row(self) -> None:
        """to_leaderboard_row() and __str__() must use the overridden label."""
        bench = self._make_bench(n_scenarios=1, n_episodes=2)
        summary = bench.run(label="my_policy")
        self.assertEqual(summary.to_leaderboard_row()["policy"], "my_policy")

    def test_config_label_used_when_no_label_kwarg(self) -> None:
        cfg = BenchConfig(
            n_eval_episodes=2,
            n_scenarios=1,
            max_steps_per_episode=5,
            baseline_label="from_config",
        )
        bench = WargamesBench(cfg)
        summary = bench.run()
        self.assertEqual(summary.results[0].policy_label, "from_config")

    def test_default_config_created_when_none(self) -> None:
        bench = WargamesBench(None)
        self.assertIsInstance(bench.config, BenchConfig)

    def test_reproducibility_same_seed(self) -> None:
        """Two runs with the same seed should agree within tolerance."""
        bench = self._make_bench(n_scenarios=2, n_episodes=20)
        summary1 = bench.run()
        summary2 = bench.run()
        self.assertTrue(summary1.is_reproducible(summary2))

    def test_make_env_returns_env(self) -> None:
        bench = WargamesBench(BenchConfig())
        scenario = BenchScenario.from_dict(BENCH_SCENARIOS[0])
        env = bench._make_env(scenario)
        self.assertTrue(hasattr(env, "reset"))
        self.assertTrue(hasattr(env, "step"))
        self.assertTrue(hasattr(env, "close"))
        env.close()


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


class TestMain(unittest.TestCase):

    def test_main_dry_run(self) -> None:
        ret = main(["--episodes", "2", "--scenarios", "2"])
        self.assertEqual(ret, 0)

    def test_main_with_label(self) -> None:
        ret = main(["--episodes", "2", "--scenarios", "1", "--label", "test_baseline"])
        self.assertEqual(ret, 0)

    def test_main_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.md"
            ret = main(["--episodes", "2", "--scenarios", "1", "--output", str(out)])
            self.assertEqual(ret, 0)
            self.assertTrue(out.exists())


# ---------------------------------------------------------------------------
# Public re-exports from benchmarks.__init__
# ---------------------------------------------------------------------------


class TestPublicReexports(unittest.TestCase):

    def test_bench_scenarios_exported(self) -> None:
        self.assertIs(benchmarks.BENCH_SCENARIOS, BENCH_SCENARIOS)

    def test_classes_exported(self) -> None:
        for name in ("BenchScenario", "BenchConfig", "BenchResult", "BenchSummary", "WargamesBench"):
            self.assertTrue(hasattr(benchmarks, name), f"benchmarks.{name} not found")

    def test_main_exported(self) -> None:
        self.assertTrue(callable(benchmarks.main))


if __name__ == "__main__":
    unittest.main()
