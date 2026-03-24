"""Tests for E11.1 — Historical Battle Database & Scenario Importer.

Covers:
* :class:`~envs.scenarios.importer.BatchScenarioImporter` (JSON and CSV)
* :class:`~envs.scenarios.importer.BattleRecord`
* :func:`~envs.scenarios.importer._record_to_scenario` via ``to_scenario``
* :class:`~training.historical_benchmark.HistoricalBenchmark`
* :class:`~training.historical_benchmark.BenchmarkSummary`
* Acceptance criterion: importer handles ≥ 50 engagements without errors
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.scenarios.historical import (
    HistoricalScenario,
    OutcomeComparator,
    ScenarioUnit,
)
from envs.scenarios.importer import (
    BatchScenarioImporter,
    BattleRecord,
    _parse_json_entry,
    _parse_csv_row,
    _record_to_scenario,
)
from envs.sim.battalion import Battalion
from envs.sim.engine import EpisodeResult, SimEngine
from envs.sim.terrain import TerrainMap
from training.historical_benchmark import (
    BenchmarkEntry,
    BenchmarkSummary,
    HistoricalBenchmark,
    _render_markdown,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BATTLES_JSON = PROJECT_ROOT / "data" / "historical" / "battles.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_record(**kwargs) -> BattleRecord:
    defaults = dict(
        battle_id="test_1800",
        name="Test Battle (1800)",
        date="1800-01-01",
        location="Testland",
        description="A test battle.",
        source="Test Source",
        factions={"blue": "Blue Force", "red": "Red Force"},
        terrain={"type": "flat", "width": 1000.0, "height": 1000.0, "rows": 10, "cols": 10, "seed": 0, "n_hills": 1, "n_forests": 1},
        blue_units=[
            {"id": "blue_0", "x": 300.0, "y": 500.0, "theta": 0.0, "strength": 1.0}
        ],
        red_units=[
            {"id": "red_0", "x": 700.0, "y": 500.0, "theta": 3.1416, "strength": 1.0}
        ],
        historical_outcome={"winner": 0, "blue_casualties": 0.20, "red_casualties": 0.40, "duration_steps": 400, "description": "Blue won."},
    )
    defaults.update(kwargs)
    return BattleRecord(**defaults)


# ---------------------------------------------------------------------------
# BattleRecord
# ---------------------------------------------------------------------------


class TestBattleRecord(unittest.TestCase):
    def test_to_scenario_returns_historical_scenario(self) -> None:
        rec = _minimal_record()
        scenario = rec.to_scenario()
        self.assertIsInstance(scenario, HistoricalScenario)

    def test_to_scenario_name(self) -> None:
        rec = _minimal_record(name="Battle of X")
        self.assertEqual(rec.to_scenario().name, "Battle of X")

    def test_to_scenario_date(self) -> None:
        rec = _minimal_record(date="1800-01-01")
        self.assertEqual(rec.to_scenario().date, "1800-01-01")

    def test_to_scenario_winner(self) -> None:
        for winner in (0, 1, None):
            with self.subTest(winner=winner):
                rec = _minimal_record(historical_outcome={"winner": winner, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300})
                self.assertEqual(rec.to_scenario().historical_outcome.winner, winner)

    def test_to_scenario_blue_units_count(self) -> None:
        rec = _minimal_record()
        self.assertEqual(rec.to_scenario().n_blue, 1)

    def test_to_scenario_red_units_count(self) -> None:
        rec = _minimal_record()
        self.assertEqual(rec.to_scenario().n_red, 1)

    def test_to_scenario_flat_terrain(self) -> None:
        rec = _minimal_record(terrain={"type": "flat"})
        terrain = rec.to_scenario().build_terrain()
        self.assertIsInstance(terrain, TerrainMap)

    def test_to_scenario_generated_terrain(self) -> None:
        rec = _minimal_record(terrain={"type": "generated", "seed": 42, "n_hills": 2, "n_forests": 1})
        terrain = rec.to_scenario().build_terrain()
        self.assertIsInstance(terrain, TerrainMap)

    def test_to_scenario_faction_blue(self) -> None:
        rec = _minimal_record(factions={"blue": "Grande Armee", "red": "Coalition"})
        self.assertEqual(rec.to_scenario().faction_blue, "Grande Armee")

    def test_to_scenario_faction_red(self) -> None:
        rec = _minimal_record(factions={"blue": "Grande Armee", "red": "Coalition"})
        self.assertEqual(rec.to_scenario().faction_red, "Coalition")

    def test_to_scenario_build_battalions(self) -> None:
        rec = _minimal_record()
        blue, red = rec.to_scenario().build_battalions()
        for b in blue + red:
            self.assertIsInstance(b, Battalion)

    def test_named_theta_in_unit(self) -> None:
        for direction, expected in [("east", 0.0), ("west", math.pi), ("north", math.pi / 2), ("south", -math.pi / 2)]:
            with self.subTest(direction=direction):
                rec = _minimal_record(
                    blue_units=[{"id": "u", "x": 500.0, "y": 500.0, "theta": direction, "strength": 1.0}]
                )
                unit = rec.to_scenario().blue_units[0]
                self.assertAlmostEqual(unit.theta, expected, places=4)

    def test_unknown_theta_raises_value_error(self) -> None:
        rec = _minimal_record(
            blue_units=[{"id": "u", "x": 500.0, "y": 500.0, "theta": "northeast", "strength": 1.0}]
        )
        with self.assertRaises(ValueError):
            rec.to_scenario()

    def test_invalid_winner_value_raises_value_error(self) -> None:
        rec = _minimal_record(historical_outcome={"winner": 2, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300})
        with self.assertRaises(ValueError):
            rec.to_scenario()

    def test_negative_winner_value_raises_value_error(self) -> None:
        rec = _minimal_record(historical_outcome={"winner": -1, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300})
        with self.assertRaises(ValueError):
            rec.to_scenario()

    def test_string_winner_non_numeric_raises_value_error(self) -> None:
        rec = _minimal_record(historical_outcome={"winner": "red", "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300})
        with self.assertRaises(ValueError):
            rec.to_scenario()


# ---------------------------------------------------------------------------
# BatchScenarioImporter — JSON
# ---------------------------------------------------------------------------


class TestBatchScenarioImporterJSON(unittest.TestCase):
    def _make_json_file(self, records: list) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8")
        json.dump(records, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_load_records_single_entry(self) -> None:
        raw = [{"id": "b1", "name": "Battle 1", "date": "1800-01-01",
                "historical_outcome": {"winner": 0, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300}}]
        path = self._make_json_file(raw)
        try:
            imp = BatchScenarioImporter(path)
            records = imp.load_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].battle_id, "b1")
        finally:
            path.unlink(missing_ok=True)

    def test_load_all_returns_scenarios(self) -> None:
        raw = [{"id": "b1", "name": "Battle 1", "date": "1800-01-01",
                "units": {"blue": [{"id": "u", "x": 300.0, "y": 500.0, "theta": 0.0, "strength": 1.0}],
                          "red":  [{"id": "v", "x": 700.0, "y": 500.0, "theta": 3.1416, "strength": 1.0}]},
                "historical_outcome": {"winner": 0, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300}}]
        path = self._make_json_file(raw)
        try:
            imp = BatchScenarioImporter(path)
            scenarios = imp.load_all()
            self.assertEqual(len(scenarios), 1)
            self.assertIsInstance(scenarios[0], HistoricalScenario)
        finally:
            path.unlink(missing_ok=True)

    def test_load_by_source(self) -> None:
        raw = [
            {"id": "b1", "name": "B1", "date": "1800-01-01", "source": "Source A",
             "historical_outcome": {"winner": 0, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300},
             "units": {"blue": [{"id": "u", "x": 300.0, "y": 500.0, "theta": 0.0, "strength": 1.0}],
                       "red":  [{"id": "v", "x": 700.0, "y": 500.0, "theta": 0.0, "strength": 1.0}]}},
            {"id": "b2", "name": "B2", "date": "1800-06-01", "source": "Source B",
             "historical_outcome": {"winner": 1, "blue_casualties": 0.3, "red_casualties": 0.1, "duration_steps": 200},
             "units": {"blue": [{"id": "u", "x": 300.0, "y": 500.0, "theta": 0.0, "strength": 1.0}],
                       "red":  [{"id": "v", "x": 700.0, "y": 500.0, "theta": 0.0, "strength": 1.0}]}},
        ]
        path = self._make_json_file(raw)
        try:
            imp = BatchScenarioImporter(path)
            source_a = imp.load_by_source("Source A")
            self.assertEqual(len(source_a), 1)
            self.assertEqual(source_a[0].name, "B1")
        finally:
            path.unlink(missing_ok=True)

    def test_load_by_id_found(self) -> None:
        raw = [{"id": "target", "name": "Target Battle", "date": "1800-01-01",
                "historical_outcome": {"winner": 0, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300},
                "units": {"blue": [{"id": "u", "x": 300.0, "y": 500.0, "theta": 0.0, "strength": 1.0}],
                          "red":  [{"id": "v", "x": 700.0, "y": 500.0, "theta": 0.0, "strength": 1.0}]}}]
        path = self._make_json_file(raw)
        try:
            imp = BatchScenarioImporter(path)
            scenario = imp.load_by_id("target")
            self.assertIsNotNone(scenario)
            self.assertEqual(scenario.name, "Target Battle")
        finally:
            path.unlink(missing_ok=True)

    def test_load_by_id_not_found(self) -> None:
        raw = [{"id": "b1", "name": "B1", "date": "1800-01-01",
                "historical_outcome": {"winner": 0, "blue_casualties": 0.1, "red_casualties": 0.2, "duration_steps": 300},
                "units": {"blue": [{"id": "u", "x": 300.0, "y": 500.0, "theta": 0.0, "strength": 1.0}],
                          "red":  [{"id": "v", "x": 700.0, "y": 500.0, "theta": 0.0, "strength": 1.0}]}}]
        path = self._make_json_file(raw)
        try:
            imp = BatchScenarioImporter(path)
            result = imp.load_by_id("nonexistent")
            self.assertIsNone(result)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_file_raises_file_not_found(self) -> None:
        imp = BatchScenarioImporter("/nonexistent/battles.json")
        with self.assertRaises(FileNotFoundError):
            imp.load_records()

    def test_non_list_json_raises_value_error(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"key": "value"}, f)
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                BatchScenarioImporter(path).load_records()
        finally:
            path.unlink(missing_ok=True)

    def test_unsupported_extension_raises_value_error(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        try:
            imp = BatchScenarioImporter(path)
            # load_records raises ValueError for unknown extension
            with self.assertRaises((ValueError, FileNotFoundError)):
                imp.load_records()
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# BatchScenarioImporter — CSV
# ---------------------------------------------------------------------------


class TestBatchScenarioImporterCSV(unittest.TestCase):
    def _make_csv_file(self, rows: List[dict]) -> Path:
        if not rows:
            raise ValueError("rows must not be empty")
        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False, encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            path = Path(f.name)
        return path

    def test_load_records_csv(self) -> None:
        rows = [{"id": "c1", "name": "CSV Battle", "date": "1800-01-01",
                 "winner": "0", "blue_casualties": "0.1", "red_casualties": "0.2",
                 "duration_steps": "300", "source": "Test"}]
        path = self._make_csv_file(rows)
        try:
            imp = BatchScenarioImporter(path)
            records = imp.load_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].name, "CSV Battle")
        finally:
            path.unlink(missing_ok=True)

    def test_csv_null_winner(self) -> None:
        rows = [{"id": "c1", "name": "Draw Battle", "date": "1800-01-01",
                 "winner": "null", "blue_casualties": "0.3", "red_casualties": "0.3",
                 "duration_steps": "400"}]
        path = self._make_csv_file(rows)
        try:
            imp = BatchScenarioImporter(path)
            scenarios = imp.load_all()
            self.assertIsNone(scenarios[0].historical_outcome.winner)
        finally:
            path.unlink(missing_ok=True)

    def test_csv_draw_winner_string(self) -> None:
        rows = [{"id": "c1", "name": "Draw Battle", "date": "1800-01-01",
                 "winner": "draw", "blue_casualties": "0.3", "red_casualties": "0.3",
                 "duration_steps": "400"}]
        path = self._make_csv_file(rows)
        try:
            imp = BatchScenarioImporter(path)
            scenarios = imp.load_all()
            self.assertIsNone(scenarios[0].historical_outcome.winner)
        finally:
            path.unlink(missing_ok=True)

    def test_csv_default_units_created(self) -> None:
        rows = [{"id": "c1", "name": "Minimal Battle", "date": "1800-01-01",
                 "winner": "0", "blue_casualties": "0.1", "red_casualties": "0.2",
                 "duration_steps": "300"}]
        path = self._make_csv_file(rows)
        try:
            imp = BatchScenarioImporter(path)
            scenario = imp.load_all()[0]
            self.assertGreater(scenario.n_blue, 0)
            self.assertGreater(scenario.n_red, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_csv_multiple_rows(self) -> None:
        rows = [
            {"id": "c1", "name": "Battle A", "date": "1800-01-01", "winner": "0",
             "blue_casualties": "0.1", "red_casualties": "0.2", "duration_steps": "300"},
            {"id": "c2", "name": "Battle B", "date": "1800-06-01", "winner": "1",
             "blue_casualties": "0.3", "red_casualties": "0.1", "duration_steps": "200"},
            {"id": "c3", "name": "Battle C", "date": "1800-12-01", "winner": "null",
             "blue_casualties": "0.2", "red_casualties": "0.2", "duration_steps": "400"},
        ]
        path = self._make_csv_file(rows)
        try:
            imp = BatchScenarioImporter(path)
            scenarios = imp.load_all()
            self.assertEqual(len(scenarios), 3)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Full battles.json database
# ---------------------------------------------------------------------------


class TestBattlesDatabase(unittest.TestCase):
    """Verify the bundled battles.json meets acceptance criteria."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.importer = BatchScenarioImporter(BATTLES_JSON)
        cls.records = cls.importer.load_records()
        cls.scenarios = cls.importer.load_all()

    def test_database_has_at_least_50_records(self) -> None:
        self.assertGreaterEqual(len(self.records), 50)

    def test_all_records_load_without_error(self) -> None:
        # If setUpClass succeeded this already holds, but explicit assertion is clear.
        self.assertEqual(len(self.scenarios), len(self.records))

    def test_all_scenarios_have_blue_units(self) -> None:
        for s in self.scenarios:
            with self.subTest(name=s.name):
                self.assertGreater(s.n_blue, 0)

    def test_all_scenarios_have_red_units(self) -> None:
        for s in self.scenarios:
            with self.subTest(name=s.name):
                self.assertGreater(s.n_red, 0)

    def test_all_unit_strengths_in_range(self) -> None:
        for s in self.scenarios:
            for u in s.blue_units + s.red_units:
                with self.subTest(scenario=s.name, unit=u.unit_id):
                    self.assertGreaterEqual(u.strength, 0.0)
                    self.assertLessEqual(u.strength, 1.0)

    def test_all_casualties_in_range(self) -> None:
        for s in self.scenarios:
            ho = s.historical_outcome
            with self.subTest(name=s.name):
                self.assertGreaterEqual(ho.blue_casualties, 0.0)
                self.assertLessEqual(ho.blue_casualties, 1.0)
                self.assertGreaterEqual(ho.red_casualties, 0.0)
                self.assertLessEqual(ho.red_casualties, 1.0)

    def test_winner_values_are_valid(self) -> None:
        for s in self.scenarios:
            with self.subTest(name=s.name):
                winner = s.historical_outcome.winner
                self.assertIn(winner, (0, 1, None))

    def test_dates_are_nonempty_strings(self) -> None:
        for s in self.scenarios:
            with self.subTest(name=s.name):
                self.assertIsInstance(s.date, str)
                self.assertTrue(s.date)

    def test_build_terrain_succeeds_for_all(self) -> None:
        for s in self.scenarios:
            with self.subTest(name=s.name):
                terrain = s.build_terrain()
                self.assertIsInstance(terrain, TerrainMap)

    def test_build_battalions_succeeds_for_all(self) -> None:
        for s in self.scenarios:
            with self.subTest(name=s.name):
                blue, red = s.build_battalions()
                for b in blue + red:
                    self.assertIsInstance(b, Battalion)

    def test_unique_battle_ids(self) -> None:
        ids = [r.battle_id for r in self.records]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate battle IDs found")

    def test_load_by_source_napoleon_battles(self) -> None:
        napoleon = self.importer.load_by_source("Napoleon's Battles")
        self.assertGreater(len(napoleon), 0)

    def test_load_by_source_corsican_ogre(self) -> None:
        corsican = self.importer.load_by_source("Corsican Ogre")
        self.assertGreater(len(corsican), 0)

    def test_load_by_source_nafziger(self) -> None:
        nafziger = self.importer.load_by_source("Nafziger OOBs")
        self.assertGreater(len(nafziger), 0)

    def test_load_waterloo_by_id(self) -> None:
        scenario = self.importer.load_by_id("waterloo_1815")
        self.assertIsNotNone(scenario)
        self.assertIn("Waterloo", scenario.name)

    def test_load_austerlitz_by_id(self) -> None:
        scenario = self.importer.load_by_id("austerlitz_1805")
        self.assertIsNotNone(scenario)
        self.assertIn("Austerlitz", scenario.name)


# ---------------------------------------------------------------------------
# Simulation: all 50+ scenarios run to completion without errors
# ---------------------------------------------------------------------------


class TestAllScenariosRunnable(unittest.TestCase):
    """Acceptance criterion: all 50+ scenarios complete without exceptions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.importer = BatchScenarioImporter(BATTLES_JSON)
        cls.scenarios = cls.importer.load_all()
        cls.records = cls.importer.load_records()

    def _run_1v1(self, scenario: HistoricalScenario) -> EpisodeResult:
        blue, red = scenario.build_battalions()
        terrain = scenario.build_terrain()
        rng = np.random.default_rng(42)
        return SimEngine(blue[0], red[0], terrain=terrain, rng=rng).run()

    def test_all_scenarios_run_without_exception(self) -> None:
        """Acceptance criterion: importer handles ≥ 50 engagements without errors."""
        errors = []
        for rec, scenario in zip(self.records, self.scenarios):
            try:
                result = self._run_1v1(scenario)
                self.assertGreater(result.steps, 0)
                self.assertGreaterEqual(result.blue_strength, 0.0)
                self.assertLessEqual(result.blue_strength, 1.0)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{rec.battle_id}: {exc}")

        if errors:
            self.fail(
                f"Scenarios that failed to run:\n" + "\n".join(errors)
            )

    def test_at_least_50_scenarios_ran(self) -> None:
        self.assertGreaterEqual(len(self.scenarios), 50)

    def test_outcome_comparisons_are_valid(self) -> None:
        """All comparison results should be finite and in-range."""
        for scenario in self.scenarios[:10]:  # spot-check first 10
            with self.subTest(name=scenario.name):
                result = self._run_1v1(scenario)
                cmp = OutcomeComparator(scenario.historical_outcome).compare(result)
                self.assertGreaterEqual(cmp.fidelity_score, 0.0)
                self.assertLessEqual(cmp.fidelity_score, 1.0)
                self.assertTrue(math.isfinite(cmp.fidelity_score))


# ---------------------------------------------------------------------------
# HistoricalBenchmark
# ---------------------------------------------------------------------------


class TestHistoricalBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bench = HistoricalBenchmark(battles_path=BATTLES_JSON, seed=0)
        cls.summary = bench.run()

    def test_total_matches_database_size(self) -> None:
        self.assertEqual(self.summary.total, len(BatchScenarioImporter(BATTLES_JSON).load_records()))

    def test_all_passed(self) -> None:
        self.assertEqual(self.summary.failed, 0, msg="Some scenarios failed in the benchmark")

    def test_importer_criterion(self) -> None:
        """Acceptance: importer handles ≥ 50 without errors."""
        self.assertTrue(
            self.summary.meets_importer_criterion,
            msg=f"Importer criterion failed: total={self.summary.total}, passed={self.summary.passed}",
        )

    def test_mean_fidelity_in_range(self) -> None:
        self.assertGreaterEqual(self.summary.mean_fidelity, 0.0)
        self.assertLessEqual(self.summary.mean_fidelity, 1.0)

    def test_winner_match_rate_in_range(self) -> None:
        self.assertGreaterEqual(self.summary.winner_match_rate, 0.0)
        self.assertLessEqual(self.summary.winner_match_rate, 1.0)

    def test_elapsed_under_two_hours(self) -> None:
        self.assertLess(self.summary.total_elapsed_seconds, 7200)

    def test_entries_count(self) -> None:
        self.assertEqual(len(self.summary.entries), self.summary.total)

    def test_entry_fields(self) -> None:
        for entry in self.summary.entries:
            with self.subTest(battle_id=entry.battle_id):
                self.assertTrue(entry.passed)
                self.assertIsNotNone(entry.comparison)
                self.assertGreaterEqual(entry.elapsed_seconds, 0.0)

    def test_write_markdown_produces_file(self) -> None:
        import tempfile
        bench = HistoricalBenchmark(battles_path=BATTLES_JSON, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = bench.write_markdown(self.summary, output_path=Path(tmpdir) / "benchmark.md")
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
            self.assertIn("Historical Benchmark Results", content)
            self.assertIn("Per-Scenario Results", content)
            self.assertIn("Waterloo", content)

    def test_markdown_contains_summary_table(self) -> None:
        md = _render_markdown(self.summary)
        self.assertIn("Total scenarios", md)
        self.assertIn("Winner match rate", md)
        self.assertIn("Importer criterion", md)


# ---------------------------------------------------------------------------
# BenchmarkEntry
# ---------------------------------------------------------------------------


class TestBenchmarkEntry(unittest.TestCase):
    def _make_entry(self, **kwargs) -> BenchmarkEntry:
        defaults = dict(
            battle_id="test",
            scenario_name="Test",
            date="1800-01-01",
            source="Test Source",
            historical_winner=0,
        )
        defaults.update(kwargs)
        return BenchmarkEntry(**defaults)

    def test_passed_true_when_no_error(self) -> None:
        entry = self._make_entry()
        self.assertTrue(entry.passed)

    def test_passed_false_when_error(self) -> None:
        entry = self._make_entry(error="something went wrong")
        self.assertFalse(entry.passed)

    def test_fidelity_score_zero_on_error(self) -> None:
        entry = self._make_entry(error="fail")
        self.assertEqual(entry.fidelity_score, 0.0)

    def test_winner_matches_false_when_no_comparison(self) -> None:
        entry = self._make_entry()
        self.assertFalse(entry.winner_matches)


# ---------------------------------------------------------------------------
# BenchmarkSummary
# ---------------------------------------------------------------------------


class TestBenchmarkSummary(unittest.TestCase):
    def _make_summary(self, **kwargs) -> BenchmarkSummary:
        defaults = dict(
            total=50,
            passed=50,
            winner_match_rate=0.65,
            mean_fidelity=0.5,
            std_fidelity=0.1,
            total_elapsed_seconds=120.0,
        )
        defaults.update(kwargs)
        return BenchmarkSummary(**defaults)

    def test_failed_count(self) -> None:
        s = self._make_summary(total=52, passed=50)
        self.assertEqual(s.failed, 2)

    def test_meets_importer_criterion_true(self) -> None:
        s = self._make_summary(total=52, passed=52)
        self.assertTrue(s.meets_importer_criterion)

    def test_meets_importer_criterion_false_too_few(self) -> None:
        s = self._make_summary(total=49, passed=49)
        self.assertFalse(s.meets_importer_criterion)

    def test_meets_importer_criterion_false_with_errors(self) -> None:
        s = self._make_summary(total=52, passed=50)
        self.assertFalse(s.meets_importer_criterion)

    def test_meets_outcome_criterion_true(self) -> None:
        s = self._make_summary(winner_match_rate=0.65)
        self.assertTrue(s.meets_outcome_criterion)

    def test_meets_outcome_criterion_false(self) -> None:
        s = self._make_summary(winner_match_rate=0.55)
        self.assertFalse(s.meets_outcome_criterion)

    def test_meets_outcome_criterion_exactly_60(self) -> None:
        s = self._make_summary(winner_match_rate=0.60)
        self.assertTrue(s.meets_outcome_criterion)


if __name__ == "__main__":
    unittest.main()
