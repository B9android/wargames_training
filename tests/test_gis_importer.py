# SPDX-License-Identifier: MIT
"""Tests for E11.2 — GIS Terrain Import (real-world maps).

Covers:
* :class:`~data.gis.terrain_importer.BattleSiteBounds`
* :class:`~data.gis.terrain_importer.SRTMImporter`
* :class:`~data.gis.terrain_importer.OSMLayerImporter` (synthetic + XML)
* :class:`~data.gis.terrain_importer.GISTerrainBuilder`
* :func:`~data.gis.terrain_importer.build_all_battle_terrains`
* :func:`~data.gis.terrain_importer._normalise_elevation`
* GIS terrain type in :class:`~envs.scenarios.historical.TerrainConfig`
  and :meth:`~envs.scenarios.historical.HistoricalScenario.build_terrain`
* :class:`~envs.scenarios.historical.ScenarioLoader` for GIS YAML files
* :class:`~training.transfer_benchmark.TransferBenchmark` (dry-run)
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gis.terrain_importer import (
    BATTLE_SITES,
    BattleSiteBounds,
    GISTerrainBuilder,
    OSMLayerImporter,
    OSMLayers,
    SRTMImporter,
    _normalise_elevation,
    build_all_battle_terrains,
)
from envs.scenarios.historical import (
    HistoricalScenario,
    ScenarioLoader,
    TerrainConfig,
)
from envs.sim.terrain import TerrainMap

# Path to GIS scenario YAML files
_GIS_YAML_DIR = (
    PROJECT_ROOT / "configs" / "scenarios" / "historical" / "gis"
)


# ---------------------------------------------------------------------------
# BattleSiteBounds
# ---------------------------------------------------------------------------


class TestBattleSiteBounds(unittest.TestCase):
    def test_all_four_sites_present(self) -> None:
        for site in ("waterloo", "austerlitz", "borodino", "salamanca"):
            self.assertIn(site, BATTLE_SITES)

    def test_width_and_height_positive(self) -> None:
        for site, bounds in BATTLE_SITES.items():
            self.assertGreater(bounds.width_m, 0.0, f"{site} width")
            self.assertGreater(bounds.height_m, 0.0, f"{site} height")

    def test_width_roughly_correct_scale(self) -> None:
        # All sites should span at least 5 km and at most 20 km
        for site, bounds in BATTLE_SITES.items():
            self.assertGreater(bounds.width_m, 5_000, f"{site} width < 5 km")
            self.assertLess(bounds.width_m, 20_000, f"{site} width > 20 km")
            self.assertGreater(bounds.height_m, 5_000, f"{site} height < 5 km")
            self.assertLess(bounds.height_m, 20_000, f"{site} height > 20 km")

    def test_lat_lon_ordering(self) -> None:
        for site, b in BATTLE_SITES.items():
            self.assertLess(b.lat_min, b.lat_max, f"{site} lat ordering")
            self.assertLess(b.lon_min, b.lon_max, f"{site} lon ordering")

    def test_synthetic_elevation_shape(self) -> None:
        for site, bounds in BATTLE_SITES.items():
            elev = bounds.synthetic_elevation(10, 12)
            self.assertEqual(elev.shape, (10, 12), f"{site} shape")

    def test_synthetic_elevation_normalised(self) -> None:
        for site, bounds in BATTLE_SITES.items():
            elev = bounds.synthetic_elevation(20, 20)
            self.assertAlmostEqual(float(elev.max()), 1.0, places=5, msg=site)
            self.assertGreaterEqual(float(elev.min()), 0.0, msg=site)

    def test_synthetic_cover_shape(self) -> None:
        for site, bounds in BATTLE_SITES.items():
            cov = bounds.synthetic_cover(10, 12)
            self.assertEqual(cov.shape, (10, 12), f"{site} shape")

    def test_synthetic_cover_in_range(self) -> None:
        for site, bounds in BATTLE_SITES.items():
            cov = bounds.synthetic_cover(20, 20)
            self.assertGreaterEqual(float(cov.min()), 0.0, msg=site)
            self.assertLessEqual(float(cov.max()), 1.0, msg=site)

    def test_unknown_site_raises_in_builder(self) -> None:
        with self.assertRaises(KeyError):
            GISTerrainBuilder(site="hogwarts")

    def test_waterloo_bounds_correct_region(self) -> None:
        b = BATTLE_SITES["waterloo"]
        # Belgium: lat ~50.7, lon ~4.4
        self.assertAlmostEqual(b.lat_min, 50.66, places=1)
        self.assertAlmostEqual(b.lon_min, 4.36, places=1)


# ---------------------------------------------------------------------------
# _normalise_elevation
# ---------------------------------------------------------------------------


class TestNormaliseElevation(unittest.TestCase):
    def test_normalise_puts_values_in_0_1(self) -> None:
        arr = np.array([[100.0, 150.0], [200.0, 50.0]], dtype=np.float32)
        result = _normalise_elevation(arr)
        self.assertAlmostEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0)

    def test_flat_array_returns_zeros(self) -> None:
        arr = np.full((4, 4), 200.0, dtype=np.float32)
        result = _normalise_elevation(arr)
        self.assertTrue(np.all(result == 0.0))

    def test_output_dtype_is_float32(self) -> None:
        arr = np.linspace(0, 100, 16).reshape(4, 4)
        result = _normalise_elevation(arr)
        self.assertEqual(result.dtype, np.float32)

    def test_preserves_shape(self) -> None:
        arr = np.ones((3, 5), dtype=np.float32) * 50.0
        arr[0, 0] = 100.0
        result = _normalise_elevation(arr)
        self.assertEqual(result.shape, (3, 5))


# ---------------------------------------------------------------------------
# SRTMImporter
# ---------------------------------------------------------------------------


class TestSRTMImporter(unittest.TestCase):
    def test_synthetic_returns_correct_shape(self) -> None:
        imp = SRTMImporter(site="waterloo", rows=20, cols=25)
        arr = imp.load()
        self.assertEqual(arr.shape, (20, 25))

    def test_synthetic_returns_float32(self) -> None:
        arr = SRTMImporter(site="austerlitz", rows=10, cols=10).load()
        self.assertEqual(arr.dtype, np.float32)

    def test_synthetic_has_positive_elevation_range(self) -> None:
        for site in BATTLE_SITES:
            arr = SRTMImporter(site=site, rows=20, cols=20).load()
            self.assertGreater(float(arr.max() - arr.min()), 0.0, msg=site)

    def test_unknown_site_raises(self) -> None:
        with self.assertRaises(KeyError):
            SRTMImporter(site="narnia", rows=10, cols=10)

    def test_missing_srtm_path_falls_back_to_synthetic(self) -> None:
        imp = SRTMImporter(
            site="borodino",
            rows=15,
            cols=15,
            srtm_path="/nonexistent/path/borodino.tif",
        )
        arr = imp.load()
        self.assertEqual(arr.shape, (15, 15))

    def test_salamanca_elevation_in_plausible_range(self) -> None:
        # Salamanca: 780–870 m ASL
        arr = SRTMImporter(site="salamanca", rows=10, cols=10).load()
        self.assertGreaterEqual(float(arr.min()), 700.0)
        self.assertLessEqual(float(arr.max()), 950.0)

    def test_waterloo_elevation_in_plausible_range(self) -> None:
        # Waterloo: 80–160 m ASL
        arr = SRTMImporter(site="waterloo", rows=10, cols=10).load()
        self.assertGreaterEqual(float(arr.min()), 50.0)
        self.assertLessEqual(float(arr.max()), 200.0)


# ---------------------------------------------------------------------------
# OSMLayerImporter (synthetic)
# ---------------------------------------------------------------------------


class TestOSMLayerImporterSynthetic(unittest.TestCase):
    def test_synthetic_layers_return_correct_shape(self) -> None:
        imp = OSMLayerImporter(site="waterloo", rows=20, cols=25)
        layers = imp.load()
        self.assertEqual(layers.road_mask.shape, (20, 25))
        self.assertEqual(layers.forest_mask.shape, (20, 25))
        self.assertEqual(layers.settlement_mask.shape, (20, 25))

    def test_road_mask_values_binary_or_zero(self) -> None:
        imp = OSMLayerImporter(site="waterloo", rows=30, cols=30)
        layers = imp.load()
        unique = np.unique(layers.road_mask)
        self.assertTrue(
            set(unique.tolist()).issubset({0.0, 1.0}),
            f"road_mask has unexpected values: {unique}",
        )

    def test_forest_mask_in_range(self) -> None:
        for site in BATTLE_SITES:
            imp = OSMLayerImporter(site=site, rows=20, cols=20)
            layers = imp.load()
            self.assertGreaterEqual(float(layers.forest_mask.min()), 0.0)
            self.assertLessEqual(float(layers.forest_mask.max()), 1.0)

    def test_settlement_mask_in_range(self) -> None:
        for site in BATTLE_SITES:
            imp = OSMLayerImporter(site=site, rows=20, cols=20)
            layers = imp.load()
            self.assertGreaterEqual(float(layers.settlement_mask.min()), 0.0)
            self.assertLessEqual(float(layers.settlement_mask.max()), 1.0)

    def test_all_four_sites_produce_layers(self) -> None:
        for site in BATTLE_SITES:
            imp = OSMLayerImporter(site=site, rows=10, cols=10)
            layers = imp.load()
            self.assertIsInstance(layers, OSMLayers, msg=site)


# ---------------------------------------------------------------------------
# OSMLayerImporter (XML parsing)
# ---------------------------------------------------------------------------


def _write_minimal_osm(path: Path) -> None:
    """Write a tiny OSM XML file with one road and one forest way."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <!-- Waterloo bounding box: lat 50.66–50.74, lon 4.36–4.46 -->
  <node id="1" lat="50.67" lon="4.37"/>
  <node id="2" lat="50.68" lon="4.38"/>
  <node id="3" lat="50.70" lon="4.40"/>
  <node id="4" lat="50.71" lon="4.41"/>
  <node id="5" lat="50.70" lon="4.39"/>
  <node id="6" lat="50.68" lon="4.39"/>
  <way id="10">
    <nd ref="1"/>
    <nd ref="2"/>
    <nd ref="3"/>
    <tag k="highway" v="primary"/>
  </way>
  <way id="11">
    <nd ref="3"/>
    <nd ref="4"/>
    <nd ref="5"/>
    <nd ref="6"/>
    <nd ref="3"/>
    <tag k="natural" v="wood"/>
  </way>
</osm>"""
    path.write_text(content, encoding="utf-8")


class TestOSMLayerImporterXML(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.osm_path = Path(self._tmpdir.name) / "waterloo.osm"
        _write_minimal_osm(self.osm_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_road_mask_has_nonzero_values(self) -> None:
        imp = OSMLayerImporter(
            site="waterloo", rows=20, cols=20, osm_path=self.osm_path
        )
        layers = imp.load()
        self.assertGreater(float(layers.road_mask.sum()), 0.0)

    def test_forest_mask_has_nonzero_values(self) -> None:
        imp = OSMLayerImporter(
            site="waterloo", rows=20, cols=20, osm_path=self.osm_path
        )
        layers = imp.load()
        self.assertGreater(float(layers.forest_mask.sum()), 0.0)

    def test_settlement_mask_is_zero_with_no_settlement_tags(self) -> None:
        imp = OSMLayerImporter(
            site="waterloo", rows=20, cols=20, osm_path=self.osm_path
        )
        layers = imp.load()
        self.assertAlmostEqual(float(layers.settlement_mask.sum()), 0.0)

    def test_masks_in_valid_range(self) -> None:
        imp = OSMLayerImporter(
            site="waterloo", rows=20, cols=20, osm_path=self.osm_path
        )
        layers = imp.load()
        for mask, name in [
            (layers.road_mask, "road"),
            (layers.forest_mask, "forest"),
            (layers.settlement_mask, "settlement"),
        ]:
            self.assertGreaterEqual(float(mask.min()), 0.0, msg=f"{name} min")
            self.assertLessEqual(float(mask.max()), 1.0, msg=f"{name} max")


# ---------------------------------------------------------------------------
# GISTerrainBuilder
# ---------------------------------------------------------------------------


class TestGISTerrainBuilder(unittest.TestCase):
    def test_build_returns_terrain_map(self) -> None:
        builder = GISTerrainBuilder(site="waterloo", rows=20, cols=20)
        terrain = builder.build()
        self.assertIsInstance(terrain, TerrainMap)

    def test_terrain_dimensions_match_site_bounds(self) -> None:
        for site in BATTLE_SITES:
            builder = GISTerrainBuilder(site=site, rows=20, cols=20)
            terrain = builder.build()
            bounds = BATTLE_SITES[site]
            self.assertAlmostEqual(
                terrain.width, bounds.width_m, delta=10.0, msg=f"{site} width"
            )
            self.assertAlmostEqual(
                terrain.height, bounds.height_m, delta=10.0, msg=f"{site} height"
            )

    def test_elevation_grid_shape(self) -> None:
        builder = GISTerrainBuilder(site="austerlitz", rows=30, cols=25)
        terrain = builder.build()
        self.assertEqual(terrain.elevation.shape, (30, 25))

    def test_cover_grid_shape(self) -> None:
        builder = GISTerrainBuilder(site="borodino", rows=30, cols=25)
        terrain = builder.build()
        self.assertEqual(terrain.cover.shape, (30, 25))

    def test_elevation_normalised_to_0_1(self) -> None:
        for site in BATTLE_SITES:
            builder = GISTerrainBuilder(site=site, rows=20, cols=20)
            terrain = builder.build()
            self.assertAlmostEqual(
                float(terrain.elevation.max()), 1.0, places=4, msg=site
            )
            self.assertGreaterEqual(float(terrain.elevation.min()), 0.0, msg=site)

    def test_cover_in_0_1(self) -> None:
        for site in BATTLE_SITES:
            builder = GISTerrainBuilder(site=site, rows=20, cols=20)
            terrain = builder.build()
            self.assertGreaterEqual(float(terrain.cover.min()), 0.0, msg=site)
            self.assertLessEqual(float(terrain.cover.max()), 1.0, msg=site)

    def test_build_road_network_data_shape(self) -> None:
        builder = GISTerrainBuilder(site="salamanca", rows=20, cols=20)
        road = builder.build_road_network_data()
        self.assertEqual(road.shape, (20, 20))

    def test_build_road_network_data_binary(self) -> None:
        builder = GISTerrainBuilder(site="waterloo", rows=10, cols=10)
        road = builder.build_road_network_data()
        unique = np.unique(road)
        self.assertTrue(set(unique.tolist()).issubset({0.0, 1.0}))

    def test_waterloo_ridge_replicates_la_haye_sainte(self) -> None:
        """Acceptance criterion: La Haye Sainte ridge line is higher than
        the French southern position (E11.2 acceptance criterion)."""
        builder = GISTerrainBuilder(site="waterloo", rows=40, cols=40)
        terrain = builder.build()
        elev = terrain.elevation  # shape (40, 40)
        # Row index for y ≈ 0.60 (ridge) and y ≈ 0.30 (French south)
        ridge_row = int(0.60 * 40)
        south_row = int(0.30 * 40)
        ridge_mean = float(elev[ridge_row, :].mean())
        south_mean = float(elev[south_row, :].mean())
        self.assertGreater(
            ridge_mean,
            south_mean,
            "Ridge elevation should exceed the southern French position",
        )

    def test_cover_weight_defaults(self) -> None:
        b = GISTerrainBuilder(site="waterloo", rows=10, cols=10)
        self.assertEqual(b.forest_cover_weight, 1.0)
        self.assertEqual(b.settlement_cover_weight, 0.5)

    def test_unknown_site_raises(self) -> None:
        with self.assertRaises(KeyError):
            GISTerrainBuilder(site="waterloo_fake")


# ---------------------------------------------------------------------------
# build_all_battle_terrains
# ---------------------------------------------------------------------------


class TestBuildAllBattleTerrains(unittest.TestCase):
    def test_returns_four_sites(self) -> None:
        terrains = build_all_battle_terrains(rows=10, cols=10)
        self.assertEqual(set(terrains.keys()), set(BATTLE_SITES.keys()))

    def test_all_values_are_terrain_maps(self) -> None:
        terrains = build_all_battle_terrains(rows=10, cols=10)
        for site, tm in terrains.items():
            self.assertIsInstance(tm, TerrainMap, msg=site)

    def test_custom_resolution(self) -> None:
        terrains = build_all_battle_terrains(rows=15, cols=15)
        for site, tm in terrains.items():
            self.assertEqual(tm.elevation.shape, (15, 15), msg=site)


# ---------------------------------------------------------------------------
# TerrainConfig GIS support
# ---------------------------------------------------------------------------


class TestTerrainConfigGIS(unittest.TestCase):
    def test_gis_fields_default_empty(self) -> None:
        cfg = TerrainConfig()
        self.assertEqual(cfg.gis_site, "")
        self.assertEqual(cfg.gis_data_dir, "")

    def test_gis_terrain_type_builds_terrain_map(self) -> None:
        cfg = TerrainConfig(terrain_type="gis", gis_site="waterloo", rows=10, cols=10)
        scenario = HistoricalScenario(
            name="test",
            date="1815-06-18",
            description="",
            faction_blue="blue",
            faction_red="red",
            terrain_config=cfg,
        )
        terrain = scenario.build_terrain()
        self.assertIsInstance(terrain, TerrainMap)

    def test_gis_terrain_empty_site_raises(self) -> None:
        cfg = TerrainConfig(terrain_type="gis", gis_site="", rows=10, cols=10)
        scenario = HistoricalScenario(
            name="test",
            date="1815-06-18",
            description="",
            faction_blue="blue",
            faction_red="red",
            terrain_config=cfg,
        )
        with self.assertRaises(ValueError):
            scenario.build_terrain()

    def test_all_four_sites_build_terrain(self) -> None:
        for site in BATTLE_SITES:
            cfg = TerrainConfig(
                terrain_type="gis", gis_site=site, rows=10, cols=10
            )
            scenario = HistoricalScenario(
                name="test",
                date="1815-01-01",
                description="",
                faction_blue="blue",
                faction_red="red",
                terrain_config=cfg,
            )
            terrain = scenario.build_terrain()
            self.assertIsInstance(terrain, TerrainMap, msg=site)

    def test_gis_terrain_elevation_shape(self) -> None:
        cfg = TerrainConfig(
            terrain_type="gis", gis_site="austerlitz", rows=15, cols=20
        )
        scenario = HistoricalScenario(
            name="test",
            date="1805-12-02",
            description="",
            faction_blue="blue",
            faction_red="red",
            terrain_config=cfg,
        )
        terrain = scenario.build_terrain()
        self.assertEqual(terrain.elevation.shape, (15, 20))


# ---------------------------------------------------------------------------
# ScenarioLoader for GIS YAML files
# ---------------------------------------------------------------------------


class TestGISScenarioYAMLLoader(unittest.TestCase):
    """Load each GIS YAML scenario and verify structure."""

    _YAML_FILES = {
        "waterloo": "waterloo_gis.yaml",
        "austerlitz": "austerlitz_gis.yaml",
        "borodino": "borodino_gis.yaml",
        "salamanca": "salamanca_gis.yaml",
    }

    def _load(self, filename: str) -> HistoricalScenario:
        path = _GIS_YAML_DIR / filename
        self.assertTrue(path.exists(), f"YAML file missing: {path}")
        return ScenarioLoader(path).load()

    def test_waterloo_gis_loads(self) -> None:
        scenario = self._load("waterloo_gis.yaml")
        self.assertIn("Waterloo", scenario.name)

    def test_austerlitz_gis_loads(self) -> None:
        scenario = self._load("austerlitz_gis.yaml")
        self.assertIn("Austerlitz", scenario.name)

    def test_borodino_gis_loads(self) -> None:
        scenario = self._load("borodino_gis.yaml")
        self.assertIn("Borodino", scenario.name)

    def test_salamanca_gis_loads(self) -> None:
        scenario = self._load("salamanca_gis.yaml")
        self.assertIn("Salamanca", scenario.name)

    def test_all_scenarios_have_gis_terrain_type(self) -> None:
        for site, filename in self._YAML_FILES.items():
            scenario = self._load(filename)
            self.assertEqual(
                scenario.terrain_config.terrain_type,
                "gis",
                msg=f"{site} terrain_type",
            )
            self.assertEqual(
                scenario.terrain_config.gis_site,
                site,
                msg=f"{site} gis_site",
            )

    def test_all_scenarios_have_units(self) -> None:
        for site, filename in self._YAML_FILES.items():
            scenario = self._load(filename)
            self.assertGreater(
                scenario.n_blue, 0, msg=f"{site} has no blue units"
            )
            self.assertGreater(
                scenario.n_red, 0, msg=f"{site} has no red units"
            )

    def test_all_scenarios_build_terrain(self) -> None:
        for site, filename in self._YAML_FILES.items():
            scenario = self._load(filename)
            terrain = scenario.build_terrain()
            self.assertIsInstance(terrain, TerrainMap, msg=site)
            self.assertGreater(terrain.width, 0.0, msg=site)
            self.assertGreater(terrain.height, 0.0, msg=site)

    def test_waterloo_has_correct_factions(self) -> None:
        scenario = self._load("waterloo_gis.yaml")
        self.assertIn("Wellington", scenario.faction_blue)
        self.assertIn("Napoleon", scenario.faction_red)

    def test_historical_outcomes_present(self) -> None:
        for site, filename in self._YAML_FILES.items():
            scenario = self._load(filename)
            outcome = scenario.historical_outcome
            self.assertGreaterEqual(outcome.blue_casualties, 0.0, msg=site)
            self.assertGreaterEqual(outcome.red_casualties, 0.0, msg=site)

    def test_waterloo_gis_grid_resolution(self) -> None:
        scenario = self._load("waterloo_gis.yaml")
        self.assertEqual(scenario.terrain_config.rows, 40)
        self.assertEqual(scenario.terrain_config.cols, 40)


# ---------------------------------------------------------------------------
# TransferBenchmark (dry-run, no checkpoint)
# ---------------------------------------------------------------------------


class TestTransferBenchmarkDryRun(unittest.TestCase):
    def _make_bench(self, site: str = "waterloo", n_episodes: int = 3):
        from training.transfer_benchmark import TransferBenchmark, TransferEvalConfig

        cfg = TransferEvalConfig(
            site=site,
            n_eval_episodes=n_episodes,
            max_steps_per_episode=50,
            finetune_steps=0,
            rows=10,
            cols=10,
        )
        return TransferBenchmark(cfg)

    def test_run_returns_summary(self) -> None:
        from training.transfer_benchmark import TransferSummary

        bench = self._make_bench()
        summary = bench.run(policy=None)
        self.assertIsInstance(summary, TransferSummary)

    def test_summary_has_all_conditions(self) -> None:
        bench = self._make_bench()
        summary = bench.run(policy=None)
        self.assertEqual(summary.procedural.condition, "procedural_baseline")
        self.assertEqual(summary.zero_shot.condition, "zero_shot_gis")
        self.assertEqual(summary.finetuned.condition, "finetuned_gis")

    def test_win_rates_in_0_1(self) -> None:
        bench = self._make_bench(n_episodes=5)
        summary = bench.run(policy=None)
        for cond in (summary.procedural, summary.zero_shot, summary.finetuned):
            self.assertGreaterEqual(cond.win_rate, 0.0)
            self.assertLessEqual(cond.win_rate, 1.0)

    def test_n_episodes_matches_config(self) -> None:
        bench = self._make_bench(n_episodes=4)
        summary = bench.run(policy=None)
        self.assertEqual(summary.procedural.n_episodes, 4)
        self.assertEqual(summary.zero_shot.n_episodes, 4)
        self.assertEqual(summary.finetuned.n_episodes, 4)

    def test_finetuned_steps_used_zero_when_no_policy(self) -> None:
        bench = self._make_bench()
        summary = bench.run(policy=None)
        self.assertEqual(summary.finetuned.finetune_steps_used, 0)

    def test_zero_shot_drop_defined(self) -> None:
        bench = self._make_bench()
        summary = bench.run(policy=None)
        # drop can be any float; just check it's finite
        self.assertTrue(math.isfinite(summary.zero_shot_drop))

    def test_all_sites_run(self) -> None:
        from training.transfer_benchmark import TransferBenchmark, TransferEvalConfig

        for site in BATTLE_SITES:
            cfg = TransferEvalConfig(
                site=site, n_eval_episodes=2, max_steps_per_episode=20,
                rows=10, cols=10,
            )
            bench = TransferBenchmark(cfg)
            summary = bench.run(policy=None)
            self.assertEqual(summary.site, site)

    def test_write_markdown(self) -> None:
        import os

        bench = self._make_bench(n_episodes=2)
        summary = bench.run(policy=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = bench.write_markdown(summary, path=Path(tmpdir) / "report.md")
            self.assertTrue(out.exists())
            content = out.read_text()
            self.assertIn("waterloo", content.lower())
            self.assertIn("Waterloo", content)

    def test_str_representation_has_win_rates(self) -> None:
        bench = self._make_bench(n_episodes=2)
        summary = bench.run(policy=None)
        text = str(summary)
        self.assertIn("Procedural baseline", text)
        self.assertIn("Zero-shot", text)
        self.assertIn("Fine-tuned", text)

    def test_missing_policy_path_falls_back_gracefully(self) -> None:
        bench = self._make_bench(n_episodes=2)
        # Passing a non-existent path should not crash
        summary = bench.run(policy="/tmp/nonexistent_policy.zip")
        self.assertIsNotNone(summary)


if __name__ == "__main__":
    unittest.main()
