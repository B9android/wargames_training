"""Unit tests for envs/sim/terrain_engine.py."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.terrain import TerrainMap
from envs.sim.terrain_engine import HeightmapLoader, TerrainEngine


# ---------------------------------------------------------------------------
# TerrainEngine factories
# ---------------------------------------------------------------------------


class TestTerrainEngineFactories(unittest.TestCase):
    def test_flat_returns_engine(self) -> None:
        eng = TerrainEngine.flat(1000.0, 1000.0)
        self.assertIsInstance(eng, TerrainEngine)
        self.assertAlmostEqual(eng.width, 1000.0)
        self.assertAlmostEqual(eng.height, 1000.0)

    def test_from_terrain_map_wraps_map(self) -> None:
        tm = TerrainMap.flat(500.0, 500.0)
        eng = TerrainEngine.from_terrain_map(tm)
        self.assertIs(eng.terrain_map, tm)

    def test_from_arrays_stores_dimensions(self) -> None:
        elev = np.zeros((4, 4), dtype=np.float32)
        cov = np.zeros((4, 4), dtype=np.float32)
        eng = TerrainEngine.from_arrays(300.0, 200.0, elev, cov)
        self.assertAlmostEqual(eng.width, 300.0)
        self.assertAlmostEqual(eng.height, 200.0)

    def test_from_arrays_rejects_1d(self) -> None:
        elev = np.zeros(9, dtype=np.float32)
        cov = np.zeros((3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            TerrainEngine.from_arrays(100.0, 100.0, elev, cov)

    def test_elevation_and_cover_properties(self) -> None:
        elev = np.array([[0.5, 0.8], [0.2, 1.0]], dtype=np.float32)
        cov = np.array([[0.0, 0.3], [0.7, 1.0]], dtype=np.float32)
        eng = TerrainEngine.from_arrays(200.0, 200.0, elev, cov)
        np.testing.assert_array_almost_equal(eng.elevation, elev)
        np.testing.assert_array_almost_equal(eng.cover, cov)


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------


class TestSlope(unittest.TestCase):
    def test_flat_terrain_slope_is_zero(self) -> None:
        eng = TerrainEngine.flat(500.0, 500.0, rows=5, cols=5)
        self.assertAlmostEqual(eng.slope(250.0, 250.0), 0.0)

    def test_slope_no_elevation_returns_zero(self) -> None:
        tm = TerrainMap(width=500.0, height=500.0, elevation=None, cover=None)
        eng = TerrainEngine.from_terrain_map(tm)
        self.assertAlmostEqual(eng.slope(100.0, 100.0), 0.0)

    def test_single_cell_slope_is_zero(self) -> None:
        elev = np.array([[1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(eng.slope(50.0, 50.0), 0.0)

    def test_known_slope_x_direction(self) -> None:
        # 1 row × 3 cols: elevations [0, 0.5, 1.0] over 300 m width
        # Central difference at centre: (1.0 - 0.0) / (2 * 100) = 0.005
        elev = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(300.0, 100.0, elev, cov)
        s = eng.slope(150.0, 50.0)  # centre cell
        self.assertAlmostEqual(s, 0.005, places=4)

    def test_slope_nonnegative_everywhere(self) -> None:
        rng = np.random.default_rng(99)
        eng = TerrainEngine.generate_random(rng, 500.0, 500.0, ruggedness=0.8)
        for x in np.linspace(0.0, 500.0, 5):
            for y in np.linspace(0.0, 500.0, 5):
                self.assertGreaterEqual(eng.slope(float(x), float(y)), 0.0)


# ---------------------------------------------------------------------------
# Bresenham LOS — acceptance criteria: blocked by ridgelines in ≥ 3 cases
# ---------------------------------------------------------------------------


class TestBresenhamLOS(unittest.TestCase):
    """Acceptance criterion: LOS correctly blocked by ridgelines in ≥ 3 unit test cases."""

    # Case 1 — horizontal ridge in a 1×3 grid
    def test_blocked_by_central_ridge_horizontal(self) -> None:
        elev = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(300.0, 100.0, elev, cov)
        # Both endpoints at elevation 0; central ridge at 10 → blocked
        self.assertFalse(eng.bresenham_los(0.0, 50.0, 300.0, 50.0))

    # Case 2 — vertical ridge in a 3×1 grid
    def test_blocked_by_central_ridge_vertical(self) -> None:
        elev = np.array([[0.0], [10.0], [0.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(100.0, 300.0, elev, cov)
        self.assertFalse(eng.bresenham_los(50.0, 0.0, 50.0, 300.0))

    # Case 3 — diagonal terrain with a ridge on one side
    def test_blocked_by_ridge_in_larger_grid(self) -> None:
        # 1-row, 5-col: flat(0)–flat(0)–ridge(20)–flat(0)–flat(0)
        elev = np.array([[0.0, 0.0, 20.0, 0.0, 0.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(500.0, 100.0, elev, cov)
        self.assertFalse(eng.bresenham_los(0.0, 50.0, 500.0, 50.0))

    def test_unblocked_flat_terrain(self) -> None:
        eng = TerrainEngine.flat(500.0, 500.0, rows=10, cols=10)
        self.assertTrue(eng.bresenham_los(0.0, 0.0, 500.0, 500.0))

    def test_unblocked_no_elevation_grid(self) -> None:
        tm = TerrainMap(width=500.0, height=500.0, elevation=None, cover=None)
        eng = TerrainEngine.from_terrain_map(tm)
        self.assertTrue(eng.bresenham_los(0.0, 0.0, 500.0, 500.0))

    def test_unblocked_when_both_endpoints_elevated(self) -> None:
        # Both endpoints at height 10 and so is the ridge — no blocking
        elev = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(300.0, 100.0, elev, cov)
        self.assertTrue(eng.bresenham_los(0.0, 50.0, 300.0, 50.0))

    def test_same_start_end_always_unblocked(self) -> None:
        elev = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(300.0, 100.0, elev, cov)
        self.assertTrue(eng.bresenham_los(150.0, 50.0, 150.0, 50.0))


# ---------------------------------------------------------------------------
# Movement cost
# ---------------------------------------------------------------------------


class TestMovementCost(unittest.TestCase):
    def test_flat_terrain_returns_full_speed(self) -> None:
        eng = TerrainEngine.flat(500.0, 500.0, rows=4, cols=4)
        self.assertAlmostEqual(eng.movement_cost(100.0, 100.0), 1.0)

    def test_cost_bounded_between_hill_factor_and_one(self) -> None:
        rng = np.random.default_rng(7)
        eng = TerrainEngine.generate_random(rng, 500.0, 500.0, ruggedness=0.7)
        for x in np.linspace(0.0, 500.0, 8):
            for y in np.linspace(0.0, 500.0, 8):
                cost = eng.movement_cost(float(x), float(y), hill_speed_factor=0.5)
                self.assertGreater(cost, 0.0)
                self.assertLessEqual(cost, 1.0 + 1e-6)

    def test_max_elev_cell_cost_lte_hill_speed_factor(self) -> None:
        # Single-cell map at max elevation → movement_cost == hill_speed_factor
        # (slope is 0 for a single cell, so only elev penalty applies)
        elev = np.array([[1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(100.0, 100.0, elev, cov)
        cost = eng.movement_cost(50.0, 50.0, hill_speed_factor=0.5)
        self.assertAlmostEqual(cost, 0.5, places=5)

    def test_invalid_hill_speed_factor_raises(self) -> None:
        eng = TerrainEngine.flat(100.0, 100.0)
        with self.assertRaises(ValueError):
            eng.movement_cost(50.0, 50.0, hill_speed_factor=0.0)


# ---------------------------------------------------------------------------
# Procedural terrain generator — ruggedness parameter
# ---------------------------------------------------------------------------


class TestGenerateRandom(unittest.TestCase):
    def _rng(self, seed: int = 0) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_returns_engine_instance(self) -> None:
        eng = TerrainEngine.generate_random(self._rng(), 1000.0, 1000.0)
        self.assertIsInstance(eng, TerrainEngine)

    def test_dimensions_preserved(self) -> None:
        eng = TerrainEngine.generate_random(self._rng(), 600.0, 400.0)
        self.assertAlmostEqual(eng.width, 600.0)
        self.assertAlmostEqual(eng.height, 400.0)

    def test_grid_shape(self) -> None:
        eng = TerrainEngine.generate_random(self._rng(), 1000.0, 1000.0, rows=8, cols=12)
        self.assertEqual(eng.elevation.shape, (8, 12))
        self.assertEqual(eng.cover.shape, (8, 12))

    def test_ruggedness_zero_produces_flat_elevation(self) -> None:
        eng = TerrainEngine.generate_random(self._rng(), 1000.0, 1000.0, ruggedness=0.0)
        np.testing.assert_array_equal(eng.elevation, np.zeros_like(eng.elevation))

    def test_ruggedness_one_produces_nonzero_elevation(self) -> None:
        eng = TerrainEngine.generate_random(self._rng(1), 1000.0, 1000.0, ruggedness=1.0)
        self.assertGreater(float(eng.elevation.max()), 0.0)

    def test_elevation_normalised_to_unit_interval(self) -> None:
        for rugg in [0.2, 0.5, 0.8, 1.0]:
            eng = TerrainEngine.generate_random(self._rng(42), 1000.0, 1000.0, ruggedness=rugg)
            self.assertGreaterEqual(float(eng.elevation.min()), 0.0)
            self.assertLessEqual(float(eng.elevation.max()), 1.0 + 1e-6)

    def test_cover_clipped_to_unit_interval(self) -> None:
        eng = TerrainEngine.generate_random(self._rng(2), 1000.0, 1000.0, num_forests=5)
        self.assertGreaterEqual(float(eng.cover.min()), 0.0)
        self.assertLessEqual(float(eng.cover.max()), 1.0 + 1e-6)

    def test_same_seed_reproducible(self) -> None:
        a = TerrainEngine.generate_random(self._rng(42), 1000.0, 1000.0)
        b = TerrainEngine.generate_random(self._rng(42), 1000.0, 1000.0)
        np.testing.assert_array_equal(a.elevation, b.elevation)

    def test_different_seeds_different_terrain(self) -> None:
        a = TerrainEngine.generate_random(self._rng(1), 1000.0, 1000.0)
        b = TerrainEngine.generate_random(self._rng(2), 1000.0, 1000.0)
        self.assertFalse(np.array_equal(a.elevation, b.elevation))

    def test_rugged_has_higher_slope_than_flat(self) -> None:
        """Higher ruggedness should produce steeper average terrain."""
        rng_flat = np.random.default_rng(10)
        rng_rugged = np.random.default_rng(10)
        flat_eng = TerrainEngine.generate_random(rng_flat, 1000.0, 1000.0,
                                                 rows=20, cols=20, ruggedness=0.0)
        rugged_eng = TerrainEngine.generate_random(rng_rugged, 1000.0, 1000.0,
                                                   rows=20, cols=20, ruggedness=1.0)
        # Average slope over a sample grid
        xs = np.linspace(50.0, 950.0, 10)
        ys = np.linspace(50.0, 950.0, 10)
        flat_slopes = [flat_eng.slope(float(x), float(y)) for x in xs for y in ys]
        rugged_slopes = [rugged_eng.slope(float(x), float(y)) for x in xs for y in ys]
        self.assertGreater(sum(rugged_slopes), sum(flat_slopes))

    def test_invalid_ruggedness_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainEngine.generate_random(self._rng(), 1000.0, 1000.0, ruggedness=-0.1)
        with self.assertRaises(ValueError):
            TerrainEngine.generate_random(self._rng(), 1000.0, 1000.0, ruggedness=1.1)

    def test_invalid_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainEngine.generate_random(self._rng(), 1000.0, 1000.0, rows=0)

    def test_invalid_forest_cover_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainEngine.generate_random(self._rng(), 1000.0, 1000.0, forest_cover=1.5)


# ---------------------------------------------------------------------------
# HeightmapLoader
# ---------------------------------------------------------------------------


class TestHeightmapLoader(unittest.TestCase):
    def test_from_array_returns_engine(self) -> None:
        elev = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = HeightmapLoader.from_array(200.0, 200.0, elev, cov)
        self.assertIsInstance(eng, TerrainEngine)

    def test_from_array_dimensions(self) -> None:
        elev = np.zeros((5, 5), dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = HeightmapLoader.from_array(500.0, 300.0, elev, cov)
        self.assertAlmostEqual(eng.width, 500.0)
        self.assertAlmostEqual(eng.height, 300.0)

    def test_from_array_elevation_correct(self) -> None:
        elev = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = HeightmapLoader.from_array(200.0, 200.0, elev, cov)
        # top-left cell (row=0, col=0) → 0.2
        self.assertAlmostEqual(eng.get_elevation(0.0, 0.0), 0.2)

    def test_from_procedural_returns_engine(self) -> None:
        rng = np.random.default_rng(5)
        eng = HeightmapLoader.from_procedural(rng, 1000.0, 1000.0, ruggedness=0.5)
        self.assertIsInstance(eng, TerrainEngine)

    def test_from_procedural_ruggedness_zero_flat(self) -> None:
        rng = np.random.default_rng(5)
        eng = HeightmapLoader.from_procedural(rng, 1000.0, 1000.0, ruggedness=0.0)
        np.testing.assert_array_equal(eng.elevation, np.zeros_like(eng.elevation))

    def test_from_array_rejects_mismatched_shapes(self) -> None:
        elev = np.zeros((3, 3), dtype=np.float32)
        cov = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            HeightmapLoader.from_array(100.0, 100.0, elev, cov)


# ---------------------------------------------------------------------------
# Delegated TerrainMap methods
# ---------------------------------------------------------------------------


class TestDelegation(unittest.TestCase):
    def test_get_elevation_delegates(self) -> None:
        elev = np.array([[5.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(eng.get_elevation(50.0, 50.0), 5.0)

    def test_get_cover_delegates(self) -> None:
        elev = np.zeros((1, 1), dtype=np.float32)
        cov = np.array([[0.75]], dtype=np.float32)
        eng = TerrainEngine.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(eng.get_cover(50.0, 50.0), 0.75)

    def test_apply_cover_modifier_delegates(self) -> None:
        elev = np.zeros((1, 1), dtype=np.float32)
        cov = np.array([[0.5]], dtype=np.float32)
        eng = TerrainEngine.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(eng.apply_cover_modifier(50.0, 50.0, 1.0), 0.5)

    def test_get_speed_modifier_delegates(self) -> None:
        elev = np.array([[1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(eng.get_speed_modifier(50.0, 50.0, hill_speed_factor=0.5), 0.5)

    def test_line_of_sight_delegates(self) -> None:
        elev = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        eng = TerrainEngine.from_arrays(300.0, 100.0, elev, cov)
        self.assertFalse(eng.line_of_sight(0.0, 50.0, 300.0, 50.0, num_samples=30))


if __name__ == "__main__":
    unittest.main()
