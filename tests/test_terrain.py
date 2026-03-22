"""Unit tests for envs/sim/terrain.py."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.terrain import TerrainMap


# ---------------------------------------------------------------------------
# TerrainMap.flat
# ---------------------------------------------------------------------------


class TestTerrainMapFlat(unittest.TestCase):
    def test_flat_stores_dimensions(self) -> None:
        tm = TerrainMap.flat(width=500.0, height=400.0)
        self.assertAlmostEqual(tm.width, 500.0)
        self.assertAlmostEqual(tm.height, 400.0)

    def test_flat_elevation_is_zero_everywhere(self) -> None:
        tm = TerrainMap.flat(width=200.0, height=200.0, rows=4, cols=4)
        self.assertIsNotNone(tm.elevation)
        self.assertTrue(np.all(tm.elevation == 0.0))

    def test_flat_cover_is_zero_everywhere(self) -> None:
        tm = TerrainMap.flat(width=200.0, height=200.0, rows=4, cols=4)
        self.assertIsNotNone(tm.cover)
        self.assertTrue(np.all(tm.cover == 0.0))

    def test_flat_line_of_sight_always_true(self) -> None:
        tm = TerrainMap.flat(width=500.0, height=500.0)
        self.assertTrue(tm.line_of_sight(0.0, 0.0, 500.0, 500.0))
        self.assertTrue(tm.line_of_sight(100.0, 100.0, 400.0, 400.0))


# ---------------------------------------------------------------------------
# TerrainMap.from_arrays
# ---------------------------------------------------------------------------


class TestTerrainMapFromArrays(unittest.TestCase):
    def _make_arrays(self, rows: int = 3, cols: int = 3):
        elev = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
        cov = np.linspace(0.0, 1.0, rows * cols, dtype=np.float32).reshape(rows, cols)
        return elev, cov

    def test_from_arrays_stores_dimensions(self) -> None:
        elev, cov = self._make_arrays()
        tm = TerrainMap.from_arrays(300.0, 300.0, elev, cov)
        self.assertAlmostEqual(tm.width, 300.0)
        self.assertAlmostEqual(tm.height, 300.0)

    def test_from_arrays_elevation_array_is_copied(self) -> None:
        elev, cov = self._make_arrays()
        tm = TerrainMap.from_arrays(300.0, 300.0, elev, cov)
        elev[0, 0] = 999.0
        self.assertNotEqual(float(tm.elevation[0, 0]), 999.0)

    def test_from_arrays_cover_array_is_copied(self) -> None:
        elev, cov = self._make_arrays()
        original_value = float(cov[0, 0])
        tm = TerrainMap.from_arrays(300.0, 300.0, elev, cov)
        cov[0, 0] = original_value + 0.5  # shift by 0.5 — well beyond float noise
        self.assertAlmostEqual(float(tm.cover[0, 0]), original_value)

    def test_from_arrays_raises_for_mismatched_shapes(self) -> None:
        elev = np.zeros((3, 3), dtype=np.float32)
        cov = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            TerrainMap.from_arrays(100.0, 100.0, elev, cov)

    def test_from_arrays_raises_for_1d_elevation(self) -> None:
        elev = np.zeros(9, dtype=np.float32)
        cov = np.zeros((3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            TerrainMap.from_arrays(100.0, 100.0, elev, cov)


# ---------------------------------------------------------------------------
# max_elevation property
# ---------------------------------------------------------------------------


class TestMaxElevation(unittest.TestCase):
    def test_flat_terrain_max_elevation_is_zero(self) -> None:
        tm = TerrainMap.flat(500.0, 500.0, rows=4, cols=4)
        self.assertAlmostEqual(tm.max_elevation, 0.0)

    def test_no_elevation_grid_returns_zero(self) -> None:
        tm = TerrainMap(width=500.0, height=500.0, elevation=None, cover=None)
        self.assertAlmostEqual(tm.max_elevation, 0.0)

    def test_returns_correct_max(self) -> None:
        elev = np.array([[0.0, 0.5], [0.8, 1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        tm = TerrainMap.from_arrays(200.0, 200.0, elev, cov)
        self.assertAlmostEqual(tm.max_elevation, 1.0)

    def test_max_elevation_matches_speed_modifier_normalisation(self) -> None:
        """max_elevation is the same value used by get_speed_modifier."""
        elev = np.array([[0.0, 0.3], [0.6, 1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        tm = TerrainMap.from_arrays(200.0, 200.0, elev, cov)
        # At max-elev cell (1.0), speed modifier should equal hill_speed_factor
        max_mod = tm.get_speed_modifier(150.0, 150.0, hill_speed_factor=0.4)
        self.assertAlmostEqual(max_mod, 0.4, places=5)
        self.assertAlmostEqual(tm.max_elevation, 1.0)


# ---------------------------------------------------------------------------
# get_elevation
# ---------------------------------------------------------------------------


class TestGetElevation(unittest.TestCase):
    def test_returns_zero_for_flat_terrain(self) -> None:
        tm = TerrainMap.flat(500.0, 500.0)
        self.assertAlmostEqual(tm.get_elevation(100.0, 100.0), 0.0)

    def test_returns_correct_value_from_grid(self) -> None:
        elev = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        tm = TerrainMap.from_arrays(200.0, 200.0, elev, cov)
        # top-left cell → row=0, col=0
        self.assertAlmostEqual(tm.get_elevation(0.0, 0.0), 0.0)
        # top-right cell → row=0, col=1
        self.assertAlmostEqual(tm.get_elevation(150.0, 0.0), 1.0)

    def test_out_of_bounds_is_clamped(self) -> None:
        elev = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        tm = TerrainMap.from_arrays(200.0, 200.0, elev, cov)
        # Negative coords should clamp to first cell
        self.assertAlmostEqual(tm.get_elevation(-50.0, -50.0), 5.0)
        # Coords beyond map should clamp to last cell
        self.assertAlmostEqual(tm.get_elevation(999.0, 999.0), 8.0)


# ---------------------------------------------------------------------------
# get_cover
# ---------------------------------------------------------------------------


class TestGetCover(unittest.TestCase):
    def test_returns_zero_for_no_cover_terrain(self) -> None:
        tm = TerrainMap.flat(500.0, 500.0)
        self.assertAlmostEqual(tm.get_cover(100.0, 100.0), 0.0)

    def test_returns_correct_cover_value(self) -> None:
        elev = np.zeros((2, 2), dtype=np.float32)
        cov = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
        tm = TerrainMap.from_arrays(200.0, 200.0, elev, cov)
        # bottom-right cell → row=1, col=1 → cover=1.0
        self.assertAlmostEqual(tm.get_cover(150.0, 150.0), 1.0)
        # top-right cell → row=0, col=1 → cover=0.5
        self.assertAlmostEqual(tm.get_cover(150.0, 0.0), 0.5)

    def test_cover_values_clamped_to_unit_interval(self) -> None:
        elev = np.zeros((1, 1), dtype=np.float32)
        cov = np.array([[2.0]], dtype=np.float32)  # over-range value
        tm = TerrainMap.from_arrays(100.0, 100.0, elev, cov)
        self.assertLessEqual(tm.get_cover(50.0, 50.0), 1.0)
        self.assertGreaterEqual(tm.get_cover(50.0, 50.0), 0.0)


# ---------------------------------------------------------------------------
# line_of_sight
# ---------------------------------------------------------------------------


class TestLineOfSight(unittest.TestCase):
    def test_flat_terrain_always_has_los(self) -> None:
        tm = TerrainMap.flat(500.0, 500.0, rows=10, cols=10)
        self.assertTrue(tm.line_of_sight(0.0, 0.0, 500.0, 500.0))

    def test_blocked_by_ridge(self) -> None:
        # 1-row, 3-col terrain: flat–high–flat
        elev = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        # Width = 300 so columns map to x ≈ 0, 100, 200
        tm = TerrainMap.from_arrays(300.0, 100.0, elev, cov)
        # Endpoints both at elevation 0; the middle ridge is at 10 — blocks LOS
        self.assertFalse(tm.line_of_sight(0.0, 50.0, 300.0, 50.0, num_samples=30))

    def test_unblocked_when_both_endpoints_elevated(self) -> None:
        # Same ridge but both endpoints are also at height 10
        elev = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        tm = TerrainMap.from_arrays(300.0, 100.0, elev, cov)
        self.assertTrue(tm.line_of_sight(0.0, 50.0, 300.0, 50.0, num_samples=30))


# ---------------------------------------------------------------------------
# apply_cover_modifier
# ---------------------------------------------------------------------------


class TestApplyCoverModifier(unittest.TestCase):
    def test_no_cover_returns_full_damage(self) -> None:
        tm = TerrainMap.flat(500.0, 500.0)
        self.assertAlmostEqual(tm.apply_cover_modifier(100.0, 100.0, 0.5), 0.5)

    def test_full_cover_returns_zero_damage(self) -> None:
        elev = np.zeros((1, 1), dtype=np.float32)
        cov = np.array([[1.0]], dtype=np.float32)
        tm = TerrainMap.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(tm.apply_cover_modifier(50.0, 50.0, 0.5), 0.0)

    def test_partial_cover_reduces_damage(self) -> None:
        elev = np.zeros((1, 1), dtype=np.float32)
        cov = np.array([[0.4]], dtype=np.float32)
        tm = TerrainMap.from_arrays(100.0, 100.0, elev, cov)
        result = tm.apply_cover_modifier(50.0, 50.0, 1.0)
        self.assertAlmostEqual(result, 0.6)


# ---------------------------------------------------------------------------
# get_speed_modifier
# ---------------------------------------------------------------------------


class TestGetSpeedModifier(unittest.TestCase):
    def test_flat_terrain_returns_full_speed(self) -> None:
        tm = TerrainMap.flat(500.0, 500.0, rows=4, cols=4)
        self.assertAlmostEqual(tm.get_speed_modifier(100.0, 100.0), 1.0)

    def test_no_elevation_grid_returns_full_speed(self) -> None:
        tm = TerrainMap(width=500.0, height=500.0, elevation=None, cover=None)
        self.assertAlmostEqual(tm.get_speed_modifier(100.0, 100.0), 1.0)

    def test_max_elevation_cell_returns_hill_speed_factor(self) -> None:
        # Single-cell map with elevation=1.0 — should return hill_speed_factor
        elev = np.array([[1.0]], dtype=np.float32)
        cov = np.zeros((1, 1), dtype=np.float32)
        tm = TerrainMap.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(tm.get_speed_modifier(50.0, 50.0, hill_speed_factor=0.5), 0.5)

    def test_intermediate_elevation_interpolates(self) -> None:
        # 1×2 grid: left=0.0, right=1.0
        elev = np.array([[0.0, 1.0]], dtype=np.float32)
        cov = np.zeros_like(elev)
        tm = TerrainMap.from_arrays(200.0, 100.0, elev, cov)
        # Left cell (flat) → full speed
        self.assertAlmostEqual(tm.get_speed_modifier(50.0, 50.0, hill_speed_factor=0.5), 1.0)
        # Right cell (max elev) → hill_speed_factor
        self.assertAlmostEqual(tm.get_speed_modifier(150.0, 50.0, hill_speed_factor=0.5), 0.5)

    def test_hill_speed_factor_one_disables_penalty(self) -> None:
        elev = np.array([[1.0]], dtype=np.float32)
        cov = np.zeros((1, 1), dtype=np.float32)
        tm = TerrainMap.from_arrays(100.0, 100.0, elev, cov)
        self.assertAlmostEqual(tm.get_speed_modifier(50.0, 50.0, hill_speed_factor=1.0), 1.0)

    def test_speed_modifier_bounded_between_factor_and_one(self) -> None:
        rng = np.random.default_rng(7)
        tm = TerrainMap.generate_random(rng=rng, width=500.0, height=500.0)
        for x in np.linspace(0.0, 500.0, 10):
            for y in np.linspace(0.0, 500.0, 10):
                mod = tm.get_speed_modifier(float(x), float(y), hill_speed_factor=0.5)
                self.assertGreaterEqual(mod, 0.5 - 1e-6)
                self.assertLessEqual(mod, 1.0 + 1e-6)

    def test_invalid_hill_speed_factor_raises(self) -> None:
        tm = TerrainMap.flat(100.0, 100.0)
        with self.assertRaises(ValueError):
            tm.get_speed_modifier(50.0, 50.0, hill_speed_factor=0.0)
        with self.assertRaises(ValueError):
            tm.get_speed_modifier(50.0, 50.0, hill_speed_factor=-0.1)
        with self.assertRaises(ValueError):
            tm.get_speed_modifier(50.0, 50.0, hill_speed_factor=1.1)


# ---------------------------------------------------------------------------
# TerrainMap.generate_random
# ---------------------------------------------------------------------------


class TestGenerateRandom(unittest.TestCase):
    def _rng(self, seed: int = 0) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_returns_terrain_map_instance(self) -> None:
        tm = TerrainMap.generate_random(self._rng(), 1000.0, 1000.0)
        self.assertIsInstance(tm, TerrainMap)

    def test_dimensions_are_correct(self) -> None:
        tm = TerrainMap.generate_random(self._rng(), 500.0, 300.0)
        self.assertAlmostEqual(tm.width, 500.0)
        self.assertAlmostEqual(tm.height, 300.0)

    def test_grid_shape_matches_rows_cols(self) -> None:
        tm = TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, rows=10, cols=15)
        self.assertEqual(tm.elevation.shape, (10, 15))
        self.assertEqual(tm.cover.shape, (10, 15))

    def test_elevation_normalised_to_unit_interval(self) -> None:
        tm = TerrainMap.generate_random(self._rng(1), 1000.0, 1000.0, num_hills=5)
        self.assertGreaterEqual(float(tm.elevation.min()), 0.0)
        self.assertLessEqual(float(tm.elevation.max()), 1.0 + 1e-6)

    def test_cover_clipped_to_unit_interval(self) -> None:
        tm = TerrainMap.generate_random(self._rng(2), 1000.0, 1000.0, num_forests=5)
        self.assertGreaterEqual(float(tm.cover.min()), 0.0)
        self.assertLessEqual(float(tm.cover.max()), 1.0 + 1e-6)

    def test_same_seed_produces_same_terrain(self) -> None:
        tm_a = TerrainMap.generate_random(self._rng(42), 1000.0, 1000.0)
        tm_b = TerrainMap.generate_random(self._rng(42), 1000.0, 1000.0)
        np.testing.assert_array_equal(tm_a.elevation, tm_b.elevation)
        np.testing.assert_array_equal(tm_a.cover, tm_b.cover)

    def test_different_seeds_produce_different_terrain(self) -> None:
        tm_a = TerrainMap.generate_random(self._rng(1), 1000.0, 1000.0)
        tm_b = TerrainMap.generate_random(self._rng(2), 1000.0, 1000.0)
        # It is astronomically unlikely that two different seeds produce
        # identical elevation grids.
        self.assertFalse(np.array_equal(tm_a.elevation, tm_b.elevation))

    def test_zero_hills_produces_flat_elevation(self) -> None:
        tm = TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, num_hills=0)
        np.testing.assert_array_equal(tm.elevation, np.zeros_like(tm.elevation))

    def test_zero_forests_produces_zero_cover(self) -> None:
        tm = TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, num_forests=0)
        np.testing.assert_array_equal(tm.cover, np.zeros_like(tm.cover))

    def test_elevation_has_nonzero_values_with_hills(self) -> None:
        tm = TerrainMap.generate_random(self._rng(3), 1000.0, 1000.0, num_hills=3)
        self.assertGreater(float(tm.elevation.max()), 0.0)

    def test_cover_has_nonzero_values_with_forests(self) -> None:
        tm = TerrainMap.generate_random(self._rng(4), 1000.0, 1000.0, num_forests=3)
        self.assertGreater(float(tm.cover.max()), 0.0)

    def test_invalid_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, rows=0)

    def test_invalid_cols_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, cols=0)

    def test_negative_num_hills_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, num_hills=-1)

    def test_negative_num_forests_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, num_forests=-1)

    def test_invalid_forest_cover_raises(self) -> None:
        with self.assertRaises(ValueError):
            TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, forest_cover=-0.1)
        with self.assertRaises(ValueError):
            TerrainMap.generate_random(self._rng(), 1000.0, 1000.0, forest_cover=1.1)


if __name__ == "__main__":
    unittest.main()
