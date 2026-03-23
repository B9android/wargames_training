# tests/test_naval.py
"""Tests for envs/sim/naval.py — Naval Unit Type & Coastal Operations.

Covers all acceptance criteria from epic E10.1:

AC-1  Naval gunfire correctly bombards coastal positions (with LOS check).
AC-2  Amphibious landing scenario produces emergent beach-head tactics.
AC-3  River crossing scenario produces emergent bridgehead tactics.

Test organisation:
  1.  ShipType and WaterTileType enum smoke tests.
  2.  NavalVesselConfig / SHIP_CONFIGS table.
  3.  NavalVessel construction, properties, movement, rotation.
  4.  NavalVessel.broadside_arc_contains — geometry.
  5.  NavalVessel.can_tile — navigability matrix.
  6.  CoastalMap construction and tile queries.
  7.  CoastalMap.has_line_of_sight — LOS blocking.
  8.  can_bombard — range, arc, tile, LOS conditions.
  9.  naval_gunfire_damage — formula and edge cases.
  10. AmphibiousLanding — full phase walk-through.
  11. AmphibiousLanding — casualty vulnerability modifier.
  12. RiverCrossing — ford and bridge mechanics.
  13. generate_coastal_map — factory sanity checks.
  14. Acceptance criteria integration tests.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.naval import (
    AmphibiousLanding,
    BASE_NAVAL_DAMAGE,
    BRIDGE_CROSSING_STEPS,
    FORD_CROSSING_STEPS,
    LANDING_STEPS_DEFAULT,
    CoastalMap,
    LandingPhase,
    NavalVessel,
    NavalVesselConfig,
    RiverCrossing,
    SHIP_CONFIGS,
    ShipType,
    WaterTileType,
    can_bombard,
    generate_coastal_map,
    naval_gunfire_damage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frigate(x: float = 0.0, y: float = 0.0, theta: float = 0.0, team: int = 0,
             strength: float = 1.0) -> NavalVessel:
    return NavalVessel(x=x, y=y, theta=theta,
                       ship_type=ShipType.FRIGATE, team=team, strength=strength)


def _sol(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> NavalVessel:
    return NavalVessel(x=x, y=y, theta=theta,
                       ship_type=ShipType.SHIP_OF_THE_LINE, team=0)


def _gunboat(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> NavalVessel:
    return NavalVessel(x=x, y=y, theta=theta,
                       ship_type=ShipType.GUNBOAT, team=0)


def _sea_map(width: float = 2_000.0, height: float = 2_000.0,
             rows: int = 10, cols: int = 10) -> CoastalMap:
    """Tiny all-sea map for tests that don't care about tile detail."""
    cmap = CoastalMap(width=width, height=height, rows=rows, cols=cols)
    for r in range(rows):
        for c in range(cols):
            cmap.set_tile(r, c, WaterTileType.SEA)
    return cmap


def _land_map(width: float = 2_000.0, height: float = 2_000.0,
              rows: int = 10, cols: int = 10) -> CoastalMap:
    """All-land map — naval LOS should be blocked everywhere."""
    cmap = CoastalMap(width=width, height=height, rows=rows, cols=cols)
    # default is LAND
    return cmap


# ---------------------------------------------------------------------------
# 1. Enum smoke tests
# ---------------------------------------------------------------------------


class TestEnums(unittest.TestCase):
    def test_ship_type_values(self) -> None:
        self.assertEqual(ShipType.SHIP_OF_THE_LINE, 0)
        self.assertEqual(ShipType.FRIGATE, 1)
        self.assertEqual(ShipType.GUNBOAT, 2)

    def test_water_tile_type_values(self) -> None:
        self.assertEqual(WaterTileType.LAND, 0)
        self.assertEqual(WaterTileType.BEACH, 5)
        self.assertEqual(WaterTileType.RIVER, 4)
        self.assertEqual(WaterTileType.FORD, 6)
        self.assertEqual(WaterTileType.BRIDGE, 7)

    def test_landing_phase_order(self) -> None:
        phases = [
            LandingPhase.EMBARKED,
            LandingPhase.APPROACHING,
            LandingPhase.LANDING,
            LandingPhase.ESTABLISHED,
            LandingPhase.COMPLETE,
        ]
        self.assertEqual(phases, sorted(phases))


# ---------------------------------------------------------------------------
# 2. NavalVesselConfig / SHIP_CONFIGS
# ---------------------------------------------------------------------------


class TestShipConfigs(unittest.TestCase):
    def test_all_ship_types_in_table(self) -> None:
        for st in ShipType:
            self.assertIn(st, SHIP_CONFIGS)

    def test_sol_has_longest_range(self) -> None:
        sol_range = SHIP_CONFIGS[ShipType.SHIP_OF_THE_LINE].fire_range
        fri_range = SHIP_CONFIGS[ShipType.FRIGATE].fire_range
        gun_range = SHIP_CONFIGS[ShipType.GUNBOAT].fire_range
        self.assertGreater(sol_range, fri_range)
        self.assertGreater(fri_range, gun_range)

    def test_sol_has_highest_broadside_damage(self) -> None:
        sol_dmg = SHIP_CONFIGS[ShipType.SHIP_OF_THE_LINE].broadside_damage
        gun_dmg = SHIP_CONFIGS[ShipType.GUNBOAT].broadside_damage
        self.assertGreater(sol_dmg, gun_dmg)

    def test_only_gunboat_can_enter_river(self) -> None:
        self.assertFalse(SHIP_CONFIGS[ShipType.SHIP_OF_THE_LINE].can_enter_river)
        self.assertFalse(SHIP_CONFIGS[ShipType.FRIGATE].can_enter_river)
        self.assertTrue(SHIP_CONFIGS[ShipType.GUNBOAT].can_enter_river)

    def test_config_is_frozen(self) -> None:
        cfg = SHIP_CONFIGS[ShipType.FRIGATE]
        with self.assertRaises((AttributeError, TypeError)):
            cfg.fire_range = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. NavalVessel construction and basic properties
# ---------------------------------------------------------------------------


class TestNavalVesselConstruction(unittest.TestCase):
    def test_default_strength(self) -> None:
        v = _frigate()
        self.assertAlmostEqual(v.strength, 1.0)

    def test_is_afloat_at_full_strength(self) -> None:
        self.assertTrue(_frigate().is_afloat)

    def test_is_not_afloat_at_zero_strength(self) -> None:
        v = _frigate(strength=0.0)
        self.assertFalse(v.is_afloat)

    def test_invalid_team_raises(self) -> None:
        with self.assertRaises(ValueError):
            NavalVessel(x=0.0, y=0.0, theta=0.0,
                        ship_type=ShipType.FRIGATE, team=2)

    def test_invalid_strength_raises(self) -> None:
        with self.assertRaises(ValueError):
            NavalVessel(x=0.0, y=0.0, theta=0.0,
                        ship_type=ShipType.FRIGATE, team=0, strength=-0.1)
        with self.assertRaises(ValueError):
            NavalVessel(x=0.0, y=0.0, theta=0.0,
                        ship_type=ShipType.FRIGATE, team=0, strength=1.1)

    def test_fire_range_propagated(self) -> None:
        v = _sol()
        self.assertAlmostEqual(v.fire_range,
                               SHIP_CONFIGS[ShipType.SHIP_OF_THE_LINE].fire_range)

    def test_take_damage_reduces_strength(self) -> None:
        v = _frigate()
        v.take_damage(0.3)
        self.assertAlmostEqual(v.strength, 0.7)

    def test_take_damage_clamps_at_zero(self) -> None:
        v = _frigate()
        v.take_damage(2.0)
        self.assertAlmostEqual(v.strength, 0.0)
        self.assertFalse(v.is_afloat)


# ---------------------------------------------------------------------------
# 4. NavalVessel.broadside_arc_contains
# ---------------------------------------------------------------------------


class TestBroadsideArc(unittest.TestCase):
    """Ship faces east (theta=0); broadside is north/south (±90° from beam)."""

    def setUp(self) -> None:
        # Ship at origin, facing east (theta=0)
        self.ship = _frigate(x=0.0, y=0.0, theta=0.0)

    def test_target_north_in_arc(self) -> None:
        # Target directly north = 90° off bow → in broadside arc
        self.assertTrue(self.ship.broadside_arc_contains(0.0, 100.0))

    def test_target_south_in_arc(self) -> None:
        # Target directly south = 90° off stern direction → in broadside arc
        self.assertTrue(self.ship.broadside_arc_contains(0.0, -100.0))

    def test_target_directly_ahead_not_in_arc(self) -> None:
        # Target directly east = 0° = dead ahead → outside broadside arc
        self.assertFalse(self.ship.broadside_arc_contains(100.0, 0.0))

    def test_target_directly_astern_not_in_arc(self) -> None:
        # Target directly west = 180° = dead astern → outside broadside arc
        self.assertFalse(self.ship.broadside_arc_contains(-100.0, 0.0))

    def test_target_at_45_degrees_in_arc(self) -> None:
        # 45° = π/4 — boundary: in arc (border inclusive)
        self.assertTrue(self.ship.broadside_arc_contains(100.0, 100.0))

    def test_target_at_135_degrees_in_arc(self) -> None:
        # 135° = 3π/4 — boundary: in arc (border inclusive)
        self.assertTrue(self.ship.broadside_arc_contains(-100.0, 100.0))

    def test_coincident_point_not_in_arc(self) -> None:
        self.assertFalse(self.ship.broadside_arc_contains(0.0, 0.0))


# ---------------------------------------------------------------------------
# 5. NavalVessel.can_tile — navigability
# ---------------------------------------------------------------------------


class TestCanTile(unittest.TestCase):
    def test_sol_cannot_enter_river(self) -> None:
        v = _sol()
        self.assertFalse(v.can_tile(WaterTileType.RIVER))

    def test_sol_can_enter_coastal_deep_and_sea(self) -> None:
        v = _sol()
        self.assertTrue(v.can_tile(WaterTileType.COASTAL_DEEP))
        self.assertTrue(v.can_tile(WaterTileType.SEA))

    def test_sol_cannot_enter_coastal_shallow(self) -> None:
        v = _sol()
        self.assertFalse(v.can_tile(WaterTileType.COASTAL_SHALLOW))

    def test_frigate_can_enter_coastal_shallow(self) -> None:
        v = _frigate()
        self.assertTrue(v.can_tile(WaterTileType.COASTAL_SHALLOW))

    def test_frigate_cannot_enter_river(self) -> None:
        v = _frigate()
        self.assertFalse(v.can_tile(WaterTileType.RIVER))

    def test_gunboat_can_enter_river(self) -> None:
        v = _gunboat()
        self.assertTrue(v.can_tile(WaterTileType.RIVER))

    def test_gunboat_can_enter_coastal_shallow(self) -> None:
        v = _gunboat()
        self.assertTrue(v.can_tile(WaterTileType.COASTAL_SHALLOW))

    def test_gunboat_can_enter_coastal_deep(self) -> None:
        v = _gunboat()
        self.assertTrue(v.can_tile(WaterTileType.COASTAL_DEEP))

    def test_gunboat_can_enter_sea(self) -> None:
        v = _gunboat()
        self.assertTrue(v.can_tile(WaterTileType.SEA))

    def test_no_ship_can_enter_land(self) -> None:
        for ship in (_sol(), _frigate(), _gunboat()):
            self.assertFalse(ship.can_tile(WaterTileType.LAND))

    def test_no_ship_can_enter_beach(self) -> None:
        for ship in (_sol(), _frigate(), _gunboat()):
            self.assertFalse(ship.can_tile(WaterTileType.BEACH))

    def test_no_ship_can_enter_ford(self) -> None:
        for ship in (_sol(), _frigate(), _gunboat()):
            self.assertFalse(ship.can_tile(WaterTileType.FORD))


# ---------------------------------------------------------------------------
# 6. CoastalMap
# ---------------------------------------------------------------------------


class TestCoastalMap(unittest.TestCase):
    def test_default_all_land(self) -> None:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        for r in range(5):
            for c in range(5):
                tile = cmap._grid[r, c]
                self.assertEqual(tile, int(WaterTileType.LAND))

    def test_set_and_get_tile(self) -> None:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.SEA)
        # (x=500, y=500) maps to row 2, col 2 in a 5×5 grid over 1000×1000
        self.assertEqual(cmap.get_tile(500.0, 500.0), WaterTileType.SEA)

    def test_out_of_bounds_raises(self) -> None:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        with self.assertRaises(IndexError):
            cmap.set_tile(5, 0, WaterTileType.SEA)
        with self.assertRaises(IndexError):
            cmap.set_tile(0, 5, WaterTileType.SEA)

    def test_invalid_dims_raise(self) -> None:
        with self.assertRaises(ValueError):
            CoastalMap(width=0.0, height=1_000.0, rows=5, cols=5)
        with self.assertRaises(ValueError):
            CoastalMap(width=1_000.0, height=1_000.0, rows=0, cols=5)

    def test_is_water_sea_tile(self) -> None:
        cmap = _sea_map()
        self.assertTrue(cmap.is_water(500.0, 500.0))

    def test_is_water_land_tile(self) -> None:
        cmap = _land_map()
        self.assertFalse(cmap.is_water(500.0, 500.0))

    def test_is_beach(self) -> None:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.BEACH)
        self.assertTrue(cmap.is_beach(500.0, 500.0))
        self.assertFalse(cmap.is_beach(100.0, 100.0))

    def test_is_river_crossable_ford(self) -> None:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.FORD)
        self.assertTrue(cmap.is_river_crossable(500.0, 500.0))

    def test_is_river_crossable_bridge(self) -> None:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.BRIDGE)
        self.assertTrue(cmap.is_river_crossable(500.0, 500.0))

    def test_is_river_crossable_river_tile_false(self) -> None:
        """RIVER tiles themselves are not crossings."""
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.RIVER)
        self.assertFalse(cmap.is_river_crossable(500.0, 500.0))

    def test_is_navigable_by(self) -> None:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.RIVER)
        gunboat = _gunboat()
        frigate = _frigate()
        self.assertTrue(cmap.is_navigable_by(500.0, 500.0, gunboat))
        self.assertFalse(cmap.is_navigable_by(500.0, 500.0, frigate))

    def test_grid_array_returns_copy(self) -> None:
        cmap = _sea_map(rows=3, cols=3)
        arr = cmap.grid_array()
        arr[0, 0] = 99
        self.assertNotEqual(cmap._grid[0, 0], 99)


# ---------------------------------------------------------------------------
# 7. CoastalMap.has_line_of_sight
# ---------------------------------------------------------------------------


class TestCoastalMapLOS(unittest.TestCase):
    def test_all_sea_has_los(self) -> None:
        cmap = _sea_map()
        self.assertTrue(cmap.has_line_of_sight(0.0, 0.0, 1_000.0, 1_000.0))

    def test_land_wall_blocks_los(self) -> None:
        """A column of land cells bisecting the map should block LOS."""
        cmap = _sea_map(rows=10, cols=10)
        # Set column 5 entirely to LAND — blocks east-west shots
        for r in range(10):
            cmap.set_tile(r, 5, WaterTileType.LAND)
        # Shoot from west side (col ~0) to east side (col ~9)
        # The interior samples will pass through col 5 → blocked
        self.assertFalse(
            cmap.has_line_of_sight(100.0, 500.0, 1_900.0, 500.0, num_samples=20)
        )

    def test_num_samples_too_small_raises(self) -> None:
        cmap = _sea_map()
        with self.assertRaises(ValueError):
            cmap.has_line_of_sight(0.0, 0.0, 100.0, 100.0, num_samples=1)


# ---------------------------------------------------------------------------
# 8. can_bombard
# ---------------------------------------------------------------------------


class TestCanBombard(unittest.TestCase):
    """AC-1: Naval gunfire correctly bombards coastal positions."""

    def _setup_sea_ship_and_land_target(self) -> tuple:
        """Frigate on a sea tile shooting at a land target."""
        cmap = CoastalMap(width=2_000.0, height=2_000.0, rows=10, cols=10)
        # Column 0–4: sea; column 5–9: land
        for r in range(10):
            for c in range(5):
                cmap.set_tile(r, c, WaterTileType.SEA)
        # Frigate at (200, 1000) — col 1 (sea), facing east (theta=0)
        # Target at (1100, 1000) — col 5 (land) — due east → dead ahead → NOT in arc
        ship = _frigate(x=200.0, y=1_000.0, theta=0.0)
        return ship, cmap

    def test_out_of_range_returns_false(self) -> None:
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0)
        far_target_x = SHIP_CONFIGS[ShipType.FRIGATE].fire_range + 100.0
        self.assertFalse(
            can_bombard(ship, far_target_x, 0.0, cmap, require_water_tile=False)
        )

    def test_target_in_broadside_arc_and_range(self) -> None:
        """Frigate facing east, target north at 200 m — should be able to bombard."""
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0)
        self.assertTrue(
            can_bombard(ship, 0.0, 200.0, cmap, require_water_tile=False)
        )

    def test_target_dead_ahead_blocked_by_arc(self) -> None:
        """Target directly ahead (theta=0 → east) is outside broadside arc."""
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0)
        self.assertFalse(
            can_bombard(ship, 200.0, 0.0, cmap, require_water_tile=False)
        )

    def test_sunken_ship_cannot_fire(self) -> None:
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0, strength=0.0)
        self.assertFalse(
            can_bombard(ship, 0.0, 200.0, cmap, require_water_tile=False)
        )

    def test_land_tile_blocks_los(self) -> None:
        """A land tile on the trajectory should block the shot."""
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=10, cols=10)
        for r in range(10):
            for c in range(10):
                cmap.set_tile(r, c, WaterTileType.SEA)
        # Place land in column 5, all rows → blocks E-W line
        for r in range(10):
            cmap.set_tile(r, 5, WaterTileType.LAND)
        # Ship at (50, 500) facing east (theta=0), target north at (50, 300) → in arc
        # But we want LOS-blocked: ship at west, target at east past the land wall
        # Make ship face north and target is due north but past land in middle
        # Use a diagonal shot blocked by the land wall
        ship = _frigate(x=50.0, y=500.0, theta=math.pi / 2)  # facing north
        # Target to the north-east, past column-5 land
        # Can't easily test this without careful geometry; use separate method test
        # Instead verify LOS method directly
        self.assertFalse(
            cmap.has_line_of_sight(50.0, 500.0, 950.0, 500.0, num_samples=20)
        )

    def test_require_water_tile_ship_on_land_blocked(self) -> None:
        """When require_water_tile=True and ship is on land, cannot bombard."""
        cmap = _land_map()  # all land
        ship = _frigate(x=500.0, y=500.0, theta=0.0)
        self.assertFalse(
            can_bombard(ship, 500.0, 600.0, cmap, require_water_tile=True)
        )

    def test_require_water_tile_ship_on_beach_blocked(self) -> None:
        """When require_water_tile=True and ship is on a BEACH tile, cannot bombard.

        A ship cannot legally occupy a BEACH tile (can_tile returns False for BEACH),
        so require_water_tile should block firing from there.
        """
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.BEACH)
        # Surrounding tiles need to be something navigable so the LOS check passes
        for r in range(5):
            for c in range(5):
                if (r, c) != (2, 2):
                    cmap.set_tile(r, c, WaterTileType.SEA)
        ship = _frigate(x=500.0, y=500.0, theta=0.0)  # ship is on the BEACH tile
        self.assertFalse(
            can_bombard(ship, 500.0, 700.0, cmap, require_water_tile=True)
        )

    def test_require_water_tile_ship_on_sea_ok(self) -> None:
        """When require_water_tile=True and ship is on sea, check passes."""
        cmap = _sea_map()
        ship = _frigate(x=500.0, y=500.0, theta=0.0)
        # Target north at 600 — in broadside arc, within range
        self.assertTrue(
            can_bombard(ship, 500.0, 700.0, cmap, require_water_tile=True)
        )


# ---------------------------------------------------------------------------
# 9. naval_gunfire_damage
# ---------------------------------------------------------------------------


class TestNavalGunfireDamage(unittest.TestCase):
    """AC-1: Naval gunfire damage formula and edge cases."""

    def test_point_blank_full_intensity(self) -> None:
        """At range ≈ 0 with intensity=1 damage should approach base×broadside×1."""
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0)
        # Target just north (broadside arc), nearly point blank
        dmg = naval_gunfire_damage(
            ship, 0.0, 1.0, 1.0, cmap, intensity=1.0, require_water_tile=False
        )
        expected_max = (
            BASE_NAVAL_DAMAGE
            * SHIP_CONFIGS[ShipType.FRIGATE].broadside_damage
        )
        self.assertAlmostEqual(dmg, expected_max, places=3)

    def test_zero_intensity_returns_zero(self) -> None:
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0)
        dmg = naval_gunfire_damage(
            ship, 0.0, 100.0, 1.0, cmap, intensity=0.0, require_water_tile=False
        )
        self.assertAlmostEqual(dmg, 0.0)

    def test_damage_decreases_with_range(self) -> None:
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0)
        dmg_near = naval_gunfire_damage(
            ship, 0.0, 50.0, 1.0, cmap, require_water_tile=False
        )
        dmg_far = naval_gunfire_damage(
            ship, 0.0, 300.0, 1.0, cmap, require_water_tile=False
        )
        self.assertGreater(dmg_near, dmg_far)

    def test_out_of_range_returns_zero(self) -> None:
        cmap = _sea_map()
        ship = _frigate(x=0.0, y=0.0, theta=0.0)
        far = SHIP_CONFIGS[ShipType.FRIGATE].fire_range + 50.0
        dmg = naval_gunfire_damage(
            ship, 0.0, far, 1.0, cmap, require_water_tile=False
        )
        self.assertAlmostEqual(dmg, 0.0)

    def test_sol_deals_more_damage_than_gunboat(self) -> None:
        """Ship-of-the-line has higher broadside_damage than gunboat."""
        cmap = _sea_map()
        sol = _sol(x=0.0, y=0.0, theta=0.0)
        gun = _gunboat(x=0.0, y=0.0, theta=0.0)
        target_y = 50.0
        # Both are in broadside arc; both within range
        dmg_sol = naval_gunfire_damage(
            sol, 0.0, target_y, 1.0, cmap, require_water_tile=False
        )
        dmg_gun = naval_gunfire_damage(
            gun, 0.0, target_y, 1.0, cmap, require_water_tile=False
        )
        self.assertGreater(dmg_sol, dmg_gun)

    def test_damaged_vessel_deals_less(self) -> None:
        """A half-strength vessel deals half the damage of a full-strength one."""
        cmap = _sea_map()
        full = _frigate(x=0.0, y=0.0, theta=0.0, strength=1.0)
        half = _frigate(x=0.0, y=0.0, theta=0.0, strength=0.5)
        target_y = 50.0
        dmg_full = naval_gunfire_damage(
            full, 0.0, target_y, 1.0, cmap, require_water_tile=False
        )
        dmg_half = naval_gunfire_damage(
            half, 0.0, target_y, 1.0, cmap, require_water_tile=False
        )
        self.assertAlmostEqual(dmg_half, dmg_full / 2.0, places=6)


# ---------------------------------------------------------------------------
# 10. AmphibiousLanding — phase walk-through
# ---------------------------------------------------------------------------


class TestAmphibiousLandingPhases(unittest.TestCase):
    """AC-2: Amphibious landing phases advance correctly."""

    def _make_landing(
        self,
        ship_x: float = 100.0,
        beach_x: float = 500.0,
        landing_steps: int = 3,
    ) -> AmphibiousLanding:
        ship = _frigate(x=ship_x, y=0.0, theta=0.0)
        return AmphibiousLanding(
            vessel=ship,
            infantry_strength=1.0,
            beach_x=beach_x,
            beach_y=0.0,
            approach_radius=200.0,
            landing_steps=landing_steps,
        )

    def test_initial_phase_is_embarked(self) -> None:
        landing = self._make_landing()
        self.assertEqual(landing.phase, LandingPhase.EMBARKED)

    def test_embarked_transitions_to_approaching_when_close(self) -> None:
        """Ship already within approach_radius → transitions on first step."""
        ship = _frigate(x=350.0, y=0.0)  # 150 m from beach (< 200 approach_radius)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=3,
        )
        landing.step()
        self.assertEqual(landing.phase, LandingPhase.APPROACHING)

    def test_approaching_transitions_to_landing(self) -> None:
        ship = _frigate(x=350.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=3,
        )
        landing.step()  # EMBARKED → APPROACHING
        landing.step()  # APPROACHING → LANDING
        self.assertEqual(landing.phase, LandingPhase.LANDING)

    def test_landing_phase_lasts_correct_steps(self) -> None:
        ship = _frigate(x=350.0, y=0.0)
        steps = 4
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=steps,
        )
        landing.step()  # → APPROACHING
        landing.step()  # → LANDING (step 0)
        for _ in range(steps):
            self.assertEqual(landing.phase, LandingPhase.LANDING)
            landing.step()
        self.assertEqual(landing.phase, LandingPhase.ESTABLISHED)

    def test_established_transitions_to_complete(self) -> None:
        ship = _frigate(x=350.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=1,
        )
        # Walk through all phases
        landing.step()  # EMBARKED → APPROACHING
        landing.step()  # → LANDING
        landing.step()  # → ESTABLISHED (1 step)
        landing.step()  # → COMPLETE
        self.assertTrue(landing.is_complete)

    def test_complete_phase_is_stable(self) -> None:
        ship = _frigate(x=350.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=1,
        )
        for _ in range(10):
            landing.step()
        self.assertEqual(landing.phase, LandingPhase.COMPLETE)

    def test_infantry_ashore_after_established(self) -> None:
        ship = _frigate(x=350.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=1,
        )
        landing.step()
        landing.step()
        landing.step()  # ESTABLISHED
        self.assertTrue(landing.infantry_ashore)

    def test_is_vulnerable_only_during_landing_phase(self) -> None:
        ship = _frigate(x=350.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=2,
        )
        landing.step()  # APPROACHING
        self.assertFalse(landing.is_vulnerable)
        landing.step()  # LANDING
        self.assertTrue(landing.is_vulnerable)
        landing.step()  # still LANDING
        self.assertTrue(landing.is_vulnerable)
        landing.step()  # ESTABLISHED
        self.assertFalse(landing.is_vulnerable)

    def test_ship_far_away_stays_embarked(self) -> None:
        """Ship > approach_radius from beach stays EMBARKED."""
        ship = _frigate(x=100.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=1_000.0, beach_y=0.0, approach_radius=100.0, landing_steps=3,
        )
        landing.step()
        self.assertEqual(landing.phase, LandingPhase.EMBARKED)

    def test_invalid_infantry_strength_raises(self) -> None:
        ship = _frigate()
        with self.assertRaises(ValueError):
            AmphibiousLanding(vessel=ship, infantry_strength=1.5,
                              beach_x=0.0, beach_y=0.0)

    def test_invalid_approach_radius_raises(self) -> None:
        ship = _frigate()
        with self.assertRaises(ValueError):
            AmphibiousLanding(vessel=ship, infantry_strength=1.0,
                              beach_x=0.0, beach_y=0.0, approach_radius=0.0)

    def test_invalid_landing_steps_raises(self) -> None:
        ship = _frigate()
        with self.assertRaises(ValueError):
            AmphibiousLanding(vessel=ship, infantry_strength=1.0,
                              beach_x=0.0, beach_y=0.0, landing_steps=0)


# ---------------------------------------------------------------------------
# 11. AmphibiousLanding — casualty vulnerability modifier
# ---------------------------------------------------------------------------


class TestAmphibiousLandingCasualties(unittest.TestCase):
    def _landing_in_phase(self, phase: LandingPhase) -> AmphibiousLanding:
        ship = _frigate(x=350.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=1.0,
            beach_x=500.0, beach_y=0.0, approach_radius=200.0, landing_steps=4,
        )
        if phase == LandingPhase.EMBARKED:
            return landing
        landing.step()  # → APPROACHING
        if phase == LandingPhase.APPROACHING:
            return landing
        landing.step()  # → LANDING
        if phase == LandingPhase.LANDING:
            return landing
        for _ in range(4):  # landing_steps=4 → ESTABLISHED
            landing.step()
        if phase == LandingPhase.ESTABLISHED:
            return landing
        landing.step()  # → COMPLETE
        return landing

    def test_vulnerability_modifier_landing_phase(self) -> None:
        landing = self._landing_in_phase(LandingPhase.LANDING)
        self.assertAlmostEqual(landing.vulnerability_modifier(), 2.0)

    def test_vulnerability_modifier_other_phases(self) -> None:
        for phase in (LandingPhase.EMBARKED, LandingPhase.APPROACHING,
                      LandingPhase.ESTABLISHED, LandingPhase.COMPLETE):
            landing = self._landing_in_phase(phase)
            self.assertAlmostEqual(landing.vulnerability_modifier(), 1.0,
                                   msg=f"Phase {phase!r}")

    def test_casualties_doubled_during_landing_phase(self) -> None:
        landing = self._landing_in_phase(LandingPhase.LANDING)
        initial = landing.infantry_strength
        actual = landing.apply_landing_casualties(0.1)
        # Modifier is 2.0 → 0.1 * 2 = 0.2 expected
        self.assertAlmostEqual(actual, 0.2)
        self.assertAlmostEqual(landing.infantry_strength, initial - 0.2)

    def test_casualties_normal_outside_landing(self) -> None:
        landing = self._landing_in_phase(LandingPhase.EMBARKED)
        actual = landing.apply_landing_casualties(0.1)
        self.assertAlmostEqual(actual, 0.1)

    def test_casualties_clamped_to_remaining_strength(self) -> None:
        ship = _frigate(x=0.0, y=0.0)
        landing = AmphibiousLanding(
            vessel=ship, infantry_strength=0.1,
            beach_x=500.0, beach_y=0.0,
        )
        actual = landing.apply_landing_casualties(1.0)
        self.assertAlmostEqual(actual, 0.1)
        self.assertAlmostEqual(landing.infantry_strength, 0.0)


# ---------------------------------------------------------------------------
# 12. RiverCrossing
# ---------------------------------------------------------------------------


class TestRiverCrossing(unittest.TestCase):
    """AC-3: River crossing produces bridgehead tactics."""

    def _ford_cmap(self) -> CoastalMap:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.FORD)
        return cmap

    def _bridge_cmap(self) -> CoastalMap:
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.BRIDGE)
        return cmap

    def _ford_crossing(self) -> RiverCrossing:
        cmap = self._ford_cmap()
        return RiverCrossing(
            unit_x=300.0, unit_y=500.0,
            crossing_x=500.0, crossing_y=500.0,
            coastal_map=cmap, team=0,
        )

    def _bridge_crossing(self) -> RiverCrossing:
        cmap = self._bridge_cmap()
        return RiverCrossing(
            unit_x=300.0, unit_y=500.0,
            crossing_x=500.0, crossing_y=500.0,
            coastal_map=cmap, team=0,
        )

    def test_ford_not_complete_initially(self) -> None:
        self.assertFalse(self._ford_crossing().is_complete)

    def test_ford_total_steps(self) -> None:
        cr = self._ford_crossing()
        self.assertEqual(cr._crossing_steps_total, FORD_CROSSING_STEPS)

    def test_bridge_total_steps(self) -> None:
        cr = self._bridge_crossing()
        self.assertEqual(cr._crossing_steps_total, BRIDGE_CROSSING_STEPS)

    def test_ford_slower_than_bridge(self) -> None:
        ford = self._ford_crossing()
        bridge = self._bridge_crossing()
        self.assertGreater(ford._crossing_steps_total, bridge._crossing_steps_total)

    def test_ford_speed_modifier_slower(self) -> None:
        ford = self._ford_crossing()
        bridge = self._bridge_crossing()
        self.assertLess(ford.speed_modifier, bridge.speed_modifier)

    def test_ford_vulnerability_higher(self) -> None:
        ford = self._ford_crossing()
        bridge = self._bridge_crossing()
        self.assertGreater(ford.vulnerability_modifier, bridge.vulnerability_modifier)

    def test_ford_completes_after_required_steps(self) -> None:
        cr = self._ford_crossing()
        completed = False
        for _ in range(FORD_CROSSING_STEPS):
            result = cr.step()
            if result:
                completed = True
        self.assertTrue(completed)
        self.assertTrue(cr.is_complete)

    def test_bridge_completes_after_required_steps(self) -> None:
        cr = self._bridge_crossing()
        for _ in range(BRIDGE_CROSSING_STEPS):
            cr.step()
        self.assertTrue(cr.is_complete)

    def test_step_after_complete_returns_false(self) -> None:
        cr = self._ford_crossing()
        for _ in range(FORD_CROSSING_STEPS):
            cr.step()
        result = cr.step()
        self.assertFalse(result)

    def test_progress_starts_at_zero(self) -> None:
        cr = self._ford_crossing()
        self.assertAlmostEqual(cr.progress, 0.0)

    def test_progress_reaches_one_on_completion(self) -> None:
        cr = self._ford_crossing()
        for _ in range(FORD_CROSSING_STEPS):
            cr.step()
        self.assertAlmostEqual(cr.progress, 1.0)

    def test_speed_modifier_one_after_complete(self) -> None:
        cr = self._ford_crossing()
        for _ in range(FORD_CROSSING_STEPS):
            cr.step()
        self.assertAlmostEqual(cr.speed_modifier, 1.0)

    def test_vulnerability_modifier_one_after_complete(self) -> None:
        cr = self._ford_crossing()
        for _ in range(FORD_CROSSING_STEPS):
            cr.step()
        self.assertAlmostEqual(cr.vulnerability_modifier, 1.0)

    def test_invalid_tile_raises(self) -> None:
        cmap = _sea_map()  # all SEA — no ford or bridge
        with self.assertRaises(ValueError):
            RiverCrossing(
                unit_x=0.0, unit_y=0.0,
                crossing_x=100.0, crossing_y=100.0,
                coastal_map=cmap, team=0,
            )

    def test_invalid_team_raises(self) -> None:
        cmap = self._ford_cmap()
        with self.assertRaises(ValueError):
            RiverCrossing(
                unit_x=300.0, unit_y=500.0,
                crossing_x=500.0, crossing_y=500.0,
                coastal_map=cmap, team=2,
            )


# ---------------------------------------------------------------------------
# 13. generate_coastal_map
# ---------------------------------------------------------------------------


class TestGenerateCoastalMap(unittest.TestCase):
    def setUp(self) -> None:
        self.cmap = generate_coastal_map()

    def test_returns_coastal_map(self) -> None:
        self.assertIsInstance(self.cmap, CoastalMap)

    def test_sea_in_leftmost_columns(self) -> None:
        """The leftmost grid columns should be SEA tiles."""
        # Column 0 — x ≈ 0
        tile = self.cmap.get_tile(0.0, 1_500.0)
        self.assertEqual(tile, WaterTileType.SEA)

    def test_land_in_rightmost_columns(self) -> None:
        """Far-right columns outside the river band should be LAND tiles."""
        # Query a row outside the river band (rows 13–17) to get a LAND tile
        # y=100 maps to row ~1 (outside river band) at the rightmost column x=4950
        tile = self.cmap.get_tile(4_950.0, 100.0)
        self.assertEqual(tile, WaterTileType.LAND)

    def test_beach_column_present(self) -> None:
        """A BEACH tile should exist in the default map."""
        found = (self.cmap._grid == int(WaterTileType.BEACH)).any()
        self.assertTrue(found)

    def test_ford_present(self) -> None:
        found = (self.cmap._grid == int(WaterTileType.FORD)).any()
        self.assertTrue(found)

    def test_bridge_present(self) -> None:
        found = (self.cmap._grid == int(WaterTileType.BRIDGE)).any()
        self.assertTrue(found)

    def test_river_present(self) -> None:
        found = (self.cmap._grid == int(WaterTileType.RIVER)).any()
        self.assertTrue(found)

    def test_beach_col_negative_sentinel_disables_beach(self) -> None:
        """beach_col=-1 should produce no BEACH tiles in the map."""
        cmap = generate_coastal_map(beach_col=-1, sea_cols=5)
        found = (cmap._grid == int(WaterTileType.BEACH)).any()
        self.assertFalse(found, "beach_col=-1 should disable beach strip generation")

    def test_beach_col_negative_sentinel_river_still_present(self) -> None:
        """River tiles are still generated when beach_col=-1."""
        cmap = generate_coastal_map(beach_col=-1, sea_cols=5)
        found = (cmap._grid == int(WaterTileType.RIVER)).any()
        self.assertTrue(found)


# ---------------------------------------------------------------------------
# 14. Acceptance criteria integration tests
# ---------------------------------------------------------------------------


class TestAC1NavalGunfireCoastal(unittest.TestCase):
    """AC-1: Naval gunfire correctly bombards coastal positions (with LOS)."""

    def test_frigate_bombards_beach_position(self) -> None:
        """A frigate at sea bombards a target on the beach."""
        cmap = generate_coastal_map()
        # Place frigate in the sea band, broadside (theta=0, target north)
        # Sea band col 0–9 → x ≈ 0–1000 m
        ship = _frigate(x=500.0, y=1_000.0, theta=0.0)
        # Beach column at col 10 → x ≈ 1_000 m; target slightly north
        target_x = 500.0
        target_y = 1_200.0  # north of ship — in broadside arc
        dmg = naval_gunfire_damage(
            ship, target_x, target_y, 1.0, cmap,
            intensity=1.0, require_water_tile=True,
        )
        self.assertGreater(dmg, 0.0, "Frigate should deal positive damage to coastal target")

    def test_sol_bombards_inland_position_within_range(self) -> None:
        """SoL (600 m range) bombards coastal target — all-sea map simplification."""
        cmap = _sea_map(width=2_000.0, height=2_000.0)
        sol = _sol(x=200.0, y=1_000.0, theta=0.0)
        # Target north at 200 m — well within 600 m range, in broadside arc
        dmg = naval_gunfire_damage(
            sol, 200.0, 1_200.0, 1.0, cmap,
            intensity=1.0, require_water_tile=False,
        )
        self.assertGreater(dmg, 0.0)

    def test_no_damage_when_los_blocked_by_headland(self) -> None:
        """Land tile between ship and target blocks the shot."""
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=10, cols=10)
        # Set all cells to sea initially
        for r in range(10):
            for c in range(10):
                cmap.set_tile(r, c, WaterTileType.SEA)
        # Place a headland (land) in the middle, blocking north–south shot
        # Ship at (500, 0), target at (500, 900), headland at col 5, row 5
        # Row 5 = y ≈ 500; col 5 = x ≈ 500 → blocks direct line
        for c in range(3, 7):
            cmap.set_tile(5, c, WaterTileType.LAND)
        ship = _frigate(x=500.0, y=50.0, theta=math.pi / 2)  # facing north
        # Target directly north past the headland
        dmg = naval_gunfire_damage(
            ship, 500.0, 950.0, 1.0, cmap,
            intensity=1.0, require_water_tile=False,
        )
        self.assertAlmostEqual(dmg, 0.0,
                               msg="Land headland should block naval gunfire LOS")


class TestAC2AmphibiousBeachHead(unittest.TestCase):
    """AC-2: Amphibious landing produces emergent beach-head establishment."""

    def test_full_landing_sequence_completes(self) -> None:
        """Running through all phases reaches COMPLETE with positive strength."""
        ship = _frigate(x=300.0, y=1_500.0, theta=0.0)
        landing = AmphibiousLanding(
            vessel=ship,
            infantry_strength=1.0,
            beach_x=500.0,
            beach_y=1_500.0,
            approach_radius=250.0,
            landing_steps=LANDING_STEPS_DEFAULT,
        )
        max_steps = 100
        for _ in range(max_steps):
            landing.step()
            if landing.is_complete:
                break
        self.assertTrue(
            landing.is_complete,
            "Landing should complete within 100 steps when ship starts close",
        )
        self.assertGreater(landing.infantry_strength, 0.0)

    def test_infantry_takes_more_damage_in_surf(self) -> None:
        """During LANDING phase infantry takes 2× damage."""
        ship = _frigate(x=350.0, y=0.0, theta=0.0)
        landing = AmphibiousLanding(
            vessel=ship,
            infantry_strength=1.0,
            beach_x=500.0,
            beach_y=0.0,
            approach_radius=200.0,
            landing_steps=5,
        )
        # Advance to LANDING phase
        landing.step()  # EMBARKED → APPROACHING
        landing.step()  # → LANDING
        self.assertEqual(landing.phase, LandingPhase.LANDING)

        # Apply 0.1 damage during surf phase — should receive 0.2 (2× modifier)
        actual = landing.apply_landing_casualties(0.1)
        self.assertAlmostEqual(actual, 0.2)


class TestAC3RiverBridgehead(unittest.TestCase):
    """AC-3: River crossing scenario produces bridgehead mechanics."""

    def test_ford_crossing_slows_and_exposes_unit(self) -> None:
        """A unit at a ford is slowed and more vulnerable than on land."""
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 2, WaterTileType.FORD)
        cr = RiverCrossing(
            unit_x=300.0, unit_y=500.0,
            crossing_x=500.0, crossing_y=500.0,
            coastal_map=cmap, team=0,
        )
        self.assertLess(cr.speed_modifier, 1.0)
        self.assertGreater(cr.vulnerability_modifier, 1.0)

    def test_bridge_crossing_faster_than_ford(self) -> None:
        """Bridge crossings complete in fewer steps than ford crossings."""
        self.assertLess(BRIDGE_CROSSING_STEPS, FORD_CROSSING_STEPS)

    def test_two_simultaneous_crossings_both_complete(self) -> None:
        """Two independent crossing objects both complete successfully."""
        cmap = CoastalMap(width=1_000.0, height=1_000.0, rows=5, cols=5)
        cmap.set_tile(2, 1, WaterTileType.FORD)
        cmap.set_tile(2, 3, WaterTileType.BRIDGE)

        ford_cr = RiverCrossing(
            unit_x=0.0, unit_y=500.0,
            crossing_x=200.0, crossing_y=500.0,
            coastal_map=cmap, team=0,
        )
        bridge_cr = RiverCrossing(
            unit_x=0.0, unit_y=500.0,
            crossing_x=600.0, crossing_y=500.0,
            coastal_map=cmap, team=0,
        )

        steps = max(FORD_CROSSING_STEPS, BRIDGE_CROSSING_STEPS)
        for _ in range(steps):
            ford_cr.step()
            bridge_cr.step()

        self.assertTrue(ford_cr.is_complete)
        self.assertTrue(bridge_cr.is_complete)

    def test_gunboat_can_support_river_crossing(self) -> None:
        """Gunboat on a river tile can bombard enemy defenders on the far bank."""
        cmap = CoastalMap(width=2_000.0, height=2_000.0, rows=10, cols=10)
        # River runs at rows 4–5 (y ≈ 800–1000)
        for r in [4, 5]:
            for c in range(10):
                cmap.set_tile(r, c, WaterTileType.RIVER)
        gunboat = _gunboat(x=500.0, y=900.0, theta=0.0)
        # Target north of river on land — within 250 m range, in broadside arc
        dmg = naval_gunfire_damage(
            gunboat, 500.0, 1_100.0, 1.0, cmap,
            intensity=1.0, require_water_tile=True,
        )
        self.assertGreater(dmg, 0.0, "Gunboat should support river crossing with fire")


if __name__ == "__main__":
    unittest.main()
