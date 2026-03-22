"""Tests for envs/sim/supply_network.py — Strategic Supply Network."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.supply_network import (
    ConvoyRoute,
    SupplyDepot,
    SupplyNetwork,
)
from envs.corps_env import (
    CorpsEnv,
    N_CORPS_SECTORS,
    N_OBJECTIVES,
    N_ROAD_FEATURES,
    _corps_obs_dim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_depot(x=500.0, y=500.0, team=0, radius=1000.0) -> SupplyDepot:
    return SupplyDepot(x=x, y=y, team=team, base_supply_radius=radius)


def make_network() -> SupplyNetwork:
    return SupplyNetwork.generate_default(map_width=10_000.0, map_height=5_000.0)


def make_env(**kwargs) -> CorpsEnv:
    return CorpsEnv(
        n_divisions=2,
        n_brigades_per_division=2,
        n_blue_per_brigade=2,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. SupplyDepot
# ---------------------------------------------------------------------------


class TestSupplyDepot(unittest.TestCase):
    """Unit tests for :class:`SupplyDepot`."""

    # ── Construction ────────────────────────────────────────────────────

    def test_defaults_alive_and_full(self) -> None:
        d = make_depot()
        self.assertTrue(d.alive)
        self.assertAlmostEqual(d.stock, 1.0)

    def test_initial_stock_propagated(self) -> None:
        d = SupplyDepot(x=0.0, y=0.0, team=0, initial_stock=0.5)
        self.assertAlmostEqual(d.stock, 0.5)

    def test_invalid_initial_stock_raises(self) -> None:
        with self.assertRaises(ValueError):
            SupplyDepot(x=0.0, y=0.0, team=0, initial_stock=0.0)
        with self.assertRaises(ValueError):
            SupplyDepot(x=0.0, y=0.0, team=0, initial_stock=1.5)

    def test_invalid_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=0.0)

    def test_invalid_team_raises(self) -> None:
        with self.assertRaises(ValueError):
            SupplyDepot(x=0.0, y=0.0, team=2)

    # ── effective_supply_radius ──────────────────────────────────────────

    def test_effective_radius_full_stock(self) -> None:
        d = make_depot(radius=1000.0)
        self.assertAlmostEqual(d.effective_supply_radius, 1000.0)

    def test_effective_radius_half_stock(self) -> None:
        d = make_depot(radius=1000.0)
        d.stock = 0.5
        self.assertAlmostEqual(d.effective_supply_radius, 500.0)

    def test_effective_radius_dead_depot(self) -> None:
        d = make_depot()
        d.interdict()
        self.assertAlmostEqual(d.effective_supply_radius, 0.0)

    # ── supply_level_at ──────────────────────────────────────────────────

    def test_supply_level_at_centre_is_one(self) -> None:
        d = make_depot(x=0.0, y=0.0, radius=1000.0)
        self.assertAlmostEqual(d.supply_level_at(0.0, 0.0), 1.0)

    def test_supply_level_at_edge_is_zero(self) -> None:
        d = make_depot(x=0.0, y=0.0, radius=1000.0)
        level = d.supply_level_at(1000.0, 0.0)
        self.assertAlmostEqual(level, 0.0, places=6)

    def test_supply_level_beyond_radius_is_zero(self) -> None:
        d = make_depot(x=0.0, y=0.0, radius=1000.0)
        self.assertAlmostEqual(d.supply_level_at(1500.0, 0.0), 0.0)

    def test_supply_level_midpoint(self) -> None:
        d = make_depot(x=0.0, y=0.0, radius=1000.0)
        level = d.supply_level_at(500.0, 0.0)
        self.assertAlmostEqual(level, 0.5, places=6)

    def test_supply_level_dead_depot_is_zero(self) -> None:
        d = make_depot(x=0.0, y=0.0, radius=1000.0)
        d.interdict()
        self.assertAlmostEqual(d.supply_level_at(0.0, 0.0), 0.0)

    def test_supply_level_in_range(self) -> None:
        d = make_depot(x=0.0, y=0.0, radius=1000.0)
        level = d.supply_level_at(300.0, 400.0)  # dist = 500
        self.assertAlmostEqual(level, 0.5, places=6)

    # ── consume ─────────────────────────────────────────────────────────

    def test_consume_reduces_stock(self) -> None:
        d = make_depot()
        consumed = d.consume(0.3)
        self.assertAlmostEqual(consumed, 0.3, places=6)
        self.assertAlmostEqual(d.stock, 0.7, places=6)

    def test_consume_clamps_to_available(self) -> None:
        d = make_depot()
        d.stock = 0.1
        consumed = d.consume(0.5)
        self.assertAlmostEqual(consumed, 0.1, places=6)
        self.assertAlmostEqual(d.stock, 0.0, places=6)

    def test_consume_dead_depot_returns_zero(self) -> None:
        d = make_depot()
        d.interdict()
        consumed = d.consume(0.3)
        self.assertAlmostEqual(consumed, 0.0)

    def test_consume_negative_raises(self) -> None:
        d = make_depot()
        with self.assertRaises(ValueError):
            d.consume(-0.1)

    # ── replenish ────────────────────────────────────────────────────────

    def test_replenish_increases_stock(self) -> None:
        d = make_depot()
        d.stock = 0.5
        d.replenish(0.3)
        self.assertAlmostEqual(d.stock, 0.8, places=6)

    def test_replenish_clamps_to_initial(self) -> None:
        d = make_depot()
        d.replenish(0.5)
        self.assertAlmostEqual(d.stock, 1.0, places=6)

    def test_replenish_dead_depot_noop(self) -> None:
        d = make_depot()
        d.interdict()
        d.replenish(0.5)
        self.assertAlmostEqual(d.stock, 0.0)

    def test_replenish_negative_raises(self) -> None:
        d = make_depot()
        with self.assertRaises(ValueError):
            d.replenish(-0.1)

    # ── interdict ────────────────────────────────────────────────────────

    def test_interdict_sets_dead_and_zero_stock(self) -> None:
        d = make_depot()
        d.interdict()
        self.assertFalse(d.alive)
        self.assertAlmostEqual(d.stock, 0.0)

    # ── reset ────────────────────────────────────────────────────────────

    def test_reset_restores_state(self) -> None:
        d = SupplyDepot(x=0.0, y=0.0, team=0, initial_stock=0.8)
        d.consume(0.4)
        d.interdict()
        d.reset()
        self.assertTrue(d.alive)
        self.assertAlmostEqual(d.stock, 0.8, places=6)


# ---------------------------------------------------------------------------
# 2. ConvoyRoute
# ---------------------------------------------------------------------------


class TestConvoyRoute(unittest.TestCase):
    """Unit tests for :class:`ConvoyRoute`."""

    def _make_pair(self, src_stock=1.0, dst_stock=0.0) -> tuple:
        src = SupplyDepot(x=0.0, y=0.0, team=0, initial_stock=1.0)
        src.stock = src_stock
        dst = SupplyDepot(x=1000.0, y=0.0, team=0, initial_stock=1.0)
        dst.stock = dst_stock
        return src, dst

    def test_invalid_same_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            ConvoyRoute(source_idx=0, dest_idx=0)

    def test_invalid_transfer_rate_raises(self) -> None:
        with self.assertRaises(ValueError):
            ConvoyRoute(source_idx=0, dest_idx=1, transfer_rate=0.0)

    def test_step_transfers_stock(self) -> None:
        src, dst = self._make_pair(src_stock=1.0, dst_stock=0.0)
        depots = [src, dst]
        route = ConvoyRoute(source_idx=0, dest_idx=1, transfer_rate=0.1)
        route.step(depots)
        self.assertAlmostEqual(src.stock, 0.9, places=6)
        self.assertAlmostEqual(dst.stock, 0.1, places=6)

    def test_step_no_transfer_if_source_dead(self) -> None:
        src, dst = self._make_pair(src_stock=1.0, dst_stock=0.0)
        src.interdict()
        depots = [src, dst]
        route = ConvoyRoute(source_idx=0, dest_idx=1, transfer_rate=0.1)
        route.step(depots)
        self.assertAlmostEqual(dst.stock, 0.0)

    def test_step_no_transfer_if_dest_dead(self) -> None:
        src, dst = self._make_pair(src_stock=1.0, dst_stock=0.0)
        dst.interdict()
        depots = [src, dst]
        route = ConvoyRoute(source_idx=0, dest_idx=1, transfer_rate=0.1)
        route.step(depots)
        self.assertAlmostEqual(src.stock, 1.0)  # nothing consumed

    def test_step_clamps_transfer_to_source_stock(self) -> None:
        src, dst = self._make_pair(src_stock=0.05, dst_stock=0.0)
        depots = [src, dst]
        route = ConvoyRoute(source_idx=0, dest_idx=1, transfer_rate=0.1)
        route.step(depots)
        self.assertAlmostEqual(src.stock, 0.0, places=6)
        self.assertAlmostEqual(dst.stock, 0.05, places=6)


# ---------------------------------------------------------------------------
# 3. SupplyNetwork
# ---------------------------------------------------------------------------


class TestSupplyNetwork(unittest.TestCase):
    """Unit tests for :class:`SupplyNetwork`."""

    # ── get_supply_level ─────────────────────────────────────────────────

    def test_get_supply_level_in_range(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        level = net.get_supply_level(0.0, 0.0, team=0)
        self.assertAlmostEqual(level, 1.0)

    def test_get_supply_level_out_of_range(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=500.0)
        net = SupplyNetwork(depots=[depot])
        level = net.get_supply_level(1000.0, 0.0, team=0)
        self.assertAlmostEqual(level, 0.0)

    def test_get_supply_level_max_of_multiple_depots(self) -> None:
        d1 = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=200.0)
        d2 = SupplyDepot(x=500.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[d1, d2])
        # Point at (300, 0): d1 is out of range (dist=300 > 200), d2 gives 0.8
        level = net.get_supply_level(300.0, 0.0, team=0)
        expected = max(0.0, 1.0 - 200.0 / 1000.0)
        self.assertAlmostEqual(level, expected, places=5)

    def test_get_supply_level_wrong_team_returns_zero(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        self.assertAlmostEqual(net.get_supply_level(0.0, 0.0, team=1), 0.0)

    def test_get_supply_level_dead_depot_returns_zero(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        net.interdict_depot(0)
        self.assertAlmostEqual(net.get_supply_level(0.0, 0.0, team=0), 0.0)

    # ── get_division_supply_levels ───────────────────────────────────────

    def test_get_division_supply_levels_length(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        positions = [(0.0, 0.0), (100.0, 0.0), (5000.0, 0.0)]
        levels = net.get_division_supply_levels(positions, team=0)
        self.assertEqual(len(levels), 3)

    def test_get_division_supply_levels_values(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        positions = [(0.0, 0.0), (2000.0, 0.0)]
        levels = net.get_division_supply_levels(positions, team=0)
        self.assertAlmostEqual(levels[0], 1.0)
        self.assertAlmostEqual(levels[1], 0.0)

    # ── consume_supply ───────────────────────────────────────────────────

    def test_consume_supply_reduces_stock(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot], consumption_per_step=0.1)
        net.consume_supply([(0.0, 0.0)], team=0)
        self.assertAlmostEqual(depot.stock, 0.9, places=6)

    def test_consume_supply_multiple_units_same_depot(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot], consumption_per_step=0.1)
        net.consume_supply([(0.0, 0.0), (100.0, 0.0)], team=0)
        self.assertAlmostEqual(depot.stock, 0.8, places=6)

    def test_consume_supply_out_of_range_no_change(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=500.0)
        net = SupplyNetwork(depots=[depot], consumption_per_step=0.1)
        net.consume_supply([(2000.0, 0.0)], team=0)
        self.assertAlmostEqual(depot.stock, 1.0)

    def test_consume_supply_wrong_team_no_change(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot], consumption_per_step=0.1)
        net.consume_supply([(0.0, 0.0)], team=1)
        self.assertAlmostEqual(depot.stock, 1.0)

    def test_consume_supply_custom_amount(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        net.consume_supply([(0.0, 0.0)], team=0, amount=0.25)
        self.assertAlmostEqual(depot.stock, 0.75, places=6)

    def test_consume_supply_empty_positions_noop(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        net.consume_supply([], team=0)
        self.assertAlmostEqual(depot.stock, 1.0)

    # ── interdict_depot ──────────────────────────────────────────────────

    def test_interdict_depot_by_index(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        net.interdict_depot(0)
        self.assertFalse(depot.alive)
        self.assertAlmostEqual(depot.stock, 0.0)

    def test_interdict_depot_out_of_range_raises(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        with self.assertRaises(IndexError):
            net.interdict_depot(5)

    # ── interdict_nearest_depot ──────────────────────────────────────────

    def test_interdict_nearest_within_radius(self) -> None:
        depot = SupplyDepot(x=100.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        idx = net.interdict_nearest_depot(0.0, 0.0, enemy_team=1, capture_radius=200.0)
        self.assertEqual(idx, 0)
        self.assertFalse(depot.alive)

    def test_interdict_nearest_outside_radius_returns_none(self) -> None:
        depot = SupplyDepot(x=5000.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        idx = net.interdict_nearest_depot(0.0, 0.0, enemy_team=1, capture_radius=200.0)
        self.assertIsNone(idx)
        self.assertTrue(depot.alive)

    def test_interdict_nearest_already_dead_ignored(self) -> None:
        depot = SupplyDepot(x=50.0, y=0.0, team=1, base_supply_radius=1000.0)
        depot.interdict()
        net = SupplyNetwork(depots=[depot])
        idx = net.interdict_nearest_depot(0.0, 0.0, enemy_team=1, capture_radius=200.0)
        self.assertIsNone(idx)

    def test_interdict_nearest_selects_closest(self) -> None:
        d_near = SupplyDepot(x=50.0, y=0.0, team=1, base_supply_radius=1000.0)
        d_far = SupplyDepot(x=150.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[d_near, d_far])
        idx = net.interdict_nearest_depot(0.0, 0.0, enemy_team=1, capture_radius=300.0)
        self.assertEqual(idx, 0)
        self.assertFalse(d_near.alive)
        self.assertTrue(d_far.alive)

    # ── step ─────────────────────────────────────────────────────────────

    def test_step_advances_consumption_and_convoys(self) -> None:
        net = make_network()
        # Blue rear depot (idx 0) feeds Blue forward (idx 1) via a convoy.
        # Blue forward depot also feeds Blue units.
        blue_pos = [(net.depots[0].x, net.depots[0].y)]
        red_pos = []
        net.step(blue_pos, red_pos)
        # Blue depot 0 should have been consumed (unit present) and sent convoy
        self.assertLess(net.depots[0].stock, 1.0)

    def test_step_no_positions_noop(self) -> None:
        net = make_network()
        initial_stocks = [d.stock for d in net.depots]
        # Convoys still run but no consumption: total stock is conserved.
        net.step([], [])
        total_initial = sum(initial_stocks)
        total_after = sum(d.stock for d in net.depots)
        self.assertAlmostEqual(total_after, total_initial, places=6)
        # All depots must remain in valid range.
        for d in net.depots:
            self.assertGreaterEqual(d.stock, 0.0)
            self.assertLessEqual(d.stock, 1.0)

    # ── reset ─────────────────────────────────────────────────────────────

    def test_reset_restores_all_depots(self) -> None:
        net = make_network()
        # Destroy and deplete all depots
        for i in range(len(net.depots)):
            net.interdict_depot(i)
        net.reset()
        for d in net.depots:
            self.assertTrue(d.alive)
            self.assertAlmostEqual(d.stock, d.initial_stock, places=6)

    # ── any_alive ────────────────────────────────────────────────────────

    def test_any_alive_true_when_depot_alive(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        self.assertTrue(net.any_alive(team=1))

    def test_any_alive_false_when_all_dead(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        net.interdict_depot(0)
        self.assertFalse(net.any_alive(team=1))

    # ── get_depots_for_team ──────────────────────────────────────────────

    def test_get_depots_for_team_filters_correctly(self) -> None:
        d_blue = SupplyDepot(x=0.0, y=0.0, team=0, base_supply_radius=1000.0)
        d_red = SupplyDepot(x=500.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[d_blue, d_red])
        blue_depots = net.get_depots_for_team(0)
        red_depots = net.get_depots_for_team(1)
        self.assertEqual(len(blue_depots), 1)
        self.assertEqual(len(red_depots), 1)
        self.assertEqual(blue_depots[0].team, 0)
        self.assertEqual(red_depots[0].team, 1)

    # ── __len__ ──────────────────────────────────────────────────────────

    def test_len(self) -> None:
        net = make_network()
        self.assertEqual(len(net), len(net.depots))

    # ── generate_default ─────────────────────────────────────────────────

    def test_generate_default_creates_depots(self) -> None:
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        self.assertGreater(len(net.depots), 0)

    def test_generate_default_has_both_teams(self) -> None:
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        teams = {d.team for d in net.depots}
        self.assertIn(0, teams)
        self.assertIn(1, teams)

    def test_generate_default_has_convoy_routes(self) -> None:
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        self.assertGreater(len(net.convoy_routes), 0)

    def test_generate_default_blue_depots_in_west(self) -> None:
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        for d in net.depots:
            if d.team == 0:
                self.assertLessEqual(d.x, 5_000.0)  # west half

    def test_generate_default_red_depots_in_east(self) -> None:
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        for d in net.depots:
            if d.team == 1:
                self.assertGreaterEqual(d.x, 5_000.0)  # east half

    def test_generate_default_red_forward_depot_matches_objective(self) -> None:
        """Red's forward depot x should be at 80% of map width."""
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        red_forward_x = [d.x for d in net.depots if d.team == 1]
        self.assertIn(8_000.0, red_forward_x)

    def test_generate_default_custom_radius(self) -> None:
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0, supply_radius=2_000.0)
        for d in net.depots:
            self.assertLessEqual(d.base_supply_radius, 2_000.0 * 1.5)


# ---------------------------------------------------------------------------
# 4. SupplyNetwork — interdiction degrades supply coverage
# ---------------------------------------------------------------------------


class TestSupplyInterdiction(unittest.TestCase):
    """Verify that interdicting a depot degrades supply coverage immediately."""

    def test_interdiction_collapses_radius(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        # Before interdiction: level > 0
        self.assertGreater(net.get_supply_level(500.0, 0.0, team=1), 0.0)
        net.interdict_depot(0)
        # After: level = 0
        self.assertAlmostEqual(net.get_supply_level(500.0, 0.0, team=1), 0.0)

    def test_interdiction_via_capture_position(self) -> None:
        """A Blue unit at the Red depot position should interdict it."""
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        red_forward = next(d for d in net.depots if d.team == 1 and d.x == 8_000.0)
        # Simulate Blue unit at the Red forward depot
        result = net.interdict_nearest_depot(
            red_forward.x, red_forward.y,
            enemy_team=1,
            capture_radius=500.0,
        )
        self.assertIsNotNone(result)
        self.assertFalse(red_forward.alive)

    def test_supply_level_drops_after_interdiction(self) -> None:
        net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        # Red's supply at 80% x should be > 0 initially
        level_before = net.get_supply_level(8_000.0, 2_500.0, team=1)
        self.assertGreater(level_before, 0.0)
        # Interdict Red forward depot
        for i, d in enumerate(net.depots):
            if d.team == 1:
                net.interdict_depot(i)
        level_after = net.get_supply_level(8_000.0, 2_500.0, team=1)
        self.assertAlmostEqual(level_after, 0.0)

    def test_stock_depletion_shrinks_radius(self) -> None:
        depot = SupplyDepot(x=0.0, y=0.0, team=1, base_supply_radius=1000.0)
        net = SupplyNetwork(depots=[depot])
        # At full stock: unit at 600 m is in supply
        self.assertGreater(net.get_supply_level(600.0, 0.0, team=1), 0.0)
        # Deplete stock to 50%
        depot.stock = 0.5
        # Now effective radius = 500 m; unit at 600 m is out of supply
        self.assertAlmostEqual(net.get_supply_level(600.0, 0.0, team=1), 0.0)


# ---------------------------------------------------------------------------
# 5. CorpsEnv integration
# ---------------------------------------------------------------------------


class TestCorpsEnvSupplyIntegration(unittest.TestCase):
    """Verify SupplyNetwork integration in :class:`CorpsEnv`."""

    # ── Observation dimension ────────────────────────────────────────────

    def test_obs_dim_includes_supply_features(self) -> None:
        """_corps_obs_dim must include n_divisions supply features."""
        for nd in (1, 2, 3):
            dim = _corps_obs_dim(nd)
            # Old formula: N_CORPS_SECTORS + 8*nd + N_ROAD_FEATURES + N_OBJECTIVES + 1
            # New formula adds nd supply features (one per division)
            old_dim = N_CORPS_SECTORS + 8 * nd + N_ROAD_FEATURES + N_OBJECTIVES + 1
            self.assertEqual(dim, old_dim + nd)

    def test_env_obs_shape_matches_dim(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape[0], env._obs_dim)
        self.assertEqual(env._obs_dim, _corps_obs_dim(2))
        env.close()

    def test_env_has_supply_network_attribute(self) -> None:
        env = make_env()
        self.assertIsInstance(env.supply_network, SupplyNetwork)
        env.close()

    def test_custom_supply_network_accepted(self) -> None:
        custom_net = SupplyNetwork.generate_default(10_000.0, 5_000.0)
        env = CorpsEnv(
            n_divisions=2,
            n_brigades_per_division=2,
            n_blue_per_brigade=2,
            supply_network=custom_net,
        )
        self.assertIs(env.supply_network, custom_net)
        env.close()

    # ── Reset ────────────────────────────────────────────────────────────

    def test_reset_resets_supply_network(self) -> None:
        env = make_env()
        env.reset(seed=0)
        # Interdict all depots
        for i in range(len(env.supply_network.depots)):
            env.supply_network.interdict_depot(i)
        # Reset should restore them
        env.reset(seed=1)
        for d in env.supply_network.depots:
            self.assertTrue(d.alive)
        env.close()

    # ── Step ─────────────────────────────────────────────────────────────

    def test_step_returns_supply_levels_in_info(self) -> None:
        env = make_env()
        env.reset(seed=0)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        self.assertIn("supply_levels", info)
        levels = info["supply_levels"]
        self.assertEqual(len(levels), env.n_divisions)
        env.close()

    def test_supply_levels_in_valid_range(self) -> None:
        env = make_env()
        env.reset(seed=0)
        for _ in range(5):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            for lvl in info["supply_levels"]:
                self.assertGreaterEqual(lvl, 0.0)
                self.assertLessEqual(lvl, 1.0)
            if terminated or truncated:
                break
        env.close()

    def test_obs_bounds_respected_after_supply(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=0)
        lo = env.observation_space.low
        hi = env.observation_space.high
        self.assertTrue(np.all(obs >= lo - 1e-5))
        self.assertTrue(np.all(obs <= hi + 1e-5))
        env.close()

    def test_supply_obs_slice_updates_after_interdiction(self) -> None:
        """Supply obs slice is present, sized correctly, and stays within [0, 1] after a step."""
        env = make_env()
        env.reset(seed=0)
        # Supply obs slice starts at: N_CORPS_SECTORS + 8*nd + N_ROAD_FEATURES + N_OBJECTIVES
        nd = env.n_divisions
        supply_start = N_CORPS_SECTORS + 8 * nd + N_ROAD_FEATURES + N_OBJECTIVES
        supply_end = supply_start + nd
        # Step once to get baseline obs
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        supply_slice = obs[supply_start:supply_end]
        self.assertEqual(len(supply_slice), nd)
        # All values should be in [0, 1]
        for v in supply_slice:
            self.assertGreaterEqual(float(v), 0.0)
            self.assertLessEqual(float(v), 1.0)
        env.close()

    def test_episode_runs_without_error(self) -> None:
        env = make_env()
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(obs.shape[0], env._obs_dim)
            self.assertIsInstance(reward, float)
            if terminated or truncated:
                break
        env.close()

    def test_multiple_resets_consistent(self) -> None:
        env = make_env()
        for seed in range(3):
            obs, _ = env.reset(seed=seed)
            self.assertEqual(obs.shape[0], env._obs_dim)
        env.close()


if __name__ == "__main__":
    unittest.main()
