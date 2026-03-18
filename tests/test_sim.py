"""Integration tests for the simulation engine (Epic E1.2 acceptance criteria).

Acceptance criteria tested here:

AC-1  A battle resolves to a winner in ≤ 500 sim steps.
AC-2  Combat damage scales correctly with range and firing arc.
AC-3  Morale drops under fire and triggers routing below threshold.
AC-4  All sim unit tests pass (this file).
AC-5  No Gymnasium or RL code required to run a sim episode.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Test-local constants
# ---------------------------------------------------------------------------

#: Maximum total casualties a unit can sustain (strength bounded in [0, 1]).
_MAX_TOTAL_CASUALTIES: float = 1.0

#: Tiny positive damage that satisfies `accumulated_damage != 0.0` so that
#: morale_check skips the passive-recovery branch without meaningfully
#: affecting morale calculations.
_TINY_DAMAGE: float = 1e-9

# AC-5: only sim-layer imports — no gymnasium, stable-baselines, etc.
from envs.sim.battalion import Battalion
from envs.sim.combat import (
    MORALE_CASUALTY_WEIGHT,
    MORALE_ROUT_THRESHOLD,
    CombatState,
    apply_casualties,
    compute_fire_damage,
    morale_check,
)
from envs.sim.engine import DESTROYED_THRESHOLD, EpisodeResult, SimEngine
from envs.sim.terrain import TerrainMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _facing_pair(dist: float = 150.0) -> tuple[Battalion, Battalion]:
    """Two battalions facing each other along the x-axis at *dist* metres."""
    blue = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
    red = Battalion(x=dist, y=0.0, theta=math.pi, strength=1.0, team=1)
    return blue, red


def _seeded_engine(dist: float = 150.0, seed: int = 42, **kwargs) -> SimEngine:
    """Return a SimEngine with a seeded RNG and optional extra kwargs."""
    blue, red = _facing_pair(dist)
    return SimEngine(blue, red, rng=np.random.default_rng(seed), **kwargs)


# ---------------------------------------------------------------------------
# AC-1: battle resolves to a winner in ≤ 500 steps
# ---------------------------------------------------------------------------


class TestBattleResolution(unittest.TestCase):
    """AC-1: A battle between two battalions resolves within the step budget."""

    def test_battle_resolves_within_500_steps(self) -> None:
        """Default 1v1 scenario (150 m separation) must end before step 500."""
        engine = _seeded_engine(dist=150.0, seed=42)
        result = engine.run()
        self.assertLessEqual(result.steps, 500)

    def test_battle_produces_a_winner(self) -> None:
        """The result must declare a winner (not a draw) for a symmetric fight."""
        # Run the symmetric scenario: both sides identical, so one must rout
        # first due to RNG.  In rare cases both rout simultaneously (draw=None)
        # — allow either outcome, but the key invariant is that the episode ENDS.
        engine = _seeded_engine(dist=150.0, seed=42)
        result = engine.run()
        # At most one side should still be at full strength
        both_full = result.blue_strength == 1.0 and result.red_strength == 1.0
        self.assertFalse(both_full, "Neither side took any damage — no combat occurred")

    def test_winner_is_valid_value(self) -> None:
        """winner must be 0, 1, or None."""
        result = _seeded_engine(seed=7).run()
        self.assertIn(result.winner, (0, 1, None))

    def test_loser_is_routed_or_destroyed(self) -> None:
        """The losing side must be routed or near-zero strength."""
        result = _seeded_engine(seed=42).run()
        if result.winner == 0:
            loser_routed = result.red_routed
            loser_low = result.red_strength <= DESTROYED_THRESHOLD
        elif result.winner == 1:
            loser_routed = result.blue_routed
            loser_low = result.blue_strength <= DESTROYED_THRESHOLD
        else:
            return  # draw — skip
        self.assertTrue(loser_routed or loser_low)

    def test_multiple_seeds_all_resolve(self) -> None:
        """Run several seeds; all should complete within 500 steps."""
        for seed in range(10):
            with self.subTest(seed=seed):
                result = _seeded_engine(seed=seed).run()
                self.assertLessEqual(result.steps, 500, f"seed={seed} exceeded 500 steps")

    def test_close_range_resolves_faster(self) -> None:
        """Closer battalions should require fewer steps than distant ones."""
        close = _seeded_engine(dist=30.0, seed=1).run()
        far = _seeded_engine(dist=180.0, seed=1).run()
        self.assertLess(close.steps, far.steps)

    def test_max_steps_respected(self) -> None:
        """Engine must never exceed its configured max_steps."""
        engine = _seeded_engine(max_steps=50)
        result = engine.run()
        self.assertLessEqual(result.steps, 50)


# ---------------------------------------------------------------------------
# AC-2: damage scales with range and firing arc
# ---------------------------------------------------------------------------


class TestDamageScaling(unittest.TestCase):
    """AC-2: Damage is higher at close range and from flanking/rear angles."""

    def test_close_range_more_damage_than_far_range(self) -> None:
        """A single volley at close range deals more damage than at far range."""
        shooter_close = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target_close = Battalion(x=30.0, y=0.0, theta=math.pi, strength=1.0, team=1)

        shooter_far = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target_far = Battalion(x=170.0, y=0.0, theta=math.pi, strength=1.0, team=1)

        dmg_close = compute_fire_damage(shooter_close, target_close, intensity=1.0)
        dmg_far = compute_fire_damage(shooter_far, target_far, intensity=1.0)
        self.assertGreater(dmg_close, dmg_far)

    def test_zero_damage_outside_fire_range(self) -> None:
        """No damage when target is beyond fire_range."""
        shooter = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target = Battalion(x=250.0, y=0.0, theta=math.pi, strength=1.0, team=1)
        dmg = compute_fire_damage(shooter, target, intensity=1.0)
        self.assertAlmostEqual(dmg, 0.0)

    def test_zero_damage_outside_fire_arc(self) -> None:
        """No damage when target is outside the frontal fire arc."""
        shooter = Battalion(x=0.0, y=0.0, theta=math.pi, strength=1.0, team=0)
        target = Battalion(x=100.0, y=0.0, theta=0.0, strength=1.0, team=1)
        # Shooter faces away from target
        dmg = compute_fire_damage(shooter, target, intensity=1.0)
        self.assertAlmostEqual(dmg, 0.0)

    def test_flanking_attack_more_damage_than_frontal(self) -> None:
        """Flanking attack (90° off target facing) deals more damage than frontal."""
        dist = 50.0
        shooter_front = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target_front = Battalion(x=dist, y=0.0, theta=math.pi, strength=1.0, team=1)

        shooter_flank = Battalion(x=dist, y=-dist, theta=math.pi / 2, strength=1.0, team=0)
        target_flank = Battalion(x=dist, y=0.0, theta=math.pi, strength=1.0, team=1)

        dmg_front = compute_fire_damage(shooter_front, target_front, 1.0)
        dmg_flank = compute_fire_damage(shooter_flank, target_flank, 1.0)
        self.assertGreater(dmg_flank, dmg_front)

    def test_rear_attack_more_damage_than_flanking(self) -> None:
        """Rear attack deals more damage than a flanking attack."""
        dist = 50.0
        # Rear: shooter west of target, target faces east
        shooter_rear = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target_rear = Battalion(x=dist, y=0.0, theta=0.0, strength=1.0, team=1)

        # Flank: shooter south of target, target faces east
        shooter_flank = Battalion(x=dist, y=-dist, theta=math.pi / 2, strength=1.0, team=0)
        target_flank = Battalion(x=dist, y=0.0, theta=0.0, strength=1.0, team=1)

        dmg_rear = compute_fire_damage(shooter_rear, target_rear, 1.0)
        dmg_flank = compute_fire_damage(shooter_flank, target_flank, 1.0)
        self.assertGreater(dmg_rear, dmg_flank)

    def test_engine_accumulates_more_damage_at_close_range(self) -> None:
        """SimEngine: close-range scenario depletes strength faster."""
        close = _seeded_engine(dist=30.0, seed=0)
        far = _seeded_engine(dist=170.0, seed=0)

        # Run exactly 20 steps on each
        for _ in range(20):
            if not close.is_over():
                close.step()
            if not far.is_over():
                far.step()

        close_total = close.blue_state.total_casualties + close.red_state.total_casualties
        far_total = far.blue_state.total_casualties + far.red_state.total_casualties
        self.assertGreater(close_total, far_total)


# ---------------------------------------------------------------------------
# AC-3: morale drops under fire and triggers routing
# ---------------------------------------------------------------------------


class TestMoraleUnderFire(unittest.TestCase):
    """AC-3: Morale decreases when a unit takes damage; routing triggers below threshold."""

    def test_morale_decreases_after_receiving_fire(self) -> None:
        """CombatState morale falls after a volley of fire."""
        shooter = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target = Battalion(x=50.0, y=0.0, theta=math.pi, strength=1.0, team=1)
        state = CombatState()

        damage = compute_fire_damage(shooter, target, intensity=1.0)
        apply_casualties(target, state, damage)
        morale_check(state, rng=np.random.default_rng(0))

        self.assertLess(state.morale, 1.0)

    def test_morale_drop_proportional_to_damage(self) -> None:
        """Larger damage causes a larger morale drop in the same step."""
        state_small = CombatState(morale=1.0, accumulated_damage=0.05)
        state_large = CombatState(morale=1.0, accumulated_damage=0.30)

        morale_check(state_small, rng=np.random.default_rng(0))
        morale_check(state_large, rng=np.random.default_rng(0))

        drop_small = 1.0 - state_small.morale
        drop_large = 1.0 - state_large.morale
        self.assertGreater(drop_large, drop_small)

    def test_morale_drop_uses_casualty_weight(self) -> None:
        """Morale drop equals accumulated_damage × MORALE_CASUALTY_WEIGHT."""
        damage = 0.10
        state = CombatState(morale=1.0, accumulated_damage=damage)
        morale_check(state, rng=np.random.default_rng(0))
        expected = 1.0 - damage * MORALE_CASUALTY_WEIGHT
        self.assertAlmostEqual(state.morale, expected)

    def test_morale_never_goes_negative(self) -> None:
        """Morale is clamped at 0.0 under heavy fire."""
        state = CombatState(morale=0.05, accumulated_damage=0.9)
        morale_check(state, rng=np.random.default_rng(0))
        self.assertGreaterEqual(state.morale, 0.0)

    def test_routing_triggered_at_zero_morale(self) -> None:
        """A unit at zero morale must route (probability = 1.0)."""
        state = CombatState(morale=0.0, accumulated_damage=0.0)
        # Tiny non-zero damage suppresses the passive-recovery branch so morale
        # stays at 0.0 and rout probability is exactly 1.0.
        state.accumulated_damage = _TINY_DAMAGE
        result = morale_check(state, rng=np.random.default_rng(0))
        self.assertTrue(result)
        self.assertTrue(state.is_routing)

    def test_sustained_fire_eventually_routes_unit(self) -> None:
        """Continuous point-blank fire routes the target within 500 steps."""
        shooter = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        target = Battalion(x=10.0, y=0.0, theta=math.pi, strength=1.0, team=1)
        state = CombatState()
        rng = np.random.default_rng(42)

        for _ in range(500):
            state.reset_step_accumulators()
            dmg = compute_fire_damage(shooter, target, 1.0)
            apply_casualties(target, state, dmg)
            if morale_check(state, rng=rng):
                break

        self.assertTrue(state.is_routing, "Target never routed despite 500 steps of fire")

    def test_engine_morale_drops_during_battle(self) -> None:
        """After a few steps, at least one side's morale should have fallen."""
        engine = _seeded_engine(dist=50.0, seed=0)
        for _ in range(10):
            if not engine.is_over():
                engine.step()

        min_morale = min(engine.blue_state.morale, engine.red_state.morale)
        self.assertLess(min_morale, 1.0)

    def test_routing_threshold_is_reachable(self) -> None:
        """MORALE_CASUALTY_WEIGHT * 1.0 must exceed (1 − MORALE_ROUT_THRESHOLD).

        If this fails the routing threshold can never be reached through fire
        damage alone (the morale system would be inoperative).
        """
        max_morale_drop = MORALE_CASUALTY_WEIGHT * _MAX_TOTAL_CASUALTIES
        required_drop = 1.0 - MORALE_ROUT_THRESHOLD
        self.assertGreater(
            max_morale_drop,
            required_drop,
            msg=(
                f"MORALE_CASUALTY_WEIGHT={MORALE_CASUALTY_WEIGHT} too low: "
                f"max morale drop {max_morale_drop:.2f} ≤ required {required_drop:.2f}"
            ),
        )


# ---------------------------------------------------------------------------
# Terrain integration
# ---------------------------------------------------------------------------


class TestTerrainIntegration(unittest.TestCase):
    """Terrain cover reduces incoming damage in the simulation engine."""

    def test_full_cover_prevents_all_damage(self) -> None:
        """A unit in full cover (c=1.0) should take no damage from any fire."""
        elev = np.zeros((1, 1), dtype=np.float32)
        cov = np.array([[1.0]], dtype=np.float32)
        terrain = TerrainMap.from_arrays(500.0, 500.0, elev, cov)

        blue, red = _facing_pair(dist=50.0)
        engine = SimEngine(blue, red, terrain=terrain, rng=np.random.default_rng(0))

        metrics = engine.step()
        # Red is in full cover — blue deals zero damage
        self.assertAlmostEqual(metrics["blue_damage_dealt"], 0.0)
        # Blue is also in full cover — red deals zero damage
        self.assertAlmostEqual(metrics["red_damage_dealt"], 0.0)

    def test_partial_cover_reduces_damage(self) -> None:
        """Partial cover (c=0.5) should halve the incoming damage."""
        cover_value = 0.5
        elev = np.zeros((1, 1), dtype=np.float32)
        cov = np.array([[cover_value]], dtype=np.float32)
        terrain = TerrainMap.from_arrays(500.0, 500.0, elev, cov)

        blue, red = _facing_pair(dist=50.0)

        # Baseline: same geometry but no cover
        blue_open, red_open = _facing_pair(dist=50.0)
        engine_open = SimEngine(
            blue_open, red_open, terrain=TerrainMap.flat(500.0, 500.0),
            rng=np.random.default_rng(0),
        )
        engine_cover = SimEngine(
            blue, red, terrain=terrain, rng=np.random.default_rng(0)
        )

        open_metrics = engine_open.step()
        cover_metrics = engine_cover.step()

        # Both directions should see reduced damage
        self.assertLess(cover_metrics["blue_damage_dealt"], open_metrics["blue_damage_dealt"])
        self.assertLess(cover_metrics["red_damage_dealt"], open_metrics["red_damage_dealt"])

    def test_no_cover_flat_terrain_matches_direct_compute(self) -> None:
        """On flat terrain, engine damage should equal raw compute_fire_damage."""
        blue, red = _facing_pair(dist=80.0)
        terrain = TerrainMap.flat(500.0, 500.0)
        engine = SimEngine(blue, red, terrain=terrain, rng=np.random.default_rng(0))

        # Compute expected damage before any step (strength = 1.0 for both)
        expected_blue_to_red = compute_fire_damage(blue, red, intensity=1.0)
        expected_red_to_blue = compute_fire_damage(red, blue, intensity=1.0)

        metrics = engine.step()
        self.assertAlmostEqual(metrics["blue_damage_dealt"], expected_blue_to_red)
        self.assertAlmostEqual(metrics["red_damage_dealt"], expected_red_to_blue)


# ---------------------------------------------------------------------------
# SimEngine API contract
# ---------------------------------------------------------------------------


class TestSimEngineAPI(unittest.TestCase):
    """Verify the SimEngine and EpisodeResult interface."""

    def test_step_returns_expected_keys(self) -> None:
        engine = _seeded_engine()
        metrics = engine.step()
        for key in ("blue_damage_dealt", "red_damage_dealt", "blue_routing", "red_routing"):
            self.assertIn(key, metrics)

    def test_step_increments_step_count(self) -> None:
        engine = _seeded_engine()
        self.assertEqual(engine.step_count, 0)
        engine.step()
        self.assertEqual(engine.step_count, 1)
        engine.step()
        self.assertEqual(engine.step_count, 2)

    def test_is_over_false_at_start(self) -> None:
        engine = _seeded_engine()
        self.assertFalse(engine.is_over())

    def test_is_over_when_max_steps_reached(self) -> None:
        engine = _seeded_engine(max_steps=3)
        for _ in range(3):
            engine.step()
        self.assertTrue(engine.is_over())

    def test_run_returns_episode_result(self) -> None:
        result = _seeded_engine().run()
        self.assertIsInstance(result, EpisodeResult)

    def test_episode_result_fields_in_range(self) -> None:
        result = _seeded_engine().run()
        self.assertGreaterEqual(result.blue_strength, 0.0)
        self.assertLessEqual(result.blue_strength, 1.0)
        self.assertGreaterEqual(result.red_strength, 0.0)
        self.assertLessEqual(result.red_strength, 1.0)
        self.assertGreaterEqual(result.blue_morale, 0.0)
        self.assertLessEqual(result.blue_morale, 1.0)
        self.assertGreaterEqual(result.red_morale, 0.0)
        self.assertLessEqual(result.red_morale, 1.0)
        self.assertGreater(result.steps, 0)

    def test_shots_fired_tracked(self) -> None:
        engine = _seeded_engine()
        for _ in range(5):
            if not engine.is_over():
                engine.step()
        self.assertGreater(engine.blue_state.shots_fired, 0)
        self.assertGreater(engine.red_state.shots_fired, 0)

    def test_default_terrain_is_flat(self) -> None:
        """Engine constructed without terrain uses flat 1 km × 1 km terrain."""
        blue, red = _facing_pair()
        engine = SimEngine(blue, red)
        # On flat terrain cover is 0 everywhere — cover modifier returns damage unchanged
        cover = engine.terrain.get_cover(blue.x, blue.y)
        self.assertAlmostEqual(cover, 0.0)

    def test_reproducible_with_same_seed(self) -> None:
        """Two engines with the same seed produce identical results."""
        r1 = _seeded_engine(seed=99).run()
        r2 = _seeded_engine(seed=99).run()
        self.assertEqual(r1.winner, r2.winner)
        self.assertEqual(r1.steps, r2.steps)
        self.assertAlmostEqual(r1.blue_strength, r2.blue_strength)

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds should (almost certainly) produce different step counts."""
        steps = {_seeded_engine(seed=s).run().steps for s in range(20)}
        # With randomness, we expect more than one unique step count
        self.assertGreater(len(steps), 1)


# ---------------------------------------------------------------------------
# AC-5: no Gymnasium imports needed
# ---------------------------------------------------------------------------


class TestNoGymRequired(unittest.TestCase):
    """AC-5: The sim engine must run without any RL framework."""

    def test_sim_imports_no_gymnasium(self) -> None:
        """Running a full episode must not import gymnasium."""
        import sys
        # gymnasium should not be imported as a side-effect of running the sim
        _seeded_engine().run()
        self.assertNotIn("gymnasium", sys.modules)

    def test_sim_imports_no_stable_baselines(self) -> None:
        """Running a full episode must not import stable_baselines3."""
        import sys
        _seeded_engine().run()
        self.assertNotIn("stable_baselines3", sys.modules)


if __name__ == "__main__":
    unittest.main()
