# tests/test_morale.py
"""Tests for envs/sim/morale.py — morale state machine with stressors.

Covers:
* MoraleConfig validation
* compute_flank_stressor — flanking and rear-attack morale penalties
* compute_recovery — distance, friendly-support, and commander bonuses
* cohesion_modifier — formation cohesion degradation curve
* update_morale — full per-step morale update (stressors + recovery + routing)
* rout_velocity — forced rout movement direction and speed
* is_dispersed — dispersal check at morale zero
* Integration with CombatState (apply_casualties → update_morale pipeline)
* Acceptance criteria: rout after sustained flanking fire; routed units flee
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

from envs.sim.battalion import Battalion
from envs.sim.combat import CombatState, apply_casualties, MORALE_CASUALTY_WEIGHT
from envs.sim.morale import (
    DEFAULT_COHESION_THRESHOLD,
    DEFAULT_ROUT_THRESHOLD,
    FLANK_STRESSOR_MULT,
    REAR_STRESSOR_MULT,
    MoraleConfig,
    cohesion_modifier,
    compute_flank_stressor,
    compute_recovery,
    is_dispersed,
    rout_velocity,
    update_morale,
)
from envs.battalion_env import BattalionEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(morale: float = 1.0, is_routing: bool = False) -> CombatState:
    """Return a fresh CombatState with a given morale."""
    s = CombatState(morale=morale, is_routing=is_routing)
    return s


def _default_config() -> MoraleConfig:
    return MoraleConfig()


def _seeded_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# MoraleConfig
# ---------------------------------------------------------------------------


class TestMoraleConfig(unittest.TestCase):
    """Verify MoraleConfig defaults and validation."""

    def test_default_cohesion_threshold(self) -> None:
        c = MoraleConfig()
        self.assertEqual(c.cohesion_threshold, DEFAULT_COHESION_THRESHOLD)

    def test_default_rout_threshold(self) -> None:
        c = MoraleConfig()
        self.assertEqual(c.rout_threshold, DEFAULT_ROUT_THRESHOLD)

    def test_default_recovery_rates_positive(self) -> None:
        c = MoraleConfig()
        self.assertGreater(c.base_recovery_rate, 0.0)
        self.assertGreater(c.distance_recovery_bonus, 0.0)
        self.assertGreater(c.friendly_support_bonus, 0.0)
        self.assertGreater(c.commander_proximity_bonus, 0.0)

    def test_rout_threshold_must_be_less_than_cohesion(self) -> None:
        with self.assertRaises(ValueError):
            # rout_threshold equal to cohesion_threshold → invalid
            MoraleConfig(cohesion_threshold=0.3, rout_threshold=0.3)

    def test_rout_threshold_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(cohesion_threshold=0.5, rout_threshold=0.0)

    def test_cohesion_threshold_must_not_exceed_one(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(cohesion_threshold=1.1, rout_threshold=0.25)

    def test_negative_recovery_rate_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(base_recovery_rate=-0.01)

    def test_negative_distance_recovery_bonus_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(distance_recovery_bonus=-0.01)

    def test_negative_friendly_support_bonus_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(friendly_support_bonus=-0.01)

    def test_negative_commander_proximity_bonus_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(commander_proximity_bonus=-0.01)

    def test_negative_rally_threshold_multiplier_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(rally_threshold_multiplier=-1.0)

    def test_rally_probability_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(rally_probability=1.5)

    def test_rally_probability_below_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(rally_probability=-0.1)

    def test_zero_rout_speed_multiplier_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoraleConfig(rout_speed_multiplier=0.0)

    def test_custom_valid_config(self) -> None:
        c = MoraleConfig(
            cohesion_threshold=0.6,
            rout_threshold=0.2,
            base_recovery_rate=0.02,
            commander_range=500.0,
        )
        self.assertEqual(c.cohesion_threshold, 0.6)
        self.assertEqual(c.rout_threshold, 0.2)

    def test_configurable_recovery_rate(self) -> None:
        """Morale recovery rate should be configurable (acceptance criterion)."""
        easy = MoraleConfig(base_recovery_rate=0.05)
        hard = MoraleConfig(base_recovery_rate=0.001)
        # Recovery from far away should be higher in easy config
        easy_recovery = compute_recovery(1000.0, easy)
        hard_recovery = compute_recovery(1000.0, hard)
        self.assertGreater(easy_recovery, hard_recovery)


# ---------------------------------------------------------------------------
# compute_flank_stressor
# ---------------------------------------------------------------------------


class TestComputeFlankStressor(unittest.TestCase):
    """Verify flanking and rear stressor penalties."""

    def test_frontal_attack_no_penalty(self) -> None:
        """Frontal hit should produce zero extra stressor."""
        # Target at (100, 0) facing east (θ=0); attacker at (200, 0) is in FRONT
        # (target's front faces east, so attacker to the east is frontal)
        penalty = compute_flank_stressor(
            attacker_x=200.0, attacker_y=0.0,
            target_x=100.0, target_y=0.0,
            target_theta=0.0,
            base_damage=0.05,
        )
        self.assertAlmostEqual(penalty, 0.0)

    def test_flank_attack_applies_penalty(self) -> None:
        """Attack from the side should apply FLANK_STRESSOR_MULT penalty."""
        # Attacker directly to the north, target facing east → flanking
        penalty = compute_flank_stressor(
            attacker_x=100.0, attacker_y=100.0,
            target_x=100.0, target_y=0.0,
            target_theta=0.0,      # facing east
            base_damage=0.1,
        )
        expected = FLANK_STRESSOR_MULT * 0.1
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_rear_attack_applies_double_penalty(self) -> None:
        """Attack from the rear should apply REAR_STRESSOR_MULT penalty."""
        # Target at (100, 0) facing east; attacker at (0, 0) is BEHIND the target
        penalty = compute_flank_stressor(
            attacker_x=0.0, attacker_y=0.0,
            target_x=100.0, target_y=0.0,
            target_theta=0.0,      # facing east; attacker is to the west (rear)
            base_damage=0.1,
        )
        expected = REAR_STRESSOR_MULT * 0.1
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_zero_damage_yields_zero_penalty(self) -> None:
        """Zero base_damage should always produce zero penalty."""
        penalty = compute_flank_stressor(
            attacker_x=100.0, attacker_y=100.0,
            target_x=100.0, target_y=0.0,
            target_theta=0.0,
            base_damage=0.0,
        )
        self.assertAlmostEqual(penalty, 0.0)

    def test_rear_penalty_exceeds_flank_penalty(self) -> None:
        """Rear stressor must be larger than flank stressor for same damage."""
        flank = compute_flank_stressor(
            attacker_x=100.0, attacker_y=100.0,
            target_x=100.0, target_y=0.0,
            target_theta=0.0,
            base_damage=0.05,
        )
        rear = compute_flank_stressor(
            attacker_x=0.0, attacker_y=0.0,
            target_x=100.0, target_y=0.0,
            target_theta=0.0,
            base_damage=0.05,
        )
        self.assertGreater(rear, flank)


# ---------------------------------------------------------------------------
# compute_recovery
# ---------------------------------------------------------------------------


class TestComputeRecovery(unittest.TestCase):
    """Verify morale recovery from distance, support, and commander."""

    def test_base_recovery_always_present(self) -> None:
        c = MoraleConfig()
        recovery = compute_recovery(enemy_dist=0.0, config=c)
        self.assertGreaterEqual(recovery, c.base_recovery_rate)

    def test_negative_enemy_dist_clamped_to_zero(self) -> None:
        """Negative enemy_dist should be clamped and not produce negative recovery."""
        c = MoraleConfig()
        recovery_zero = compute_recovery(enemy_dist=0.0, config=c)
        recovery_negative = compute_recovery(enemy_dist=-100.0, config=c)
        self.assertAlmostEqual(recovery_zero, recovery_negative, places=6)
        self.assertGreaterEqual(recovery_negative, 0.0)

    def test_distant_enemy_increases_recovery(self) -> None:
        c = MoraleConfig()
        near = compute_recovery(enemy_dist=10.0, config=c)
        far = compute_recovery(enemy_dist=c.safe_distance, config=c)
        self.assertGreater(far, near)

    def test_recovery_capped_at_safe_distance(self) -> None:
        """Recovery should not grow beyond what safe_distance gives."""
        c = MoraleConfig()
        at_safe = compute_recovery(enemy_dist=c.safe_distance, config=c)
        beyond_safe = compute_recovery(enemy_dist=c.safe_distance * 10, config=c)
        self.assertAlmostEqual(at_safe, beyond_safe, places=6)

    def test_friendly_support_adds_bonus(self) -> None:
        c = MoraleConfig()
        without = compute_recovery(enemy_dist=500.0, config=c, friendly_dist=None)
        with_friend = compute_recovery(
            enemy_dist=500.0, config=c, friendly_dist=c.commander_range * 0.5
        )
        self.assertAlmostEqual(with_friend - without, c.friendly_support_bonus, places=6)

    def test_friendly_support_only_when_in_range(self) -> None:
        c = MoraleConfig()
        out_of_range = compute_recovery(
            enemy_dist=500.0, config=c, friendly_dist=c.commander_range * 2
        )
        base = compute_recovery(enemy_dist=500.0, config=c, friendly_dist=None)
        self.assertAlmostEqual(out_of_range, base, places=6)

    def test_commander_proximity_adds_bonus(self) -> None:
        c = MoraleConfig()
        without = compute_recovery(enemy_dist=500.0, config=c, commander_dist=None)
        with_co = compute_recovery(
            enemy_dist=500.0, config=c, commander_dist=c.commander_range * 0.5
        )
        self.assertAlmostEqual(
            with_co - without, c.commander_proximity_bonus, places=6
        )

    def test_commander_out_of_range_no_bonus(self) -> None:
        c = MoraleConfig()
        without = compute_recovery(enemy_dist=500.0, config=c, commander_dist=None)
        out_of_range = compute_recovery(
            enemy_dist=500.0, config=c, commander_dist=c.commander_range + 1.0
        )
        self.assertAlmostEqual(out_of_range, without, places=6)

    def test_all_bonuses_stack(self) -> None:
        """All three bonuses (distance, friendly, commander) should stack."""
        c = MoraleConfig()
        none = compute_recovery(enemy_dist=0.0, config=c)
        all_bonuses = compute_recovery(
            enemy_dist=c.safe_distance,
            config=c,
            friendly_dist=0.0,
            commander_dist=0.0,
        )
        self.assertGreater(all_bonuses, none)


# ---------------------------------------------------------------------------
# cohesion_modifier
# ---------------------------------------------------------------------------


class TestCohesionModifier(unittest.TestCase):
    """Verify cohesion effectiveness modifier curve."""

    def test_full_morale_returns_one(self) -> None:
        c = MoraleConfig()
        self.assertAlmostEqual(cohesion_modifier(1.0, c), 1.0)

    def test_at_threshold_returns_one(self) -> None:
        c = MoraleConfig()
        self.assertAlmostEqual(cohesion_modifier(c.cohesion_threshold, c), 1.0)

    def test_half_threshold_returns_between_half_and_one(self) -> None:
        c = MoraleConfig()
        val = cohesion_modifier(c.cohesion_threshold / 2, c)
        self.assertGreater(val, 0.5)
        self.assertLess(val, 1.0)

    def test_zero_morale_returns_half(self) -> None:
        """At morale=0 cohesion should be 0.5 (unit still partially functional)."""
        c = MoraleConfig()
        self.assertAlmostEqual(cohesion_modifier(0.0, c), 0.5)

    def test_above_threshold_always_full_cohesion(self) -> None:
        c = MoraleConfig()
        for morale in [0.6, 0.8, 1.0]:
            self.assertAlmostEqual(cohesion_modifier(morale, c), 1.0, msg=f"morale={morale}")

    def test_cohesion_monotone_with_morale(self) -> None:
        """Higher morale must give equal or higher cohesion modifier."""
        c = MoraleConfig()
        values = [cohesion_modifier(m / 10.0, c) for m in range(11)]
        for i in range(len(values) - 1):
            self.assertLessEqual(values[i], values[i + 1])


# ---------------------------------------------------------------------------
# update_morale
# ---------------------------------------------------------------------------


class TestUpdateMorale(unittest.TestCase):
    """Verify the full per-step morale update logic."""

    def test_damage_reduces_morale(self) -> None:
        """Accumulated damage should reduce morale via MORALE_CASUALTY_WEIGHT."""
        config = MoraleConfig()
        state = _make_state(morale=1.0)
        state.accumulated_damage = 0.1
        update_morale(state, enemy_dist=500.0, config=config, rng=_seeded_rng())
        expected_morale = max(0.0, 1.0 - 0.1 * MORALE_CASUALTY_WEIGHT)
        self.assertAlmostEqual(state.morale, expected_morale, places=6)

    def test_flank_penalty_adds_to_morale_loss(self) -> None:
        """Flank penalty should add to the morale hit beyond casualty weight."""
        config = MoraleConfig()
        # State A: damage only
        state_a = _make_state(morale=1.0)
        state_a.accumulated_damage = 0.05
        update_morale(state_a, enemy_dist=500.0, config=config, rng=_seeded_rng(0))

        # State B: same damage + flank penalty
        state_b = _make_state(morale=1.0)
        state_b.accumulated_damage = 0.05
        update_morale(
            state_b, enemy_dist=500.0, config=config,
            flank_penalty=0.03, rng=_seeded_rng(0)
        )

        self.assertLess(state_b.morale, state_a.morale)

    def test_recovery_when_not_under_fire(self) -> None:
        """Morale should recover when accumulated_damage is zero."""
        config = MoraleConfig()
        state = _make_state(morale=0.7)
        state.accumulated_damage = 0.0
        update_morale(state, enemy_dist=config.safe_distance, config=config)
        self.assertGreater(state.morale, 0.7)

    def test_no_recovery_under_fire(self) -> None:
        """Morale should NOT recover when the unit is taking damage."""
        config = MoraleConfig(base_recovery_rate=0.1)
        state = _make_state(morale=0.9)
        state.accumulated_damage = 0.01
        morale_before = state.morale - 0.01 * MORALE_CASUALTY_WEIGHT
        update_morale(state, enemy_dist=500.0, config=config)
        self.assertAlmostEqual(state.morale, max(0.0, morale_before), places=6)

    def test_morale_clamped_to_zero(self) -> None:
        """Morale should never go below zero."""
        config = MoraleConfig()
        state = _make_state(morale=0.05)
        state.accumulated_damage = 1.0  # massive hit
        update_morale(state, enemy_dist=500.0, config=config)
        self.assertGreaterEqual(state.morale, 0.0)

    def test_morale_clamped_to_one(self) -> None:
        """Morale should never exceed one."""
        config = MoraleConfig(base_recovery_rate=1.0)
        state = _make_state(morale=0.99)
        state.accumulated_damage = 0.0
        update_morale(state, enemy_dist=config.safe_distance, config=config)
        self.assertLessEqual(state.morale, 1.0)

    def test_routes_below_rout_threshold_probabilistically(self) -> None:
        """Unit should eventually route after enough morale damage."""
        config = MoraleConfig()
        rng = np.random.default_rng(42)
        state = _make_state(morale=config.rout_threshold * 0.1)  # very low morale
        routed = False
        for _ in range(100):
            state.accumulated_damage = 0.0
            result = update_morale(state, enemy_dist=50.0, config=config, rng=rng)
            if result:
                routed = True
                break
        self.assertTrue(routed, "Unit should route at very low morale")

    def test_does_not_route_at_high_morale(self) -> None:
        """Unit should not route at high morale."""
        config = MoraleConfig()
        rng = np.random.default_rng(0)
        state = _make_state(morale=0.9)
        for _ in range(100):
            state.accumulated_damage = 0.0
            update_morale(state, enemy_dist=500.0, config=config, rng=rng)
        self.assertFalse(state.is_routing)

    def test_routing_unit_can_rally(self) -> None:
        """A routing unit with high morale should eventually rally."""
        config = MoraleConfig(rally_probability=1.0)  # always rally if morale gated
        rng = np.random.default_rng(0)
        rally_gate_morale = config.rout_threshold * config.rally_threshold_multiplier + 0.1
        state = _make_state(morale=rally_gate_morale, is_routing=True)
        state.accumulated_damage = 0.0
        result = update_morale(state, enemy_dist=1000.0, config=config, rng=rng)
        self.assertFalse(result, "Unit should rally with high morale and 100% rally_probability")

    def test_routing_unit_does_not_rally_below_gate(self) -> None:
        """Routing unit below rally gate should never rally."""
        config = MoraleConfig(rally_probability=1.0)
        rng = np.random.default_rng(0)
        # Morale below the rally gate (half-way between zero and rally gate)
        below_gate_morale = config.rout_threshold * config.rally_threshold_multiplier * 0.5
        state = _make_state(morale=below_gate_morale, is_routing=True)
        state.accumulated_damage = 0.0
        for _ in range(10):
            update_morale(state, enemy_dist=50.0, config=config, rng=rng)
        self.assertTrue(state.is_routing)


# ---------------------------------------------------------------------------
# Acceptance criteria: sustained flanking fire causes routing
# ---------------------------------------------------------------------------


class TestFlankingRouting(unittest.TestCase):
    """AC: Battalion routs reliably after sustained flanking fire."""

    def test_routs_after_sustained_flanking_fire(self) -> None:
        """Sustained flanking fire should drive morale to rout level."""
        config = MoraleConfig()
        rng = np.random.default_rng(42)

        # Unit with moderate morale taking flanking fire each step.
        # Target at (100, 0) facing east (θ=0); attacker to the north at
        # (100, 100) → 90° flank hit.
        state = _make_state(morale=1.0)
        damage_per_step = 0.02   # moderate damage from the flank

        routed = False
        for _ in range(200):
            state.reset_step_accumulators()
            state.accumulated_damage = damage_per_step

            # Attacker to the north (flanking) of eastward-facing target
            flank = compute_flank_stressor(
                attacker_x=100.0, attacker_y=100.0,
                target_x=100.0, target_y=0.0,
                target_theta=0.0,
                base_damage=damage_per_step,
            )
            result = update_morale(
                state,
                enemy_dist=100.0,
                config=config,
                flank_penalty=flank,
                rng=rng,
            )
            if result:
                routed = True
                break

        self.assertTrue(routed, "Unit should rout after sustained flanking fire")

    def test_frontal_fire_is_slower_to_route(self) -> None:
        """Frontal fire at same damage should take more steps to route than flanking."""
        config = MoraleConfig()
        damage_per_step = 0.02
        # Flanking penalty (attacker to north of eastward-facing target)
        flank_penalty = compute_flank_stressor(
            attacker_x=100.0, attacker_y=100.0,
            target_x=100.0, target_y=0.0,
            target_theta=0.0,
            base_damage=damage_per_step,
        )

        def steps_to_rout(penalty: float, seed: int) -> int:
            rng = np.random.default_rng(seed)
            state = _make_state(morale=1.0)
            for step in range(500):
                state.reset_step_accumulators()
                state.accumulated_damage = damage_per_step
                result = update_morale(
                    state,
                    enemy_dist=100.0,
                    config=config,
                    flank_penalty=penalty,
                    rng=rng,
                )
                if result:
                    return step
            return 500

        frontal_steps = steps_to_rout(0.0, seed=42)
        flanking_steps = steps_to_rout(flank_penalty, seed=42)
        self.assertLessEqual(flanking_steps, frontal_steps,
                             "Flanking fire should cause faster routing")


# ---------------------------------------------------------------------------
# rout_velocity
# ---------------------------------------------------------------------------


class TestRoutVelocity(unittest.TestCase):
    """Verify routing movement direction and speed."""

    def test_velocity_points_away_from_enemy(self) -> None:
        """Rout velocity should point directly away from the enemy."""
        config = MoraleConfig()
        # Unit at origin, enemy to the east — should flee west
        vx, vy = rout_velocity(0.0, 0.0, 100.0, 0.0, max_speed=50.0, config=config)
        self.assertLess(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0, places=6)

    def test_velocity_magnitude(self) -> None:
        """Speed should equal max_speed × rout_speed_multiplier."""
        config = MoraleConfig(rout_speed_multiplier=2.0)
        max_speed = 50.0
        vx, vy = rout_velocity(0.0, 0.0, 100.0, 0.0, max_speed=max_speed, config=config)
        speed = math.sqrt(vx ** 2 + vy ** 2)
        self.assertAlmostEqual(speed, max_speed * 2.0, places=5)

    def test_velocity_diagonal_enemy(self) -> None:
        """Should correctly compute direction for a diagonal enemy position."""
        config = MoraleConfig()
        # Unit at origin, enemy at (100, 100) — should flee SW
        vx, vy = rout_velocity(0.0, 0.0, 100.0, 100.0, max_speed=50.0, config=config)
        self.assertLess(vx, 0.0)
        self.assertLess(vy, 0.0)

    def test_velocity_unit_at_enemy_position(self) -> None:
        """Degenerate case: unit exactly at enemy position should not raise."""
        config = MoraleConfig()
        vx, vy = rout_velocity(100.0, 100.0, 100.0, 100.0, max_speed=50.0, config=config)
        speed = math.sqrt(vx ** 2 + vy ** 2)
        self.assertGreater(speed, 0.0, "Should still produce non-zero velocity")

    def test_rout_moves_unit_away(self) -> None:
        """Applying rout velocity for one step should increase distance to enemy."""
        config = MoraleConfig()
        unit_x, unit_y = 0.0, 0.0
        enemy_x, enemy_y = 100.0, 0.0
        dt = 0.1

        vx, vy = rout_velocity(unit_x, unit_y, enemy_x, enemy_y, max_speed=50.0, config=config)
        new_x = unit_x + vx * dt
        new_y = unit_y + vy * dt

        old_dist = math.sqrt((unit_x - enemy_x) ** 2 + (unit_y - enemy_y) ** 2)
        new_dist = math.sqrt((new_x - enemy_x) ** 2 + (new_y - enemy_y) ** 2)
        self.assertGreater(new_dist, old_dist)


# ---------------------------------------------------------------------------
# is_dispersed
# ---------------------------------------------------------------------------


class TestIsDispersed(unittest.TestCase):
    """Verify dispersal check."""

    def test_not_dispersed_at_full_morale(self) -> None:
        state = _make_state(morale=1.0)
        self.assertFalse(is_dispersed(state))

    def test_not_dispersed_at_low_but_nonzero_morale(self) -> None:
        state = _make_state(morale=0.01)
        self.assertFalse(is_dispersed(state))

    def test_dispersed_at_zero_morale(self) -> None:
        state = _make_state(morale=0.0)
        self.assertTrue(is_dispersed(state))

    def test_dispersed_after_morale_hits_zero(self) -> None:
        """Morale reaching zero through update_morale should trigger dispersal."""
        config = MoraleConfig()
        state = _make_state(morale=0.01)
        state.accumulated_damage = 1.0  # enough to zero morale
        update_morale(state, enemy_dist=10.0, config=config)
        self.assertTrue(is_dispersed(state))


# ---------------------------------------------------------------------------
# Integration: apply_casualties → update_morale pipeline
# ---------------------------------------------------------------------------


class TestMoraleIntegration(unittest.TestCase):
    """Integration tests: CombatState flows correctly through the morale pipeline."""

    def test_casualties_drive_routing(self) -> None:
        """Enough accumulated casualties should eventually cause routing."""
        config = MoraleConfig()
        rng = np.random.default_rng(1)

        blue = Battalion(x=200.0, y=500.0, theta=0.0, strength=1.0, team=0)
        blue_state = CombatState()

        routed = False
        for _ in range(300):
            blue_state.reset_step_accumulators()
            apply_casualties(blue, blue_state, 0.015)  # steady damage

            routing = update_morale(
                blue_state,
                enemy_dist=150.0,
                config=config,
                rng=rng,
            )
            if routing:
                routed = True
                break

        self.assertTrue(routed, "Sustained casualties should eventually cause routing")

    def test_morale_recovery_when_far_from_enemy(self) -> None:
        """Damaged morale should recover when the unit is far from the enemy."""
        config = MoraleConfig(base_recovery_rate=0.05, distance_recovery_bonus=0.05)
        rng = np.random.default_rng(0)

        state = _make_state(morale=0.6)

        # Simulate many steps without damage, far from enemy
        for _ in range(20):
            state.accumulated_damage = 0.0
            update_morale(
                state,
                enemy_dist=config.safe_distance,
                config=config,
                rng=rng,
            )

        self.assertGreater(state.morale, 0.6, "Morale should recover over time away from enemy")

    def test_commander_proximity_accelerates_recovery(self) -> None:
        """Recovery with commander nearby should be faster than without."""
        config = MoraleConfig()
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)

        state_without_co = _make_state(morale=0.5)
        state_with_co = _make_state(morale=0.5)

        steps = 10
        for _ in range(steps):
            state_without_co.accumulated_damage = 0.0
            update_morale(state_without_co, enemy_dist=500.0, config=config, rng=rng_a)

            state_with_co.accumulated_damage = 0.0
            update_morale(
                state_with_co, enemy_dist=500.0, config=config,
                commander_dist=config.commander_range * 0.5, rng=rng_b,
            )

        self.assertGreater(
            state_with_co.morale,
            state_without_co.morale,
            "Commander proximity should accelerate morale recovery",
        )

    def test_rout_movement_occurs_on_routing_step_in_env(self) -> None:
        """In BattalionEnv with morale_config, rout movement must occur on the
        same step routing is triggered (not just on subsequent steps)."""
        config = MoraleConfig(rout_speed_multiplier=2.0)
        env = BattalionEnv(
            morale_config=config,
            randomize_terrain=False,
            curriculum_level=1,  # Red stationary, does not fire
        )
        env.reset(seed=42)

        # Force Blue into routing state on first step of the episode
        env.blue_state.is_routing = True
        env.blue.routed = True
        env.blue_state.morale = 0.1  # low but non-zero

        blue_x_before = env.blue.x
        blue_y_before = env.blue.y
        red_x = env.red.x
        red_y = env.red.y

        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)

        # After the step, Blue must have moved away from Red
        dist_before = math.sqrt((blue_x_before - red_x) ** 2 + (blue_y_before - red_y) ** 2)
        dist_after = math.sqrt((env.blue.x - red_x) ** 2 + (env.blue.y - red_y) ** 2)
        self.assertGreater(
            dist_after, dist_before,
            "Routing Blue should move away from Red in the same step routing is set",
        )
        env.close()


if __name__ == "__main__":
    unittest.main()
