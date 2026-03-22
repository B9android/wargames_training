"""Tests for envs/sim/weapons.py — weapon profiles and reload state machine.

Historical accuracy cross-validation against Nafziger (1988):
    Musket hit probability at 50 m / 150 m / 300 m within ±5 percentage
    points of the reference values.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.weapons import (
    CANNON,
    HOWITZER,
    MUSKET,
    NAFZIGER_MUSKET_150M,
    NAFZIGER_MUSKET_300M,
    NAFZIGER_MUSKET_50M,
    RIFLE,
    RangeBand,
    ReloadMachine,
    ReloadStatus,
    WeaponProfile,
    WeaponType,
    get_range_band,
    hit_probability,
    suppression_morale_penalty,
    synchronized_volley,
    volley_readiness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Maximum allowed absolute deviation from Nafziger (1988) reference values.
NAFZIGER_TOLERANCE: float = 0.05


# ---------------------------------------------------------------------------
# WeaponProfile validation
# ---------------------------------------------------------------------------


class TestWeaponProfiles(unittest.TestCase):
    """Verify that pre-defined weapon profiles have sensible field values."""

    def _assert_valid_profile(self, profile: WeaponProfile, name: str) -> None:
        self.assertGreater(profile.max_range, 0.0, f"{name}.max_range must be > 0")
        self.assertGreater(profile.close_range, 0.0, f"{name}.close_range must be > 0")
        self.assertGreater(
            profile.effective_range,
            profile.close_range,
            f"{name}.effective_range must exceed close_range",
        )
        self.assertLess(
            profile.effective_range,
            profile.max_range,
            f"{name}.effective_range must be < max_range",
        )
        self.assertGreater(profile.base_accuracy, 0.0, f"{name}.base_accuracy must be > 0")
        self.assertGreater(profile.decay_rate, 0.0, f"{name}.decay_rate must be > 0")
        self.assertGreater(profile.reload_steps, 0, f"{name}.reload_steps must be ≥ 1")
        self.assertGreater(profile.fire_steps, 0, f"{name}.fire_steps must be ≥ 1")

    def test_musket_profile_valid(self) -> None:
        self._assert_valid_profile(MUSKET, "MUSKET")

    def test_rifle_profile_valid(self) -> None:
        self._assert_valid_profile(RIFLE, "RIFLE")

    def test_cannon_profile_valid(self) -> None:
        self._assert_valid_profile(CANNON, "CANNON")

    def test_howitzer_profile_valid(self) -> None:
        self._assert_valid_profile(HOWITZER, "HOWITZER")

    def test_musket_reload_steps_in_historical_range(self) -> None:
        self.assertGreaterEqual(MUSKET.reload_steps, 2)
        self.assertLessEqual(MUSKET.reload_steps, 4)

    def test_cannon_reload_steps_in_historical_range(self) -> None:
        self.assertGreaterEqual(CANNON.reload_steps, 6)
        self.assertLessEqual(CANNON.reload_steps, 10)

    def test_howitzer_reload_steps_in_historical_range(self) -> None:
        self.assertGreaterEqual(HOWITZER.reload_steps, 6)
        self.assertLessEqual(HOWITZER.reload_steps, 10)

    def test_weapon_types(self) -> None:
        self.assertEqual(MUSKET.weapon_type, WeaponType.MUSKET)
        self.assertEqual(RIFLE.weapon_type, WeaponType.RIFLE)
        self.assertEqual(CANNON.weapon_type, WeaponType.CANNON)
        self.assertEqual(HOWITZER.weapon_type, WeaponType.HOWITZER)

    def test_artillery_has_suppression_radius(self) -> None:
        self.assertGreater(CANNON.suppression_radius, 0.0)
        self.assertGreater(HOWITZER.suppression_radius, 0.0)

    def test_small_arms_no_suppression_radius(self) -> None:
        self.assertEqual(MUSKET.suppression_radius, 0.0)
        self.assertEqual(RIFLE.suppression_radius, 0.0)


# ---------------------------------------------------------------------------
# Historical accuracy cross-validation (Nafziger 1988)
# ---------------------------------------------------------------------------


class TestNafzigerAccuracy(unittest.TestCase):
    """Musket hit probability within ±5 pp of Nafziger (1988) reference."""

    def _check_prob(
        self,
        distance: float,
        reference: float,
        label: str,
    ) -> None:
        prob = hit_probability(MUSKET, distance)
        self.assertAlmostEqual(
            prob,
            reference,
            delta=NAFZIGER_TOLERANCE,
            msg=(
                f"Musket P(hit @ {label}) = {prob:.3f} deviates more than "
                f"±{NAFZIGER_TOLERANCE} from Nafziger reference {reference:.2f}"
            ),
        )

    def test_musket_accuracy_at_50m(self) -> None:
        """P(hit | 50 m) ≈ 0.55 (Nafziger, 1988)."""
        self._check_prob(50.0, NAFZIGER_MUSKET_50M, "50 m")

    def test_musket_accuracy_at_150m(self) -> None:
        """P(hit | 150 m) ≈ 0.25 (Nafziger, 1988)."""
        self._check_prob(150.0, NAFZIGER_MUSKET_150M, "150 m")

    def test_musket_accuracy_at_300m(self) -> None:
        """P(hit | 300 m) ≈ 0.08 (Nafziger, 1988)."""
        self._check_prob(300.0, NAFZIGER_MUSKET_300M, "300 m")

    def test_accuracy_decreases_monotonically_with_range(self) -> None:
        ranges = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
        probs = [hit_probability(MUSKET, r) for r in ranges]
        for i in range(len(probs) - 1):
            self.assertGreaterEqual(
                probs[i],
                probs[i + 1],
                msg=f"P({ranges[i]}) < P({ranges[i+1]}): accuracy should not increase with range",
            )

    def test_accuracy_zero_beyond_max_range(self) -> None:
        self.assertEqual(hit_probability(MUSKET, MUSKET.max_range + 1.0), 0.0)
        self.assertEqual(hit_probability(MUSKET, 1000.0), 0.0)

    def test_accuracy_non_negative(self) -> None:
        for dist in [0.0, 50.0, 150.0, 300.0, 500.0]:
            self.assertGreaterEqual(hit_probability(MUSKET, dist), 0.0)

    def test_formation_modifier_increases_probability(self) -> None:
        base = hit_probability(MUSKET, 100.0, formation_modifier=1.0)
        with_modifier = hit_probability(MUSKET, 100.0, formation_modifier=1.5)
        self.assertGreater(with_modifier, base)

    def test_formation_modifier_zero_returns_zero(self) -> None:
        self.assertEqual(hit_probability(MUSKET, 50.0, formation_modifier=0.0), 0.0)

    def test_negative_distance_treated_as_zero(self) -> None:
        """Negative distances should be clamped to zero range."""
        prob_at_zero = hit_probability(MUSKET, 0.0)
        prob_at_neg = hit_probability(MUSKET, -10.0)
        self.assertAlmostEqual(prob_at_neg, prob_at_zero)


# ---------------------------------------------------------------------------
# Range bands
# ---------------------------------------------------------------------------


class TestRangeBands(unittest.TestCase):
    """Verify get_range_band classification."""

    def test_zero_range_is_close(self) -> None:
        self.assertEqual(get_range_band(MUSKET, 0.0), RangeBand.CLOSE)

    def test_at_close_boundary_is_close(self) -> None:
        self.assertEqual(get_range_band(MUSKET, MUSKET.close_range), RangeBand.CLOSE)

    def test_just_above_close_boundary_is_effective(self) -> None:
        self.assertEqual(get_range_band(MUSKET, MUSKET.close_range + 1.0), RangeBand.EFFECTIVE)

    def test_at_effective_boundary_is_effective(self) -> None:
        self.assertEqual(get_range_band(MUSKET, MUSKET.effective_range), RangeBand.EFFECTIVE)

    def test_just_above_effective_boundary_is_extreme(self) -> None:
        self.assertEqual(
            get_range_band(MUSKET, MUSKET.effective_range + 1.0), RangeBand.EXTREME
        )

    def test_at_max_range_is_extreme(self) -> None:
        self.assertEqual(get_range_band(MUSKET, MUSKET.max_range), RangeBand.EXTREME)

    def test_beyond_max_range_is_out_of_range(self) -> None:
        self.assertEqual(get_range_band(MUSKET, MUSKET.max_range + 1.0), RangeBand.OUT_OF_RANGE)

    def test_negative_distance_treated_as_zero(self) -> None:
        self.assertEqual(get_range_band(MUSKET, -5.0), RangeBand.CLOSE)

    def test_cannon_range_bands(self) -> None:
        self.assertEqual(get_range_band(CANNON, 100.0), RangeBand.CLOSE)
        self.assertEqual(get_range_band(CANNON, 300.0), RangeBand.EFFECTIVE)
        self.assertEqual(get_range_band(CANNON, 600.0), RangeBand.EXTREME)
        self.assertEqual(get_range_band(CANNON, 800.0), RangeBand.OUT_OF_RANGE)


# ---------------------------------------------------------------------------
# Reload state machine
# ---------------------------------------------------------------------------


class TestReloadMachine(unittest.TestCase):
    """Verify the LOADED → FIRING → RELOADING → LOADED cycle."""

    def test_initial_status_is_loaded(self) -> None:
        m = ReloadMachine(MUSKET)
        self.assertEqual(m.status, ReloadStatus.LOADED)
        self.assertTrue(m.is_loaded)

    def test_fire_returns_true_when_loaded(self) -> None:
        m = ReloadMachine(MUSKET)
        self.assertTrue(m.fire())

    def test_status_after_fire_is_firing(self) -> None:
        m = ReloadMachine(MUSKET)
        m.fire()
        self.assertEqual(m.status, ReloadStatus.FIRING)
        self.assertFalse(m.is_loaded)

    def test_fire_blocked_during_firing(self) -> None:
        """Reload state machine must block firing while weapon is firing."""
        m = ReloadMachine(MUSKET)
        m.fire()
        self.assertFalse(m.fire(), "Should not be able to fire while already firing")

    def test_fire_blocked_during_reloading(self) -> None:
        """Reload state machine must block firing during reload cycle."""
        m = ReloadMachine(MUSKET)
        m.fire()
        # Advance through fire_steps to reach RELOADING
        for _ in range(MUSKET.fire_steps):
            m.step()
        self.assertEqual(m.status, ReloadStatus.RELOADING)
        self.assertFalse(m.fire(), "Should not be able to fire while reloading")

    def test_transitions_to_reloading_after_fire_steps(self) -> None:
        m = ReloadMachine(MUSKET)
        m.fire()
        for _ in range(MUSKET.fire_steps):
            status = m.step()
        self.assertEqual(status, ReloadStatus.RELOADING)

    def test_transitions_to_loaded_after_reload_steps(self) -> None:
        m = ReloadMachine(MUSKET)
        m.fire()
        # Complete fire phase
        for _ in range(MUSKET.fire_steps):
            m.step()
        # Complete reload phase
        for _ in range(MUSKET.reload_steps):
            status = m.step()
        self.assertEqual(status, ReloadStatus.LOADED)
        self.assertTrue(m.is_loaded)

    def test_full_cycle_musket(self) -> None:
        """Full LOADED → FIRING → RELOADING → LOADED cycle for musket."""
        m = ReloadMachine(MUSKET)
        self.assertEqual(m.status, ReloadStatus.LOADED)
        self.assertTrue(m.fire())
        self.assertEqual(m.status, ReloadStatus.FIRING)

        for _ in range(MUSKET.fire_steps):
            m.step()
        self.assertEqual(m.status, ReloadStatus.RELOADING)

        for _ in range(MUSKET.reload_steps):
            m.step()
        self.assertEqual(m.status, ReloadStatus.LOADED)
        self.assertTrue(m.is_loaded)

    def test_full_cycle_artillery(self) -> None:
        """Artillery reload cycle is longer than musket cycle."""
        cannon_machine = ReloadMachine(CANNON)
        cannon_machine.fire()
        for _ in range(CANNON.fire_steps):
            cannon_machine.step()
        self.assertEqual(cannon_machine.status, ReloadStatus.RELOADING)
        for _ in range(CANNON.reload_steps):
            cannon_machine.step()
        self.assertEqual(cannon_machine.status, ReloadStatus.LOADED)

    def test_artillery_reload_longer_than_musket(self) -> None:
        """Artillery reload steps must exceed musket reload steps."""
        self.assertGreater(CANNON.reload_steps, MUSKET.reload_steps)

    def test_step_when_loaded_is_noop(self) -> None:
        m = ReloadMachine(MUSKET)
        for _ in range(5):
            status = m.step()
        self.assertEqual(status, ReloadStatus.LOADED)

    def test_steps_remaining_decrements(self) -> None:
        m = ReloadMachine(MUSKET)
        m.fire()
        initial_steps = m.steps_remaining
        self.assertEqual(initial_steps, MUSKET.fire_steps)
        m.step()
        if MUSKET.fire_steps > 1:
            self.assertEqual(m.steps_remaining, initial_steps - 1)

    def test_reset_returns_to_loaded(self) -> None:
        m = ReloadMachine(MUSKET)
        m.fire()
        m.step()
        m.reset()
        self.assertEqual(m.status, ReloadStatus.LOADED)
        self.assertEqual(m.steps_remaining, 0)

    def test_can_fire_again_after_full_cycle(self) -> None:
        m = ReloadMachine(MUSKET)
        m.fire()
        for _ in range(MUSKET.fire_steps + MUSKET.reload_steps):
            m.step()
        self.assertTrue(m.is_loaded)
        self.assertTrue(m.fire(), "Should be able to fire again after reload")


# ---------------------------------------------------------------------------
# Suppression effect
# ---------------------------------------------------------------------------


class TestSuppressionEffect(unittest.TestCase):
    """Artillery suppression (morale penalty without casualties)."""

    def test_cannon_suppression_at_zero_distance(self) -> None:
        penalty = suppression_morale_penalty(CANNON, 0.0)
        self.assertAlmostEqual(penalty, CANNON.suppression_morale_penalty)

    def test_cannon_suppression_decreases_with_distance(self) -> None:
        p_near = suppression_morale_penalty(CANNON, 10.0)
        p_far = suppression_morale_penalty(CANNON, 40.0)
        self.assertGreater(p_near, p_far)

    def test_cannon_suppression_zero_at_radius(self) -> None:
        penalty = suppression_morale_penalty(CANNON, CANNON.suppression_radius)
        self.assertEqual(penalty, 0.0)

    def test_cannon_suppression_zero_beyond_radius(self) -> None:
        penalty = suppression_morale_penalty(CANNON, CANNON.suppression_radius + 10.0)
        self.assertEqual(penalty, 0.0)

    def test_musket_no_suppression(self) -> None:
        """Small arms must not produce suppression."""
        self.assertEqual(suppression_morale_penalty(MUSKET, 0.0), 0.0)
        self.assertEqual(suppression_morale_penalty(MUSKET, 50.0), 0.0)

    def test_rifle_no_suppression(self) -> None:
        self.assertEqual(suppression_morale_penalty(RIFLE, 0.0), 0.0)

    def test_howitzer_suppression_at_zero_distance(self) -> None:
        penalty = suppression_morale_penalty(HOWITZER, 0.0)
        self.assertAlmostEqual(penalty, HOWITZER.suppression_morale_penalty)

    def test_howitzer_larger_suppression_radius_than_cannon(self) -> None:
        self.assertGreater(HOWITZER.suppression_radius, CANNON.suppression_radius)

    def test_negative_distance_treated_as_zero(self) -> None:
        p_zero = suppression_morale_penalty(CANNON, 0.0)
        p_neg = suppression_morale_penalty(CANNON, -5.0)
        self.assertAlmostEqual(p_neg, p_zero)

    def test_suppression_non_negative(self) -> None:
        for d in [0.0, 10.0, 25.0, 50.0, 100.0]:
            self.assertGreaterEqual(suppression_morale_penalty(CANNON, d), 0.0)


# ---------------------------------------------------------------------------
# Volley synchronisation
# ---------------------------------------------------------------------------


class TestVolleySynchronisation(unittest.TestCase):
    """Coordinated fire window for a formed line."""

    def test_all_loaded_volleys_successfully(self) -> None:
        machines = [ReloadMachine(MUSKET) for _ in range(6)]
        results = synchronized_volley(machines)
        self.assertEqual(results, [True] * 6)

    def test_none_loaded_produces_no_fire(self) -> None:
        machines = [ReloadMachine(MUSKET) for _ in range(4)]
        # Fire once to put all into non-loaded state
        for m in machines:
            m.fire()
        results = synchronized_volley(machines)
        self.assertEqual(results, [False] * 4)

    def test_partial_readiness(self) -> None:
        machines = [ReloadMachine(MUSKET) for _ in range(4)]
        # Put first two into reloading
        machines[0].fire()
        machines[1].fire()
        results = synchronized_volley(machines)
        self.assertEqual(results, [False, False, True, True])

    def test_volley_readiness_all_loaded(self) -> None:
        machines = [ReloadMachine(MUSKET) for _ in range(5)]
        self.assertAlmostEqual(volley_readiness(machines), 1.0)

    def test_volley_readiness_none_loaded(self) -> None:
        machines = [ReloadMachine(MUSKET) for _ in range(5)]
        for m in machines:
            m.fire()
        self.assertAlmostEqual(volley_readiness(machines), 0.0)

    def test_volley_readiness_half_loaded(self) -> None:
        machines = [ReloadMachine(MUSKET) for _ in range(4)]
        machines[0].fire()
        machines[1].fire()
        self.assertAlmostEqual(volley_readiness(machines), 0.5)

    def test_volley_readiness_empty_sequence(self) -> None:
        self.assertEqual(volley_readiness([]), 0.0)

    def test_after_volley_machines_are_not_loaded(self) -> None:
        machines = [ReloadMachine(MUSKET) for _ in range(3)]
        synchronized_volley(machines)
        for m in machines:
            self.assertFalse(m.is_loaded)

    def test_volley_after_reload_cycle(self) -> None:
        """After completing a full reload, a second volley can be fired."""
        machines = [ReloadMachine(MUSKET) for _ in range(3)]
        synchronized_volley(machines)
        # Advance all through the full reload cycle
        for _ in range(MUSKET.fire_steps + MUSKET.reload_steps):
            for m in machines:
                m.step()
        results = synchronized_volley(machines)
        self.assertEqual(results, [True, True, True])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_hit_probability_at_zero_range_is_bounded(self) -> None:
        prob = hit_probability(MUSKET, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertGreaterEqual(prob, 0.0)

    def test_hit_probability_at_max_range(self) -> None:
        prob = hit_probability(MUSKET, MUSKET.max_range)
        # Should still be non-negative
        self.assertGreaterEqual(prob, 0.0)

    def test_weapon_profiles_are_immutable(self) -> None:
        """WeaponProfile is frozen; mutations should raise."""
        with self.assertRaises((AttributeError, TypeError)):
            MUSKET.max_range = 500.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
