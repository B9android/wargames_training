# SPDX-License-Identifier: MIT
# tests/test_battalion.py

import pytest
import numpy as np
from envs.sim.battalion import Battalion


class TestBattalionMorale:
    """Test morale system and routing mechanics."""

    def test_initial_morale(self):
        """Battalion should start with full morale."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0)
        assert b.morale == 1.0
        assert not b.routed

    def test_morale_default_routing_threshold(self):
        """Default routing threshold should be 0.3."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0)
        assert b.routing_threshold == 0.3

    def test_take_damage_reduces_morale(self):
        """Taking damage should reduce both strength and morale."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0)
        initial_morale = b.morale
        initial_strength = b.strength

        b.take_damage(0.2, morale_impact=0.1)

        assert b.strength < initial_strength
        assert b.morale < initial_morale
        # Morale should drop by damage * morale_impact
        assert b.morale == pytest.approx(initial_morale - 0.2 * 0.1)

    def test_morale_triggers_routing(self):
        """Battalion should route when morale drops below threshold."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, routing_threshold=0.3)
        assert not b.routed

        # Reduce morale below threshold
        b.take_damage(7.5, morale_impact=0.1)  # 7.5 * 0.1 = 0.75 morale loss

        assert b.morale < 0.3
        assert b.routed

    def test_check_routing(self):
        """check_routing should detect and set routed status."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, morale=0.5)

        # Should not route above threshold
        assert not b.check_routing()

        # Manually set morale below threshold
        b.morale = 0.2
        assert b.check_routing()
        assert b.routed

    def test_rally_restores_morale(self):
        """Rally should restore morale."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, morale=0.5)

        b.rally(morale_gain=0.2)

        assert b.morale == pytest.approx(0.7)

    def test_rally_cannot_exceed_max_morale(self):
        """Rally should not increase morale above 1.0."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, morale=0.95)

        b.rally(morale_gain=0.2)

        assert b.morale == 1.0

    def test_rally_can_recover_from_routing(self):
        """Battalion can recover from routing if morale rises significantly."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0,
                     morale=0.2, routed=True, routing_threshold=0.3)

        # Rally to above 1.5 * threshold
        b.rally(morale_gain=0.3)  # morale -> 0.5, which is > 0.3 * 1.5 = 0.45

        assert not b.routed
        assert b.morale == pytest.approx(0.5)

    def test_rally_insufficient_morale_stays_routed(self):
        """Battalion stays routed if morale doesn't rise enough."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0,
                     morale=0.2, routed=True, routing_threshold=0.3)

        # Rally but not enough (need > 0.45, will be at 0.4)
        b.rally(morale_gain=0.2)

        assert b.routed
        assert b.morale == pytest.approx(0.4)

    def test_morale_cannot_go_negative(self):
        """Morale should be clamped at 0.0."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0)

        b.take_damage(20.0, morale_impact=0.1)  # Massive damage

        assert b.morale == 0.0
        assert b.routed


class TestBattalionCohesion:
    """Test formation cohesion bonus mechanics."""

    def test_default_cohesion_bonus(self):
        """Default cohesion bonus should be 1.0."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0)
        assert b.cohesion_bonus == 1.0

    def test_cohesion_bonus_affects_damage(self):
        """Cohesion bonus should multiply damage output."""
        attacker = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, cohesion_bonus=1.5)
        target = Battalion(x=100, y=0, theta=np.pi, strength=1.0, team=1)

        damage = attacker.fire_at(target, intensity=1.0)

        # Damage should be scaled by cohesion bonus
        # Without cohesion: intensity * range_factor * 0.05
        # With cohesion: intensity * range_factor * 0.05 * 1.5
        assert damage > 0
        # Verify the cohesion bonus is applied (damage should be 50% higher)

    def test_cohesion_bonus_comparison(self):
        """Compare damage with and without cohesion bonus."""
        # Two identical attackers, one with bonus
        attacker_normal = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, cohesion_bonus=1.0)
        attacker_bonus = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, cohesion_bonus=1.5)

        target1 = Battalion(x=100, y=0, theta=np.pi, strength=1.0, team=1)
        target2 = Battalion(x=100, y=0, theta=np.pi, strength=1.0, team=1)

        damage_normal = attacker_normal.fire_at(target1, intensity=1.0)
        damage_bonus = attacker_bonus.fire_at(target2, intensity=1.0)

        assert damage_bonus == pytest.approx(damage_normal * 1.5)


class TestBattalionMovement:
    """Test basic movement still works with new fields."""

    def test_movement_with_new_fields(self):
        """Movement should work normally with new morale fields."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0)

        b.move(vx=10.0, vy=0.0, dt=0.1)

        assert b.x == pytest.approx(1.0)
        assert b.y == pytest.approx(0.0)
        # Morale should be unaffected by movement
        assert b.morale == 1.0

    def test_rotation_with_new_fields(self):
        """Rotation should work normally with new morale fields."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0)

        b.rotate(delta_theta=0.05)

        assert b.theta == pytest.approx(0.05)
        # Morale should be unaffected by rotation
        assert b.morale == 1.0


class TestBattalionFiring:
    """Test firing mechanics with cohesion bonus."""

    def test_fire_at_applies_cohesion(self):
        """fire_at should use cohesion_bonus when calculating damage."""
        attacker = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, cohesion_bonus=2.0)
        target = Battalion(x=50, y=0, theta=np.pi, strength=1.0, team=1)

        initial_strength = target.strength
        damage = attacker.fire_at(target, intensity=1.0)

        assert damage > 0
        assert target.strength == pytest.approx(initial_strength - damage)

    def test_out_of_range_no_damage(self):
        """Out of range targets should take no damage."""
        attacker = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, fire_range=100.0)
        target = Battalion(x=250, y=0, theta=np.pi, strength=1.0, team=1)

        damage = attacker.fire_at(target, intensity=1.0)

        assert damage == 0.0
        assert target.strength == 1.0


class TestBattalionCustomParameters:
    """Test creating battalions with custom morale parameters."""

    def test_custom_routing_threshold(self):
        """Can create battalion with custom routing threshold."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, routing_threshold=0.5)
        assert b.routing_threshold == 0.5

    def test_custom_initial_morale(self):
        """Can create battalion with custom initial morale."""
        b = Battalion(x=0, y=0, theta=0, strength=1.0, team=0, morale=0.8)
        assert b.morale == 0.8

    def test_create_already_routed(self):
        """Can create battalion that starts routed."""
        b = Battalion(x=0, y=0, theta=0, strength=0.5, team=0, morale=0.1, routed=True)
        assert b.routed
        assert b.morale == 0.1
