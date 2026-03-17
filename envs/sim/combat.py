# Firepower, damage, morale

import numpy as np

from envs.sim.battalion import Battalion

# Morale penalty scale: strength loss * scale → morale penalty (capped at 1.0)
MORALE_DAMAGE_SCALE = 0.5

# Base multiplier that converts (intensity × range_factor) into a strength loss fraction.
BASE_DAMAGE_MULTIPLIER = 0.05


def range_factor(dist: float, fire_range: float) -> float:
    """Linear damage falloff: 1.0 at zero range, 0.0 at max fire_range, clipped to [0, 1].

    Raises ValueError if fire_range is not positive.
    """
    if fire_range <= 0:
        raise ValueError(f"fire_range must be positive, got {fire_range}")
    return float(np.clip(1.0 - dist / fire_range, 0.0, 1.0))


def in_fire_range(attacker: Battalion, target: Battalion) -> bool:
    """Return True if target is within attacker's fire range."""
    dx = target.x - attacker.x
    dy = target.y - attacker.y
    dist = np.sqrt(dx**2 + dy**2)
    return bool(dist <= attacker.fire_range)


def in_fire_arc(attacker: Battalion, target: Battalion) -> bool:
    """Return True if target lies within attacker's frontal fire arc."""
    dx = target.x - attacker.x
    dy = target.y - attacker.y
    angle_to_target = np.arctan2(dy, dx)
    angle_diff = abs((angle_to_target - attacker.theta + np.pi) % (2 * np.pi) - np.pi)
    return bool(angle_diff < attacker.fire_arc)


def compute_damage(attacker: Battalion, target: Battalion, intensity: float) -> float:
    """Compute damage dealt by attacker on target at given intensity.

    Returns 0.0 if the target is out of range or outside the frontal fire arc.
    Damage falls off linearly from full intensity at zero range to zero at fire_range.
    Negative intensity is clamped to zero (no accidental healing).
    """
    intensity = max(0.0, intensity)
    if not attacker.can_fire_at(target):
        return 0.0
    dx = target.x - attacker.x
    dy = target.y - attacker.y
    dist = np.sqrt(dx**2 + dy**2)
    rf = range_factor(dist, attacker.fire_range)
    return intensity * rf * BASE_DAMAGE_MULTIPLIER


def resolve_fire(attacker: Battalion, target: Battalion, intensity: float) -> float:
    """Apply combat: compute damage and reduce target strength. Returns damage dealt."""
    damage = compute_damage(attacker, target, intensity)
    target.strength = max(0.0, target.strength - damage)
    return damage


def morale_penalty(strength_before: float, strength_after: float) -> float:
    """Compute a morale penalty based on strength loss.

    Returns a value in [0, 1] representing how much morale is affected.
    A unit that takes heavy casualties suffers a proportionally large morale hit.
    """
    loss = max(0.0, strength_before - strength_after)
    return float(np.clip(loss * MORALE_DAMAGE_SCALE, 0.0, 1.0))
