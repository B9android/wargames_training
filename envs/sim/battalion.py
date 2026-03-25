# SPDX-License-Identifier: MIT
# envs/sim/battalion.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from envs.sim.weapons import WeaponProfile
    from envs.sim.formations import Formation

@dataclass
class Battalion:
    x: float          # position
    y: float
    theta: float      # facing angle (radians)
    strength: float   # 0.0 to 1.0
    team: int         # 0 = blue, 1 = red

    # Morale and routing
    morale: float = 1.0           # 0.0 to 1.0, decreases under fire
    routing_threshold: float = 0.3  # morale below this triggers routing
    routed: bool = False          # True if battalion has routed

    # Formation cohesion bonus (multiplicative modifier to effectiveness)
    cohesion_bonus: float = 1.0   # 1.0 = normal, >1.0 = bonus from formation

    # Physical limits
    max_speed: float = 50.0      # meters per second (scaled)
    max_turn_rate: float = 0.1   # radians per step
    fire_range: float = 200.0    # meters
    fire_arc: float = np.pi / 4  # ±45° frontal arc

    # Optional weapon profile — when set, enables range-band accuracy and
    # reload-cycle mechanics via envs.sim.weapons.
    weapon_profile: Optional["WeaponProfile"] = field(default=None, repr=False)

    # Formation system — discrete tactical state and transition tracking.
    # formation is stored as int (Formation enum value) to avoid circular
    # imports; use Formation(self.formation) to access the enum.
    formation: int = 0              # Formation.LINE = 0 (default)
    formation_transition_steps: int = 0   # steps remaining in current transition
    target_formation: Optional[int] = None  # Formation value being transitioned to

    def move(self, vx: float, vy: float, dt: float = 0.1):
        """Apply velocity, clamp to max speed."""
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            vx, vy = vx / speed * self.max_speed, vy / speed * self.max_speed
        self.x += vx * dt
        self.y += vy * dt

    def rotate(self, delta_theta: float):
        """Apply rotation, clamp to max turn rate."""
        delta_theta = np.clip(delta_theta, -self.max_turn_rate, self.max_turn_rate)
        self.theta = (self.theta + delta_theta) % (2 * np.pi)

    def can_fire_at(self, target: 'Battalion') -> bool:
        """Check if target is in range and within frontal fire arc."""
        dx = target.x - self.x
        dy = target.y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > self.fire_range:
            return False
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = abs((angle_to_target - self.theta + np.pi) % (2 * np.pi) - np.pi)
        return angle_diff < self.fire_arc

    def fire_at(self, target: 'Battalion', intensity: float) -> float:
        """Deal damage. Returns damage dealt."""
        if not self.can_fire_at(target):
            return 0.0
        dx = target.x - self.x
        dy = target.y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        # Damage falls off with range
        range_factor = 1.0 - (dist / self.fire_range)
        # Apply cohesion bonus to damage output
        damage = intensity * range_factor * 0.05 * self.cohesion_bonus
        target.strength = max(0.0, target.strength - damage)
        return damage

    def take_damage(self, damage: float, morale_impact: float = 0.1):
        """
        Apply damage to strength and reduce morale.

        Args:
            damage: Amount of strength damage to take
            morale_impact: Morale loss per unit of damage (default 0.1)
        """
        self.strength = max(0.0, self.strength - damage)
        # Morale drops proportionally to damage taken
        morale_loss = damage * morale_impact
        self.morale = max(0.0, self.morale - morale_loss)
        # Check if battalion should route
        if self.morale <= self.routing_threshold and not self.routed:
            self.routed = True

    def check_routing(self) -> bool:
        """
        Check if battalion should route based on morale.

        Returns:
            True if battalion has routed, False otherwise
        """
        if not self.routed and self.morale <= self.routing_threshold:
            self.routed = True
        return self.routed

    def rally(self, morale_gain: float = 0.1):
        """
        Attempt to rally a routing or demoralized battalion.

        Args:
            morale_gain: Amount of morale to restore
        """
        self.morale = min(1.0, self.morale + morale_gain)
        # Can only rally if morale recovers significantly above threshold
        if self.routed and self.morale > self.routing_threshold * 1.5:
            self.routed = False

