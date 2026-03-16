# envs/sim/battalion.py

import numpy as np
from dataclasses import dataclass

@dataclass
class Battalion:
    x: float          # position
    y: float
    theta: float      # facing angle (radians)
    strength: float   # 0.0 to 1.0
    team: int         # 0 = blue, 1 = red

    # Physical limits
    max_speed: float = 50.0      # meters per second (scaled)
    max_turn_rate: float = 0.1   # radians per step
    fire_range: float = 200.0    # meters
    fire_arc: float = np.pi / 4  # ±45° frontal arc

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
        damage = intensity * range_factor * 0.05  # tune this
        target.strength = max(0.0, target.strength - damage)
        return damage

