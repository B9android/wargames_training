# envs/reward.py
"""Shaped reward computation for BattalionEnv.

Provides configurable reward components that supply a dense learning signal
to supplement the sparse win/loss outcome:

* ``delta_enemy_strength`` — reward proportional to damage dealt to Red
* ``delta_own_strength``   — penalty proportional to damage taken by Blue
* ``survival_bonus``       — small per-step reward scaled by Blue's remaining strength
* ``win_bonus``            — terminal bonus when Blue wins (Red routed/destroyed)
* ``loss_penalty``         — terminal penalty when Blue loses (Blue routed/destroyed)
* ``time_penalty``         — per-step negative reward to discourage stalling

Typical usage::

    from envs.reward import RewardWeights, compute_reward

    weights = RewardWeights()
    components = compute_reward(
        dmg_b2r=0.05,
        dmg_r2b=0.02,
        blue_strength=0.9,
        blue_won=False,
        blue_lost=False,
        weights=weights,
    )
    total_reward = components.total
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RewardWeights", "RewardComponents", "compute_reward"]


@dataclass
class RewardWeights:
    """Configurable multipliers for each reward component.

    All weights are applied by multiplying against their respective raw
    component value before summing into the total reward.  Set a weight
    to ``0.0`` to disable that component entirely.

    Parameters
    ----------
    delta_enemy_strength:
        Multiplier applied to the fraction of enemy strength destroyed in a
        step (``dmg_b2r``).  Encourages the agent to deal damage.
    delta_own_strength:
        Multiplier applied to the fraction of own strength lost in a step
        (``dmg_r2b``).  The contribution is negated before summing.
    survival_bonus:
        Per-step bonus scaled by Blue's current strength.  Set to ``0.0``
        (default) to disable; a small positive value (e.g. ``0.005``)
        rewards staying alive longer.
    win_bonus:
        Terminal reward added when Blue wins (Red routed or destroyed).
    loss_penalty:
        Terminal reward added when Blue loses (Blue routed or destroyed).
        Should be negative.
    time_penalty:
        Constant added every step.  A small negative value (e.g. ``-0.01``)
        discourages unnecessary stalling.
    """

    delta_enemy_strength: float = 5.0
    delta_own_strength: float = 5.0
    survival_bonus: float = 0.0
    win_bonus: float = 10.0
    loss_penalty: float = -10.0
    time_penalty: float = -0.01


@dataclass
class RewardComponents:
    """Per-component reward breakdown for a single environment step.

    Use ``components.total`` to obtain the scalar reward to return from
    ``env.step()``.  The individual fields can be logged to W&B for
    analysis of the learning signal.
    """

    delta_enemy_strength: float = 0.0
    delta_own_strength: float = 0.0
    survival_bonus: float = 0.0
    win_bonus: float = 0.0
    loss_penalty: float = 0.0
    time_penalty: float = 0.0

    @property
    def total(self) -> float:
        """Scalar sum of all reward components."""
        return (
            self.delta_enemy_strength
            + self.delta_own_strength
            + self.survival_bonus
            + self.win_bonus
            + self.loss_penalty
            + self.time_penalty
        )

    def as_dict(self) -> dict[str, float]:
        """Return a ``{component_name: value}`` mapping suitable for logging."""
        return {
            "reward/delta_enemy_strength": self.delta_enemy_strength,
            "reward/delta_own_strength": self.delta_own_strength,
            "reward/survival_bonus": self.survival_bonus,
            "reward/win_bonus": self.win_bonus,
            "reward/loss_penalty": self.loss_penalty,
            "reward/time_penalty": self.time_penalty,
            "reward/total": self.total,
        }


def compute_reward(
    *,
    dmg_b2r: float,
    dmg_r2b: float,
    blue_strength: float,
    blue_won: bool,
    blue_lost: bool,
    weights: RewardWeights,
) -> RewardComponents:
    """Compute all reward components for a single environment step.

    Parameters
    ----------
    dmg_b2r:
        Damage dealt by Blue to Red this step (strength-fraction units,
        typically in ``[0, 1]``).
    dmg_r2b:
        Damage dealt by Red to Blue this step.
    blue_strength:
        Blue's current strength after casualties (used for the survival
        bonus), in ``[0, 1]``.
    blue_won:
        ``True`` when the episode terminates with Red defeated.
    blue_lost:
        ``True`` when the episode terminates with Blue defeated.
    weights:
        :class:`RewardWeights` multipliers for each component.

    Returns
    -------
    RewardComponents
        Individual reward components; call ``.total`` for the scalar sum
        or ``.as_dict()`` for a loggable mapping.
    """
    comps = RewardComponents()
    comps.delta_enemy_strength = dmg_b2r * weights.delta_enemy_strength
    comps.delta_own_strength = -dmg_r2b * weights.delta_own_strength
    comps.survival_bonus = blue_strength * weights.survival_bonus
    comps.time_penalty = weights.time_penalty
    if blue_won:
        comps.win_bonus = weights.win_bonus
    if blue_lost:
        comps.loss_penalty = weights.loss_penalty
    return comps
