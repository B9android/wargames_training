# envs/metrics/coordination.py
"""Quantitative coordination metrics for multi-battalion combat episodes.

Provides three scalar metrics that measure emergent tactical behaviours
in multi-agent battalion combat.  All functions operate on plain sequences
of :class:`~envs.sim.battalion.Battalion` objects so they can be called
from any environment wrapper, trainer, or analysis notebook.

Metric overview
---------------

``flanking_ratio``
    Fraction of in-range blue-attacks-on-red delivered from outside the
    target's frontal arc.  A value > 0.3 indicates meaningful flanking.

``fire_concentration``
    Degree to which blue fire is focused on a single red target.  Computed
    as the fraction of firing Blue units that aim at the most-targeted Red
    unit.  Ranges from ``0.0`` (no blue unit can fire) to ``1.0`` (all
    firing blues concentrate on one target).

``mutual_support_score``
    Average fraction of ally battalions within *support_radius* of each
    Blue unit.  Ranges from ``0.0`` (no unit has any nearby ally) to
    ``1.0`` (every unit is within support range of every other unit).

Typical usage::

    from envs.metrics.coordination import compute_all

    blue = [b for b in env.battalions if b.team == 0 and not b.routed]
    red  = [b for b in env.battalions if b.team == 1 and not b.routed]
    metrics = compute_all(blue, red)
    # {'coordination/flanking_ratio': 0.5,
    #  'coordination/fire_concentration': 0.75,
    #  'coordination/mutual_support_score': 0.33}
"""

from __future__ import annotations

import math
from typing import Sequence

from envs.sim.battalion import Battalion

__all__ = [
    "flanking_ratio",
    "fire_concentration",
    "mutual_support_score",
    "compute_all",
]


def flanking_ratio(
    blue: Sequence[Battalion],
    red: Sequence[Battalion],
) -> float:
    """Fraction of in-range blue attacks delivered from outside red's frontal arc.

    For every (blue_attacker, red_target) pair where the blue unit is within
    *red_target.fire_range* of the red unit, we test whether the blue unit
    sits outside the red unit's ``±fire_arc`` frontal sector.  Such a position
    forces the red unit to rotate before it can return fire — the defining
    property of a successful flank.

    A ``flanking_ratio > 0.3`` is considered meaningful flanking behaviour
    (i.e. at least 30 % of potential engagements are from the flank/rear).

    Parameters
    ----------
    blue:
        Sequence of Blue :class:`~envs.sim.battalion.Battalion` objects
        (active or otherwise; filtering by ``routed``/``strength`` is the
        caller's responsibility).
    red:
        Sequence of Red :class:`~envs.sim.battalion.Battalion` objects.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``; ``0.0`` when no in-range pairs exist.
    """
    total_pairs = 0
    flanking_pairs = 0
    for attacker in blue:
        for target in red:
            dx = attacker.x - target.x
            dy = attacker.y - target.y
            dist = math.hypot(dx, dy)
            if dist > target.fire_range:
                continue
            # Angle from red's perspective to the blue attacker
            angle_to_attacker = math.atan2(dy, dx)
            angle_diff = abs(
                (angle_to_attacker - target.theta + math.pi) % (2 * math.pi) - math.pi
            )
            total_pairs += 1
            if angle_diff >= target.fire_arc:  # outside frontal arc → flanking
                flanking_pairs += 1
    return flanking_pairs / total_pairs if total_pairs > 0 else 0.0


def fire_concentration(
    blue: Sequence[Battalion],
    red: Sequence[Battalion],
) -> float:
    """Degree to which blue fire is focused on a single red target.

    For each Blue battalion we identify its highest-priority Red target: the
    nearest Red unit that the Blue unit can fire at (within range and inside
    its own frontal arc, per :meth:`~envs.sim.battalion.Battalion.can_fire_at`).
    We then count how many Blue units select each Red target and return the
    fraction that targets the most-targeted Red unit.

    *   ``1.0`` — all firing Blue units concentrate on one Red target.
    *   ``1/k`` — fire is spread evenly across ``k`` Red targets.
    *   ``0.0`` — no Blue unit can fire at any Red unit.

    Parameters
    ----------
    blue:
        Active Blue :class:`~envs.sim.battalion.Battalion` sequence.
    red:
        Active Red :class:`~envs.sim.battalion.Battalion` sequence.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.
    """
    if not blue or not red:
        return 0.0

    target_counts: dict[int, int] = {}
    firing_blues = 0

    for attacker in blue:
        best_dist = float("inf")
        best_idx = -1
        for idx, target in enumerate(red):
            if not attacker.can_fire_at(target):
                continue
            dx = target.x - attacker.x
            dy = target.y - attacker.y
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx >= 0:
            target_counts[best_idx] = target_counts.get(best_idx, 0) + 1
            firing_blues += 1

    if firing_blues == 0:
        return 0.0

    max_count = max(target_counts.values())
    return max_count / firing_blues


def mutual_support_score(
    blue: Sequence[Battalion],
    support_radius: float = 300.0,
) -> float:
    """Average fraction of ally battalions within mutual-support range.

    Two Blue units are considered to be in mutual support when the Euclidean
    distance between them is ≤ *support_radius*.  The score is the mean over
    all Blue units of the fraction of *other* Blue units within that radius.

    *   ``1.0`` — every Blue unit is within support range of every other.
    *   ``0.0`` — no Blue unit has any ally within support range (or fewer
        than 2 Blue units are present).

    Parameters
    ----------
    blue:
        Active Blue :class:`~envs.sim.battalion.Battalion` sequence.
    support_radius:
        Distance threshold in the same units as
        :attr:`~envs.sim.battalion.Battalion.x` / ``y`` (metres).
        Defaults to 300 m (1.5 × the default fire range).

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.
    """
    n = len(blue)
    if n < 2:
        return 0.0

    total_fraction = 0.0
    for i, unit in enumerate(blue):
        nearby = sum(
            1
            for j, other in enumerate(blue)
            if i != j
            and math.hypot(unit.x - other.x, unit.y - other.y) <= support_radius
        )
        total_fraction += nearby / (n - 1)
    return total_fraction / n


def compute_all(
    blue: Sequence[Battalion],
    red: Sequence[Battalion],
    support_radius: float = 300.0,
) -> dict[str, float]:
    """Compute all three coordination metrics and return as a loggable dict.

    Parameters
    ----------
    blue:
        Active Blue :class:`~envs.sim.battalion.Battalion` sequence.
    red:
        Active Red :class:`~envs.sim.battalion.Battalion` sequence.
    support_radius:
        Distance threshold for :func:`mutual_support_score` (metres).

    Returns
    -------
    dict[str, float]
        Keys:

        * ``"coordination/flanking_ratio"``
        * ``"coordination/fire_concentration"``
        * ``"coordination/mutual_support_score"``
    """
    return {
        "coordination/flanking_ratio": flanking_ratio(blue, red),
        "coordination/fire_concentration": fire_concentration(blue, red),
        "coordination/mutual_support_score": mutual_support_score(
            blue, support_radius
        ),
    }
