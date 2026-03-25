# SPDX-License-Identifier: MIT
# envs/options.py
"""Semi-MDP Options framework.

Defines the :class:`Option` dataclass and the six-element macro-action
vocabulary used by :class:`~envs.smdp_wrapper.SMDPWrapper`.

Macro-action vocabulary
-----------------------
Each ``Option`` has three components following Sutton, Precup & Singh (1999):

* *Initiation set* ``I(s) → bool``  — can the option start in state ``s``?
* *Policy* ``π(s) → a``             — primitive action while the option is active.
* *Termination* ``β(s, k) → bool``  — should the option end after ``k`` steps?

The six standard macro-actions are:

.. list-table::
   :header-rows: 1

   * - Index
     - Name
     - Behaviour
   * - 0
     - ``advance_sector``
     - Forward movement with suppression fire.
   * - 1
     - ``defend_position``
     - Hold ground, maximum sustained fire.
   * - 2
     - ``flank_left``
     - Move while rotating counter-clockwise (left flank).
   * - 3
     - ``flank_right``
     - Move while rotating clockwise (right flank).
   * - 4
     - ``withdraw``
     - Retreat at full speed, no fire.
   * - 5
     - ``concentrate_fire``
     - Stationary with maximum fire, slight tracking rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import numpy as np

__all__ = [
    "MacroAction",
    "Option",
    "make_default_options",
    # Self-state observation index constants
    "OBS_SELF_X",
    "OBS_SELF_Y",
    "OBS_COS_THETA",
    "OBS_SIN_THETA",
    "OBS_STRENGTH",
    "OBS_MORALE",
]

# ---------------------------------------------------------------------------
# Self-state observation indices
# These are fixed for every observation regardless of team size:
#   obs[0:6] = [x/w, y/h, cos θ, sin θ, strength, morale]
# ---------------------------------------------------------------------------

OBS_SELF_X: int = 0
OBS_SELF_Y: int = 1
OBS_COS_THETA: int = 2
OBS_SIN_THETA: int = 3
OBS_STRENGTH: int = 4
OBS_MORALE: int = 5

# ---------------------------------------------------------------------------
# Thresholds and tactic constants used by the default option set
# ---------------------------------------------------------------------------

_LOW_MORALE: float = 0.35
_LOW_STRENGTH: float = 0.40
_ROUT_THRESHOLD: float = 0.25
_SAFE_MORALE: float = 0.60

# Primitive-action components for each macro-action
_ADVANCE_SPEED: float = 0.8       # forward intensity for advance_sector
_SUPPRESSION_FIRE: float = 0.2    # fire intensity while advancing
_FLANK_SPEED: float = 0.6         # forward intensity for flanking manoeuvres
_TRACKING_ROTATE: float = 0.1     # slight rotation for concentrate_fire


# ---------------------------------------------------------------------------
# Macro-action enum
# ---------------------------------------------------------------------------


class MacroAction(IntEnum):
    """Indices for the six standard macro-actions in the SMDP vocabulary."""

    ADVANCE_SECTOR = 0
    DEFEND_POSITION = 1
    FLANK_LEFT = 2
    FLANK_RIGHT = 3
    WITHDRAW = 4
    CONCENTRATE_FIRE = 5


# ---------------------------------------------------------------------------
# Option dataclass
# ---------------------------------------------------------------------------


@dataclass
class Option:
    """An SMDP Option — a temporally-extended macro-action.

    Parameters
    ----------
    name:
        Human-readable label used in logging and debugging.
    initiation_set:
        Callable ``(obs: np.ndarray) -> bool``.  Returns ``True`` if this
        option may be initiated from the current local observation.
    policy:
        Callable ``(obs: np.ndarray) -> np.ndarray``.  Maps the agent's
        current local observation to a primitive action of shape ``(3,)``:
        ``[move ∈ [-1,1], rotate ∈ [-1,1], fire ∈ [0,1]]``.
    termination:
        Callable ``(obs: np.ndarray, steps_active: int) -> bool``.  Returns
        ``True`` when the option should end.  ``steps_active`` counts how
        many primitive steps this option has been running (≥ 1 on first
        call after the initial step).
    max_steps:
        Hard cap on option duration in primitive steps.  The option is
        forced to terminate once ``steps_active >= max_steps`` regardless
        of the ``termination`` callable.
    """

    name: str
    initiation_set: Callable[[np.ndarray], bool]
    policy: Callable[[np.ndarray], np.ndarray]
    termination: Callable[[np.ndarray, int], bool]
    max_steps: int = 50

    def can_initiate(self, obs: np.ndarray) -> bool:
        """Return ``True`` if this option can be initiated from *obs*."""
        return bool(self.initiation_set(obs))

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return a primitive action from this option's policy for *obs*.

        The returned action is validated to ensure it matches the expected
        shape ``(3,)`` and contains only finite values.  This provides
        earlier and more informative errors than downstream index failures
        in the environment step logic.

        Raises
        ------
        ValueError
            If the policy returns an action with shape other than ``(3,)``
            or containing non-finite values (NaN or Inf).
        """
        action = np.asarray(self.policy(obs), dtype=np.float32)

        if action.shape != (3,):
            raise ValueError(
                f"Option '{self.name}' policy returned action with shape "
                f"{action.shape!r}, but expected shape (3,)."
            )

        if not np.all(np.isfinite(action)):
            raise ValueError(
                f"Option '{self.name}' policy returned non-finite action "
                f"values: {action!r}"
            )

        return action

    def should_terminate(self, obs: np.ndarray, steps_active: int) -> bool:
        """Return ``True`` if this option should terminate.

        Terminates when the hard cap ``max_steps`` is reached **or** the
        caller-supplied ``termination`` callable returns ``True``.
        """
        if steps_active >= self.max_steps:
            return True
        return bool(self.termination(obs, steps_active))


# ---------------------------------------------------------------------------
# Default option factory
# ---------------------------------------------------------------------------


def make_default_options(max_steps: int = 30) -> list[Option]:
    """Return the six-element default macro-action vocabulary.

    All option policies use only the agent's **self-state** (``obs[0:6]``)
    which occupies fixed positions in every observation regardless of team
    size.  This makes the options environment-agnostic.

    Parameters
    ----------
    max_steps:
        Maximum primitive steps per option execution (time-limit
        termination).  Flanking options use ``max_steps // 2`` as their
        cap to keep them short and decisive.

    Returns
    -------
    list[Option]
        Six ``Option`` objects in :class:`MacroAction` index order:
        ``[advance_sector, defend_position, flank_left, flank_right,
        withdraw, concentrate_fire]``.
    """
    if max_steps < 1:
        raise ValueError(
            f"max_steps must be >= 1, got {max_steps}. "
            "Non-positive values would create options that terminate immediately "
            "and make flanking durations inconsistent with the documented behaviour."
        )

    # ------------------------------------------------------------------
    # Shared termination predicates
    # ------------------------------------------------------------------

    def _routing(obs: np.ndarray, _steps: int) -> bool:
        """Unit is routing (morale below rout threshold)."""
        return float(obs[OBS_MORALE]) < _ROUT_THRESHOLD

    def _low_strength(obs: np.ndarray, _steps: int) -> bool:
        """Unit has taken significant casualties."""
        return float(obs[OBS_STRENGTH]) < _LOW_STRENGTH

    def _recovered(obs: np.ndarray, _steps: int) -> bool:
        """Unit morale has recovered enough to stop withdrawing."""
        return float(obs[OBS_MORALE]) > _SAFE_MORALE

    # Flanking options run for half the normal duration
    flank_max = max(1, max_steps // 2)

    # ------------------------------------------------------------------
    # ADVANCE_SECTOR (index 0)
    # Move forward aggressively with suppression fire.
    # Distinct pattern: high forward speed + moderate fire.
    # ------------------------------------------------------------------
    advance = Option(
        name="advance_sector",
        initiation_set=lambda obs: float(obs[OBS_MORALE]) >= _LOW_MORALE,
        policy=lambda obs: np.array([_ADVANCE_SPEED, 0.0, _SUPPRESSION_FIRE], dtype=np.float32),
        termination=lambda obs, s: _routing(obs, s) or _low_strength(obs, s),
        max_steps=max_steps,
    )

    # ------------------------------------------------------------------
    # DEFEND_POSITION (index 1)
    # Hold ground with maximum sustained fire, no movement.
    # Distinct pattern: zero movement + full fire.
    # ------------------------------------------------------------------
    defend = Option(
        name="defend_position",
        initiation_set=lambda obs: True,
        policy=lambda obs: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        termination=_routing,
        max_steps=max_steps,
    )

    # ------------------------------------------------------------------
    # FLANK_LEFT (index 2)
    # Move while rotating counter-clockwise to attack the enemy's flank.
    # Distinct pattern: moderate forward + full left rotation + no fire.
    # ------------------------------------------------------------------
    flank_left = Option(
        name="flank_left",
        initiation_set=lambda obs: float(obs[OBS_MORALE]) >= _LOW_MORALE,
        policy=lambda obs: np.array([_FLANK_SPEED, -1.0, 0.0], dtype=np.float32),
        termination=_routing,
        max_steps=flank_max,
    )

    # ------------------------------------------------------------------
    # FLANK_RIGHT (index 3)
    # Move while rotating clockwise to attack the enemy's flank.
    # Distinct pattern: moderate forward + full right rotation + no fire.
    # ------------------------------------------------------------------
    flank_right = Option(
        name="flank_right",
        initiation_set=lambda obs: float(obs[OBS_MORALE]) >= _LOW_MORALE,
        policy=lambda obs: np.array([_FLANK_SPEED, 1.0, 0.0], dtype=np.float32),
        termination=_routing,
        max_steps=flank_max,
    )

    # ------------------------------------------------------------------
    # WITHDRAW (index 4)
    # Retreat at full speed away from enemies; no fire.
    # Initiation gated on low morale or low strength.
    # Distinct pattern: full backward speed + no fire.
    # ------------------------------------------------------------------
    withdraw = Option(
        name="withdraw",
        initiation_set=lambda obs: (
            float(obs[OBS_MORALE]) < _LOW_MORALE
            or float(obs[OBS_STRENGTH]) < _LOW_STRENGTH
        ),
        policy=lambda obs: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        termination=_recovered,
        max_steps=max_steps,
    )

    # ------------------------------------------------------------------
    # CONCENTRATE_FIRE (index 5)
    # Stationary with maximum sustained fire and slight tracking rotation.
    # Distinct pattern: zero movement + tracking rotation + full fire.
    # ------------------------------------------------------------------
    concentrate = Option(
        name="concentrate_fire",
        initiation_set=lambda obs: True,
        policy=lambda obs: np.array([0.0, _TRACKING_ROTATE, 1.0], dtype=np.float32),
        termination=lambda obs, s: _routing(obs, s) or _low_strength(obs, s),
        max_steps=max_steps,
    )

    return [advance, defend, flank_left, flank_right, withdraw, concentrate]
