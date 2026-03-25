# SPDX-License-Identifier: MIT
"""Adaptive temporal abstraction for hierarchical RL (E3.5).

This module provides :class:`AdaptiveTemporalScheduler` which adjusts the
*temporal abstraction ratio* — how many primitive battalion steps each
brigade macro-step spans — based on episode progress.

The ratio directly controls the ``max_steps`` parameter passed to
:func:`~envs.options.make_default_options`, and therefore determines how long
each selected option can run before the brigade agent must issue a new
macro-command.

Typical use in training::

    from training.adaptive_temporal import AdaptiveTemporalScheduler

    scheduler = AdaptiveTemporalScheduler(base_ratio=10, adaptation="fixed")

    # At the start of a new episode, update the environment's options:
    options = scheduler.make_options(episode_progress=0.0)
    env._options = options  # or pass at BrigadeEnv construction time

    # Mid-episode, refresh options to reflect the current progress:
    options = scheduler.make_options(episode_progress=0.5)
    env._options = options

Temporal abstraction ratios and their characteristics
------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Ratio
     - Option duration (primitive steps)
     - Characteristic
   * - 5
     - Very short — almost primitive
     - Commander retains fine-grained control; low temporal abstraction
   * - 10
     - Short–medium
     - Balanced; recommended default after E3.5 sweep
   * - 20
     - Medium–long
     - True macro-commands; brigade acts on slower timescale
   * - 50
     - Long
     - Very coarse; battalion policies dominate; hierarchy benefit is high

The E3.5 sweep tests ratios 5, 10, 20, and 50 to find the optimal value.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from envs.options import Option, make_default_options

__all__ = [
    "AdaptiveTemporalScheduler",
    "AdaptationStrategy",
    "SWEEP_RATIOS",
]

log = logging.getLogger(__name__)

#: Temporal abstraction ratios evaluated in the E3.5 sweep.
#: These values correspond exactly to the ``env.temporal_ratio`` parameter
#: values listed in ``configs/sweeps/temporal_ratio_sweep.yaml``.
SWEEP_RATIOS: tuple[int, ...] = (5, 10, 20, 50)

#: Adaptation strategy type alias.
AdaptationStrategy = Literal["fixed", "linear_decrease", "linear_increase"]


class AdaptiveTemporalScheduler:
    """Adapt the temporal abstraction ratio over the course of an episode.

    The *temporal abstraction ratio* K controls how many primitive battalion
    steps each brigade macro-step spans (i.e. the ``max_steps`` hard cap for
    every option).  This scheduler can keep K fixed throughout training or
    linearly interpolate it from one bound to the other as the episode
    progresses.

    Parameters
    ----------
    base_ratio:
        The fixed ratio used when ``adaptation="fixed"``.  Not used by the
        linear strategies, which interpolate between ``min_ratio`` and
        ``max_ratio`` regardless of this value.
    min_ratio:
        Lower bound on the ratio (inclusive).  Must be ``>= 1``.
        For ``"linear_decrease"``: ratio at episode end (progress = 1).
        For ``"linear_increase"``: ratio at episode start (progress = 0).
    max_ratio:
        Upper bound on the ratio (inclusive).  Must be ``>= min_ratio``.
        For ``"linear_decrease"``: ratio at episode start (progress = 0).
        For ``"linear_increase"``: ratio at episode end (progress = 1).
    adaptation:
        One of:

        ``"fixed"``
            K = ``base_ratio`` throughout the episode.
        ``"linear_decrease"``
            K decreases linearly from ``max_ratio`` at episode start
            (progress = 0) to ``min_ratio`` at episode end (progress = 1).
            Models coarse manoeuvring early in the episode and fine-grained
            control during the decisive engagement phase.
        ``"linear_increase"``
            K increases linearly from ``min_ratio`` to ``max_ratio``.
            Models tight control at episode start expanding to longer
            committed options as the battle stabilises.

    Examples
    --------
    Fixed ratio (use in sweep):

    >>> sched = AdaptiveTemporalScheduler(base_ratio=10)
    >>> sched.get_ratio(0.0)
    10
    >>> sched.get_ratio(1.0)
    10

    Linearly decreasing ratio:

    >>> sched = AdaptiveTemporalScheduler(
    ...     min_ratio=5, max_ratio=20, adaptation="linear_decrease"
    ... )
    >>> sched.get_ratio(0.0)   # episode start → max_ratio
    20
    >>> sched.get_ratio(1.0)   # episode end → min_ratio
    5
    """

    def __init__(
        self,
        base_ratio: int = 10,
        min_ratio: int = 5,
        max_ratio: int = 20,
        adaptation: AdaptationStrategy = "fixed",
    ) -> None:
        if int(base_ratio) < 1:
            raise ValueError(f"base_ratio must be >= 1, got {base_ratio}")
        if int(min_ratio) < 1:
            raise ValueError(f"min_ratio must be >= 1, got {min_ratio}")
        if int(max_ratio) < int(min_ratio):
            raise ValueError(
                f"max_ratio ({max_ratio}) must be >= min_ratio ({min_ratio})"
            )
        valid_strategies: tuple[str, ...] = (
            "fixed",
            "linear_decrease",
            "linear_increase",
        )
        if adaptation not in valid_strategies:
            raise ValueError(
                f"adaptation must be one of {valid_strategies!r}, "
                f"got {adaptation!r}"
            )

        self.base_ratio: int = int(base_ratio)
        self.min_ratio: int = int(min_ratio)
        self.max_ratio: int = int(max_ratio)
        self.adaptation: AdaptationStrategy = adaptation

        # Initialise current_ratio to the value that get_ratio(0.0) would return
        # so the property is always consistent before the first call.
        if adaptation == "linear_decrease":
            self._current_ratio: int = self.max_ratio
        elif adaptation == "linear_increase":
            self._current_ratio = self.min_ratio
        else:
            self._current_ratio = self.base_ratio

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_ratio(self) -> int:
        """The temporal ratio most recently returned by :meth:`get_ratio`."""
        return self._current_ratio

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_ratio(self, episode_progress: float) -> int:
        """Return the temporal ratio for the given episode progress.

        Parameters
        ----------
        episode_progress:
            Scalar in ``[0.0, 1.0]`` where ``0.0`` is the episode start and
            ``1.0`` is the episode end.  Values outside this range are
            clipped silently.

        Returns
        -------
        int
            Current option ``max_steps`` cap (always ``>= 1``).
        """
        p = float(np.clip(episode_progress, 0.0, 1.0))

        if self.adaptation == "fixed":
            ratio_f = float(self.base_ratio)
        elif self.adaptation == "linear_decrease":
            # max_ratio at p=0 → min_ratio at p=1
            ratio_f = self.max_ratio + p * (self.min_ratio - self.max_ratio)
        else:  # linear_increase
            # min_ratio at p=0 → max_ratio at p=1
            ratio_f = self.min_ratio + p * (self.max_ratio - self.min_ratio)

        self._current_ratio = max(1, round(ratio_f))
        return self._current_ratio

    def make_options(self, episode_progress: float = 0.0) -> list[Option]:
        """Return a fresh option vocabulary sized for the given episode progress.

        Calls :func:`~envs.options.make_default_options` with ``max_steps``
        set to the value returned by :meth:`get_ratio`.

        Parameters
        ----------
        episode_progress:
            Scalar in ``[0.0, 1.0]``.

        Returns
        -------
        list[Option]
            Six :class:`~envs.options.Option` objects whose ``max_steps`` are
            derived from the current temporal ratio.  Most options use this
            ratio as their cap directly; flanking options (``flank_left``,
            ``flank_right``) use a shorter cap (``ratio // 2``) as defined
            by :func:`~envs.options.make_default_options`.
        """
        ratio = self.get_ratio(episode_progress)
        log.debug(
            "AdaptiveTemporalScheduler: progress=%.3f → ratio=%d (adaptation=%s)",
            episode_progress,
            ratio,
            self.adaptation,
        )
        return make_default_options(max_steps=ratio)

    def wandb_config(self) -> dict:
        """Return a dict of scheduler parameters suitable for W&B ``run.config``.

        Usage::

            import wandb
            sched = AdaptiveTemporalScheduler(base_ratio=10)
            wandb.init(config={**other_cfg, **sched.wandb_config()})

        Returns
        -------
        dict
            Keys: ``temporal_ratio`` (initial ratio for the chosen strategy),
            ``temporal_ratio_min``, ``temporal_ratio_max``,
            ``temporal_adaptation``.
        """
        return {
            "temporal_ratio": self.current_ratio,
            "temporal_ratio_min": self.min_ratio,
            "temporal_ratio_max": self.max_ratio,
            "temporal_adaptation": self.adaptation,
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveTemporalScheduler("
            f"base_ratio={self.base_ratio}, "
            f"min_ratio={self.min_ratio}, "
            f"max_ratio={self.max_ratio}, "
            f"adaptation={self.adaptation!r})"
        )
