# training/hrl_curriculum.py
"""E3.4 — Hierarchical Curriculum: bottom-up training scheduler.

Implements the three-phase curriculum for the HRL hierarchy:

    Phase 1 — Battalion Training (MAPPO 2v2)
    Phase 2 — Brigade Training   (PPO on BrigadeEnv, frozen battalion policies)
    Phase 3 — Division Training  (PPO on DivisionEnv, frozen brigade policies)

Each phase has **dual promotion criteria**:

1. **Win-rate criterion** — rolling win rate over the last ``win_rate_window``
   episodes must reach ``win_rate_threshold``.
2. **Elo criterion** — the agent's Elo rating (maintained by an optional
   :class:`~training.elo.EloRegistry`) must reach ``elo_threshold`` when
   evaluated against ``elo_opponent``.

Both criteria must be satisfied simultaneously to promote when they are
enabled. The Elo criterion is only applied when ``elo_threshold`` is not
``None``; to disable Elo-based promotion entirely (e.g. when no Elo registry
is available or Elo is not being tracked), set ``elo_threshold=None`` so that
promotion depends solely on the win-rate criterion.

Typical usage::

    from training.hrl_curriculum import HRLPhase, HRLCurriculumScheduler

    scheduler = HRLCurriculumScheduler(
        win_rate_threshold=0.70,
        win_rate_window=50,
        elo_threshold=800.0,
    )

    # After each evaluation episode:
    scheduler.record_episode(win=True)

    # After computing Elo:
    scheduler.update_elo(elo_registry, agent_name="phase1_run_v1",
                         opponent="scripted_l3", win_rate=0.72, n_episodes=20)

    if scheduler.should_promote():
        new_phase = scheduler.promote()
        print(f"Promoted to {new_phase.name}")
"""

from __future__ import annotations

import logging
from collections import deque
from enum import IntEnum
from typing import Deque, Optional

log = logging.getLogger(__name__)

__all__ = [
    "HRLPhase",
    "HRLCurriculumScheduler",
    "PHASE_LABELS",
    "PHASE_DESCRIPTIONS",
]


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class HRLPhase(IntEnum):
    """Ordered phases of the HRL bottom-up curriculum.

    The integer value acts as the phase index (1-based for human readability).
    """

    PHASE_1_BATTALION = 1  #: Train battalion-level MAPPO agents
    PHASE_2_BRIGADE = 2    #: Train brigade-level PPO with frozen battalions
    PHASE_3_DIVISION = 3   #: Train division-level PPO with frozen brigades


#: Human-readable phase labels.
PHASE_LABELS: dict[HRLPhase, str] = {
    HRLPhase.PHASE_1_BATTALION: "Phase 1 — Battalion",
    HRLPhase.PHASE_2_BRIGADE:   "Phase 2 — Brigade",
    HRLPhase.PHASE_3_DIVISION:  "Phase 3 — Division",
}

#: Short phase keys (used in W&B metric names).
PHASE_KEYS: dict[HRLPhase, str] = {
    HRLPhase.PHASE_1_BATTALION: "battalion",
    HRLPhase.PHASE_2_BRIGADE:   "brigade",
    HRLPhase.PHASE_3_DIVISION:  "division",
}

#: One-line description of what each phase trains.
PHASE_DESCRIPTIONS: dict[HRLPhase, str] = {
    HRLPhase.PHASE_1_BATTALION: (
        "Train MAPPO battalion agents (2v2). "
        "Freeze checkpoint on promotion."
    ),
    HRLPhase.PHASE_2_BRIGADE: (
        "Train PPO brigade commander with frozen battalion policies. "
        "Freeze checkpoint on promotion."
    ),
    HRLPhase.PHASE_3_DIVISION: (
        "Train PPO division commander with frozen brigade policies. "
        "Final phase — curriculum complete on criterion."
    ),
}


# ---------------------------------------------------------------------------
# HRLCurriculumScheduler
# ---------------------------------------------------------------------------


class HRLCurriculumScheduler:
    """Tracks episode outcomes and enforces dual promotion criteria.

    Dual criteria
    -------------
    * **Win-rate** — rolling mean over the last ``win_rate_window`` episodes
      must be ``>= win_rate_threshold``.
    * **Elo** (optional) — when ``elo_threshold`` is set and a rating has
      been recorded via :meth:`update_elo`, the agent's Elo rating must be
      ``>= elo_threshold``.

    Parameters
    ----------
    win_rate_threshold:
        Minimum rolling win rate (in ``[0, 1]``) required to promote.
    win_rate_window:
        Rolling window size for win-rate computation.
    elo_threshold:
        Minimum Elo rating required to promote.  Set to ``None`` to disable
        the Elo criterion.
    initial_phase:
        Starting phase.  Defaults to :attr:`HRLPhase.PHASE_1_BATTALION`.
    """

    def __init__(
        self,
        win_rate_threshold: float = 0.70,
        win_rate_window: int = 50,
        elo_threshold: Optional[float] = 800.0,
        initial_phase: HRLPhase = HRLPhase.PHASE_1_BATTALION,
    ) -> None:
        if not (0.0 < win_rate_threshold <= 1.0):
            raise ValueError(
                f"win_rate_threshold must be in (0, 1], got {win_rate_threshold}"
            )
        if win_rate_window < 1:
            raise ValueError(
                f"win_rate_window must be >= 1, got {win_rate_window}"
            )

        self.win_rate_threshold = float(win_rate_threshold)
        self.win_rate_window = int(win_rate_window)
        self.elo_threshold: Optional[float] = (
            float(elo_threshold) if elo_threshold is not None else None
        )
        self._phase: HRLPhase = initial_phase

        # Rolling window of episode outcomes (True=win, False=loss/draw)
        self._outcomes: Deque[bool] = deque(maxlen=win_rate_window)

        # Most-recently observed Elo rating for the current phase's agent
        self._current_elo: Optional[float] = None

        # Cumulative episode counter (resets on promotion)
        self._phase_episodes: int = 0
        self._total_episodes: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> HRLPhase:
        """The current curriculum phase."""
        return self._phase

    @property
    def phase_label(self) -> str:
        """Human-readable label for the current phase."""
        return PHASE_LABELS[self._phase]

    @property
    def phase_key(self) -> str:
        """Short key for the current phase (e.g. ``"battalion"``)."""
        return PHASE_KEYS[self._phase]

    @property
    def is_final_phase(self) -> bool:
        """``True`` when the scheduler is at the last phase (Division)."""
        return self._phase == HRLPhase.PHASE_3_DIVISION

    @property
    def total_episodes(self) -> int:
        """Total episodes recorded across all phases."""
        return self._total_episodes

    @property
    def phase_episodes(self) -> int:
        """Episodes recorded in the current phase (resets on promotion)."""
        return self._phase_episodes

    # ------------------------------------------------------------------
    # Episode tracking
    # ------------------------------------------------------------------

    def record_episode(self, win: bool) -> None:
        """Record the outcome of a completed episode.

        Parameters
        ----------
        win:
            ``True`` if the Blue team won, ``False`` otherwise.
        """
        self._outcomes.append(bool(win))
        self._phase_episodes += 1
        self._total_episodes += 1

    def win_rate(self) -> float:
        """Return the rolling win rate over the last ``win_rate_window`` episodes.

        Returns ``0.0`` if no episodes have been recorded yet.
        """
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    # ------------------------------------------------------------------
    # Elo tracking
    # ------------------------------------------------------------------

    def update_elo(
        self,
        elo_registry: object,
        agent_name: str,
        opponent: str,
        win_rate: float,
        n_episodes: int = 20,
    ) -> float:
        """Update the Elo registry and cache the agent's new rating.

        Parameters
        ----------
        elo_registry:
            An :class:`~training.elo.EloRegistry` instance.
        agent_name:
            Registry key for the current phase's agent.
        opponent:
            Registry key for the evaluation opponent (e.g. ``"scripted_l3"``).
        win_rate:
            Win rate over the evaluation batch (in ``[0, 1]``).
        n_episodes:
            Number of evaluation episodes played.

        Returns
        -------
        float
            Elo rating delta (positive = improved).
        """
        delta = elo_registry.update(
            agent=agent_name,
            opponent=opponent,
            outcome=float(win_rate),
            n_games=n_episodes,
        )
        self._current_elo = elo_registry.get_rating(agent_name)
        log.info(
            "HRL Elo update [%s]: agent=%s elo=%.1f delta=%+.1f",
            self.phase_key,
            agent_name,
            self._current_elo,
            delta,
        )
        return delta

    def set_elo(self, elo: float) -> None:
        """Directly set the cached Elo rating (useful for testing or manual init).

        Parameters
        ----------
        elo:
            Elo rating to record for the current phase.
        """
        self._current_elo = float(elo)

    # ------------------------------------------------------------------
    # Promotion logic
    # ------------------------------------------------------------------

    def _win_rate_met(self) -> bool:
        """Return ``True`` if the win-rate criterion is satisfied."""
        if len(self._outcomes) < self.win_rate_window:
            return False
        return self.win_rate() >= self.win_rate_threshold

    def _elo_met(self) -> bool:
        """Return ``True`` if the Elo criterion is satisfied (or not required)."""
        if self.elo_threshold is None:
            return True
        if self._current_elo is None:
            return False
        return self._current_elo >= self.elo_threshold

    def should_promote(self) -> bool:
        """Return ``True`` if **all** promotion criteria are satisfied.

        Criteria checked:
        * The rolling win-rate window is full (``win_rate_window`` episodes).
        * Rolling win rate ``>= win_rate_threshold``.
        * Elo rating ``>= elo_threshold`` (if configured).
        * Not already at the final phase.
        """
        if self.is_final_phase:
            return False
        return self._win_rate_met() and self._elo_met()

    def promotion_status(self) -> dict:
        """Return a dict summarising the current promotion check results.

        Keys:
        * ``"phase"`` — integer phase index
        * ``"phase_label"`` — human-readable phase name
        * ``"win_rate"`` — current rolling win rate
        * ``"win_rate_met"`` — bool
        * ``"elo"`` — current cached Elo (or ``None``)
        * ``"elo_met"`` — bool
        * ``"should_promote"`` — bool (both criteria met)
        * ``"phase_episodes"`` — episodes in current phase
        * ``"total_episodes"`` — all-time episodes
        """
        return {
            "phase": int(self._phase),
            "phase_label": self.phase_label,
            "win_rate": self.win_rate(),
            "win_rate_met": self._win_rate_met(),
            "elo": self._current_elo,
            "elo_met": self._elo_met(),
            "should_promote": self.should_promote(),
            "phase_episodes": self._phase_episodes,
            "total_episodes": self._total_episodes,
        }

    def promote(self) -> HRLPhase:
        """Advance to the next phase and reset per-phase counters.

        Returns
        -------
        HRLPhase
            The new phase after promotion.

        Raises
        ------
        RuntimeError
            If called when already at the final phase.
        """
        if self.is_final_phase:
            raise RuntimeError(
                f"Cannot promote past the final curriculum phase "
                f"({self.phase_label})."
            )

        old_phase = self._phase
        self._phase = HRLPhase(int(self._phase) + 1)

        # Reset per-phase counters
        self._outcomes.clear()
        self._current_elo = None
        self._phase_episodes = 0

        log.info(
            "HRL curriculum promoted: %s → %s (total_episodes=%d)",
            PHASE_LABELS[old_phase],
            PHASE_LABELS[self._phase],
            self._total_episodes,
        )
        return self._phase

    # ------------------------------------------------------------------
    # W&B helpers
    # ------------------------------------------------------------------

    def wandb_metrics(self) -> dict:
        """Return a dict of W&B metrics for the current scheduler state.

        Keys:
        * ``hrl_curriculum/phase`` — integer phase index
        * ``hrl_curriculum/phase_label`` — e.g. ``"Phase 1 — Battalion"``
        * ``hrl_curriculum/win_rate`` — rolling win rate
        * ``hrl_curriculum/elo`` — cached Elo (or 0.0 if not set)
        * ``hrl_curriculum/phase_episodes`` — episodes in current phase
        * ``hrl_curriculum/total_episodes`` — all-time episodes
        """
        return {
            "hrl_curriculum/phase": int(self._phase),
            "hrl_curriculum/phase_label": self.phase_label,
            "hrl_curriculum/win_rate": self.win_rate(),
            "hrl_curriculum/elo": self._current_elo or 0.0,
            "hrl_curriculum/phase_episodes": self._phase_episodes,
            "hrl_curriculum/total_episodes": self._total_episodes,
        }

    def log_promotion_event(
        self,
        total_steps: int,
        wandb_run: object = None,
    ) -> None:
        """Log a phase-promotion event to W&B (and the Python logger).

        Parameters
        ----------
        total_steps:
            Current total environment step count (W&B x-axis).
        wandb_run:
            Active ``wandb.run`` object.  When ``None`` the event is only
            written to the Python logger.
        """
        metrics = self.wandb_metrics()
        metrics["hrl_curriculum/promotion_step"] = total_steps
        log.info(
            "HRL curriculum phase transition → %s at step %d "
            "(win_rate=%.3f, elo=%s, total_episodes=%d)",
            self.phase_label,
            total_steps,
            metrics["hrl_curriculum/win_rate"],
            metrics["hrl_curriculum/elo"],
            self._total_episodes,
        )
        if wandb_run is not None:
            try:
                import wandb  # local import

                wandb.log(metrics, step=total_steps)
            except Exception as exc:  # pragma: no cover
                log.warning("W&B logging failed during HRL promotion: %s", exc)
