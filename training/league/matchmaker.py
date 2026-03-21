# training/league/matchmaker.py
"""League matchmaker using Prioritized Fictitious Self-Play (PFSP) (E4.1).

The :class:`LeagueMatchmaker` combines an :class:`~training.league.agent_pool.AgentPool`
and a :class:`~training.league.match_database.MatchDatabase` to select
opponents for each league agent according to their historical win rates.

PFSP overview
-------------
For a focal agent *A* the probability of selecting opponent *O* is::

    P(O | A) ∝ f(win_rate(A, O))

where the default weighting function is ``f(w) = (1 - w)`` (hard-first).
When ``win_rate(A, O)`` is unknown (no recorded matches), the focal agent
is assumed to have a 50 % win rate against *O*, giving weight ``0.5``.

Agent-type matchup rules (AlphaStar-style)
------------------------------------------
* **MAIN_AGENT** — plays against all league members (main agents,
  exploiters, league exploiters).
* **MAIN_EXPLOITER** — plays exclusively against main agents.
* **LEAGUE_EXPLOITER** — plays against all league members.

Callers may override the opponent pool by passing an explicit
*candidate_types* list to :meth:`LeagueMatchmaker.select_opponent`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from training.league.agent_pool import AgentPool, AgentRecord, AgentType
from training.league.match_database import MatchDatabase

log = logging.getLogger(__name__)

__all__ = ["LeagueMatchmaker", "make_nash_weight_fn"]

# ---------------------------------------------------------------------------
# Default matchup rules
# ---------------------------------------------------------------------------

#: Default set of opponent types each agent role plays against.
_DEFAULT_OPPONENT_TYPES: Dict[AgentType, Optional[List[AgentType]]] = {
    AgentType.MAIN_AGENT: None,  # None → all types
    AgentType.MAIN_EXPLOITER: [AgentType.MAIN_AGENT],
    AgentType.LEAGUE_EXPLOITER: None,  # None → all types
}

#: Win rate assumed for matchups with no recorded history.
_DEFAULT_UNKNOWN_WIN_RATE: float = 0.5


# ---------------------------------------------------------------------------
# LeagueMatchmaker
# ---------------------------------------------------------------------------


class LeagueMatchmaker:
    """Select opponents for league agents using PFSP win-rate weighting.

    Parameters
    ----------
    agent_pool:
        The shared :class:`AgentPool` of league snapshots.
    match_database:
        The :class:`MatchDatabase` containing historical match outcomes.
    unknown_win_rate:
        Assumed win rate used when no historical data exists for a
        (focal, opponent) pair.  Defaults to ``0.5``.
    pfsp_weight_fn:
        Custom PFSP weighting function ``f(win_rate: float) -> float``.
        Defaults to the hard-first function ``f(w) = 1 - w``.
    """

    def __init__(
        self,
        agent_pool: AgentPool,
        match_database: MatchDatabase,
        unknown_win_rate: float = _DEFAULT_UNKNOWN_WIN_RATE,
        pfsp_weight_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        if not (0.0 <= unknown_win_rate <= 1.0):
            raise ValueError(
                f"unknown_win_rate must be in [0, 1], got {unknown_win_rate!r}"
            )
        self.agent_pool = agent_pool
        self.match_database = match_database
        self.unknown_win_rate = float(unknown_win_rate)
        self._pfsp_weight_fn = pfsp_weight_fn
        # Nash distribution weights: {agent_id: probability}.  When set,
        # overrides PFSP for opponent sampling.
        self._nash_weights: Optional[Dict[str, float]] = None
        # Shared RNG instance — avoids creating a new generator on every call
        # to select_opponent(), which would add overhead and make global
        # randomness harder to control.
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_opponent(
        self,
        focal_agent_id: str,
        *,
        candidate_types: Optional[List[Union[AgentType, str]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[AgentRecord]:
        """Select an opponent for *focal_agent_id* using PFSP.

        Parameters
        ----------
        focal_agent_id:
            ID of the agent that needs an opponent.  Must be present in
            the pool.
        candidate_types:
            Explicit list of :class:`AgentType` (or strings) to draw
            opponents from.  When ``None`` (default), the default matchup
            rules for the focal agent's type are used.
        rng:
            Optional NumPy random generator for reproducibility.

        Returns
        -------
        AgentRecord or None
            The selected opponent, or ``None`` if no eligible candidates
            exist.

        Raises
        ------
        KeyError
            If *focal_agent_id* is not found in the pool.
        """
        candidates, probs = self._candidates_and_probs(focal_agent_id, candidate_types)
        if candidates is None:
            log.debug(
                "LeagueMatchmaker: no eligible opponents for %s", focal_agent_id
            )
            return None

        _rng = rng if rng is not None else self._rng
        idx = int(_rng.choice(len(candidates), p=probs))
        chosen = candidates[idx]

        log.debug(
            "LeagueMatchmaker: selected opponent %s (type=%s) for focal %s",
            chosen.agent_id,
            chosen.agent_type.value,
            focal_agent_id,
        )
        return chosen

    def opponent_probabilities(
        self,
        focal_agent_id: str,
        *,
        candidate_types: Optional[List[Union[AgentType, str]]] = None,
    ) -> Dict[str, float]:
        """Return the PFSP sampling probability for each eligible opponent.

        Useful for debugging and visualisation.

        Parameters
        ----------
        focal_agent_id:
            ID of the focal agent.
        candidate_types:
            Same semantics as in :meth:`select_opponent`.

        Returns
        -------
        dict[str, float]
            Mapping of ``agent_id → probability``.  Sums to ``1.0`` when
            non-empty, empty dict when no candidates exist.

        Raises
        ------
        KeyError
            If *focal_agent_id* is not found in the pool.
        """
        candidates, probs = self._candidates_and_probs(focal_agent_id, candidate_types)
        if candidates is None:
            return {}
        return {c.agent_id: float(p) for c, p in zip(candidates, probs)}

    def set_weight_function(
        self, pfsp_weight_fn: Optional[Callable[[float], float]]
    ) -> None:
        """Replace the PFSP weight function used for opponent sampling.

        Parameters
        ----------
        pfsp_weight_fn:
            New PFSP weighting function ``f(win_rate: float) -> float``.
            Pass ``None`` to revert to the default hard-first function
            ``f(w) = 1 - w``.
        """
        self._pfsp_weight_fn = pfsp_weight_fn

    def set_nash_weights(
        self, nash_weights: Optional[Dict[str, float]]
    ) -> None:
        """Set a pre-computed Nash distribution as opponent sampling weights.

        When *nash_weights* is set, opponent sampling is driven by the Nash
        distribution rather than per-focal-agent PFSP win-rate weights.
        The Nash distribution is a global probability assignment over all
        agents, independent of the focal agent's win history.

        Parameters
        ----------
        nash_weights:
            Mapping of ``agent_id → probability``.  Need not be normalised;
            probabilities are re-normalised after filtering to eligible
            candidates.  Pass ``None`` to revert to PFSP sampling.
        """
        self._nash_weights = dict(nash_weights) if nash_weights is not None else None
        log.debug(
            "LeagueMatchmaker: Nash weights %s (%d agents)",
            "set" if self._nash_weights is not None else "cleared",
            len(self._nash_weights) if self._nash_weights is not None else 0,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _candidates_and_probs(
        self,
        focal_agent_id: str,
        candidate_types: Optional[List[Union[AgentType, str]]],
    ):
        """Return (candidates, probs) arrays for PFSP, or (None, None) if empty.

        Shared by :meth:`select_opponent` and :meth:`opponent_probabilities`
        to avoid duplicating the candidate-filtering and weight-calculation
        logic.
        """
        focal = self.agent_pool.get(focal_agent_id)
        recorded_wr = self.match_database.win_rates_for(focal_agent_id)

        # Resolve candidate set.
        if candidate_types is not None:
            resolved = [
                AgentType.from_str(ct) if isinstance(ct, str) else ct
                for ct in candidate_types
            ]
            candidates: List[AgentRecord] = []
            for atype in resolved:
                candidates.extend(self.agent_pool.list(agent_type=atype))
        else:
            default = _DEFAULT_OPPONENT_TYPES.get(focal.agent_type)
            if default is None:
                candidates = self.agent_pool.list()
            else:
                candidates = []
                for atype in default:
                    candidates.extend(self.agent_pool.list(agent_type=atype))

        # Exclude the focal agent itself.
        candidates = [c for c in candidates if c.agent_id != focal_agent_id]

        if not candidates:
            return None, None

        # Compute sampling weights: Nash distribution (if set) or PFSP.
        if self._nash_weights is not None:
            weights = np.array(
                [max(self._nash_weights.get(c.agent_id, 0.0), 0.0) for c in candidates],
                dtype=np.float64,
            )
        else:
            # Compute PFSP weights.
            weight_fn = self._pfsp_weight_fn if self._pfsp_weight_fn is not None else _hard_first
            weights = np.array(
                [
                    max(
                        weight_fn(
                            recorded_wr.get(c.agent_id, self.unknown_win_rate)
                        ),
                        0.0,
                    )
                    for c in candidates
                ],
                dtype=np.float64,
            )

        total = weights.sum()
        if total <= 0.0:
            weights = np.ones(len(candidates), dtype=np.float64)
            total = float(len(candidates))

        probs = weights / total
        return candidates, probs


# ---------------------------------------------------------------------------
# Default PFSP weight function
# ---------------------------------------------------------------------------


def _hard_first(win_rate: float) -> float:
    """Hard-first PFSP weight: ``f(w) = 1 - w``.

    Biases sampling towards opponents the focal agent struggles against.
    """
    return max(1.0 - float(win_rate), 0.0)


# ---------------------------------------------------------------------------
# Public helper: Nash-derived weight function
# ---------------------------------------------------------------------------


def make_nash_weight_fn(
    nash_weights: Dict[str, float],
) -> Callable[[str], float]:
    """Return a function that maps *agent_id* → Nash probability.

    Convenience factory for converting a Nash distribution
    (``{agent_id: probability}``) into a simple callable suitable for use
    wherever a per-agent weight function is needed.

    Parameters
    ----------
    nash_weights:
        Mapping of ``agent_id → probability`` as returned by
        :func:`~training.league.nash.compute_nash_distribution` combined
        with :func:`~training.league.nash.build_payoff_matrix`.

    Returns
    -------
    Callable[[str], float]
        Function ``f(agent_id) → weight``.  Returns ``0.0`` for unknown IDs.
    """
    _weights = dict(nash_weights)

    def _fn(agent_id: str) -> float:
        return max(_weights.get(agent_id, 0.0), 0.0)

    return _fn
