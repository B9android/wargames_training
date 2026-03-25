# SPDX-License-Identifier: MIT
# training/league/__init__.py
"""League training infrastructure (E4.1, E4.2, E4.3, E4.4, E4.5, E4.6).

Provides an AlphaStar-style league with:

* :class:`~training.league.agent_pool.AgentPool` — pool of agent snapshots
  (main agents, exploiters, league exploiters) with PFSP sampling.
* :class:`~training.league.match_database.MatchDatabase` — persistent
  store of historical match outcomes (JSON-backed).
* :class:`~training.league.matchmaker.LeagueMatchmaker` — PFSP-based
  matchmaking that samples opponents proportional to win-rate weights.
  Supports Nash distribution sampling via
  :meth:`~training.league.matchmaker.LeagueMatchmaker.set_nash_weights`.
* :mod:`~training.league.nash` — Nash equilibrium approximation via
  regret matching / LP; entropy metric for W&B logging.
* :mod:`~training.league.diversity` — behavioral embedding, pairwise
  cosine distances, diversity score, and :class:`DiversityTracker` for
  quantifying strategy diversity across the pool (E4.6).
* :class:`~training.league.train_main_agent.MainAgentTrainer` — MAPPO
  training loop for main agents using PFSP against the full league pool.
* :class:`~training.league.train_exploiter.MainExploiterTrainer` — MAPPO
  training loop for main exploiter agents targeting the latest main agent.
* :class:`~training.league.train_league_exploiter.LeagueExploiterTrainer` —
  MAPPO training loop for league exploiter agents using PFSP against the
  full historical pool.

Notes
-----
``MainAgentTrainer``, ``MainExploiterTrainer``, ``LeagueExploiterTrainer``,
``make_pfsp_weight_fn``, and ``compute_league_exploitability`` are imported
lazily so that the lightweight infrastructure classes (``AgentPool``,
``MatchDatabase``, ``LeagueMatchmaker``) can be imported without pulling in
the heavy training dependencies (torch, wandb, hydra, envs).  Use the
qualified import path (e.g.
``from training.league.train_league_exploiter import LeagueExploiterTrainer``)
when you need those directly, or access them via this package and they will
be loaded on first use.

The Nash module (:mod:`~training.league.nash`) is lightweight (NumPy-only,
SciPy optional) and is imported directly.
"""

from training.league.agent_pool import AgentPool, AgentRecord, AgentType
from training.league.match_database import MatchDatabase, MatchResult
from training.league.matchmaker import LeagueMatchmaker, make_nash_weight_fn
from training.league.nash import (
    build_payoff_matrix,
    compute_nash_distribution,
    nash_entropy,
)
from training.league.diversity import (
    TrajectoryBatch,
    DiversityTracker,
    embed_trajectory,
    pairwise_cosine_distances,
    diversity_score,
)

__all__ = [
    "AgentPool",
    "AgentRecord",
    "AgentType",
    "MatchDatabase",
    "MatchResult",
    "LeagueMatchmaker",
    "make_nash_weight_fn",
    "build_payoff_matrix",
    "compute_nash_distribution",
    "nash_entropy",
    "TrajectoryBatch",
    "DiversityTracker",
    "embed_trajectory",
    "pairwise_cosine_distances",
    "diversity_score",
    "MainAgentTrainer",
    "make_pfsp_weight_fn",
    "MainExploiterTrainer",
    "LeagueExploiterTrainer",
    "compute_league_exploitability",
]

# ---------------------------------------------------------------------------
# Lazy imports for heavy training symbols
# ---------------------------------------------------------------------------

_LAZY = {
    "MainAgentTrainer": "training.league.train_main_agent",
    "make_pfsp_weight_fn": "training.league.train_main_agent",
    "MainExploiterTrainer": "training.league.train_exploiter",
    "LeagueExploiterTrainer": "training.league.train_league_exploiter",
    "compute_league_exploitability": "training.league.train_league_exploiter",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        # Cache in module globals so subsequent accesses are O(1).
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
