# training/league/__init__.py
"""League training infrastructure (E4.1, E4.2).

Provides an AlphaStar-style league with:

* :class:`~training.league.agent_pool.AgentPool` — pool of agent snapshots
  (main agents, exploiters, league exploiters) with PFSP sampling.
* :class:`~training.league.match_database.MatchDatabase` — persistent
  store of historical match outcomes (JSON-backed).
* :class:`~training.league.matchmaker.LeagueMatchmaker` — PFSP-based
  matchmaking that samples opponents proportional to win-rate weights.
* :class:`~training.league.train_main_agent.MainAgentTrainer` — MAPPO
  training loop for main agents using PFSP against the full league pool.

Notes
-----
``MainAgentTrainer`` and ``make_pfsp_weight_fn`` are imported lazily so that
the lightweight infrastructure classes (``AgentPool``, ``MatchDatabase``,
``LeagueMatchmaker``) can be imported without pulling in the heavy training
dependencies (torch, wandb, hydra, envs).  Use the qualified import path
``from training.league.train_main_agent import MainAgentTrainer`` when you
need those directly, or access them via this package and they will be loaded
on first use.
"""

from training.league.agent_pool import AgentPool, AgentRecord, AgentType
from training.league.match_database import MatchDatabase, MatchResult
from training.league.matchmaker import LeagueMatchmaker

__all__ = [
    "AgentPool",
    "AgentRecord",
    "AgentType",
    "MatchDatabase",
    "MatchResult",
    "LeagueMatchmaker",
    "MainAgentTrainer",
    "make_pfsp_weight_fn",
]

# ---------------------------------------------------------------------------
# Lazy imports for heavy training symbols
# ---------------------------------------------------------------------------

_LAZY = {
    "MainAgentTrainer": "training.league.train_main_agent",
    "make_pfsp_weight_fn": "training.league.train_main_agent",
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
