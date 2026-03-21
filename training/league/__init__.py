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
"""

from training.league.agent_pool import AgentPool, AgentRecord, AgentType
from training.league.match_database import MatchDatabase, MatchResult
from training.league.matchmaker import LeagueMatchmaker
from training.league.train_main_agent import MainAgentTrainer, make_pfsp_weight_fn

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
