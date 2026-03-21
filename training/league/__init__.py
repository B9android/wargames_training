# training/league/__init__.py
"""League training infrastructure (E4.1).

Provides an AlphaStar-style league with:

* :class:`~training.league.agent_pool.AgentPool` — pool of agent snapshots
  (main agents, exploiters, league exploiters) with PFSP sampling.
* :class:`~training.league.match_database.MatchDatabase` — persistent
  store of historical match outcomes (JSON-backed).
* :class:`~training.league.matchmaker.LeagueMatchmaker` — PFSP-based
  matchmaking that samples opponents proportional to win-rate weights.
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
]
