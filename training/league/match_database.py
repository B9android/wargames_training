# training/league/match_database.py
"""Persistent match result database for league training (E4.1).

Match outcomes are stored in a newline-delimited JSON (JSONL) file so that
records accumulate across training restarts without any external database
dependency.

Classes
-------
MatchResult
    Immutable record of a single league match.
MatchDatabase
    Append-only match store with win-rate queries and cross-restart
    persistence.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

log = logging.getLogger(__name__)

__all__ = [
    "MatchResult",
    "MatchDatabase",
]

# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


class MatchResult:
    """Immutable record of a single match between two league agents.

    Parameters
    ----------
    agent_id:
        ID of the agent whose perspective defines *outcome*.
    opponent_id:
        ID of the opponent.
    outcome:
        Win rate (or win flag) of *agent_id* in this match.  Must be in
        ``[0, 1]``: ``1.0`` = full win, ``0.5`` = draw, ``0.0`` = loss.
        For single-episode matches use ``1.0`` / ``0.0``; for evaluation
        batches use the fraction of wins.
    match_id:
        Unique identifier for this result.  Auto-generated if omitted.
    timestamp:
        Unix timestamp.  Defaults to current time.
    metadata:
        Optional free-form dict (episode count, map name, …).
    """

    __slots__ = (
        "match_id",
        "agent_id",
        "opponent_id",
        "outcome",
        "timestamp",
        "metadata",
    )

    def __init__(
        self,
        agent_id: str,
        opponent_id: str,
        outcome: float,
        match_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        if not (0.0 <= outcome <= 1.0):
            raise ValueError(
                f"outcome must be in [0, 1], got {outcome!r}"
            )
        self.match_id: str = match_id if match_id is not None else str(uuid.uuid4())
        self.agent_id: str = agent_id
        self.opponent_id: str = opponent_id
        self.outcome: float = float(outcome)
        self.timestamp: float = timestamp if timestamp is not None else time.time()
        self.metadata: Dict = metadata or {}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        return {
            "match_id": self.match_id,
            "agent_id": self.agent_id,
            "opponent_id": self.opponent_id,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MatchResult":
        """Reconstruct a :class:`MatchResult` from a plain dict."""
        return cls(
            agent_id=data["agent_id"],
            opponent_id=data["opponent_id"],
            outcome=float(data["outcome"]),
            match_id=data.get("match_id"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"MatchResult(agent={self.agent_id!r}, opp={self.opponent_id!r}, "
            f"outcome={self.outcome:.3f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MatchResult):
            return NotImplemented
        return self.match_id == other.match_id

    def __hash__(self) -> int:
        return hash(self.match_id)


# ---------------------------------------------------------------------------
# MatchDatabase
# ---------------------------------------------------------------------------


class MatchDatabase:
    """Append-only store of league match results.

    Results are persisted to a newline-delimited JSON (JSONL) file; one
    JSON object per line.  This format allows atomic appends and is
    straightforwardly readable by external tools.

    Parameters
    ----------
    db_path:
        Path to the ``.jsonl`` database file.  The parent directory is
        created automatically.  Existing records are loaded at
        construction time.

    Notes
    -----
    All results loaded from disk are kept in memory for fast query
    access.  For very long training runs (millions of matches) callers
    may wish to use :meth:`prune` to discard old records and keep memory
    usage bounded.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self._results: List[MatchResult] = []
        self._load()

    # ------------------------------------------------------------------
    # Public interface — mutation
    # ------------------------------------------------------------------

    def record(
        self,
        agent_id: str,
        opponent_id: str,
        outcome: float,
        *,
        match_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> MatchResult:
        """Record a match result and persist it to disk.

        Parameters
        ----------
        agent_id:
            ID of the focal agent.
        opponent_id:
            ID of the opponent.
        outcome:
            Win rate of *agent_id* in ``[0, 1]``.
        match_id:
            Explicit match ID; auto-generated when omitted.
        metadata:
            Optional free-form dict appended to the record.

        Returns
        -------
        MatchResult
            The newly created record.
        """
        result = MatchResult(
            agent_id=agent_id,
            opponent_id=opponent_id,
            outcome=outcome,
            match_id=match_id,
            metadata=metadata,
        )
        self._results.append(result)
        self._append(result)
        log.debug(
            "MatchDatabase: recorded %s vs %s outcome=%.3f",
            agent_id,
            opponent_id,
            outcome,
        )
        return result

    def prune(self, keep_last: int) -> int:
        """Discard all but the most recent *keep_last* results from memory.

        .. note::
            This operation only affects the in-memory cache; the on-disk
            JSONL file is **not** modified (history is preserved).  To
            rebuild the file from the pruned cache call :meth:`rewrite`.

        Parameters
        ----------
        keep_last:
            Number of most-recent records to retain.

        Returns
        -------
        int
            Number of records discarded.
        """
        n_before = len(self._results)
        self._results = self._results[-keep_last:]
        discarded = n_before - len(self._results)
        log.debug("MatchDatabase: pruned %d records (kept %d)", discarded, len(self._results))
        return discarded

    def rewrite(self) -> None:
        """Overwrite the on-disk file with the current in-memory records.

        Useful after :meth:`prune` to reclaim disk space.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(r.to_dict()) for r in self._results]
        tmp = self.db_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp.replace(self.db_path)
        log.debug("MatchDatabase: rewrote %d records to %s", len(self._results), self.db_path)

    # ------------------------------------------------------------------
    # Public interface — queries
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of results in memory."""
        return len(self._results)

    def results_for(
        self,
        agent_id: str,
        opponent_id: Optional[str] = None,
    ) -> List[MatchResult]:
        """Return all results involving *agent_id*.

        Parameters
        ----------
        agent_id:
            Filter to results where ``result.agent_id == agent_id``.
        opponent_id:
            When provided, additionally filter by opponent.

        Returns
        -------
        list[MatchResult]
            Matching records in chronological order.
        """
        out = [r for r in self._results if r.agent_id == agent_id]
        if opponent_id is not None:
            out = [r for r in out if r.opponent_id == opponent_id]
        return out

    def win_rate(
        self,
        agent_id: str,
        opponent_id: str,
    ) -> Optional[float]:
        """Return the historical win rate of *agent_id* against *opponent_id*.

        Parameters
        ----------
        agent_id:
            The focal agent.
        opponent_id:
            The opponent.

        Returns
        -------
        float or None
            Mean outcome in ``[0, 1]``, or ``None`` if no results exist.
        """
        relevant = self.results_for(agent_id, opponent_id)
        if not relevant:
            return None
        return sum(r.outcome for r in relevant) / len(relevant)

    def win_rates_for(self, agent_id: str) -> Dict[str, float]:
        """Return win rates of *agent_id* against every recorded opponent.

        Parameters
        ----------
        agent_id:
            The focal agent.

        Returns
        -------
        dict[str, float]
            Mapping of ``opponent_id → mean outcome``.
        """
        opponent_ids = {
            r.opponent_id
            for r in self._results
            if r.agent_id == agent_id
        }
        return {
            opp: self.win_rate(agent_id, opp)  # type: ignore[misc]
            for opp in opponent_ids
        }

    def all_results(self) -> List[MatchResult]:
        """Return a shallow copy of all results in chronological order."""
        return list(self._results)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _append(self, result: MatchResult) -> None:
        """Atomically append a single result to the JSONL file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(result.to_dict()) + "\n"
        with self.db_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def _load(self) -> None:
        if not self.db_path.exists():
            log.debug("MatchDatabase: no file at %s, starting empty", self.db_path)
            return
        loaded = 0
        errors = 0
        for raw_line in self.db_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                self._results.append(MatchResult.from_dict(data))
                loaded += 1
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                errors += 1
                log.warning("MatchDatabase: skipped malformed line: %s", exc)
        log.info(
            "MatchDatabase: loaded %d results from %s (%d errors)",
            loaded,
            self.db_path,
            errors,
        )
