# SPDX-License-Identifier: MIT
# training/league/match_database.py
"""Persistent match result database for league training (E4.1, E7.3).

Match outcomes are stored in a newline-delimited JSON (JSONL) file so that
records accumulate across training restarts without any external database
dependency.

Classes
-------
MatchResult
    Immutable record of a single league match.  Extended in E7.3 with
    optional operational result fields for corps-scale matches
    (territory control, casualties, supply consumed).
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
    territory_control:
        Optional Blue territory-control fraction in ``[0, 1]`` at episode
        end.  Populated for corps-scale (CorpsEnv) matches.
    blue_casualties:
        Optional number of Blue unit casualties in the match.
    red_casualties:
        Optional number of Red unit casualties in the match.
    supply_consumed:
        Optional total supply consumed across all Blue divisions during the
        match (sum over ``info['supply_levels']`` deltas).
    """

    __slots__ = (
        "match_id",
        "agent_id",
        "opponent_id",
        "outcome",
        "timestamp",
        "metadata",
        "territory_control",
        "blue_casualties",
        "red_casualties",
        "supply_consumed",
    )

    def __init__(
        self,
        agent_id: str,
        opponent_id: str,
        outcome: float,
        match_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
        territory_control: Optional[float] = None,
        blue_casualties: Optional[int] = None,
        red_casualties: Optional[int] = None,
        supply_consumed: Optional[float] = None,
    ) -> None:
        if not (0.0 <= outcome <= 1.0):
            raise ValueError(
                f"outcome must be in [0, 1], got {outcome!r}"
            )
        if territory_control is not None and not (0.0 <= territory_control <= 1.0):
            raise ValueError(
                f"territory_control must be in [0, 1] or None, got {territory_control!r}"
            )
        if blue_casualties is not None and blue_casualties < 0:
            raise ValueError(
                f"blue_casualties must be non-negative or None, got {blue_casualties!r}"
            )
        if red_casualties is not None and red_casualties < 0:
            raise ValueError(
                f"red_casualties must be non-negative or None, got {red_casualties!r}"
            )
        if supply_consumed is not None and supply_consumed < 0.0:
            raise ValueError(
                f"supply_consumed must be non-negative or None, got {supply_consumed!r}"
            )
        self.match_id: str = match_id if match_id is not None else str(uuid.uuid4())
        self.agent_id: str = agent_id
        self.opponent_id: str = opponent_id
        self.outcome: float = float(outcome)
        self.timestamp: float = timestamp if timestamp is not None else time.time()
        self.metadata: Dict = metadata or {}
        self.territory_control: Optional[float] = (
            float(territory_control) if territory_control is not None else None
        )
        self.blue_casualties: Optional[int] = (
            int(blue_casualties) if blue_casualties is not None else None
        )
        self.red_casualties: Optional[int] = (
            int(red_casualties) if red_casualties is not None else None
        )
        self.supply_consumed: Optional[float] = (
            float(supply_consumed) if supply_consumed is not None else None
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        d = {
            "match_id": self.match_id,
            "agent_id": self.agent_id,
            "opponent_id": self.opponent_id,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        # Only include operational fields when they are set to keep the
        # JSONL file backward-compatible with records written before E7.3.
        if self.territory_control is not None:
            d["territory_control"] = self.territory_control
        if self.blue_casualties is not None:
            d["blue_casualties"] = self.blue_casualties
        if self.red_casualties is not None:
            d["red_casualties"] = self.red_casualties
        if self.supply_consumed is not None:
            d["supply_consumed"] = self.supply_consumed
        return d

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
            territory_control=data.get("territory_control"),
            blue_casualties=data.get("blue_casualties"),
            red_casualties=data.get("red_casualties"),
            supply_consumed=data.get("supply_consumed"),
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
        territory_control: Optional[float] = None,
        blue_casualties: Optional[int] = None,
        red_casualties: Optional[int] = None,
        supply_consumed: Optional[float] = None,
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
        territory_control:
            Optional Blue territory-control fraction in ``[0, 1]`` at
            episode end (corps-scale matches only).
        blue_casualties:
            Optional number of Blue unit casualties in the match.
        red_casualties:
            Optional number of Red unit casualties in the match.
        supply_consumed:
            Optional total supply consumed across all Blue divisions.

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
            territory_control=territory_control,
            blue_casualties=blue_casualties,
            red_casualties=red_casualties,
            supply_consumed=supply_consumed,
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
            Number of most-recent records to retain.  Must be non-negative.

        Returns
        -------
        int
            Number of records discarded.

        Raises
        ------
        ValueError
            If *keep_last* is negative.
        """
        if keep_last < 0:
            raise ValueError(f"keep_last must be non-negative, got {keep_last}")
        n_before = len(self._results)
        if keep_last == 0:
            self._results = []
        else:
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
        """Return results where *agent_id* is the focal agent (``result.agent_id``).

        Note: only results recorded from *agent_id*'s perspective are returned
        (i.e. where ``result.agent_id == agent_id``).  Results where *agent_id*
        appears as the opponent are not included.

        Parameters
        ----------
        agent_id:
            Filter to results where ``result.agent_id == agent_id``.
        opponent_id:
            When provided, additionally filter to results against this
            specific opponent.

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

        Computed in a single pass to avoid O(N×M) rescanning.

        Parameters
        ----------
        agent_id:
            The focal agent.

        Returns
        -------
        dict[str, float]
            Mapping of ``opponent_id → mean outcome``.
        """
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for r in self._results:
            if r.agent_id != agent_id:
                continue
            opp_id = r.opponent_id
            totals[opp_id] = totals.get(opp_id, 0.0) + r.outcome
            counts[opp_id] = counts.get(opp_id, 0) + 1
        return {opp_id: totals[opp_id] / counts[opp_id] for opp_id in totals}

    def all_results(self) -> List[MatchResult]:
        """Return a shallow copy of all results in chronological order."""
        return list(self._results)

    def mean_territory_control(
        self,
        agent_id: str,
        opponent_id: Optional[str] = None,
    ) -> Optional[float]:
        """Mean Blue territory-control fraction for *agent_id*'s matches.

        Only results that have ``territory_control`` populated are included.
        Returns ``None`` when no such records exist.
        """
        relevant = [
            r for r in self.results_for(agent_id, opponent_id)
            if r.territory_control is not None
        ]
        if not relevant:
            return None
        tc_values: List[float] = [r.territory_control for r in relevant]  # type: ignore[misc]
        return sum(tc_values) / len(tc_values)

    def mean_casualties(
        self,
        agent_id: str,
        opponent_id: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        """Mean Blue/Red casualties for *agent_id*'s matches.

        Each side's mean is computed independently over only the records where
        that side's casualty field is populated, so partial records do not bias
        the mean of the other side.  Returns a dict with keys ``"blue"`` and
        ``"red"`` (both floats), or ``None`` when no records with any casualty
        field exist.
        """
        all_relevant = [
            r for r in self.results_for(agent_id, opponent_id)
            if r.blue_casualties is not None or r.red_casualties is not None
        ]
        if not all_relevant:
            return None

        blue_values = [
            r.blue_casualties for r in all_relevant
            if r.blue_casualties is not None
        ]
        red_values = [
            r.red_casualties for r in all_relevant
            if r.red_casualties is not None
        ]

        mean_blue = float(sum(blue_values)) / len(blue_values) if blue_values else 0.0
        mean_red = float(sum(red_values)) / len(red_values) if red_values else 0.0
        return {"blue": mean_blue, "red": mean_red}

    def mean_supply_consumed(
        self,
        agent_id: str,
        opponent_id: Optional[str] = None,
    ) -> Optional[float]:
        """Mean supply consumed per match for *agent_id*.

        Only results that have ``supply_consumed`` populated are included.
        Returns ``None`` when no such records exist.
        """
        relevant = [
            r for r in self.results_for(agent_id, opponent_id)
            if r.supply_consumed is not None
        ]
        if not relevant:
            return None
        sc_values: List[float] = [r.supply_consumed for r in relevant]  # type: ignore[misc]
        return sum(sc_values) / len(sc_values)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _append(self, result: MatchResult) -> None:
        """Append a single result to the JSONL file.

        .. note::
            This class is designed for single-writer use.  No file locking
            is applied; concurrent writers from separate processes may
            interleave writes and corrupt JSONL lines.  Ensure only one
            process writes to the database at a time.
        """
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
        with self.db_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
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
