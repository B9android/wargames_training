# training/league/agent_pool.py
"""Agent pool for league training (E4.1).

Maintains a registry of agent snapshots — main agents, exploiters, and
league exploiters — and implements *Prioritized Fictitious Self-Play*
(PFSP) sampling.

Classes
-------
AgentType
    Enum of AlphaStar-style agent roles.
AgentRecord
    Immutable metadata for a single agent snapshot.
AgentPool
    Pool (not thread-safe; external locking required) with PFSP-weighted
    opponent sampling; state is persisted to a JSON manifest on disk so
    that it survives restarts.

PFSP weighting
--------------
The default weighting function is ``f(w) = (1 - w)`` where *w* is the
focal agent's historical win rate against a candidate opponent.  This
biases sampling towards harder opponents (low win rate).  Callers may
supply any callable ``f: float -> float`` via the *pfsp_weight_fn*
argument to :meth:`AgentPool.sample_pfsp`.

When no win-rate information is available for a candidate opponent an
effective win rate of ``0.5`` (a neutral matchup) is assumed; under the
default weighting this yields a weight of ``0.5``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "AgentType",
    "AgentRecord",
    "AgentPool",
]

# ---------------------------------------------------------------------------
# AgentType
# ---------------------------------------------------------------------------


class AgentType(str, Enum):
    """AlphaStar-style agent roles within the league.

    MAIN_AGENT
        The primary agent trained against all league members via PFSP.
    MAIN_EXPLOITER
        Specialist agent trained exclusively against main agents.  Adds
        diversity by forcing main agents to defend against targeted
        strategies.
    LEAGUE_EXPLOITER
        Generalist exploiter trained against all current league members.
        Ensures no strategy in the league goes unexploited.
    """

    MAIN_AGENT = "main_agent"
    MAIN_EXPLOITER = "main_exploiter"
    LEAGUE_EXPLOITER = "league_exploiter"

    @classmethod
    def from_str(cls, value: str) -> "AgentType":
        """Case-insensitive lookup.

        Raises
        ------
        ValueError
            If *value* does not match any :class:`AgentType` member.
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Unknown agent type {value!r}. Valid values: {valid}"
            ) from None


# ---------------------------------------------------------------------------
# AgentRecord
# ---------------------------------------------------------------------------


class AgentRecord:
    """Immutable metadata for a single agent snapshot in the pool.

    Parameters
    ----------
    agent_id:
        Unique identifier for this snapshot (UUID or caller-assigned
        string).
    agent_type:
        Role of this agent within the league.
    version:
        Monotonically increasing integer within the agent lineage.
    snapshot_path:
        Absolute or relative path to the serialised policy file on disk
        (e.g. a ``.pt`` or ``.zip`` file).
    created_at:
        Unix timestamp of when this record was created.  Defaults to the
        current time.
    metadata:
        Optional free-form dict for run IDs, config hashes, etc.
    """

    __slots__ = (
        "agent_id",
        "agent_type",
        "version",
        "snapshot_path",
        "created_at",
        "metadata",
    )

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        version: int,
        snapshot_path: Union[str, Path],
        created_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_id: str = agent_id
        self.agent_type: AgentType = agent_type
        self.version: int = int(version)
        self.snapshot_path: Path = Path(snapshot_path)
        self.created_at: float = created_at if created_at is not None else time.time()
        self.metadata: Dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "version": self.version,
            "snapshot_path": str(self.snapshot_path),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRecord":
        """Reconstruct an :class:`AgentRecord` from a plain dict."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=AgentType(data["agent_type"]),
            version=int(data["version"]),
            snapshot_path=data["snapshot_path"],
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"AgentRecord(agent_id={self.agent_id!r}, "
            f"type={self.agent_type.value}, "
            f"version={self.version}, "
            f"path={str(self.snapshot_path)!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentRecord):
            return NotImplemented
        return self.agent_id == other.agent_id

    def __hash__(self) -> int:
        return hash(self.agent_id)


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------


class AgentPool:
    """Registry of agent snapshots for league training.

    The pool stores :class:`AgentRecord` objects indexed by ``agent_id``.
    Records are serialised to a JSON manifest at *pool_manifest* so that
    the pool survives training restarts.

    Parameters
    ----------
    pool_manifest:
        Path to the JSON manifest file.  The parent directory is created
        automatically.  If the file already exists its contents are loaded
        at construction time.
    max_size:
        Maximum number of snapshots the pool may hold.  When the pool is
        full, :meth:`add` raises a :class:`RuntimeError` unless *force* is
        set.  Defaults to ``200`` (comfortably above the ≥ 50 requirement).

    Notes
    -----
    Concurrency: this class is **not** thread-safe.  External locking is
    required when multiple threads share a single :class:`AgentPool` instance.
    """

    def __init__(
        self,
        pool_manifest: Union[str, Path],
        max_size: int = 200,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.pool_manifest = Path(pool_manifest)
        self.max_size = int(max_size)
        # agent_id → AgentRecord (insertion-ordered as of Python 3.7)
        self._records: Dict[str, AgentRecord] = {}
        self._rng: np.random.Generator = np.random.default_rng()
        self._load()

    # ------------------------------------------------------------------
    # Public interface — mutation
    # ------------------------------------------------------------------

    def add(
        self,
        snapshot_path: Union[str, Path],
        agent_type: Union[AgentType, str] = AgentType.MAIN_AGENT,
        *,
        agent_id: Optional[str] = None,
        version: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> AgentRecord:
        # Note: path existence is not validated here; callers are responsible
        # for ensuring the snapshot file is reachable before registering it.
        """Register a new snapshot in the pool and persist the manifest.

        Parameters
        ----------
        snapshot_path:
            Path to the saved policy file.
        agent_type:
            Role of the agent; accepts :class:`AgentType` or a plain string.
        agent_id:
            Explicit unique identifier.  If omitted a UUID4 string is
            generated automatically.
        version:
            Version number embedded in the record.
        metadata:
            Optional free-form dict (run ID, config hash, …).
        force:
            When ``True`` the oldest record is silently evicted if the
            pool is at capacity.  When ``False`` (default) a
            :class:`RuntimeError` is raised instead.

        Returns
        -------
        AgentRecord
            The newly created record.

        Raises
        ------
        ValueError
            If an agent with the same *agent_id* already exists.
        RuntimeError
            If the pool is at capacity and *force* is ``False``.
        """
        if isinstance(agent_type, str):
            agent_type = AgentType.from_str(agent_type)

        if agent_id is None:
            agent_id = str(uuid.uuid4())

        if agent_id in self._records:
            raise ValueError(
                f"Agent with id {agent_id!r} already exists in the pool. "
                "Use a different agent_id or remove the existing record first."
            )

        if len(self._records) >= self.max_size:
            if not force:
                raise RuntimeError(
                    f"AgentPool is at capacity ({self.max_size}). "
                    "Set force=True to evict the oldest entry."
                )
            # Evict oldest entries (first insertion) until under capacity.
            # This also handles the case where a manifest was created with a
            # larger max_size and the pool is already over the new limit.
            while len(self._records) >= self.max_size:
                oldest_id = next(iter(self._records))
                self._records.pop(oldest_id)
                log.debug("AgentPool: evicted oldest record %s", oldest_id)

        record = AgentRecord(
            agent_id=agent_id,
            agent_type=agent_type,
            version=version,
            snapshot_path=snapshot_path,
            metadata=metadata,
        )
        self._records[agent_id] = record
        self._save()
        log.info(
            "AgentPool: added %s (type=%s, version=%d, pool=%d/%d)",
            agent_id,
            agent_type.value,
            version,
            len(self._records),
            self.max_size,
        )
        return record

    def remove(self, agent_id: str) -> None:
        """Remove the record identified by *agent_id* and persist the manifest.

        Parameters
        ----------
        agent_id:
            ID of the record to remove.

        Raises
        ------
        KeyError
            If no record with the given *agent_id* exists.
        """
        if agent_id not in self._records:
            raise KeyError(f"No agent with id {agent_id!r} in the pool.")
        self._records.pop(agent_id)
        self._save()
        log.info("AgentPool: removed %s", agent_id)

    # ------------------------------------------------------------------
    # Public interface — queries
    # ------------------------------------------------------------------

    def get(self, agent_id: str) -> AgentRecord:
        """Return the :class:`AgentRecord` for *agent_id*.

        Raises
        ------
        KeyError
            If *agent_id* is not in the pool.
        """
        try:
            return self._records[agent_id]
        except KeyError:
            raise KeyError(f"No agent with id {agent_id!r} in the pool.") from None

    def list(
        self, agent_type: Optional[Union[AgentType, str]] = None
    ) -> List[AgentRecord]:
        """Return all records, optionally filtered by *agent_type*.

        Parameters
        ----------
        agent_type:
            When provided, only records of this type are returned.
            Accepts :class:`AgentType` or a plain string.

        Returns
        -------
        list[AgentRecord]
            Records in insertion order.
        """
        if agent_type is not None:
            if isinstance(agent_type, str):
                agent_type = AgentType.from_str(agent_type)
            return [r for r in self._records.values() if r.agent_type == agent_type]
        return list(self._records.values())

    @property
    def size(self) -> int:
        """Current number of records in the pool."""
        return len(self._records)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # PFSP sampling
    # ------------------------------------------------------------------

    def sample_pfsp(
        self,
        win_rates: Optional[Dict[str, float]] = None,
        *,
        exclude_ids: Optional[List[str]] = None,
        agent_type: Optional[Union[AgentType, str]] = None,
        pfsp_weight_fn: Optional[Callable[[float], float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[AgentRecord]:
        """Sample one agent using Prioritized Fictitious Self-Play (PFSP).

        Candidates are weighted by ``pfsp_weight_fn(win_rate)`` where
        *win_rate* is the focal agent's historical win rate against each
        candidate.  Candidates without a recorded win rate are assumed to
        have a neutral win rate of ``0.5``, giving weight ``0.5`` under the
        default hard-first function (consistent with
        :class:`LeagueMatchmaker`'s default ``unknown_win_rate``).

        The default weighting function ``f(w) = (1 - w)`` (hard-first)
        biases sampling towards opponents the focal agent struggles against.

        Parameters
        ----------
        win_rates:
            Mapping of ``agent_id → win_rate`` for the focal agent.
            ``win_rate`` must be in ``[0, 1]``.  Missing keys are treated
            as an unknown win rate of ``0.5``.
        exclude_ids:
            Optional list of agent IDs to exclude from sampling (e.g. the
            focal agent itself).
        agent_type:
            When provided, restrict candidates to this type.
        pfsp_weight_fn:
            Callable ``f(win_rate: float) -> float``.  Must return a
            non-negative value.  Defaults to ``f(w) = (1 - w)``.
        rng:
            Optional NumPy random generator for reproducibility.

        Returns
        -------
        AgentRecord or None
            The sampled record, or ``None`` if no candidates exist.
        """
        candidates = self.list(agent_type=agent_type)

        if exclude_ids:
            exclude_set = set(exclude_ids)
            candidates = [r for r in candidates if r.agent_id not in exclude_set]

        if not candidates:
            return None

        weight_fn = pfsp_weight_fn if pfsp_weight_fn is not None else _default_pfsp_weight

        wr = win_rates or {}
        weights = np.array(
            [max(weight_fn(wr.get(r.agent_id, 0.5)), 0.0) for r in candidates],
            dtype=np.float64,
        )

        total = weights.sum()
        if total <= 0.0:
            # All weights are zero — fall back to uniform sampling.
            weights = np.ones(len(candidates), dtype=np.float64)
            total = float(len(candidates))

        probs = weights / total
        _rng = rng if rng is not None else self._rng
        idx = int(_rng.choice(len(candidates), p=probs))
        return candidates[idx]

    def sample_uniform(
        self,
        *,
        exclude_ids: Optional[List[str]] = None,
        agent_type: Optional[Union[AgentType, str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[AgentRecord]:
        """Sample one agent uniformly at random.

        Parameters
        ----------
        exclude_ids:
            Optional list of agent IDs to exclude.
        agent_type:
            When provided, restrict candidates to this type.
        rng:
            Optional NumPy random generator.

        Returns
        -------
        AgentRecord or None
            The sampled record, or ``None`` if no candidates exist.
        """
        return self.sample_pfsp(
            win_rates=None,
            exclude_ids=exclude_ids,
            agent_type=agent_type,
            pfsp_weight_fn=lambda _w: 1.0,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the pool manifest to disk (called automatically by mutations)."""
        self._save()

    def _save(self) -> None:
        self.pool_manifest.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_size": self.max_size,
            "agents": [r.to_dict() for r in self._records.values()],
        }
        tmp = self.pool_manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.pool_manifest)
        log.debug("AgentPool: manifest saved to %s", self.pool_manifest)

    def _load(self) -> None:
        if not self.pool_manifest.exists():
            log.debug("AgentPool: no manifest at %s, starting empty", self.pool_manifest)
            return
        try:
            raw = json.loads(self.pool_manifest.read_text(encoding="utf-8"))
            for record_data in raw.get("agents", []):
                record = AgentRecord.from_dict(record_data)
                self._records[record.agent_id] = record
            log.info(
                "AgentPool: loaded %d records from %s",
                len(self._records),
                self.pool_manifest,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("AgentPool: failed to load manifest %s: %s", self.pool_manifest, exc)


# ---------------------------------------------------------------------------
# Default PFSP weight function
# ---------------------------------------------------------------------------


def _default_pfsp_weight(win_rate: float) -> float:
    """Hard-first PFSP weight: ``f(w) = 1 - w``.

    Parameters
    ----------
    win_rate:
        The focal agent's historical win rate against the candidate
        opponent, in ``[0, 1]``.

    Returns
    -------
    float
        Weight for this candidate.  Returns ``0`` when ``win_rate = 1``
        (the focal agent always wins → no benefit from more practice).
    """
    return max(1.0 - float(win_rate), 0.0)
