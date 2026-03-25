# SPDX-License-Identifier: MIT
# training/league/diversity.py
"""Strategy diversity metrics for league training (E4.6).

Provides utilities for quantifying behavioural diversity across the league
agent pool.  Diverse leagues produce more robust main agents and prevent
premature convergence to a single strategy archetype.

The core idea is to represent each agent's behaviour as a fixed-length
*behavioral embedding* vector derived from its trajectory data (actions
observed during rollouts and the corresponding map positions).  Pairwise
cosine distances between these embeddings capture how differently two agents
behave, and the *diversity score* aggregates the full pairwise distance
matrix into a single scalar that can be logged to W&B.

Classes
-------
TrajectoryBatch
    Lightweight container for a single agent's trajectory data.
DiversityTracker
    Stateful accumulator that stores one :class:`TrajectoryBatch` per agent
    and exposes :meth:`~DiversityTracker.diversity_score` for W&B logging.

Functions
---------
embed_trajectory
    Encode a :class:`TrajectoryBatch` as a fixed-length L2-normalised
    embedding vector.
pairwise_cosine_distances
    Compute the symmetric ``(N, N)`` pairwise cosine-distance matrix for a
    batch of embedding vectors.
diversity_score
    Aggregate pairwise distances into a single scalar diversity score.

Notes
-----
Embedding design
    The embedding concatenates three feature groups:

    1. **Action histogram** — for each continuous action dimension the
       values are binned into ``n_action_bins`` equal-width buckets, yielding
       ``action_dim × n_action_bins`` normalised frequencies.
    2. **Position heatmap** — the 2-D map is divided into an
       ``n_pos_bins × n_pos_bins`` grid; cell visit counts are normalised to
       sum to 1, yielding ``n_pos_bins²`` features.
    3. **Movement statistics** — scalar summaries: mean L2 action magnitude,
       standard deviation of action magnitude, mean displacement per step,
       and fraction of steps with above-median action magnitude (aggression
       proxy).

    The full vector is L2-normalised to unit length so that cosine distance
    equals 1 − cosine similarity.

Diversity score
    ``diversity_score = mean of upper-triangle pairwise cosine distances``

    This equals ``0.0`` when all agents are behaviourally identical and
    approaches ``1.0`` (or higher for anti-correlated agents) as diversity
    increases.  For a single-agent pool the function returns ``0.0`` rather
    than raising.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "TrajectoryBatch",
    "embed_trajectory",
    "pairwise_cosine_distances",
    "diversity_score",
    "DiversityTracker",
]

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

_DEFAULT_N_ACTION_BINS: int = 8
_DEFAULT_N_POS_BINS: int = 4


# ---------------------------------------------------------------------------
# TrajectoryBatch
# ---------------------------------------------------------------------------


class TrajectoryBatch:
    """Compact container for a single agent's trajectory data.

    Parameters
    ----------
    actions:
        Array of shape ``(T, action_dim)`` containing the agent's actions
        over *T* time-steps.  Continuous actions are expected (any finite
        float values).
    positions:
        Array of shape ``(T, 2)`` containing the agent's (x, y) map
        coordinates, normalised to ``[0, 1]``.
    agent_id:
        Optional identifier for logging and tracking.

    Raises
    ------
    ValueError
        If *actions* or *positions* have incompatible shapes, or if the
        number of time-steps is zero.
    """

    __slots__ = ("actions", "positions", "agent_id")

    def __init__(
        self,
        actions: np.ndarray,
        positions: np.ndarray,
        agent_id: Optional[str] = None,
    ) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        positions = np.asarray(positions, dtype=np.float32)

        if actions.ndim != 2:
            raise ValueError(
                f"actions must be a 2-D array of shape (T, action_dim), "
                f"got shape {actions.shape}"
            )
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"positions must be a 2-D array of shape (T, 2), "
                f"got shape {positions.shape}"
            )
        if actions.shape[0] == 0:
            raise ValueError("trajectory must contain at least one time-step")
        if actions.shape[0] != positions.shape[0]:
            raise ValueError(
                f"actions and positions must have the same number of time-steps: "
                f"{actions.shape[0]} vs {positions.shape[0]}"
            )

        self.actions: np.ndarray = actions
        self.positions: np.ndarray = positions
        self.agent_id: Optional[str] = agent_id

    @property
    def n_steps(self) -> int:
        """Number of time-steps in this trajectory."""
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action space."""
        return int(self.actions.shape[1])

    def __repr__(self) -> str:
        return (
            f"TrajectoryBatch("
            f"agent_id={self.agent_id!r}, "
            f"n_steps={self.n_steps}, "
            f"action_dim={self.action_dim})"
        )


# ---------------------------------------------------------------------------
# Behavioral embedding
# ---------------------------------------------------------------------------


def embed_trajectory(
    trajectory: TrajectoryBatch,
    n_action_bins: int = _DEFAULT_N_ACTION_BINS,
    n_pos_bins: int = _DEFAULT_N_POS_BINS,
    action_clip: float = 5.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Encode a trajectory as a fixed-length L2-normalised embedding vector.

    The embedding is the concatenation of three feature groups
    (see module docstring for details):

    1. Action histogram — ``action_dim × n_action_bins`` values.
    2. Position heatmap — ``n_pos_bins² `` values.
    3. Movement statistics — 4 scalar values.

    Total dimension: ``action_dim × n_action_bins + n_pos_bins² + 4``.

    Parameters
    ----------
    trajectory:
        Source trajectory data.
    n_action_bins:
        Number of histogram bins per action dimension (default 8).
    n_pos_bins:
        Grid resolution for the position heatmap (default 4; creates a
        ``n_pos_bins × n_pos_bins`` grid).
    action_clip:
        Values in *trajectory.actions* are clipped to
        ``[-action_clip, action_clip]`` before binning so that outliers do
        not dominate a single extreme bin (default 5.0).
    epsilon:
        Small floor to avoid division by zero during L2 normalisation.

    Returns
    -------
    numpy.ndarray
        1-D float64 array of unit L2 norm, length
        ``action_dim × n_action_bins + n_pos_bins² + 4``.

    Raises
    ------
    ValueError
        If *n_action_bins* or *n_pos_bins* is less than 1.
    """
    if n_action_bins < 1:
        raise ValueError(f"n_action_bins must be >= 1, got {n_action_bins!r}")
    if n_pos_bins < 1:
        raise ValueError(f"n_pos_bins must be >= 1, got {n_pos_bins!r}")

    features: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # 1. Action histogram
    # ------------------------------------------------------------------
    actions = np.clip(
        trajectory.actions.astype(np.float64), -action_clip, action_clip
    )
    bin_edges = np.linspace(-action_clip, action_clip, n_action_bins + 1)
    for dim_idx in range(trajectory.action_dim):
        hist, _ = np.histogram(actions[:, dim_idx], bins=bin_edges)
        hist = hist.astype(np.float64)
        total = hist.sum()
        features.append(hist / (total + epsilon))

    # ------------------------------------------------------------------
    # 2. Position heatmap
    # ------------------------------------------------------------------
    # Clamp positions to [0, 1] to handle minor floating-point overshoot.
    pos = np.clip(trajectory.positions.astype(np.float64), 0.0, 1.0)
    heatmap = np.zeros((n_pos_bins, n_pos_bins), dtype=np.float64)
    # Map [0, 1] → [0, n_pos_bins) bin indices.
    xi = np.minimum((pos[:, 0] * n_pos_bins).astype(int), n_pos_bins - 1)
    yi = np.minimum((pos[:, 1] * n_pos_bins).astype(int), n_pos_bins - 1)
    for x, y in zip(xi, yi):
        heatmap[x, y] += 1.0
    total_visits = heatmap.sum()
    heatmap = heatmap / (total_visits + epsilon)
    features.append(heatmap.ravel())

    # ------------------------------------------------------------------
    # 3. Movement statistics
    # ------------------------------------------------------------------
    # Action magnitude per step — scalar proxy for activity level.
    action_mags = np.linalg.norm(actions, axis=1)  # shape (T,)
    mean_mag = float(action_mags.mean())
    std_mag = float(action_mags.std())

    # Mean step displacement (Euclidean) as proxy for agent speed.
    if trajectory.n_steps > 1:
        displacements = np.linalg.norm(
            np.diff(pos, axis=0), axis=1
        )  # shape (T-1,)
        mean_displacement = float(displacements.mean())
    else:
        mean_displacement = 0.0

    # Aggression proxy: fraction of steps with above-median action magnitude.
    if trajectory.n_steps > 1:
        median_mag = float(np.median(action_mags))
        aggression = float(np.mean(action_mags > median_mag))
    else:
        aggression = 0.5

    features.append(
        np.array([mean_mag, std_mag, mean_displacement, aggression], dtype=np.float64)
    )

    # ------------------------------------------------------------------
    # Concatenate + L2 normalise
    # ------------------------------------------------------------------
    embedding = np.concatenate(features).astype(np.float64)
    norm = np.linalg.norm(embedding)
    if norm > epsilon:
        embedding = embedding / norm
    else:
        # All-zero trajectory — return uniform unit vector.
        embedding = np.ones_like(embedding) / np.sqrt(max(len(embedding), 1))

    return embedding


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------


def pairwise_cosine_distances(embeddings: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Compute the symmetric pairwise cosine-distance matrix.

    Cosine distance is defined as ``1 − cosine_similarity(u, v)``.  For
    L2-normalised vectors (as produced by :func:`embed_trajectory`) this
    simplifies to ``1 − u · v``.

    Parameters
    ----------
    embeddings:
        2-D array of shape ``(N, D)`` where ``N`` is the number of agents
        and ``D`` is the embedding dimension.  Rows need *not* be
        L2-normalised — normalisation is applied internally.
    epsilon:
        Small floor to guard against zero-norm rows.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(N, N)`` float64 matrix of pairwise cosine distances in
        ``[0, 2]`` (though in practice the range is ``[0, 1]`` for
        non-negative embeddings).  Diagonal entries are ``0.0``.

    Raises
    ------
    ValueError
        If *embeddings* is not a non-empty 2-D array.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2-D array of shape (N, D), "
            f"got shape {embeddings.shape}"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings array must contain at least one row")

    # L2-normalise each row.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, epsilon)
    normed = embeddings / norms

    # Cosine similarity matrix: (N, N).
    similarity = normed @ normed.T
    # Clip to [-1, 1] to handle floating-point noise.
    similarity = np.clip(similarity, -1.0, 1.0)

    dist = 1.0 - similarity
    np.fill_diagonal(dist, 0.0)
    return dist


# ---------------------------------------------------------------------------
# Diversity score
# ---------------------------------------------------------------------------


def diversity_score(
    embeddings: np.ndarray,
    *,
    aggregation: str = "mean",
    epsilon: float = 1e-8,
) -> float:
    """Aggregate pairwise cosine distances into a scalar diversity score.

    Parameters
    ----------
    embeddings:
        2-D array of shape ``(N, D)`` — one embedding row per agent.
    aggregation:
        How to reduce the pairwise distance matrix to a scalar.

        ``"mean"`` (default)
            Mean of the upper-triangle entries.  Equals 0.0 for a single
            agent; approaches 1.0 for maximally diverse pools.
        ``"min"``
            Minimum pairwise distance (identifies the two most similar
            agents in the pool).
        ``"median"``
            Median of upper-triangle pairwise distances.

    epsilon:
        Passed through to :func:`pairwise_cosine_distances`.

    Returns
    -------
    float
        Scalar diversity score in ``[0, 2]`` (typical range ``[0, 1]``).
        Returns ``0.0`` for a single-agent pool without raising.

    Raises
    ------
    ValueError
        If *aggregation* is not one of the supported options.
        If *embeddings* is not a valid 2-D array.
    """
    valid_aggregations = {"mean", "min", "median"}
    if aggregation not in valid_aggregations:
        raise ValueError(
            f"aggregation must be one of {sorted(valid_aggregations)!r}, "
            f"got {aggregation!r}"
        )

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2-D array of shape (N, D), "
            f"got shape {embeddings.shape}"
        )

    n = embeddings.shape[0]
    if n <= 1:
        return 0.0

    dist_matrix = pairwise_cosine_distances(embeddings, epsilon=epsilon)

    # Extract upper-triangle entries (i < j) to avoid double-counting.
    upper_idx = np.triu_indices(n, k=1)
    upper_values = dist_matrix[upper_idx]

    if len(upper_values) == 0:
        return 0.0

    if aggregation == "mean":
        return float(np.mean(upper_values))
    elif aggregation == "min":
        return float(np.min(upper_values))
    else:  # median
        return float(np.median(upper_values))


# ---------------------------------------------------------------------------
# DiversityTracker
# ---------------------------------------------------------------------------


class DiversityTracker:
    """Stateful accumulator of per-agent trajectory embeddings.

    Stores the most recent :class:`TrajectoryBatch` (or pre-computed
    embedding) for each agent.  Call :meth:`update` after each evaluation
    rollout and :meth:`diversity_score` to get the current pool-wide
    diversity score.

    Parameters
    ----------
    n_action_bins:
        Forwarded to :func:`embed_trajectory` (default 8).
    n_pos_bins:
        Forwarded to :func:`embed_trajectory` (default 4).
    aggregation:
        Diversity aggregation method passed to :func:`diversity_score`
        (default ``"mean"``).

    Examples
    --------
    ::

        tracker = DiversityTracker()
        for agent_id, traj in agent_trajectories.items():
            tracker.update(agent_id, traj)
        score = tracker.diversity_score()
        wandb.log({"league/diversity_score": score})
    """

    def __init__(
        self,
        n_action_bins: int = _DEFAULT_N_ACTION_BINS,
        n_pos_bins: int = _DEFAULT_N_POS_BINS,
        aggregation: str = "mean",
    ) -> None:
        self.n_action_bins = int(n_action_bins)
        self.n_pos_bins = int(n_pos_bins)
        self.aggregation = aggregation

        # agent_id -> pre-computed embedding vector
        self._embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        agent_id: str,
        trajectory: TrajectoryBatch,
    ) -> None:
        """Compute and store the embedding for *agent_id*.

        Replaces any previously stored embedding for this agent so the
        tracker always reflects the most recent behaviour.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent.
        trajectory:
            Recent trajectory data for this agent.
        """
        embedding = embed_trajectory(
            trajectory,
            n_action_bins=self.n_action_bins,
            n_pos_bins=self.n_pos_bins,
        )
        self._embeddings[agent_id] = embedding
        log.debug(
            "DiversityTracker.update: stored embedding for agent=%s "
            "(pool size=%d)",
            agent_id,
            len(self._embeddings),
        )

    def update_embedding(self, agent_id: str, embedding: np.ndarray) -> None:
        """Store a pre-computed embedding directly.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent.
        embedding:
            Pre-computed embedding vector (any shape will be flattened).
        """
        self._embeddings[agent_id] = np.asarray(embedding, dtype=np.float64).ravel()

    def remove(self, agent_id: str) -> None:
        """Remove an agent's embedding from the tracker.

        No-op if *agent_id* is not present.
        """
        self._embeddings.pop(agent_id, None)

    def clear(self) -> None:
        """Remove all stored embeddings."""
        self._embeddings.clear()

    @property
    def agent_ids(self) -> List[str]:
        """List of agent IDs currently tracked."""
        return list(self._embeddings.keys())

    @property
    def pool_size(self) -> int:
        """Number of agents currently tracked."""
        return len(self._embeddings)

    def embeddings_matrix(
        self,
        agent_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """Return an ``(N, D)`` embeddings matrix and the corresponding IDs.

        Parameters
        ----------
        agent_ids:
            Ordered list of agent IDs to include.  Defaults to all tracked
            agents in insertion order.

        Returns
        -------
        (ordered_ids, matrix)
            ``ordered_ids`` is a list of agent IDs; ``matrix`` is a 2-D
            float64 array of shape ``(N, D)``.

        Raises
        ------
        KeyError
            If any requested *agent_id* is not in the tracker.
        """
        if agent_ids is None:
            agent_ids = self.agent_ids

        rows = []
        for aid in agent_ids:
            if aid not in self._embeddings:
                raise KeyError(f"Agent {aid!r} is not tracked")
            rows.append(self._embeddings[aid])

        if not rows:
            return [], np.empty((0, 0), dtype=np.float64)

        matrix = np.vstack(rows)
        return list(agent_ids), matrix

    def diversity_score(self, agent_ids: Optional[List[str]] = None) -> float:
        """Compute the pool-wide diversity score.

        Parameters
        ----------
        agent_ids:
            Subset of agents to include (default: all tracked agents).

        Returns
        -------
        float
            Diversity score in ``[0, 2]`` (typically ``[0, 1]``).  Returns
            ``0.0`` if fewer than two agents are tracked.
        """
        ids, matrix = self.embeddings_matrix(agent_ids)
        if len(ids) < 2:
            return 0.0
        return diversity_score(matrix, aggregation=self.aggregation)

    def pairwise_distances(
        self,
        agent_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """Compute the pairwise cosine-distance matrix.

        Parameters
        ----------
        agent_ids:
            Subset of agents to include (default: all tracked agents).

        Returns
        -------
        (ordered_ids, distance_matrix)
            Symmetric ``(N, N)`` distance matrix.
        """
        ids, matrix = self.embeddings_matrix(agent_ids)
        if len(ids) < 2:
            return ids, np.zeros((len(ids), len(ids)), dtype=np.float64)
        return ids, pairwise_cosine_distances(matrix)

    def __repr__(self) -> str:
        return (
            f"DiversityTracker("
            f"pool_size={self.pool_size}, "
            f"n_action_bins={self.n_action_bins}, "
            f"n_pos_bins={self.n_pos_bins}, "
            f"aggregation={self.aggregation!r})"
        )
