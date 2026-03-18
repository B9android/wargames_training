# training/elo.py
"""Elo rating system for battalion policy evaluation.

Provides a standard Elo update rule, a K-factor schedule that decays as
more games are played, and a :class:`EloRegistry` that persists ratings to
a JSON file so they survive across training runs.

Scripted baseline opponents have *fixed* seed ratings that are never
modified by calls to :meth:`EloRegistry.update`.

Usage::

    from training.elo import EloRegistry

    registry = EloRegistry("checkpoints/elo_registry.json")

    # After evaluating a checkpoint vs scripted_l3:
    delta = registry.update(
        agent="my_run_v1",
        opponent="scripted_l3",
        outcome=0.6,          # win-rate over the evaluation batch
        n_games=50,
    )
    registry.save()
    print(f"Elo delta: {delta:+.1f}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default Elo rating assigned to new, unseen agents.
DEFAULT_RATING: float = 1000.0

#: Fixed seed ratings for scripted baseline opponents.
#: These values are intentionally spread to give informative Elo deltas
#: relative to each level of difficulty.
BASELINE_RATINGS: dict[str, float] = {
    "random": 500.0,
    "scripted_l1": 600.0,
    "scripted_l2": 700.0,
    "scripted_l3": 800.0,
    "scripted_l4": 900.0,
    "scripted_l5": 1000.0,
}

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def expected_score(r_a: float, r_b: float) -> float:
    """Expected score of agent *a* against agent *b*.

    Uses the standard Elo formula with a 400-point scale factor.

    Parameters
    ----------
    r_a:
        Elo rating of agent *a*.
    r_b:
        Elo rating of agent *b*.

    Returns
    -------
    float
        A value in ``[0, 1]`` where 1 means certain win for *a*.
    """
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def k_factor(n_games: int) -> float:
    """K-factor schedule based on total rated games played by an agent.

    The K-factor controls how much each result moves the rating.  It starts
    high (fast learning) and decreases as the agent accumulates more games.

    Parameters
    ----------
    n_games:
        Total number of rated games played by the agent so far (before the
        current batch).

    Returns
    -------
    float
        K-factor to apply for the next rating update.
    """
    if n_games < 30:
        return 40.0
    if n_games < 100:
        return 20.0
    return 10.0


# ---------------------------------------------------------------------------
# EloRegistry
# ---------------------------------------------------------------------------


class EloRegistry:
    """Persistent Elo rating registry backed by a JSON file.

    Stores per-agent ratings and game counts.  Scripted baseline opponents
    (``"scripted_l1"`` … ``"scripted_l5"``, ``"random"``) have fixed seed
    ratings defined in :data:`BASELINE_RATINGS` and are **never modified**.

    Parameters
    ----------
    path:
        Path to the JSON file used for persistence.  The parent directory
        is created automatically on :meth:`save`.  Pass ``None`` to create
        an in-memory registry that cannot be saved to disk.
    """

    def __init__(
        self,
        path: Union[str, Path, None] = "checkpoints/elo_registry.json",
    ) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._ratings: dict[str, float] = {}
        self._game_counts: dict[str, int] = {}
        if self._path is not None and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_rating(self, name: str) -> float:
        """Return the current Elo rating for *name*.

        Falls back to :data:`BASELINE_RATINGS` for scripted opponents, then
        to :data:`DEFAULT_RATING` for completely unknown agents.
        """
        if name in self._ratings:
            return self._ratings[name]
        return BASELINE_RATINGS.get(name, DEFAULT_RATING)

    def get_game_count(self, name: str) -> int:
        """Return the total number of rated games played by *name*."""
        return self._game_counts.get(name, 0)

    def all_ratings(self) -> dict[str, float]:
        """Return a copy of all *stored* ratings (excludes pure baselines)."""
        return dict(self._ratings)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        agent: str,
        opponent: str,
        outcome: float,
        n_games: int = 1,
    ) -> float:
        """Update the Elo rating of *agent* after a batch of *n_games*.

        The *outcome* is the average score per game:

        * ``1.0`` — all wins
        * ``0.5`` — all draws
        * ``0.0`` — all losses

        Scripted baseline opponents' ratings are **never modified**.

        Parameters
        ----------
        agent:
            Identifier for the agent whose rating is updated.
        opponent:
            Identifier of the opponent played against.
        outcome:
            Average per-game score in ``[0, 1]``.
        n_games:
            Number of games in this batch (used to advance the game-count
            counter; the K-factor is evaluated at the *pre-update* count).

        Returns
        -------
        float
            Elo rating delta for *agent* (positive = rating increased).

        Raises
        ------
        ValueError
            If *outcome* is outside ``[0, 1]`` or *n_games* < 1.
        """
        if not 0.0 <= outcome <= 1.0:
            raise ValueError(
                f"outcome must be in [0, 1], got {outcome!r}."
            )
        if n_games < 1:
            raise ValueError(
                f"n_games must be >= 1, got {n_games!r}."
            )

        r_agent = self.get_rating(agent)
        r_opponent = self.get_rating(opponent)
        n_so_far = self.get_game_count(agent)
        k = k_factor(n_so_far)

        expected = expected_score(r_agent, r_opponent)
        delta = k * (outcome - expected)

        self._ratings[agent] = r_agent + delta
        self._game_counts[agent] = n_so_far + n_games
        return delta

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist current ratings and game counts to the JSON file.

        Raises
        ------
        ValueError
            If the registry was created without a file path (``path=None``).
        """
        if self._path is None:
            raise ValueError(
                "Cannot save: this EloRegistry was created without a file path. "
                "Pass a path to the constructor to enable persistence."
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "ratings": self._ratings,
            "game_counts": self._game_counts,
        }
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)

    def _load(self) -> None:
        """Load ratings from the existing JSON file."""
        with open(self._path, encoding="utf-8") as fh:  # type: ignore[arg-type]
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"EloRegistry: failed to parse JSON from '{self._path}': {exc}"
                ) from exc
        try:
            self._ratings = {str(k): float(v) for k, v in data.get("ratings", {}).items()}
            self._game_counts = {
                str(k): int(v) for k, v in data.get("game_counts", {}).items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"EloRegistry: invalid data types in '{self._path}': {exc}"
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EloRegistry(path={self._path!r}, "
            f"n_agents={len(self._ratings)})"
        )
