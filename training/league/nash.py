# training/league/nash.py
"""Nash equilibrium approximation for league training (E4.5).

Provides utilities for computing an approximate Nash equilibrium over the
league's agent pool and deriving sampling distributions from it.

The Nash distribution answers the question: *which mixture of agents cannot
be exploited by any other mixture?*  Sampling opponents from this distribution
gives training opponents that are theoretically grounded and ensures no
strategy goes unchallenged.

Functions
---------
compute_nash_distribution
    Approximate Nash equilibrium via regret matching (with optional LP
    refinement when SciPy is available).
nash_entropy
    Shannon entropy of a probability distribution (nats).
build_payoff_matrix
    Construct a pairwise win-rate matrix from a :class:`MatchDatabase` and
    a list of agent IDs.

Notes
-----
Zero-sum game assumption
    The payoff matrix ``M`` satisfies ``M[i, j] + M[j, i] ≈ 1`` (i.e.
    win + loss ≈ 1).  The Nash equilibrium for such a symmetric zero-sum game
    is the mixed strategy ``σ*`` that maximises the guaranteed expected
    win rate regardless of the opponent's strategy.

Regret matching convergence
    The average strategy produced by :func:`compute_nash_distribution`
    converges to the Nash equilibrium as ``n_iterations → ∞`` in any finite
    two-player zero-sum game (Blackwell's approachability theorem).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "compute_nash_distribution",
    "nash_entropy",
    "build_payoff_matrix",
]

# ---------------------------------------------------------------------------
# Nash equilibrium computation
# ---------------------------------------------------------------------------


def compute_nash_distribution(
    payoff_matrix: np.ndarray,
    n_iterations: int = 10_000,
    epsilon: float = 1e-8,
    use_lp: bool = True,
) -> np.ndarray:
    """Return an approximate Nash equilibrium distribution over agents.

    Given a payoff matrix ``M`` where ``M[i, j]`` is the win rate of agent
    ``i`` against agent ``j``, this function returns a probability
    distribution ``σ`` over agents such that no agent can improve its
    expected win rate by unilateral deviation.

    The computation uses **regret matching** (guaranteed to converge in
    finite zero-sum games) with an optional **linear programming** (LP)
    refinement when SciPy is available.

    Parameters
    ----------
    payoff_matrix:
        Square ``(N, N)`` array where ``M[i, j] ∈ [0, 1]`` is the win rate
        of agent ``i`` against agent ``j``.  For a proper zero-sum game,
        ``M[i, j] + M[j, i] ≈ 1``.
    n_iterations:
        Number of regret-matching iterations.  More iterations improve
        accuracy at the cost of compute time.  Defaults to ``10_000``.
    epsilon:
        Small floor added to prevent division by zero when normalising.
        Defaults to ``1e-8``.
    use_lp:
        If ``True`` (default) and SciPy is importable, use linear
        programming to obtain the exact Nash equilibrium.  Falls back to
        regret matching if SciPy is unavailable or the LP solver fails.

    Returns
    -------
    numpy.ndarray
        1-D probability distribution of shape ``(N,)``.  Sums to ``1.0``.

    Raises
    ------
    ValueError
        If *payoff_matrix* is not a non-empty square 2-D array.
    """
    payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
    if payoff_matrix.ndim != 2 or payoff_matrix.shape[0] != payoff_matrix.shape[1]:
        raise ValueError(
            f"payoff_matrix must be a square 2-D array, got shape {payoff_matrix.shape}"
        )
    n = payoff_matrix.shape[0]
    if n == 0:
        raise ValueError("payoff_matrix must be non-empty")

    if n == 1:
        return np.array([1.0])

    # Try LP first for exact solution.
    if use_lp:
        result = _nash_lp(payoff_matrix, epsilon)
        if result is not None:
            return result

    # Fall back to regret matching.
    return _nash_regret_matching(payoff_matrix, n_iterations, epsilon)


def _nash_lp(payoff_matrix: np.ndarray, epsilon: float) -> Optional[np.ndarray]:
    """Solve the Nash equilibrium LP for a zero-sum game.

    Formulation (row player — maximise worst-case payoff *v*)::

        maximise  v
        subject to  M.T @ σ ≥ v · 1
                    σ ≥ 0
                    1.T σ = 1

    Translated to ``scipy.optimize.linprog`` (minimisation form) with
    variables ``x = [σ_0, …, σ_{N-1}, v]``.

    Returns ``None`` if SciPy is unavailable or the solver fails.
    """
    try:
        from scipy.optimize import linprog  # type: ignore[import]
    except ImportError:
        log.debug("nash._nash_lp: scipy not available; falling back to regret matching")
        return None

    n = payoff_matrix.shape[0]

    # Objective: minimise -v  (index n in the variable vector).
    c = np.zeros(n + 1)
    c[n] = -1.0  # maximise v ↔ minimise -v

    # Inequality constraints: -M.T σ + v 1 ≤ 0  ↔  M.T σ ≥ v 1
    # Shape: (n, n+1).
    A_ub = np.hstack([-payoff_matrix.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    # Equality constraint: sum(σ) = 1  (only the first n variables).
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds: σ ≥ 0, v unconstrained.
    bounds = [(0.0, None)] * n + [(None, None)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    except Exception as exc:
        log.debug("nash._nash_lp: LP solver raised %s; falling back", exc)
        return None

    if not res.success:
        log.debug("nash._nash_lp: LP solver status=%s; falling back", res.status)
        return None

    sigma = np.array(res.x[:n], dtype=np.float64)
    sigma = np.maximum(sigma, 0.0)
    total = sigma.sum()
    if total < epsilon:
        return None
    return sigma / total


def _nash_regret_matching(
    payoff_matrix: np.ndarray,
    n_iterations: int,
    epsilon: float,
) -> np.ndarray:
    """Approximate Nash equilibrium via two-player external regret matching.

    Implements external regret minimization for both the row player
    (maximiser) and the column player (minimiser) simultaneously.  The
    average strategies of both players are guaranteed to converge to a
    Nash equilibrium in any finite two-player zero-sum game (by the
    minimax theorem and the regret-matching convergence theorem).

    Algorithm
    ---------
    For each iteration *t*:

    1. Derive current mixed strategies from accumulated positive regrets::

           σ_row ∝ max(0, R_row)   (uniform if all regrets ≤ 0)
           σ_col ∝ max(0, R_col)   (uniform if all regrets ≤ 0)

    2. Compute the expected payoff under the joint strategy::

           ev = σ_row · M · σ_col

    3. Update row-player regrets (maximiser)::

           R_row[i] += (M @ σ_col)[i] − ev

    4. Update column-player regrets (minimiser)::

           R_col[j] += ev − (M.T @ σ_row)[j]

    5. Accumulate row player's average strategy: ``S_row += σ_row``

    The Nash distribution is the normalised average row strategy
    ``S_row / ‖S_row‖₁``.  For a symmetric game both players converge to
    the same equilibrium; only the row player's average is returned.
    """
    n = payoff_matrix.shape[0]
    cum_regrets_row = np.zeros(n, dtype=np.float64)
    cum_regrets_col = np.zeros(n, dtype=np.float64)
    cum_strategy_row = np.zeros(n, dtype=np.float64)

    for _ in range(n_iterations):
        # Step 1: current strategies from positive regrets.
        pos_row = np.maximum(cum_regrets_row, 0.0)
        total_row = pos_row.sum()
        strategy_row = pos_row / total_row if total_row > epsilon else np.ones(n, dtype=np.float64) / n

        pos_col = np.maximum(cum_regrets_col, 0.0)
        total_col = pos_col.sum()
        strategy_col = pos_col / total_col if total_col > epsilon else np.ones(n, dtype=np.float64) / n

        # Step 2: expected payoff under joint strategy.
        ev = float(strategy_row @ payoff_matrix @ strategy_col)

        # Step 3: row-player (maximiser) regret update.
        #   R_row[i] += (M @ σ_col)[i] - ev
        row_action_values = payoff_matrix @ strategy_col
        cum_regrets_row += row_action_values - ev

        # Step 4: column-player (minimiser) regret update.
        #   R_col[j] += ev - (M.T @ σ_row)[j]
        col_action_values = payoff_matrix.T @ strategy_row
        cum_regrets_col += ev - col_action_values

        # Step 5: accumulate row player's average strategy.
        cum_strategy_row += strategy_row

    total = cum_strategy_row.sum()
    if total < epsilon:
        return np.ones(n, dtype=np.float64) / n
    return cum_strategy_row / total


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def nash_entropy(distribution: np.ndarray, epsilon: float = 1e-12) -> float:
    """Compute the Shannon entropy (in nats) of a probability distribution.

    High entropy indicates a diverse Nash mixture (many agents are relevant);
    low entropy indicates the Nash distribution is concentrated on a few
    agents (others have been dominated).

    Parameters
    ----------
    distribution:
        1-D probability array.  Need not be normalised — the function
        normalises internally.
    epsilon:
        Small floor to avoid ``log(0)``.  Defaults to ``1e-12``.

    Returns
    -------
    float
        Shannon entropy in nats: ``H(σ) = -Σ_i σ_i ln(σ_i)``.
        Returns ``0.0`` for a single-element distribution.
    """
    p = np.asarray(distribution, dtype=np.float64)
    total = p.sum()
    if total < epsilon:
        return 0.0
    p = p / total
    # Clip to avoid log(0).
    p = np.clip(p, epsilon, None)
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Payoff matrix builder
# ---------------------------------------------------------------------------


def build_payoff_matrix(
    agent_ids: List[str],
    win_rate_fn: Callable[[str, str], Optional[float]],
    unknown_win_rate: float = 0.5,
) -> np.ndarray:
    """Build an ``(N, N)`` payoff matrix from pairwise win-rate estimates.

    ``M[i, j]`` is the win rate of agent ``i`` against agent ``j``.

    Parameters
    ----------
    agent_ids:
        Ordered list of ``N`` agent identifiers.
    win_rate_fn:
        Callable ``f(agent_id, opponent_id) → float | None``.  Should
        return the historical win rate in ``[0, 1]`` or ``None`` when no
        data is available.  Typically wraps
        :meth:`~training.league.match_database.MatchDatabase.win_rate`.
    unknown_win_rate:
        Value used when *win_rate_fn* returns ``None``.  Defaults to
        ``0.5`` (balanced assumption).

    Returns
    -------
    numpy.ndarray
        ``(N, N)`` float64 array.  Diagonal entries are set to
        ``0.5`` (a tie against oneself).
    """
    n = len(agent_ids)
    matrix = np.full((n, n), unknown_win_rate, dtype=np.float64)
    np.fill_diagonal(matrix, 0.5)

    for i, ai in enumerate(agent_ids):
        for j, aj in enumerate(agent_ids):
            if i == j:
                continue
            wr = win_rate_fn(ai, aj)
            if wr is not None:
                matrix[i, j] = float(wr)
            else:
                # If only the reverse direction is recorded (from aj's
                # perspective), infer this entry under the constant-sum
                # assumption: M[i, j] = 1 - M[j, i].
                wr_reverse = win_rate_fn(aj, ai)
                if wr_reverse is not None:
                    matrix[i, j] = 1.0 - float(wr_reverse)
    return matrix
