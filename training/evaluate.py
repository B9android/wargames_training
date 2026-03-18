# training/evaluate.py
"""Evaluate a saved PPO checkpoint against a configurable opponent.

Loads a Stable-Baselines3 PPO model from a ``.zip`` checkpoint, runs it
against a chosen opponent in :class:`~envs.battalion_env.BattalionEnv` for a
configurable number of episodes, and reports the Blue win rate and optional
Elo delta to stdout.

Supported opponent identifiers
-------------------------------
``scripted_l1`` … ``scripted_l5``
    Built-in scripted Red opponent at the specified curriculum level.
``random``
    A Red opponent that samples uniformly random actions every step.
``<path>``
    Any file-system path to an SB3 ``.zip`` checkpoint; that model drives Red.

A **win** is defined as Red routing or being destroyed without Blue having
routed or been destroyed in the same step.  A **draw** occurs when both sides
lose simultaneously or the episode reaches the step limit with neither side
eliminated.

Usage::

    python training/evaluate.py --checkpoint checkpoints/run/final \\
        --opponent scripted_l3
    python training/evaluate.py --checkpoint checkpoints/run/final \\
        --opponent scripted_l3 --n-episodes 100 --seed 0 \\
        --elo-registry checkpoints/elo_registry.json \\
        --agent-name my_run_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np

# Ensure project root is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3 import PPO

from envs.battalion_env import BattalionEnv, DESTROYED_THRESHOLD
from training.elo import EloRegistry, BASELINE_RATINGS

# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class EvaluationResult(NamedTuple):
    """Structured result from an evaluation run.

    Attributes
    ----------
    wins:
        Number of episodes Blue won.
    draws:
        Number of episodes that ended as a draw (both sides lost or timeout).
    losses:
        Number of episodes Blue lost.
    n_episodes:
        Total episodes evaluated (``wins + draws + losses``).
    win_rate:
        ``wins / n_episodes``.
    draw_rate:
        ``draws / n_episodes``.
    loss_rate:
        ``losses / n_episodes``.
    """

    wins: int
    draws: int
    losses: int
    n_episodes: int
    win_rate: float
    draw_rate: float
    loss_rate: float


# ---------------------------------------------------------------------------
# Random Red policy helper
# ---------------------------------------------------------------------------


class _RandomPolicy:
    """Red policy that samples uniformly random actions each step.

    This satisfies the :class:`~envs.battalion_env.RedPolicy` protocol.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        action = np.array(
            [
                self._rng.uniform(-1.0, 1.0),   # move
                self._rng.uniform(-1.0, 1.0),   # rotate
                self._rng.uniform(0.0, 1.0),    # fire
            ],
            dtype=np.float32,
        )
        return action, None


# ---------------------------------------------------------------------------
# Opponent factory
# ---------------------------------------------------------------------------


def _make_env(opponent: str) -> BattalionEnv:
    """Create a :class:`BattalionEnv` configured for *opponent*.

    Parameters
    ----------
    opponent:
        One of ``"scripted_l1"`` … ``"scripted_l5"``, ``"random"``, or a
        file-system path to an SB3 ``.zip`` checkpoint.

    Returns
    -------
    BattalionEnv

    Raises
    ------
    ValueError
        If *opponent* is a scripted level with an out-of-range number.
    FileNotFoundError
        If *opponent* looks like a path but does not exist on disk.
    """
    if opponent.startswith("scripted_l"):
        try:
            level = int(opponent[len("scripted_l"):])
        except ValueError:
            raise ValueError(
                f"Invalid opponent '{opponent}'. "
                "Scripted levels must be 'scripted_l1' to 'scripted_l5'."
            )
        if not 1 <= level <= 5:
            raise ValueError(
                f"Scripted level must be between 1 and 5, got {level}."
            )
        return BattalionEnv(curriculum_level=level)

    if opponent == "random":
        return BattalionEnv(red_policy=_RandomPolicy())

    # Treat as a checkpoint path.
    opp_path = Path(opponent)
    zip_path = opp_path if opp_path.suffix == ".zip" else Path(str(opp_path) + ".zip")
    if not zip_path.exists() and not opp_path.exists():
        raise FileNotFoundError(
            f"Opponent checkpoint not found: '{opponent}'. "
            "Provide 'scripted_l1'–'scripted_l5', 'random', or a valid path."
        )
    opp_model = PPO.load(opponent)
    return BattalionEnv(red_policy=opp_model)


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------


def _classify_outcome(
    info: dict,
    env: BattalionEnv,
) -> int:
    """Return 1 (Blue win), -1 (Blue loss), or 0 (draw) for a finished episode."""
    red_lost = (
        info.get("red_routed", False)
        or env.red.strength <= DESTROYED_THRESHOLD  # type: ignore[union-attr]
    )
    blue_lost = (
        info.get("blue_routed", False)
        or env.blue.strength <= DESTROYED_THRESHOLD  # type: ignore[union-attr]
    )
    if red_lost and not blue_lost:
        return 1
    if blue_lost and not red_lost:
        return -1
    return 0


def run_episodes_with_model(
    model: Any,
    opponent: str = "scripted_l5",
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> EvaluationResult:
    """Run evaluation episodes using an already-loaded model object.

    This is useful for in-training callbacks that have direct access to a
    :class:`~stable_baselines3.PPO` model without needing to save and reload
    a checkpoint file.

    Parameters
    ----------
    model:
        Any object with a ``predict(obs, deterministic)`` method (e.g. an
        SB3 ``PPO`` instance).
    opponent:
        Opponent identifier — see module docstring for valid values.
    n_episodes:
        Number of episodes to run (must be ≥ 1).
    deterministic:
        Whether the policy acts deterministically.
    seed:
        Base random seed; episode *i* uses ``seed + i`` when provided.

    Returns
    -------
    EvaluationResult

    Raises
    ------
    ValueError
        If *n_episodes* < 1.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}.")

    env = _make_env(opponent)
    wins = draws = losses = 0

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        outcome = _classify_outcome(info, env)
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        else:
            draws += 1

    env.close()
    return EvaluationResult(
        wins=wins,
        draws=draws,
        losses=losses,
        n_episodes=n_episodes,
        win_rate=wins / n_episodes,
        draw_rate=draws / n_episodes,
        loss_rate=losses / n_episodes,
    )


def evaluate_detailed(
    checkpoint_path: str,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Optional[int] = None,
    opponent: str = "scripted_l5",
) -> EvaluationResult:
    """Load a checkpoint and run *n_episodes*, returning a full result struct.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.zip`` checkpoint (extension may be omitted).
    n_episodes:
        Number of evaluation episodes (must be ≥ 1).
    deterministic:
        Whether the policy acts deterministically.
    seed:
        Base random seed; episode *i* uses ``seed + i`` when provided.
    opponent:
        Opponent identifier — see module docstring for valid values.

    Returns
    -------
    EvaluationResult

    Raises
    ------
    ValueError
        If *n_episodes* < 1.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}.")
    env = _make_env(opponent)
    model = PPO.load(checkpoint_path, env=env)
    result = run_episodes_with_model(
        model,
        opponent=opponent,
        n_episodes=n_episodes,
        deterministic=deterministic,
        seed=seed,
    )
    env.close()
    return result


def evaluate(
    checkpoint_path: str,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Optional[int] = None,
    opponent: str = "scripted_l5",
) -> float:
    """Load a checkpoint, run *n_episodes*, and return the Blue win rate.

    This is a thin wrapper around :func:`evaluate_detailed` kept for
    backward compatibility.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.zip`` checkpoint (extension may be omitted).
    n_episodes:
        Number of evaluation episodes (must be ≥ 1).
    deterministic:
        Whether the policy acts deterministically.
    seed:
        Base random seed; episode *i* uses ``seed + i`` when provided.
    opponent:
        Opponent identifier — see module docstring for valid values.
        Defaults to ``"scripted_l5"`` (full-combat scripted Red) to match
        the previous behaviour.

    Returns
    -------
    float
        Win rate in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *n_episodes* is less than 1.
    """
    return evaluate_detailed(
        checkpoint_path,
        n_episodes=n_episodes,
        deterministic=deterministic,
        seed=seed,
        opponent=opponent,
    ).win_rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a PPO checkpoint against a chosen opponent "
            "and optionally update an Elo registry."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the SB3 .zip checkpoint (extension optional).",
    )
    parser.add_argument(
        "--opponent",
        default="scripted_l5",
        help=(
            "Opponent to evaluate against.  "
            "One of 'scripted_l1'…'scripted_l5', 'random', "
            "or a path to an SB3 .zip checkpoint.  "
            "(default: scripted_l5)"
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50, minimum: 1).",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use deterministic actions (default).",
    )
    action_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic actions instead of deterministic.",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the evaluation environment.",
    )
    parser.add_argument(
        "--elo-registry",
        default=None,
        help=(
            "Path to the Elo registry JSON file.  "
            "When provided, the agent's rating is updated and persisted.  "
            "(default: no persistence)"
        ),
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        help=(
            "Name used as the agent key in the Elo registry.  "
            "Defaults to the checkpoint path when not specified."
        ),
    )

    args = parser.parse_args(argv)
    if args.n_episodes < 1:
        parser.error(f"--n-episodes must be >= 1, got {args.n_episodes}.")

    result = evaluate_detailed(
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        seed=args.seed,
        opponent=args.opponent,
    )

    opp_elo = BASELINE_RATINGS.get(args.opponent, 1000.0)
    print(f"Opponent:  {args.opponent} (Elo: {opp_elo:.0f})")
    print(
        f"Win rate:  {result.win_rate:.2%} "
        f"({result.wins}W / {result.draws}D / {result.losses}L "
        f"in {result.n_episodes} episodes)"
    )

    # Elo delta — always compute and print when opponent is given.
    agent_name = args.agent_name or args.checkpoint
    registry_path = args.elo_registry or "checkpoints/elo_registry.json"
    registry = EloRegistry(registry_path)
    old_rating = registry.get_rating(agent_name)
    # outcome score: win=1, draw=0.5, loss=0
    outcome = (result.wins + 0.5 * result.draws) / result.n_episodes
    delta = registry.update(
        agent=agent_name,
        opponent=args.opponent,
        outcome=outcome,
        n_games=result.n_episodes,
    )
    new_rating = registry.get_rating(agent_name)
    print(
        f"Elo:       {old_rating:.1f} → {new_rating:.1f} "
        f"(Δ {delta:+.1f})"
    )

    if args.elo_registry is not None:
        registry.save()
        print(f"Registry:  saved to {registry_path}")


if __name__ == "__main__":
    main()
