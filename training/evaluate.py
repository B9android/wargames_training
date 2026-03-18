# training/evaluate.py
"""Evaluate a saved PPO checkpoint against the scripted opponent.

Loads a Stable-Baselines3 PPO model from a ``.zip`` checkpoint, runs it
against :class:`~envs.battalion_env.BattalionEnv`'s built-in scripted Red
opponent for a configurable number of episodes, and prints the Blue win rate
to stdout.

A **win** is defined as Red routing or being destroyed without Blue having
routed or been destroyed in the same step.

Usage::

    python training/evaluate.py --checkpoint checkpoints/my_run/ppo_battalion_final
    python training/evaluate.py --checkpoint checkpoints/my_run/ppo_battalion_final \\
        --n-episodes 100 --deterministic --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3 import PPO

from envs.battalion_env import BattalionEnv, DESTROYED_THRESHOLD


def evaluate(
    checkpoint_path: str,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> float:
    """Run *n_episodes* evaluation episodes and return the Blue win rate.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.zip`` checkpoint file (the ``.zip`` extension may be
        omitted — SB3 will add it automatically).
    n_episodes:
        Number of episodes to evaluate.
    deterministic:
        Whether the policy selects actions deterministically.
    seed:
        Random seed for the evaluation environment.  When ``None`` the
        environment is not seeded.

    Returns
    -------
    float
        Win rate in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *n_episodes* is less than 1.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}.")
    env = BattalionEnv()
    model = PPO.load(checkpoint_path, env=env)

    wins = 0
    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Blue wins if Red routed/destroyed but Blue did not.
        red_lost = (
            info.get("red_routed", False)
            or env.red.strength <= DESTROYED_THRESHOLD  # type: ignore[union-attr]
        )
        blue_lost = (
            info.get("blue_routed", False)
            or env.blue.strength <= DESTROYED_THRESHOLD  # type: ignore[union-attr]
        )
        if red_lost and not blue_lost:
            wins += 1

    env.close()
    return wins / n_episodes


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint against the scripted opponent.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the SB3 .zip checkpoint (extension optional).",
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

    args = parser.parse_args(argv)
    if args.n_episodes < 1:
        parser.error(f"--n-episodes must be >= 1, got {args.n_episodes}.")

    win_rate = evaluate(
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    print(f"Win rate: {win_rate:.2%} ({round(win_rate * args.n_episodes)}/{args.n_episodes})")


if __name__ == "__main__":
    main()
