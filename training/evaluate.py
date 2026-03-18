# Evaluation & replay
"""Evaluate a trained PPO checkpoint against the scripted opponent.

Runs ``n_eval_episodes`` episodes with the saved policy and prints the
win rate.  A *win* is defined as Red routing or being destroyed without
Blue routing or being destroyed in the same step.

Usage::

    python training/evaluate.py --checkpoint checkpoints/<run_id>/final.zip
    python training/evaluate.py --checkpoint checkpoints/<run_id>/final.zip --episodes 100
    python training/evaluate.py --checkpoint checkpoints/<run_id>/final.zip --no-wandb

Acceptance criteria (E1.4):
* ``python training/evaluate.py --checkpoint <path>`` prints win rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from envs.battalion_env import BattalionEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: Path,
    n_episodes: int = 20,
    seed: int = 0,
    render: bool = False,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Run ``n_episodes`` evaluation episodes against the scripted opponent.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.zip`` checkpoint saved by SB3 ``model.save()``.
    n_episodes:
        Number of evaluation episodes.
    seed:
        Base random seed; each episode uses ``seed + episode_index``.
    render:
        If ``True``, call ``env.render()`` each step (no-op for the default
        render mode).
    verbose:
        Print per-episode results if ``True``.

    Returns
    -------
    win_rate, mean_reward, mean_ep_length
        Fraction of episodes won by Blue, mean episode reward, mean length.
    """
    from stable_baselines3 import PPO

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        # Try appending .zip if caller omitted the extension
        alt = checkpoint_path.with_suffix(".zip")
        if alt.exists():
            checkpoint_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = PPO.load(str(checkpoint_path))

    env = BattalionEnv()

    wins = 0
    total_reward = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            if render:
                env.render()

        # Outcome: win if Red is done and Blue is not
        red_done = info.get("red_routed", False) or (
            env.red is not None and env.red.strength <= 0.01
        )
        blue_done = info.get("blue_routed", False) or (
            env.blue is not None and env.blue.strength <= 0.01
        )
        win = red_done and not blue_done

        if win:
            wins += 1
        total_reward += ep_reward
        total_steps += ep_steps

        if verbose:
            outcome = "WIN" if win else "LOSS/DRAW"
            print(
                f"  Episode {ep + 1:3d}/{n_episodes}: {outcome} "
                f"reward={ep_reward:+.2f}  steps={ep_steps}"
            )

    env.close()

    win_rate = wins / n_episodes
    mean_reward = total_reward / n_episodes
    mean_length = total_steps / n_episodes

    return win_rate, mean_reward, mean_length


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO checkpoint against the scripted opponent"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .zip checkpoint file (e.g. checkpoints/run_xxx/final.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging of evaluation results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode output.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}  seed: {args.seed}")
    print()

    win_rate, mean_reward, mean_length = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print()
    print(f"=== Evaluation Results ===")
    print(f"Win rate    : {win_rate * 100:.1f}%  ({int(win_rate * args.episodes)}/{args.episodes})")
    print(f"Mean reward : {mean_reward:.3f}")
    print(f"Mean length : {mean_length:.1f} steps")

    if not args.no_wandb:
        try:
            import wandb

            with wandb.init(
                project="wargames_training",
                job_type="evaluation",
                config={"checkpoint": str(args.checkpoint), "n_episodes": args.episodes},
            ) as run:
                run.summary["eval/win_rate"] = win_rate
                run.summary["eval/mean_reward"] = mean_reward
                run.summary["eval/mean_ep_length"] = mean_length
                print(f"\nEvaluation results logged to W&B: {run.get_url()}")
        except Exception as exc:
            print(f"\n[evaluate] W&B logging skipped: {exc}")


if __name__ == "__main__":
    main()

