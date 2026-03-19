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
from training.elo import EloRegistry

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


def _make_env(
    opponent: str,
    seed: Optional[int] = None,
    env_kwargs: Optional[dict] = None,
) -> BattalionEnv:
    """Create a :class:`BattalionEnv` configured for *opponent*.

    Parameters
    ----------
    opponent:
        One of ``"scripted_l1"`` … ``"scripted_l5"``, ``"random"``, or a
        file-system path to an SB3 ``.zip`` checkpoint.
    seed:
        Random seed forwarded to :class:`_RandomPolicy` when the opponent
        is ``"random"``, making random-opponent evaluation reproducible.
    env_kwargs:
        Additional keyword arguments forwarded to :class:`BattalionEnv`
        (e.g. ``map_width``, ``reward_weights``).  ``curriculum_level`` and
        ``red_policy`` are always controlled by *opponent* and are stripped
        from *env_kwargs* if present.

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
    # Strip keys that are controlled by the opponent spec to avoid conflicts.
    base: dict = {
        k: v for k, v in (env_kwargs or {}).items()
        if k not in ("curriculum_level", "red_policy")
    }

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
        return BattalionEnv(curriculum_level=level, **base)

    if opponent == "random":
        return BattalionEnv(red_policy=_RandomPolicy(seed=seed), **base)

    # Treat as a checkpoint path.
    opp_path = Path(opponent)
    zip_path = opp_path if opp_path.suffix == ".zip" else Path(str(opp_path) + ".zip")
    if not zip_path.exists() and not opp_path.exists():
        raise FileNotFoundError(
            f"Opponent checkpoint not found: '{opponent}'. "
            "Provide 'scripted_l1'–'scripted_l5', 'random', or a valid path."
        )
    opp_model = PPO.load(opponent)
    return BattalionEnv(red_policy=opp_model, **base)


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
    env: Optional[BattalionEnv] = None,
    env_kwargs: Optional[dict] = None,
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
        Also used to seed :class:`_RandomPolicy` when ``opponent="random"``.
    env:
        Pre-built :class:`BattalionEnv` to reuse.  When provided the caller
        owns the environment and is responsible for closing it.  When
        ``None``, a new environment is created and closed automatically.
    env_kwargs:
        Extra keyword arguments forwarded to :class:`BattalionEnv` when
        creating a new environment (ignored when *env* is provided).

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

    owns_env = env is None
    active_env: BattalionEnv = (
        env if env is not None
        else _make_env(opponent, seed=seed, env_kwargs=env_kwargs)
    )
    wins = draws = losses = 0

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = active_env.reset(seed=ep_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = active_env.step(action)
            done = terminated or truncated

        outcome = _classify_outcome(info, active_env)
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        else:
            draws += 1

    if owns_env:
        active_env.close()
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
    env_kwargs: Optional[dict] = None,
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
    env_kwargs:
        Extra keyword arguments forwarded to :class:`BattalionEnv`
        (e.g. ``map_width``, ``reward_weights``).

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
    # Create a single env and reuse it for both model loading and episode runs.
    env = _make_env(opponent, seed=seed, env_kwargs=env_kwargs)
    try:
        model = PPO.load(checkpoint_path, env=env)
        result = run_episodes_with_model(
            model,
            opponent=opponent,
            n_episodes=n_episodes,
            deterministic=deterministic,
            seed=seed,
            env=env,  # reuse the already-created env; caller closes below
            env_kwargs=env_kwargs,
        )
    finally:
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
# Rendered episode helper
# ---------------------------------------------------------------------------


def _run_rendered_episode(
    model: Any,
    env: BattalionEnv,
    ep_seed: Optional[int] = None,
    deterministic: bool = True,
    recorder: Optional[Any] = None,
) -> int:
    """Run a single episode with a live pygame renderer.

    Parameters
    ----------
    model:
        Loaded SB3 model (or any object with ``predict``).
    env:
        A :class:`BattalionEnv` instance — rendered via the returned
        renderer directly.
    ep_seed:
        Random seed passed to ``env.reset()``.  ``None`` means
        non-deterministic seeding.
    deterministic:
        Whether the model acts deterministically.
    recorder:
        Optional :class:`~envs.rendering.recorder.EpisodeRecorder`.  When
        provided, every step is appended to the recorder.

    Returns
    -------
    int
        Episode outcome: ``1`` (Blue win), ``-1`` (Blue loss), ``0`` (draw).
    """
    from envs.rendering.renderer import BattalionRenderer  # noqa: PLC0415

    renderer = BattalionRenderer(env.map_width, env.map_height)
    try:
        obs, _ = env.reset(seed=ep_seed)
        renderer.set_terrain(env.terrain)
        current_step = 0
        done = False
        info: dict = {}

        if recorder is not None:
            recorder.record_step(current_step, env.blue, env.red)

        while not done:
            alive = renderer.render_frame(
                env.blue,  # type: ignore[arg-type]
                env.red,   # type: ignore[arg-type]
                step=current_step,
                info=info,
            )
            if not alive:
                break
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            current_step += 1
            done = terminated or truncated

            if recorder is not None:
                recorder.record_step(
                    current_step,
                    env.blue,  # type: ignore[arg-type]
                    env.red,   # type: ignore[arg-type]
                    reward=float(reward),
                    info=info,
                )

        # Show the terminal frame
        if env.blue is not None and env.red is not None:
            renderer.render_frame(
                env.blue,
                env.red,
                step=current_step,
                info=info,
            )
    finally:
        renderer.close()

    return _classify_outcome(info, env)


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
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help=(
            "Open a pygame window and display each episode.  "
            "Requires pygame to be installed and a display to be available."
        ),
    )
    parser.add_argument(
        "--record",
        metavar="DIR",
        default=None,
        help=(
            "Save each episode trajectory as a JSON file under DIR "
            "(e.g. 'replays').  Files are named '<checkpoint>_ep<N>.json'."
        ),
    )

    args = parser.parse_args(argv)
    if args.n_episodes < 1:
        parser.error(f"--n-episodes must be >= 1, got {args.n_episodes}.")

    # ------------------------------------------------------------------
    # Rendered / recorded path
    # ------------------------------------------------------------------
    if args.render or args.record:
        env = _make_env(args.opponent, seed=args.seed)
        model = PPO.load(args.checkpoint, env=env)
        wins = draws = losses = 0
        record_dir = Path(args.record) if args.record else None
        try:
            for ep in range(args.n_episodes):
                ep_seed = None if args.seed is None else args.seed + ep
                recorder = None
                if record_dir is not None:
                    from envs.rendering.recorder import EpisodeRecorder  # noqa: PLC0415
                    recorder = EpisodeRecorder()

                if args.render:
                    outcome = _run_rendered_episode(
                        model,
                        env,
                        ep_seed=ep_seed,
                        deterministic=args.deterministic,
                        recorder=recorder,
                    )
                else:
                    # Record-only (no window)
                    obs, _ = env.reset(seed=ep_seed)
                    done = False
                    step_info: dict = {}
                    current_step = 0
                    if recorder is not None:
                        recorder.record_step(current_step, env.blue, env.red)  # type: ignore[arg-type]
                    while not done:
                        action, _ = model.predict(obs, deterministic=args.deterministic)
                        obs, reward, terminated, truncated, step_info = env.step(action)
                        current_step += 1
                        done = terminated or truncated
                        if recorder is not None:
                            recorder.record_step(current_step, env.blue, env.red, float(reward), step_info)  # type: ignore[arg-type]
                    outcome = _classify_outcome(step_info, env)

                if outcome == 1:
                    wins += 1
                elif outcome == -1:
                    losses += 1
                else:
                    draws += 1

                if recorder is not None and record_dir is not None:
                    ckpt_stem = Path(args.checkpoint).stem
                    save_path = record_dir / f"{ckpt_stem}_ep{ep:04d}.json"
                    recorder.save(save_path)
                    print(f"Recorded:  {save_path}")
        finally:
            env.close()

        result = EvaluationResult(
            wins=wins,
            draws=draws,
            losses=losses,
            n_episodes=args.n_episodes,
            win_rate=wins / args.n_episodes,
            draw_rate=draws / args.n_episodes,
            loss_rate=losses / args.n_episodes,
        )
    else:
        result = evaluate_detailed(
            checkpoint_path=args.checkpoint,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            seed=args.seed,
            opponent=args.opponent,
        )

    # Instantiate a registry for Elo computation.  When --elo-registry is
    # given we load from (and later persist to) that file; otherwise we use an
    # in-memory registry (path=None) so no file is created without explicit
    # opt-in.
    agent_name = args.agent_name or args.checkpoint
    registry = EloRegistry(args.elo_registry)  # None → in-memory

    opp_elo = registry.get_rating(args.opponent)
    print(f"Opponent:  {args.opponent} (Elo: {opp_elo:.0f})")
    print(
        f"Win rate:  {result.win_rate:.2%} "
        f"({result.wins}W / {result.draws}D / {result.losses}L "
        f"in {result.n_episodes} episodes)"
    )

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
        print(f"Registry:  saved to {args.elo_registry}")


if __name__ == "__main__":
    main()
