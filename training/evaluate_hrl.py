# training/evaluate_hrl.py
"""End-to-end HRL vs. flat MARL evaluation harness (Epic E3.7).

Implements a rigorous head-to-head tournament between the three-echelon HRL
architecture (DivisionEnv → BrigadeEnv → MultiBattalionEnv) and the v2 flat
MAPPO baseline (MAPPOPolicy on MultiBattalionEnv directly).

The evaluation protocol controls for:

* **Scenario** — both systems play the same 4v4 configuration (2 Blue brigades
  × 2 battalions each vs 2 Red brigades × 2 battalions each).
* **Episode count** — ≥ 100 episodes per system for statistical power.
* **Opponent** — both face the same Red policy (random or scripted stationary
  by default).
* **Confidence intervals** — bootstrapped 95 % CIs are reported so that
  readers can judge whether a difference in win rate is meaningful.

Typical usage::

    # Compare without checkpoints (random Blue actions — baseline smoke test)
    python training/evaluate_hrl.py --n-episodes 20

    # Full tournament with checkpoints
    python training/evaluate_hrl.py \\
        --division-checkpoint checkpoints/division/ppo_division_final.zip \\
        --mappo-checkpoint    checkpoints/mappo/team_snapshot_v000100.pt \\
        --n-episodes 100 --seed 0 \\
        --output results/hrl_vs_marl.json

Results are written as JSON to *--output* (default: stdout) and can be loaded
by ``notebooks/v3_hrl_analysis.ipynb`` for plotting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import numpy as np

# Ensure project root is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

__all__ = [
    "TournamentResult",
    "bootstrap_ci",
    "run_hrl_episodes",
    "run_flat_marl_episodes",
    "run_tournament",
]

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

#: Default number of bootstrap resamples for confidence intervals.
DEFAULT_N_BOOTSTRAP: int = 2000
#: Default confidence level for bootstrapped CIs.
DEFAULT_CI_CONFIDENCE: float = 0.95

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class TournamentResult(NamedTuple):
    """Structured result from one arm of the HRL tournament.

    Attributes
    ----------
    wins:
        Number of episodes the Blue side won.
    draws:
        Number of episodes that ended as a draw (timeout or mutual destruction).
    losses:
        Number of episodes the Blue side lost.
    n_episodes:
        Total episodes (``wins + draws + losses``).
    win_rate:
        ``wins / n_episodes``.
    draw_rate:
        ``draws / n_episodes``.
    loss_rate:
        ``losses / n_episodes``.
    ci_lower:
        Lower bound of the bootstrapped confidence interval for *win_rate*.
    ci_upper:
        Upper bound of the bootstrapped confidence interval for *win_rate*.
    ci_confidence:
        Nominal confidence level of the CI (e.g. 0.95).
    label:
        Short descriptive label (e.g. ``"HRL"`` or ``"FlatMARL"``).
    """

    wins: int
    draws: int
    losses: int
    n_episodes: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    ci_lower: float
    ci_upper: float
    ci_confidence: float
    label: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return self._asdict()


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    outcomes: Sequence[int],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CI_CONFIDENCE,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Compute a bootstrapped confidence interval for the win rate.

    Parameters
    ----------
    outcomes:
        Sequence of episode outcomes: ``1`` = win, ``0`` = draw, ``-1`` = loss.
        The win rate is estimated as ``mean(outcome == 1)``.
    n_bootstrap:
        Number of bootstrap resamples (default :data:`DEFAULT_N_BOOTSTRAP`).
    confidence:
        Nominal confidence level (default :data:`DEFAULT_CI_CONFIDENCE`).
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    (ci_lower, ci_upper)
        Bootstrapped percentile interval for the win rate.

    Raises
    ------
    ValueError
        If *outcomes* is empty, *n_bootstrap* < 1, or *confidence* not in (0, 1).
    """
    if len(outcomes) == 0:
        raise ValueError("outcomes must not be empty.")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    _rng = rng if rng is not None else np.random.default_rng()
    arr = np.asarray(outcomes)
    win_flags = (arr == 1).astype(np.float64)

    # Vectorised bootstrap: sample an (n_bootstrap × n) index matrix at once.
    n = len(win_flags)
    indices = _rng.integers(0, n, size=(n_bootstrap, n))
    boot_rates = win_flags[indices].mean(axis=1)

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_rates, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_rates, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


# ---------------------------------------------------------------------------
# HRL episode runner
# ---------------------------------------------------------------------------


def run_hrl_episodes(
    n_episodes: int = 100,
    division_checkpoint: Optional[str | Path] = None,
    n_brigades: int = 2,
    n_blue_per_brigade: int = 2,
    n_red_brigades: int = 2,
    n_red_per_brigade: int = 2,
    max_steps: int = 500,
    map_width: float = 1000.0,
    map_height: float = 1000.0,
    randomize_terrain: bool = True,
    visibility_radius: float = 600.0,
    red_random: bool = False,
    deterministic: bool = True,
    seed: Optional[int] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_confidence: float = DEFAULT_CI_CONFIDENCE,
    label: str = "HRL",
) -> TournamentResult:
    """Run *n_episodes* using the full three-echelon HRL stack.

    The Blue division commander is driven by *division_checkpoint* (an SB3 PPO
    ``.zip`` file) when provided.  When no checkpoint is supplied the division
    action space is sampled uniformly at random — useful as a baseline or smoke
    test.

    A **win** is recorded when ``info["winner"] == "blue"`` at episode end.
    A **loss** is recorded when ``info["winner"] == "red"``.  Everything else
    (timeout, draw) counts as a **draw**.

    Parameters
    ----------
    n_episodes:
        Number of evaluation episodes (must be ≥ 1).
    division_checkpoint:
        Optional path to an SB3 PPO ``.zip`` checkpoint for the division
        commander.  When ``None`` the division acts randomly.
    n_brigades:
        Number of Blue brigades.
    n_blue_per_brigade:
        Blue battalions per brigade.
    n_red_brigades:
        Number of Red brigades.
    n_red_per_brigade:
        Red battalions per Red brigade.
    max_steps:
        Maximum primitive steps per episode.
    map_width, map_height:
        Map dimensions in metres.
    randomize_terrain:
        Randomize terrain each episode.
    visibility_radius:
        Fog-of-war visibility radius in metres.
    red_random:
        Red battalions take random actions (ignored when a brigade policy is
        set on the environment).
    deterministic:
        Whether the division commander acts deterministically (only applies
        when *division_checkpoint* is provided).
    seed:
        Base random seed; episode *i* uses ``seed + i`` when provided.
    n_bootstrap:
        Number of bootstrap resamples for the CI.
    ci_confidence:
        Confidence level for the bootstrapped CI.
    label:
        Label embedded in the returned :class:`TournamentResult`.

    Returns
    -------
    TournamentResult
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}.")

    from envs.division_env import DivisionEnv

    env = DivisionEnv(
        n_brigades=n_brigades,
        n_blue_per_brigade=n_blue_per_brigade,
        n_red_brigades=n_red_brigades,
        n_red_per_brigade=n_red_per_brigade,
        max_steps=max_steps,
        map_width=map_width,
        map_height=map_height,
        randomize_terrain=randomize_terrain,
        visibility_radius=visibility_radius,
        red_random=red_random,
    )

    model = None
    if division_checkpoint is not None:
        from stable_baselines3 import PPO

        ckpt = Path(division_checkpoint)
        log.info("Loading HRL division checkpoint: %s", ckpt)
        model = PPO.load(str(ckpt), env=env)

    outcomes: list[int] = []
    wins = draws = losses = 0

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        info: dict = {}
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = env.action_space.sample()
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        winner = info.get("winner", "")
        if winner == "blue":
            wins += 1
            outcomes.append(1)
        elif winner == "red":
            losses += 1
            outcomes.append(-1)
        else:
            draws += 1
            outcomes.append(0)

    env.close()

    ci_lower, ci_upper = bootstrap_ci(
        outcomes,
        n_bootstrap=n_bootstrap,
        confidence=ci_confidence,
        rng=np.random.default_rng(None if seed is None else seed + 10_000),
    )

    return TournamentResult(
        wins=wins,
        draws=draws,
        losses=losses,
        n_episodes=n_episodes,
        win_rate=wins / n_episodes,
        draw_rate=draws / n_episodes,
        loss_rate=losses / n_episodes,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_confidence=ci_confidence,
        label=label,
    )


# ---------------------------------------------------------------------------
# Flat MARL episode runner
# ---------------------------------------------------------------------------


def run_flat_marl_episodes(
    n_episodes: int = 100,
    mappo_checkpoint: Optional[str | Path] = None,
    n_blue: int = 4,
    n_red: int = 4,
    max_steps: int = 500,
    map_width: float = 1000.0,
    map_height: float = 1000.0,
    randomize_terrain: bool = True,
    visibility_radius: float = 600.0,
    red_random: bool = False,
    deterministic: bool = True,
    seed: Optional[int] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_confidence: float = DEFAULT_CI_CONFIDENCE,
    label: str = "FlatMARL",
) -> TournamentResult:
    """Run *n_episodes* using the flat MAPPO baseline.

    The Blue team is driven by *mappo_checkpoint* (a ``MAPPOPolicy`` ``.pt``
    snapshot) when provided.  When no checkpoint is supplied the Blue team
    acts randomly — useful as a baseline or smoke test.

    The win condition mirrors the HRL runner: the episode ``info`` dict
    is inspected for ``info["winner"]`` after episode end.  As a fallback,
    survivors are counted: Blue wins if at least one Blue battalion is alive
    and no Red battalions are alive.

    Parameters
    ----------
    n_episodes:
        Number of evaluation episodes (must be ≥ 1).
    mappo_checkpoint:
        Optional path to a ``MAPPOPolicy`` ``.pt`` snapshot.  When ``None``
        the Blue team acts randomly.
    n_blue:
        Number of Blue agents.
    n_red:
        Number of Red agents.
    max_steps:
        Maximum primitive steps per episode.
    map_width, map_height:
        Map dimensions in metres.
    randomize_terrain:
        Randomize terrain each episode.
    visibility_radius:
        Fog-of-war visibility radius in metres.
    red_random:
        Red agents take random actions when ``True``; stationary otherwise.
    deterministic:
        Whether the Blue MAPPO policy acts deterministically (only when a
        checkpoint is provided).
    seed:
        Base random seed; episode *i* uses ``seed + i`` when provided.
    n_bootstrap:
        Number of bootstrap resamples for the CI.
    ci_confidence:
        Confidence level for the bootstrapped CI.
    label:
        Label embedded in the returned :class:`TournamentResult`.

    Returns
    -------
    TournamentResult
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}.")

    import torch
    from envs.multi_battalion_env import MultiBattalionEnv

    env = MultiBattalionEnv(
        n_blue=n_blue,
        n_red=n_red,
        max_steps=max_steps,
        map_width=map_width,
        map_height=map_height,
        randomize_terrain=randomize_terrain,
        visibility_radius=visibility_radius,
    )
    act_low = env._act_space.low
    act_high = env._act_space.high
    obs_dim = env._obs_dim

    policy = None
    if mappo_checkpoint is not None:
        from models.mappo_policy import MAPPOPolicy

        ckpt = Path(mappo_checkpoint)
        log.info("Loading flat MARL MAPPO checkpoint: %s", ckpt)
        data = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        policy = MAPPOPolicy(**data["kwargs"])
        policy.load_state_dict(data["state_dict"])
        policy.eval()

    outcomes: list[int] = []
    wins = draws = losses = 0

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        blue_won = False
        red_won = False

        while env.agents:
            actions: dict[str, np.ndarray] = {}
            zero_obs = np.zeros(obs_dim, dtype=np.float32)
            for i in range(n_blue):
                agent_id = f"blue_{i}"
                if agent_id in env.agents:
                    if policy is not None:
                        obs_t = torch.as_tensor(
                            obs.get(agent_id, zero_obs), dtype=torch.float32
                        ).unsqueeze(0)
                        with torch.no_grad():
                            acts_t, _ = policy.act(
                                obs_t,
                                agent_idx=i % policy.n_agents,
                                deterministic=deterministic,
                            )
                        actions[agent_id] = np.clip(
                            acts_t[0].cpu().numpy(), act_low, act_high
                        )
                    else:
                        actions[agent_id] = env.action_space(agent_id).sample()

            for i in range(n_red):
                agent_id = f"red_{i}"
                if agent_id in env.agents:
                    if red_random:
                        actions[agent_id] = env.action_space(agent_id).sample()
                    else:
                        actions[agent_id] = np.zeros(3, dtype=np.float32)

            obs, _rewards, _terminated, _truncated, _info = env.step(actions)

            # Win/loss detection after step (env.agents updated by step).
            red_alive = any(a.startswith("red_") for a in env.agents)
            blue_alive = any(a.startswith("blue_") for a in env.agents)
            if not red_alive and blue_alive and not blue_won:
                blue_won = True
            if not blue_alive and red_alive and not red_won:
                red_won = True

        if blue_won:
            wins += 1
            outcomes.append(1)
        elif red_won:
            losses += 1
            outcomes.append(-1)
        else:
            draws += 1
            outcomes.append(0)

    env.close()

    ci_lower, ci_upper = bootstrap_ci(
        outcomes,
        n_bootstrap=n_bootstrap,
        confidence=ci_confidence,
        rng=np.random.default_rng(None if seed is None else seed + 20_000),
    )

    return TournamentResult(
        wins=wins,
        draws=draws,
        losses=losses,
        n_episodes=n_episodes,
        win_rate=wins / n_episodes,
        draw_rate=draws / n_episodes,
        loss_rate=losses / n_episodes,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_confidence=ci_confidence,
        label=label,
    )


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------


def run_tournament(
    hrl_result: TournamentResult,
    flat_result: TournamentResult,
) -> dict:
    """Produce a summary dict comparing HRL and flat MARL results.

    Parameters
    ----------
    hrl_result:
        :class:`TournamentResult` from :func:`run_hrl_episodes`.
    flat_result:
        :class:`TournamentResult` from :func:`run_flat_marl_episodes`.

    Returns
    -------
    dict
        JSON-serialisable summary containing both results and a brief
        conclusion string (``"HRL wins"``, ``"Flat MARL wins"``, or
        ``"Inconclusive"``).
    """
    delta = hrl_result.win_rate - flat_result.win_rate

    # CIs overlap?
    ci_overlap = (
        hrl_result.ci_lower <= flat_result.ci_upper
        and flat_result.ci_lower <= hrl_result.ci_upper
    )

    if ci_overlap:
        conclusion = "Inconclusive (confidence intervals overlap)"
    elif delta > 0:
        conclusion = f"{hrl_result.label} outperforms {flat_result.label}"
    else:
        conclusion = f"{flat_result.label} outperforms {hrl_result.label}"

    return {
        "hrl": hrl_result.to_dict(),
        "flat_marl": flat_result.to_dict(),
        "delta_win_rate": round(delta, 4),
        "ci_overlap": ci_overlap,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the HRL evaluation harness."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="evaluate_hrl",
        description=(
            "Run a head-to-head tournament: HRL (DivisionEnv) vs "
            "flat MARL (MAPPO on MultiBattalionEnv)."
        ),
    )
    parser.add_argument(
        "--division-checkpoint",
        default=None,
        help=(
            "Path to an SB3 PPO .zip checkpoint for the HRL division "
            "commander.  When omitted the division acts randomly."
        ),
    )
    parser.add_argument(
        "--mappo-checkpoint",
        default=None,
        help=(
            "Path to a MAPPOPolicy .pt snapshot for the flat MARL baseline. "
            "When omitted Blue acts randomly."
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Episodes per arm (default: 100; acceptance criterion >= 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-brigades",
        type=int,
        default=2,
        help="Number of Blue brigades (default: 2 → 4v4 with 2 battalions each).",
    )
    parser.add_argument(
        "--n-blue-per-brigade",
        type=int,
        default=2,
        help="Blue battalions per brigade (default: 2).",
    )
    parser.add_argument(
        "--n-red-brigades",
        type=int,
        default=2,
        help="Number of Red brigades (default: 2).",
    )
    parser.add_argument(
        "--n-red-per-brigade",
        type=int,
        default=2,
        help="Red battalions per Red brigade (default: 2).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for CI (default: 2000).",
    )
    parser.add_argument(
        "--ci-confidence",
        type=float,
        default=0.95,
        help="CI confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--red-random",
        action="store_true",
        help="Red battalions act randomly (default: stationary).",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Blue acts deterministically (default).",
    )
    action_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Blue acts stochastically.",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to write the JSON results file. "
            "When omitted, results are printed to stdout."
        ),
    )

    args = parser.parse_args(argv)

    n_blue = args.n_brigades * args.n_blue_per_brigade
    n_red = args.n_red_brigades * args.n_red_per_brigade

    log.info(
        "Running HRL evaluation: n_episodes=%d  scenario=%dv%d  seed=%s",
        args.n_episodes,
        n_blue,
        n_red,
        args.seed,
    )

    # ── HRL arm ────────────────────────────────────────────────────────
    log.info("=== HRL arm ===")
    try:
        hrl_result = run_hrl_episodes(
            n_episodes=args.n_episodes,
            division_checkpoint=args.division_checkpoint,
            n_brigades=args.n_brigades,
            n_blue_per_brigade=args.n_blue_per_brigade,
            n_red_brigades=args.n_red_brigades,
            n_red_per_brigade=args.n_red_per_brigade,
            red_random=args.red_random,
            deterministic=args.deterministic,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            ci_confidence=args.ci_confidence,
        )
    except ValueError as exc:
        parser.error(str(exc))
    log.info(
        "HRL: wins=%d draws=%d losses=%d win_rate=%.3f CI=[%.3f, %.3f]",
        hrl_result.wins,
        hrl_result.draws,
        hrl_result.losses,
        hrl_result.win_rate,
        hrl_result.ci_lower,
        hrl_result.ci_upper,
    )

    # ── Flat MARL arm ──────────────────────────────────────────────────
    log.info("=== Flat MARL arm ===")
    flat_result = run_flat_marl_episodes(
        n_episodes=args.n_episodes,
        mappo_checkpoint=args.mappo_checkpoint,
        n_blue=n_blue,
        n_red=n_red,
        red_random=args.red_random,
        deterministic=args.deterministic,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        ci_confidence=args.ci_confidence,
    )
    log.info(
        "Flat MARL: wins=%d draws=%d losses=%d win_rate=%.3f CI=[%.3f, %.3f]",
        flat_result.wins,
        flat_result.draws,
        flat_result.losses,
        flat_result.win_rate,
        flat_result.ci_lower,
        flat_result.ci_upper,
    )

    # ── Tournament summary ──────────────────────────────────────────────
    summary = run_tournament(hrl_result, flat_result)
    log.info("Conclusion: %s", summary["conclusion"])

    output_json = json.dumps(summary, indent=2)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json)
        log.info("Results written to %s", out_path)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
