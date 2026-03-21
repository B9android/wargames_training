# analysis/coa_generator.py
"""Course of Action (COA) Generator (Epic E5.2).

Uses Monte-Carlo rollout to generate and rank multiple candidate courses of
action for a given scenario.  Each COA corresponds to a distinct tactical
archetype (e.g. aggressive advance, flanking, defensive stand-off) realised
by biasing the action outputs of a base policy or random sampler.

Typical usage::

    from analysis.coa_generator import COAGenerator
    from envs.battalion_env import BattalionEnv

    env = BattalionEnv(randomize_terrain=False)
    generator = COAGenerator(env=env, n_rollouts=30, seed=42)
    coas = generator.generate()          # uses random policy when none provided
    for coa in coas:
        print(coa.rank, coa.label, coa.score.win_rate)

    # Or pass a trained SB3 PPO model:
    from stable_baselines3 import PPO
    model = PPO.load("checkpoints/my_model")
    coas = generator.generate(policy=model)

The generator guarantees ``n_coas`` distinct COAs whose aggregate action
sequences differ meaningfully (they are initialised from different tactical
archetypes, not merely different random seeds).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from envs.battalion_env import BattalionEnv, DESTROYED_THRESHOLD

__all__ = [
    "COAScore",
    "CourseOfAction",
    "COAGenerator",
    "generate_coas",
    "STRATEGY_LABELS",
]

# ---------------------------------------------------------------------------
# Strategy archetypes
# ---------------------------------------------------------------------------

#: Built-in strategy labels, each corresponding to a different action bias.
STRATEGY_LABELS: Tuple[str, ...] = (
    "aggressive",
    "defensive",
    "flanking_left",
    "flanking_right",
    "probing",
    "standoff",
    "rapid_assault",
)

# Each archetype is a dict of action-space biases applied additively to the
# base-policy action before clipping to valid bounds.
# Keys: "move_bias", "rotate_bias", "fire_bias"
# Values in [-1, 1] (or [-1, 1] for move/rotate, [0, 1] for fire).
_STRATEGY_BIASES: Dict[str, Dict[str, float]] = {
    "aggressive":     {"move_bias":  0.7, "rotate_bias":  0.0, "fire_bias":  0.8},
    "defensive":      {"move_bias": -0.5, "rotate_bias":  0.0, "fire_bias":  0.4},
    "flanking_left":  {"move_bias":  0.4, "rotate_bias": -0.5, "fire_bias":  0.3},
    "flanking_right": {"move_bias":  0.4, "rotate_bias":  0.5, "fire_bias":  0.3},
    "probing":        {"move_bias":  0.2, "rotate_bias":  0.1, "fire_bias":  0.1},
    "standoff":       {"move_bias":  0.0, "rotate_bias":  0.0, "fire_bias":  1.0},
    "rapid_assault":  {"move_bias":  1.0, "rotate_bias":  0.0, "fire_bias":  1.0},
}

# Composite score weights (must sum to 1).
_WIN_RATE_WEIGHT: float = 0.5
_CASUALTY_EFFICIENCY_WEIGHT: float = 0.3   # red casualties minus blue casualties
_TERRAIN_CONTROL_WEIGHT: float = 0.2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class COAScore:
    """Scalar metrics summarising the outcomes of one COA's Monte-Carlo rollouts.

    Attributes
    ----------
    win_rate:
        Fraction of rollouts won by Blue (0–1).
    draw_rate:
        Fraction of rollouts that ended as a draw (0–1).
    loss_rate:
        Fraction of rollouts lost by Blue (0–1).
    blue_casualties:
        Mean normalised Blue strength *loss* across rollouts (0–1).
        Higher means more Blue casualties; 0 = no damage taken.
    red_casualties:
        Mean normalised Red strength *loss* across rollouts (0–1).
        Higher means Blue dealt more damage to Red.
    terrain_control:
        Mean fraction of steps in which Blue held terrain advantage
        (closer to map centre than Red), averaged across rollouts (0–1).
    composite:
        Weighted composite score used for ranking (higher is better).
    n_rollouts:
        Number of rollouts used to compute these statistics.
    """

    win_rate: float
    draw_rate: float
    loss_rate: float
    blue_casualties: float
    red_casualties: float
    terrain_control: float
    composite: float
    n_rollouts: int

    def as_dict(self) -> dict:
        """Return a plain ``dict`` representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CourseOfAction:
    """A candidate tactical plan with associated outcome predictions.

    Attributes
    ----------
    label:
        Human-readable name of the tactical archetype.
    rank:
        Rank among all generated COAs (1 = best, higher = worse).
    score:
        Aggregated outcome statistics from Monte-Carlo rollouts.
    action_summary:
        Aggregate action statistics across rollouts:
        mean ``move``, ``rotate``, and ``fire`` per quartile of the episode.
    seed:
        Base random seed used to initialise this COA's rollouts.
    """

    label: str
    rank: int
    score: COAScore
    action_summary: dict
    seed: int

    def as_dict(self) -> dict:
        """Return a JSON-serialisable ``dict``."""
        return {
            "label": self.label,
            "rank": self.rank,
            "score": self.score.as_dict(),
            "action_summary": self.action_summary,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _BiasedPolicy:
    """Wraps an optional base policy with a fixed additive action bias.

    When no base policy is supplied the Blue action is drawn from the
    environment's action space (uniform random), then the bias is applied.

    Parameters
    ----------
    base_policy:
        Optional policy with a ``predict(obs, deterministic) -> (action, state)``
        method.  Pass ``None`` to use purely random actions.
    strategy:
        Name of the archetype from :data:`STRATEGY_LABELS`.
    rng:
        NumPy random generator for random-policy fallback.
    """

    def __init__(
        self,
        base_policy: Optional[Any],
        strategy: str,
        rng: np.random.Generator,
    ) -> None:
        if strategy not in _STRATEGY_BIASES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid strategies: {sorted(_STRATEGY_BIASES)}"
            )
        self._base = base_policy
        self._biases = _STRATEGY_BIASES[strategy]
        self._rng = rng

    def predict(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, Any]:
        if self._base is not None:
            action, state = self._base.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32).copy()
        else:
            action = np.array(
                [
                    self._rng.uniform(-1.0, 1.0),
                    self._rng.uniform(-1.0, 1.0),
                    self._rng.uniform(0.0, 1.0),
                ],
                dtype=np.float32,
            )
            state = None

        # Apply additive bias, then clip to valid action bounds.
        action[0] = float(np.clip(action[0] + self._biases["move_bias"],   -1.0, 1.0))
        action[1] = float(np.clip(action[1] + self._biases["rotate_bias"], -1.0, 1.0))
        action[2] = float(np.clip(action[2] + self._biases["fire_bias"],    0.0, 1.0))
        return action, state


def _run_single_rollout(
    env: BattalionEnv,
    policy: _BiasedPolicy,
    seed: int,
) -> dict:
    """Run one episode and return per-step action and outcome statistics.

    Returns
    -------
    dict with keys:
        ``outcome``         – 1 (win), -1 (loss), 0 (draw)
        ``blue_strength``   – final Blue strength
        ``red_strength``    – final Red strength
        ``steps``           – episode length
        ``actions``         – (steps, 3) float32 array of actions taken
        ``terrain_control_frac`` – fraction of steps Blue held terrain advantage
    """
    obs, _ = env.reset(seed=seed)
    done = False
    actions: List[np.ndarray] = []
    terrain_advantage_steps = 0
    step = 0

    while not done:
        action, _ = policy.predict(obs, deterministic=False)
        obs, _reward, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        done = terminated or truncated
        step += 1

        # Terrain control: Blue is "winning" the terrain if it is closer to
        # the map centre than Red.
        if env.blue is not None and env.red is not None:
            cx = env.map_width / 2.0
            cy = env.map_height / 2.0
            d_blue = math.hypot(env.blue.x - cx, env.blue.y - cy)
            d_red  = math.hypot(env.red.x  - cx, env.red.y  - cy)
            if d_blue < d_red:
                terrain_advantage_steps += 1

    # Determine outcome
    red_lost = info.get("red_routed", False) or (
        env.red is not None and env.red.strength <= DESTROYED_THRESHOLD
    )
    blue_lost = info.get("blue_routed", False) or (
        env.blue is not None and env.blue.strength <= DESTROYED_THRESHOLD
    )
    if red_lost and not blue_lost:
        outcome = 1
    elif blue_lost and not red_lost:
        outcome = -1
    else:
        outcome = 0

    action_arr = np.stack(actions, axis=0) if actions else np.zeros((0, 3), dtype=np.float32)
    tc_frac = terrain_advantage_steps / max(step, 1)

    return {
        "outcome": outcome,
        "blue_strength": float(env.blue.strength) if env.blue is not None else 0.0,
        "red_strength":  float(env.red.strength)  if env.red  is not None else 0.0,
        "steps": step,
        "actions": action_arr,
        "terrain_control_frac": tc_frac,
    }


def _aggregate_rollouts(results: List[dict]) -> Tuple[COAScore, dict]:
    """Compute a :class:`COAScore` and action summary from multiple rollouts.

    Parameters
    ----------
    results:
        List of dicts as returned by :func:`_run_single_rollout`.

    Returns
    -------
    (COAScore, action_summary_dict)
    """
    n = len(results)
    wins  = sum(1 for r in results if r["outcome"] == 1)
    draws = sum(1 for r in results if r["outcome"] == 0)
    losses = sum(1 for r in results if r["outcome"] == -1)
    win_rate  = wins  / n
    draw_rate = draws / n
    loss_rate = losses / n

    blue_casualties = float(np.mean([1.0 - r["blue_strength"] for r in results]))
    red_casualties  = float(np.mean([1.0 - r["red_strength"]  for r in results]))
    terrain_control = float(np.mean([r["terrain_control_frac"] for r in results]))

    # Composite: win rate + casualty efficiency + terrain control
    casualty_efficiency = (red_casualties - blue_casualties + 1.0) / 2.0  # normalised to [0,1]
    composite = (
        _WIN_RATE_WEIGHT           * win_rate
        + _CASUALTY_EFFICIENCY_WEIGHT * casualty_efficiency
        + _TERRAIN_CONTROL_WEIGHT     * terrain_control
    )

    score = COAScore(
        win_rate=round(win_rate, 4),
        draw_rate=round(draw_rate, 4),
        loss_rate=round(loss_rate, 4),
        blue_casualties=round(blue_casualties, 4),
        red_casualties=round(red_casualties, 4),
        terrain_control=round(terrain_control, 4),
        composite=round(composite, 4),
        n_rollouts=n,
    )

    # Action summary: mean action per episode-quartile across all rollouts
    all_actions = [r["actions"] for r in results if len(r["actions"]) > 0]
    action_summary: dict = {}
    if all_actions:
        quartile_means: Dict[str, List[float]] = {
            "move_q1": [], "move_q2": [], "move_q3": [], "move_q4": [],
            "rotate_q1": [], "rotate_q2": [], "rotate_q3": [], "rotate_q4": [],
            "fire_q1": [], "fire_q2": [], "fire_q3": [], "fire_q4": [],
        }
        for ep_actions in all_actions:
            T = len(ep_actions)
            if T == 0:
                continue
            q = max(T // 4, 1)
            slices = [
                ep_actions[:q],
                ep_actions[q: 2 * q],
                ep_actions[2 * q: 3 * q],
                ep_actions[3 * q:],
            ]
            for qi, sl in enumerate(slices, start=1):
                if len(sl) == 0:
                    sl = ep_actions  # fall back to full episode for short episodes
                quartile_means[f"move_q{qi}"].append(float(np.mean(sl[:, 0])))
                quartile_means[f"rotate_q{qi}"].append(float(np.mean(sl[:, 1])))
                quartile_means[f"fire_q{qi}"].append(float(np.mean(sl[:, 2])))

        action_summary = {
            k: round(float(np.mean(v)), 4) if v else 0.0
            for k, v in quartile_means.items()
        }
    return score, action_summary


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class COAGenerator:
    """Generate and rank candidate Courses of Action via Monte-Carlo rollout.

    Parameters
    ----------
    env:
        A :class:`~envs.battalion_env.BattalionEnv` instance.  The
        environment is reset before each rollout; the caller retains
        ownership and is responsible for closing it.
    n_rollouts:
        Number of Monte-Carlo rollouts per COA (default 20).  More
        rollouts give more stable estimates but take longer.
    n_coas:
        Number of distinct COAs to generate (default 5, maximum is the
        number of built-in strategy archetypes, i.e. 7).
    seed:
        Base random seed for reproducibility.  Each COA uses a
        deterministic derived seed.
    strategies:
        Explicit list of strategy labels to use.  When ``None`` (the
        default) the first ``n_coas`` strategies from
        :data:`STRATEGY_LABELS` are used.
    """

    def __init__(
        self,
        env: BattalionEnv,
        n_rollouts: int = 20,
        n_coas: int = 5,
        seed: Optional[int] = None,
        strategies: Optional[Sequence[str]] = None,
    ) -> None:
        if n_rollouts < 1:
            raise ValueError(f"n_rollouts must be >= 1, got {n_rollouts}")
        if n_coas < 1:
            raise ValueError(f"n_coas must be >= 1, got {n_coas}")
        if n_coas > len(STRATEGY_LABELS):
            raise ValueError(
                f"n_coas ({n_coas}) exceeds the number of built-in strategy "
                f"archetypes ({len(STRATEGY_LABELS)}).  Pass a custom "
                f"'strategies' list or reduce n_coas."
            )
        self.env = env
        self.n_rollouts = int(n_rollouts)
        self.n_coas = int(n_coas)
        self.seed = seed

        if strategies is not None:
            invalid = [s for s in strategies if s not in _STRATEGY_BIASES]
            if invalid:
                raise ValueError(
                    f"Unknown strategy labels: {invalid}. "
                    f"Valid: {sorted(_STRATEGY_BIASES)}"
                )
            self._strategies: List[str] = list(strategies)[: self.n_coas]
        else:
            self._strategies = list(STRATEGY_LABELS[: self.n_coas])

    def generate(
        self,
        policy: Optional[Any] = None,
        deterministic: bool = False,
    ) -> List[CourseOfAction]:
        """Generate and rank candidate COAs.

        Parameters
        ----------
        policy:
            Optional trained policy (e.g. a Stable-Baselines3 ``PPO`` model).
            Must expose ``predict(obs, deterministic) -> (action, state)``.
            When ``None`` a random action policy is used (useful for smoke
            tests and scenario exploration without a trained model).
        deterministic:
            Passed through to the base policy's ``predict`` call.  Ignored
            when *policy* is ``None``.

        Returns
        -------
        list of :class:`CourseOfAction`
            Ordered from best to worst by composite score.  The list has
            exactly ``n_coas`` entries.
        """
        coa_list: List[CourseOfAction] = []

        for coa_idx, strategy in enumerate(self._strategies):
            coa_seed = (
                (self.seed * 1000 + coa_idx * 37) if self.seed is not None
                else coa_idx * 37
            )
            rng = np.random.default_rng(coa_seed)
            biased = _BiasedPolicy(
                base_policy=policy,
                strategy=strategy,
                rng=rng,
            )

            rollout_results: List[dict] = []
            for rollout_i in range(self.n_rollouts):
                ep_seed = coa_seed + rollout_i
                result = _run_single_rollout(self.env, biased, seed=ep_seed)
                rollout_results.append(result)

            score, action_summary = _aggregate_rollouts(rollout_results)
            coa_list.append(
                CourseOfAction(
                    label=strategy,
                    rank=0,   # assigned after sorting
                    score=score,
                    action_summary=action_summary,
                    seed=coa_seed,
                )
            )

        # Sort by composite score (descending) and assign ranks.
        coa_list.sort(key=lambda c: c.score.composite, reverse=True)
        for rank, coa in enumerate(coa_list, start=1):
            coa.rank = rank

        return coa_list


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def generate_coas(
    env: Optional[BattalionEnv] = None,
    policy: Optional[Any] = None,
    n_rollouts: int = 20,
    n_coas: int = 5,
    seed: Optional[int] = None,
    strategies: Optional[Sequence[str]] = None,
    env_kwargs: Optional[dict] = None,
) -> List[CourseOfAction]:
    """Generate COAs, optionally creating a temporary environment.

    This is a thin convenience wrapper around :class:`COAGenerator`.

    Parameters
    ----------
    env:
        An existing :class:`~envs.battalion_env.BattalionEnv`.  When
        ``None`` a new environment is created using *env_kwargs* and
        closed automatically when generation completes.
    policy:
        Optional trained policy (see :meth:`COAGenerator.generate`).
    n_rollouts:
        Number of Monte-Carlo rollouts per COA.
    n_coas:
        Number of distinct COAs to generate (1–7).
    seed:
        Base random seed.
    strategies:
        Explicit ordered list of strategy labels to evaluate.
    env_kwargs:
        Keyword arguments forwarded to :class:`BattalionEnv` when *env*
        is ``None``.

    Returns
    -------
    list of :class:`CourseOfAction`, best first.
    """
    owns_env = env is None
    active_env = (
        env
        if env is not None
        else BattalionEnv(**(env_kwargs or {}))
    )
    try:
        generator = COAGenerator(
            env=active_env,
            n_rollouts=n_rollouts,
            n_coas=n_coas,
            seed=seed,
            strategies=strategies,
        )
        return generator.generate(policy=policy)
    finally:
        if owns_env:
            active_env.close()
