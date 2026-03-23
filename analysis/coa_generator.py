# analysis/coa_generator.py
"""Course of Action (COA) Generator (Epics E5.2 / E9.2).

Uses Monte-Carlo rollout to generate and rank multiple candidate courses of
action for a given scenario.  Each COA corresponds to a distinct tactical
archetype (e.g. aggressive advance, flanking, defensive stand-off) realised
by biasing the action outputs of a base policy or random sampler.

Battalion-level usage::

    from analysis.coa_generator import COAGenerator
    from envs.battalion_env import BattalionEnv

    env = BattalionEnv(randomize_terrain=False)
    generator = COAGenerator(env=env, n_rollouts=30, seed=42)
    coas = generator.generate()          # uses random policy when none provided
    for coa in coas:
        print(coa.rank, coa.label, coa.score.win_rate)

Corps-level usage (E9.2)::

    from analysis.coa_generator import CorpsCOAGenerator
    from envs.corps_env import CorpsEnv

    env = CorpsEnv(n_divisions=3)
    generator = CorpsCOAGenerator(env=env, n_rollouts=10, n_coas=10, seed=42)
    coas = generator.generate()
    for coa in coas:
        print(coa.rank, coa.label, coa.score.composite)

    # Explain a COA — returns 3+ key decisions:
    explanation = generator.explain_coa(coas[0])
    for decision in explanation.key_decisions:
        print(decision)

    # Modify a COA and re-evaluate:
    from analysis.coa_generator import COAModification
    mod = COAModification(strategy_override="pincer_attack", n_rollouts=5)
    updated = generator.modify_and_evaluate(coas[0], mod)

The generator guarantees ``n_coas`` distinct COAs whose aggregate action
sequences differ meaningfully (they are initialised from different tactical
archetypes, not merely different random seeds).
"""

from __future__ import annotations

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
    # Corps-level (E9.2)
    "CorpsCOAScore",
    "CorpsCourseOfAction",
    "COAExplanation",
    "COAModification",
    "CorpsCOAGenerator",
    "generate_corps_coas",
    "CORPS_STRATEGY_LABELS",
    # Internal helpers exposed for testing
    "_CorpsStrategyPolicy",
    "_run_corps_rollout",
    "_aggregate_corps_rollouts",
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
    deterministic: bool = False,
) -> dict:
    """Run one episode and return per-step action and outcome statistics.

    Parameters
    ----------
    env:
        The battalion environment to roll out in.
    policy:
        Biased policy to use for action selection.
    seed:
        Random seed for the environment reset.
    deterministic:
        Passed through to the policy's ``predict`` call.

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
        action, _ = policy.predict(obs, deterministic=deterministic)
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
            # Deduplicate while preserving order to ensure distinct COAs.
            unique_strategies = list(dict.fromkeys(strategies))
            if len(unique_strategies) < self.n_coas:
                raise ValueError(
                    "Insufficient distinct strategy labels provided: "
                    f"expected at least {self.n_coas}, "
                    f"got {len(unique_strategies)} from {list(strategies)!r}"
                )
            self._strategies: List[str] = unique_strategies[: self.n_coas]
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
                result = _run_single_rollout(self.env, biased, seed=ep_seed,
                                            deterministic=deterministic)
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


# ===========================================================================
# Corps-level COA generation (Epic E9.2)
# ===========================================================================

# ---------------------------------------------------------------------------
# Corps strategy archetypes
# ---------------------------------------------------------------------------

# Command index constants matching the BrigadeEnv option vocabulary
# (advance_sector=0, defend_position=1, flank_left=2, flank_right=3,
#  withdraw=4, concentrate_fire=5).
_CMD_ADVANCE = 0
_CMD_DEFEND = 1
_CMD_FLANK_L = 2
_CMD_FLANK_R = 3
_CMD_WITHDRAW = 4
_CMD_FIRE = 5

#: Built-in corps-level strategy labels (10 archetypes to meet the E9.2
#: requirement of generating ≥ 10 COAs).
CORPS_STRATEGY_LABELS: Tuple[str, ...] = (
    "full_advance",
    "fortress_defense",
    "left_envelopment",
    "right_envelopment",
    "pincer_attack",
    "fire_superiority",
    "advance_and_fix",
    "feint_and_assault",
    "strategic_withdrawal",
    "rapid_exploitation",
)

# Each entry maps a strategy name to a list of preferred command indices
# per division *position* (front / mid / rear pattern, cycled if more
# divisions than pattern length).
_CORPS_STRATEGY_PATTERNS: Dict[str, List[int]] = {
    "full_advance":         [_CMD_ADVANCE, _CMD_ADVANCE, _CMD_ADVANCE],
    "fortress_defense":     [_CMD_DEFEND,  _CMD_DEFEND,  _CMD_DEFEND],
    "left_envelopment":     [_CMD_FLANK_L, _CMD_ADVANCE, _CMD_FLANK_L],
    "right_envelopment":    [_CMD_FLANK_R, _CMD_ADVANCE, _CMD_FLANK_R],
    "pincer_attack":        [_CMD_FLANK_L, _CMD_ADVANCE, _CMD_FLANK_R],
    "fire_superiority":     [_CMD_FIRE,    _CMD_FIRE,    _CMD_FIRE],
    "advance_and_fix":      [_CMD_ADVANCE, _CMD_FIRE,    _CMD_ADVANCE],
    "feint_and_assault":    [_CMD_FLANK_L, _CMD_FIRE,    _CMD_ADVANCE],
    "strategic_withdrawal": [_CMD_WITHDRAW, _CMD_DEFEND,  _CMD_WITHDRAW],
    "rapid_exploitation":   [_CMD_ADVANCE,  _CMD_FLANK_R, _CMD_FLANK_L],
}

# Probability of choosing the preferred command (vs. uniform-random).
_CORPS_BIAS_STRENGTH: float = 0.80

# Composite score weights for corps-level scoring.
_CORPS_WIN_RATE_WEIGHT: float = 0.40
_CORPS_CASUALTY_EFF_WEIGHT: float = 0.20
_CORPS_OBJECTIVE_WEIGHT: float = 0.25
_CORPS_SUPPLY_WEIGHT: float = 0.15


# ---------------------------------------------------------------------------
# Corps-level data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CorpsCOAScore:
    """Outcome metrics for one corps-level COA.

    Attributes
    ----------
    win_rate, draw_rate, loss_rate:
        Episode-outcome fractions across rollouts.
    blue_casualties:
        Mean fractional Blue unit-count loss.
    red_casualties:
        Mean fractional Red unit-count loss.
    objective_completion:
        Mean fraction of operational objectives completed by Blue (0–1).
    supply_efficiency:
        Mean end-of-episode Blue supply level (0–1).
    composite:
        Weighted composite score used for ranking.
    n_rollouts:
        Number of rollouts used.
    """

    win_rate: float
    draw_rate: float
    loss_rate: float
    blue_casualties: float
    red_casualties: float
    objective_completion: float
    supply_efficiency: float
    composite: float
    n_rollouts: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class COAExplanation:
    """Explanation of a COA highlighting key decisions.

    Attributes
    ----------
    coa_label:
        Label of the explained COA.
    key_decisions:
        Ordered list of human-readable decision strings (≥ 3 items) that
        most strongly drive the outcome for this COA.  Sorted by impact
        (highest impact first).
    command_frequency:
        Dict mapping command name → mean fraction of steps each command
        was issued across all divisions and rollouts.
    winning_patterns:
        Top-3 command sequences (as lists of command names) that appear
        most often in winning rollouts but rarely in losing ones.
    objective_timeline:
        Mean objective-reward per episode-quartile.  Keys are
        ``"q1", "q2", "q3", "q4"``.
    """

    coa_label: str
    key_decisions: List[str]
    command_frequency: Dict[str, float]
    winning_patterns: List[List[str]]
    objective_timeline: Dict[str, float]

    def as_dict(self) -> dict:
        return {
            "coa_label": self.coa_label,
            "key_decisions": self.key_decisions,
            "command_frequency": self.command_frequency,
            "winning_patterns": self.winning_patterns,
            "objective_timeline": self.objective_timeline,
        }


@dataclasses.dataclass
class CorpsCourseOfAction:
    """A candidate corps-level plan with outcome predictions.

    Attributes
    ----------
    label:
        Human-readable strategy archetype name.
    rank:
        Rank among all generated COAs (1 = best).
    score:
        Aggregated outcome statistics.
    action_summary:
        Mean command frequency per division and per episode-quartile.
    explanation:
        Optional explanation (populated by :meth:`CorpsCOAGenerator.explain_coa`).
    seed:
        Base random seed for this COA's rollouts.
    """

    label: str
    rank: int
    score: CorpsCOAScore
    action_summary: dict
    seed: int
    explanation: Optional[COAExplanation] = None

    def as_dict(self) -> dict:
        d: dict = {
            "label": self.label,
            "rank": self.rank,
            "score": self.score.as_dict(),
            "action_summary": self.action_summary,
            "seed": self.seed,
        }
        if self.explanation is not None:
            d["explanation"] = self.explanation.as_dict()
        return d


@dataclasses.dataclass
class COAModification:
    """User modifications to a COA for re-simulation.

    Attributes
    ----------
    strategy_override:
        Replace the COA's strategy archetype with this label.  If ``None``
        the original label is preserved.
    n_rollouts:
        Number of rollouts to use for re-evaluation.  Defaults to the
        generator's ``n_rollouts`` if ``None``.
    division_command_overrides:
        Per-division command-index overrides applied at every step.
        Dict mapping division index (0-based) → command index.  Divisions
        not listed use the normal strategy bias.
    """

    strategy_override: Optional[str] = None
    n_rollouts: Optional[int] = None
    division_command_overrides: Optional[Dict[int, int]] = None


# ---------------------------------------------------------------------------
# Internal helpers (corps)
# ---------------------------------------------------------------------------

#: Maps integer command indices back to human-readable names.
_CMD_NAMES: Dict[int, str] = {
    _CMD_ADVANCE:  "advance_sector",
    _CMD_DEFEND:   "defend_position",
    _CMD_FLANK_L:  "flank_left",
    _CMD_FLANK_R:  "flank_right",
    _CMD_WITHDRAW: "withdraw",
    _CMD_FIRE:     "concentrate_fire",
}


class _CorpsStrategyPolicy:
    """Samples MultiDiscrete corps actions biased toward a strategy archetype.

    Parameters
    ----------
    n_divisions:
        Number of Blue divisions (determines action shape).
    n_corps_options:
        Number of valid command indices per division.
    strategy:
        Name of the corps strategy archetype.
    rng:
        NumPy random generator.
    bias_strength:
        Probability of choosing the preferred command per division.
    division_command_overrides:
        Hard overrides: mapping division_idx → command_idx.  Overridden
        divisions always issue their fixed command.
    """

    def __init__(
        self,
        n_divisions: int,
        n_corps_options: int,
        strategy: str,
        rng: np.random.Generator,
        bias_strength: float = _CORPS_BIAS_STRENGTH,
        division_command_overrides: Optional[Dict[int, int]] = None,
    ) -> None:
        if strategy not in _CORPS_STRATEGY_PATTERNS:
            raise ValueError(
                f"Unknown corps strategy '{strategy}'.  "
                f"Valid: {sorted(_CORPS_STRATEGY_PATTERNS)}"
            )
        self._n_divisions = n_divisions
        self._n_options = n_corps_options
        self._rng = rng
        self._bias_strength = bias_strength
        self._overrides: Dict[int, int] = division_command_overrides or {}

        # Build preferred command per division based on the pattern.
        pattern = _CORPS_STRATEGY_PATTERNS[strategy]
        self._preferred: List[int] = [
            pattern[i % len(pattern)] % n_corps_options
            for i in range(n_divisions)
        ]

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Any]:
        action = np.zeros(self._n_divisions, dtype=np.int64)
        for i in range(self._n_divisions):
            if i in self._overrides:
                cmd = int(self._overrides[i]) % self._n_options
            elif self._rng.random() < self._bias_strength:
                cmd = self._preferred[i]
            else:
                cmd = int(self._rng.integers(0, self._n_options))
            action[i] = cmd
        return action, None


def _run_corps_rollout(
    env: Any,
    policy: _CorpsStrategyPolicy,
    seed: int,
    deterministic: bool = False,
) -> dict:
    """Run one CorpsEnv episode and return per-step statistics.

    Parameters
    ----------
    env:
        A :class:`~envs.corps_env.CorpsEnv` instance.
    policy:
        Corps strategy policy to use for action selection.
    seed:
        Random seed for :py:meth:`env.reset`.
    deterministic:
        Passed through to ``policy.predict`` (reserved for learned policies).

    Returns
    -------
    dict with keys:
        ``outcome``              – 1 (Blue wins), -1 (Red wins), 0 (draw/truncated)
        ``blue_units_start``     – Blue unit count at episode start
        ``red_units_start``      – Red unit count at episode start
        ``blue_units_end``       – Blue units alive at episode end
        ``red_units_end``        – Red units alive at episode end
        ``steps``                – episode length in macro-steps
        ``actions``              – (steps, n_divisions) int64 array
        ``objective_rewards``    – (steps,) per-step total objective reward
        ``supply_levels``        – (steps, n_divisions) per-step supply levels
        ``total_reward``         – cumulative episode reward
    """
    obs, info = env.reset(seed=seed)
    done = False

    # Capture initial unit counts for casualty computation.
    # Prefer explicit per-side unit counts (e.g., env.n_blue / env.n_red for CorpsEnv),
    # falling back to n_divisions and finally 1 if nothing else is available.
    initial_blue = int(
        info.get(
            "blue_units_alive",
            getattr(env, "n_blue", getattr(env, "n_divisions", 1)),
        )
    )
    initial_red = int(
        info.get(
            "red_units_alive",
            getattr(env, "n_red", getattr(env, "n_divisions", 1)),
        )
    )

    actions_list: List[np.ndarray] = []
    obj_rewards_list: List[float] = []
    supply_list: List[List[float]] = []
    total_reward = 0.0
    step = 0

    blue_end = initial_blue
    red_end = initial_red

    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        total_reward += float(reward)

        actions_list.append(action.copy())

        # Aggregate per-step objective reward.
        obj_detail = info.get("objective_rewards", {})
        step_obj_reward = sum(float(v) for v in obj_detail.values())
        obj_rewards_list.append(step_obj_reward)

        # Track supply levels.
        sl = info.get("supply_levels", [])
        supply_list.append(list(sl) if sl else [])

        blue_end = int(info.get("blue_units_alive", blue_end))
        red_end  = int(info.get("red_units_alive",  red_end))

    # Determine outcome from final unit counts.
    if red_end == 0 and blue_end > 0:
        outcome = 1
    elif blue_end == 0 and red_end > 0:
        outcome = -1
    else:
        outcome = 0

    action_arr = (
        np.stack(actions_list, axis=0)
        if actions_list
        else np.zeros((0, policy._n_divisions), dtype=np.int64)
    )
    obj_arr = np.array(obj_rewards_list, dtype=np.float32)
    # Normalise casualties to [0, 1].
    start_blue = max(initial_blue, 1)
    start_red  = max(initial_red, 1)

    # Average supply efficiency = mean over all steps and divisions.
    if supply_list and any(sl for sl in supply_list):
        flat_supply = [v for sl in supply_list for v in sl if sl]
        mean_supply = float(np.mean(flat_supply)) if flat_supply else 0.0
    else:
        mean_supply = 0.0

    return {
        "outcome": outcome,
        "blue_units_start": initial_blue,
        "red_units_start": initial_red,
        "blue_units_end": blue_end,
        "red_units_end": red_end,
        "steps": step,
        "actions": action_arr,
        "objective_rewards": obj_arr,
        "supply_levels": supply_list,
        "mean_supply": mean_supply,
        "total_reward": total_reward,
        "blue_frac_lost": 1.0 - blue_end / start_blue,
        "red_frac_lost": 1.0 - red_end / start_red,
    }


def _aggregate_corps_rollouts(
    results: List[dict],
    n_divisions: int,
    n_options: int,
) -> Tuple[CorpsCOAScore, dict]:
    """Compute a :class:`CorpsCOAScore` and action summary from rollouts.

    Returns
    -------
    (CorpsCOAScore, action_summary_dict)
    """
    n = len(results)
    wins   = sum(1 for r in results if r["outcome"] == 1)
    draws  = sum(1 for r in results if r["outcome"] == 0)
    losses = sum(1 for r in results if r["outcome"] == -1)
    win_rate   = wins   / n
    draw_rate  = draws  / n
    loss_rate  = losses / n

    blue_casualties = float(np.mean([r["blue_frac_lost"] for r in results]))
    red_casualties  = float(np.mean([r["red_frac_lost"]  for r in results]))
    obj_completion  = float(np.mean([
        float(np.sum(r["objective_rewards"])) / max(r["steps"], 1)
        for r in results
    ]))
    # Normalise obj_completion to [0, 1] via sigmoid-like clamping.
    obj_completion_norm = float(np.clip((obj_completion + 1.0) / 2.0, 0.0, 1.0))
    supply_eff = float(np.mean([r["mean_supply"] for r in results]))

    casualty_eff = (red_casualties - blue_casualties + 1.0) / 2.0
    composite = (
        _CORPS_WIN_RATE_WEIGHT     * win_rate
        + _CORPS_CASUALTY_EFF_WEIGHT * casualty_eff
        + _CORPS_OBJECTIVE_WEIGHT    * obj_completion_norm
        + _CORPS_SUPPLY_WEIGHT       * supply_eff
    )

    score = CorpsCOAScore(
        win_rate=round(win_rate, 4),
        draw_rate=round(draw_rate, 4),
        loss_rate=round(loss_rate, 4),
        blue_casualties=round(blue_casualties, 4),
        red_casualties=round(red_casualties, 4),
        objective_completion=round(obj_completion_norm, 4),
        supply_efficiency=round(supply_eff, 4),
        composite=round(composite, 4),
        n_rollouts=n,
    )

    # Action summary: command frequency per division per episode-quartile.
    cmd_names = [_CMD_NAMES.get(c, str(c)) for c in range(n_options)]
    action_summary: dict = {}
    for div_i in range(n_divisions):
        div_key = f"div_{div_i}"
        action_summary[div_key] = {}
        for qi in range(1, 5):
            qkey = f"q{qi}"
            freqs = {name: [] for name in cmd_names}
            for r in results:
                arr = r["actions"]
                T = len(arr)
                if T == 0:
                    continue
                q = max(T // 4, 1)
                slices = [arr[:q], arr[q: 2*q], arr[2*q: 3*q], arr[3*q:]]
                sl = slices[qi - 1]
                if len(sl) == 0:
                    sl = arr
                div_cmds = sl[:, div_i]
                for c_idx, cname in enumerate(cmd_names):
                    freqs[cname].append(float(np.mean(div_cmds == c_idx)))
            action_summary[div_key][qkey] = {
                k: round(float(np.mean(v)), 4) if v else 0.0
                for k, v in freqs.items()
            }

    return score, action_summary


# ---------------------------------------------------------------------------
# Main class (corps)
# ---------------------------------------------------------------------------


class CorpsCOAGenerator:
    """Generate and rank corps-level Courses of Action via Monte-Carlo rollout.

    Satisfies the E9.2 requirements:
    * Up to 10 COAs generated via :meth:`generate`.
    * COA explanation via :meth:`explain_coa` (≥ 3 key decisions per COA).
    * COA modification and re-evaluation via :meth:`modify_and_evaluate`.

    Parameters
    ----------
    env:
        A :class:`~envs.corps_env.CorpsEnv` instance.  The caller retains
        ownership and is responsible for closing it.
    n_rollouts:
        Number of Monte-Carlo rollouts per COA (default 10).
    n_coas:
        Number of distinct COAs to generate (1–10, default 10).
    seed:
        Base random seed for reproducibility.
    strategies:
        Explicit list of strategy labels to evaluate.  When ``None`` the
        first ``n_coas`` labels from :data:`CORPS_STRATEGY_LABELS` are used.
    """

    def __init__(
        self,
        env: Any,
        n_rollouts: int = 10,
        n_coas: int = 10,
        seed: Optional[int] = None,
        strategies: Optional[Sequence[str]] = None,
    ) -> None:
        if n_rollouts < 1:
            raise ValueError(f"n_rollouts must be >= 1, got {n_rollouts}")
        if n_coas < 1:
            raise ValueError(f"n_coas must be >= 1, got {n_coas}")
        if n_coas > len(CORPS_STRATEGY_LABELS):
            raise ValueError(
                f"n_coas ({n_coas}) exceeds the number of built-in corps strategy "
                f"archetypes ({len(CORPS_STRATEGY_LABELS)}).  Pass a custom "
                f"'strategies' list or reduce n_coas."
            )
        self.env = env
        self.n_rollouts = int(n_rollouts)
        self.n_coas = int(n_coas)
        self.seed = seed

        if strategies is not None:
            invalid = [s for s in strategies if s not in _CORPS_STRATEGY_PATTERNS]
            if invalid:
                raise ValueError(
                    f"Unknown corps strategy labels: {invalid}.  "
                    f"Valid: {sorted(_CORPS_STRATEGY_PATTERNS)}"
                )
            unique = list(dict.fromkeys(strategies))
            if len(unique) < self.n_coas:
                raise ValueError(
                    f"Insufficient distinct strategy labels: expected >= {self.n_coas}, "
                    f"got {len(unique)}"
                )
            self._strategies: List[str] = unique[: self.n_coas]
        else:
            self._strategies = list(CORPS_STRATEGY_LABELS[: self.n_coas])

        # Cache env properties needed for policy construction.
        self._n_divisions: int = int(getattr(env, "n_divisions", 3))
        self._n_corps_options: int = int(getattr(env, "n_corps_options", 6))

        # Store raw rollout results per strategy for explanation purposes.
        self._last_rollout_results: Dict[str, List[dict]] = {}

    def generate(
        self,
        policy: Optional[Any] = None,
        deterministic: bool = False,
    ) -> List[CorpsCourseOfAction]:
        """Generate and rank candidate corps-level COAs.

        Parameters
        ----------
        policy:
            Optional trained policy with
            ``predict(obs, deterministic) -> (action, state)``.  When
            ``None`` a strategy-biased random policy is used.
        deterministic:
            Passed through to the policy's ``predict`` call (only relevant
            for learned base policies).

        Returns
        -------
        list of :class:`CorpsCourseOfAction`
            Ordered best to worst by composite score.
        """
        coa_list: List[CorpsCourseOfAction] = []
        self._last_rollout_results = {}

        for coa_idx, strategy in enumerate(self._strategies):
            coa_seed = (
                (self.seed * 1000 + coa_idx * 37) if self.seed is not None
                else coa_idx * 37
            )
            rng = np.random.default_rng(coa_seed)
            corps_policy = _CorpsStrategyPolicy(
                n_divisions=self._n_divisions,
                n_corps_options=self._n_corps_options,
                strategy=strategy,
                rng=rng,
            )

            # If a trained base policy is supplied, wrap it with override
            # capability rather than ignoring it.
            if policy is not None:
                corps_policy = _WrappedCorpsPolicy(
                    base_policy=policy,
                    strategy_policy=corps_policy,
                    n_divisions=self._n_divisions,
                    n_corps_options=self._n_corps_options,
                    bias_strength=_CORPS_BIAS_STRENGTH,
                    rng=rng,
                )

            rollout_results: List[dict] = []
            for rollout_i in range(self.n_rollouts):
                ep_seed = coa_seed + rollout_i
                result = _run_corps_rollout(
                    self.env, corps_policy, seed=ep_seed,
                    deterministic=deterministic,
                )
                rollout_results.append(result)

            self._last_rollout_results[strategy] = rollout_results

            score, action_summary = _aggregate_corps_rollouts(
                rollout_results, self._n_divisions, self._n_corps_options
            )
            coa_list.append(
                CorpsCourseOfAction(
                    label=strategy,
                    rank=0,
                    score=score,
                    action_summary=action_summary,
                    seed=coa_seed,
                )
            )

        coa_list.sort(key=lambda c: c.score.composite, reverse=True)
        for rank, coa in enumerate(coa_list, start=1):
            coa.rank = rank

        return coa_list

    def explain_coa(self, coa: CorpsCourseOfAction) -> COAExplanation:
        """Explain the key decisions that drive a COA's outcome.

        Analyses the stored rollout results for ``coa.label`` and returns
        a :class:`COAExplanation` with ≥ 3 key decisions.

        Parameters
        ----------
        coa:
            A :class:`CorpsCourseOfAction` previously returned by
            :meth:`generate`.

        Returns
        -------
        :class:`COAExplanation`
        """
        results = self._last_rollout_results.get(coa.label, [])
        if not results:
            # Re-run rollouts if needed (e.g. after pickle/restore).
            return self._explain_from_scratch(coa)

        wins  = [r for r in results if r["outcome"] == 1]
        losses = [r for r in results if r["outcome"] == -1]
        all_r  = results

        cmd_names = [_CMD_NAMES.get(c, str(c)) for c in range(self._n_corps_options)]

        # ── Command frequency (overall) ───────────────────────────────────
        cmd_freq: Dict[str, float] = {name: 0.0 for name in cmd_names}
        total_cmds = 0
        for r in all_r:
            arr = r["actions"]
            if len(arr) == 0:
                continue
            for c_idx, cname in enumerate(cmd_names):
                cmd_freq[cname] += float(np.sum(arr == c_idx))
            total_cmds += arr.size
        if total_cmds > 0:
            cmd_freq = {k: round(v / total_cmds, 4) for k, v in cmd_freq.items()}

        # ── Objective reward timeline per quartile ─────────────────────────
        obj_timeline: Dict[str, float] = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
        for qi, qkey in enumerate(["q1", "q2", "q3", "q4"], start=1):
            vals: List[float] = []
            for r in all_r:
                arr = r["objective_rewards"]
                T = len(arr)
                if T == 0:
                    continue
                q = max(T // 4, 1)
                slices = [arr[:q], arr[q: 2*q], arr[2*q: 3*q], arr[3*q:]]
                sl = slices[qi - 1]
                if len(sl) == 0:
                    sl = arr
                vals.append(float(np.sum(sl)))
            obj_timeline[qkey] = round(float(np.mean(vals)), 4) if vals else 0.0

        # ── Winning command patterns ───────────────────────────────────────
        # For each winning rollout, extract the most frequent command per div.
        def _dominant_commands(r: dict) -> List[str]:
            arr = r["actions"]
            if len(arr) == 0:
                return []
            cmds: List[str] = []
            for div_i in range(self._n_divisions):
                col = arr[:, div_i]
                dom = int(np.bincount(col, minlength=self._n_corps_options).argmax())
                cmds.append(_CMD_NAMES.get(dom, str(dom)))
            return cmds

        pattern_counts: Dict[str, int] = {}
        for r in wins:
            pat = tuple(_dominant_commands(r))
            key = str(list(pat))
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

        # Sort by frequency; return top-3 as lists.
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        winning_patterns: List[List[str]] = []
        for key, _cnt in sorted_patterns[:3]:
            import ast
            try:
                winning_patterns.append(ast.literal_eval(key))
            except (ValueError, SyntaxError):
                winning_patterns.append([key])

        # ── Key decisions ─────────────────────────────────────────────────
        key_decisions: List[str] = []

        # Decision 1: dominant strategy command and its impact on win rate.
        dominant_cmd = max(cmd_freq, key=lambda k: cmd_freq[k])
        key_decisions.append(
            f"Issuing '{dominant_cmd}' most frequently ({cmd_freq[dominant_cmd]*100:.1f}% "
            f"of all orders) is the defining action of this COA."
        )

        # Decision 2: phase where objectives are maximally gained.
        best_phase = max(obj_timeline, key=lambda k: obj_timeline[k])
        key_decisions.append(
            f"Highest objective gain occurs in {best_phase} (score: "
            f"{obj_timeline[best_phase]:.3f}), suggesting the critical push "
            f"happens in the {'first' if best_phase == 'q1' else 'second' if best_phase == 'q2' else 'third' if best_phase == 'q3' else 'final'} quarter of the episode."
        )

        # Decision 3: casualty trade-off.
        blue_cas = coa.score.blue_casualties
        red_cas  = coa.score.red_casualties
        trade_off = "favourable" if red_cas > blue_cas else "costly"
        key_decisions.append(
            f"Casualty trade-off is {trade_off}: Blue loses "
            f"{blue_cas*100:.1f}% vs Red loses {red_cas*100:.1f}% of units."
        )

        # Decision 4: supply impact.
        key_decisions.append(
            f"Mean Blue supply efficiency is {coa.score.supply_efficiency*100:.1f}%; "
            f"{'supply is well maintained — sustaining this COA is feasible.' if coa.score.supply_efficiency >= 0.5 else 'supply is degraded — this COA strains logistics.'}"
        )

        # Decision 5: win-rate context.
        win_pct = coa.score.win_rate * 100
        key_decisions.append(
            f"This COA wins {win_pct:.1f}% of rollouts "
            f"({'high' if win_pct >= 60 else 'moderate' if win_pct >= 40 else 'low'} reliability)."
        )

        return COAExplanation(
            coa_label=coa.label,
            key_decisions=key_decisions,
            command_frequency=cmd_freq,
            winning_patterns=winning_patterns,
            objective_timeline=obj_timeline,
        )

    def _explain_from_scratch(self, coa: CorpsCourseOfAction) -> COAExplanation:
        """Fallback explanation when rollout results are unavailable."""
        return COAExplanation(
            coa_label=coa.label,
            key_decisions=[
                f"Strategy '{coa.label}' achieves composite score {coa.score.composite:.4f}.",
                f"Win rate: {coa.score.win_rate*100:.1f}%, "
                f"casualty efficiency: {(coa.score.red_casualties - coa.score.blue_casualties + 1)/2*100:.1f}%.",
                f"Objective completion: {coa.score.objective_completion*100:.1f}%, "
                f"supply efficiency: {coa.score.supply_efficiency*100:.1f}%.",
            ],
            command_frequency={},
            winning_patterns=[],
            objective_timeline={"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0},
        )

    def modify_and_evaluate(
        self,
        coa: CorpsCourseOfAction,
        modification: COAModification,
    ) -> CorpsCourseOfAction:
        """Apply user modifications to a COA and re-simulate it.

        Parameters
        ----------
        coa:
            The original :class:`CorpsCourseOfAction` to modify.
        modification:
            A :class:`COAModification` describing the changes.

        Returns
        -------
        A new :class:`CorpsCourseOfAction` with updated score and the
        modified strategy label (or original label if no override).
        """
        strategy = modification.strategy_override or coa.label
        if strategy not in _CORPS_STRATEGY_PATTERNS:
            raise ValueError(
                f"Unknown strategy override '{strategy}'.  "
                f"Valid: {sorted(_CORPS_STRATEGY_PATTERNS)}"
            )
        raw_n_rollouts = modification.n_rollouts
        if raw_n_rollouts is None:
            n_rollouts = self.n_rollouts
        else:
            if raw_n_rollouts < 1:
                raise ValueError(
                    f"n_rollouts must be at least 1, got {raw_n_rollouts!r}"
                )
            n_rollouts = raw_n_rollouts
        overrides  = modification.division_command_overrides or {}

        rng = np.random.default_rng(coa.seed + 999)  # distinct seed from original
        corps_policy = _CorpsStrategyPolicy(
            n_divisions=self._n_divisions,
            n_corps_options=self._n_corps_options,
            strategy=strategy,
            rng=rng,
            division_command_overrides=overrides,
        )

        rollout_results: List[dict] = []
        for rollout_i in range(n_rollouts):
            ep_seed = coa.seed + 999 + rollout_i
            result = _run_corps_rollout(
                self.env, corps_policy, seed=ep_seed, deterministic=False
            )
            rollout_results.append(result)

        # Store for explain_coa use.
        self._last_rollout_results[strategy] = rollout_results

        score, action_summary = _aggregate_corps_rollouts(
            rollout_results, self._n_divisions, self._n_corps_options
        )
        return CorpsCourseOfAction(
            label=strategy,
            rank=0,  # caller should re-rank if needed
            score=score,
            action_summary=action_summary,
            seed=coa.seed + 999,
        )


class _WrappedCorpsPolicy:
    """Blends a trained base policy with a strategy bias.

    At each step the base policy's action is used with probability
    ``(1 - bias_strength)``; otherwise the strategy-biased policy is used.
    """

    def __init__(
        self,
        base_policy: Any,
        strategy_policy: _CorpsStrategyPolicy,
        n_divisions: int,
        n_corps_options: int,
        bias_strength: float,
        rng: np.random.Generator,
    ) -> None:
        self._base = base_policy
        self._strat = strategy_policy
        self._n_div = n_divisions
        self._n_opt = n_corps_options
        self._bias = bias_strength
        self._rng = rng
        # Expose for _run_corps_rollout compatibility.
        self._n_divisions = n_divisions

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Any]:
        base_action, state = self._base.predict(obs, deterministic=deterministic)
        base_action = np.asarray(base_action, dtype=np.int64)
        strat_action, _ = self._strat.predict(obs, deterministic=deterministic)
        # Per-division blend.
        result = np.where(
            self._rng.random(self._n_div) < self._bias,
            strat_action,
            np.clip(base_action, 0, self._n_opt - 1),
        ).astype(np.int64)
        return result, state


# ---------------------------------------------------------------------------
# Convenience function (corps)
# ---------------------------------------------------------------------------


def generate_corps_coas(
    env: Optional[Any] = None,
    policy: Optional[Any] = None,
    n_rollouts: int = 10,
    n_coas: int = 10,
    seed: Optional[int] = None,
    strategies: Optional[Sequence[str]] = None,
    env_kwargs: Optional[dict] = None,
    explain: bool = False,
) -> List[CorpsCourseOfAction]:
    """Generate corps-level COAs, optionally creating a temporary CorpsEnv.

    Parameters
    ----------
    env:
        An existing :class:`~envs.corps_env.CorpsEnv`.  When ``None`` a new
        environment is created using *env_kwargs* and closed automatically.
    policy:
        Optional trained policy.
    n_rollouts:
        Monte-Carlo rollouts per COA (default 10).  10 COAs × 10 rollouts
        runs comfortably within the 120 s budget on CPU.
    n_coas:
        Number of COAs to generate (1–10, default 10).
    seed:
        Base random seed.
    strategies:
        Explicit ordered list of corps strategy labels to evaluate.
    env_kwargs:
        Keyword arguments forwarded to :class:`~envs.corps_env.CorpsEnv`
        when *env* is ``None``.
    explain:
        If ``True``, populate the ``explanation`` field of each
        :class:`CorpsCourseOfAction` via :meth:`CorpsCOAGenerator.explain_coa`.

    Returns
    -------
    list of :class:`CorpsCourseOfAction`, best first.
    """
    from envs.corps_env import CorpsEnv

    owns_env = env is None
    active_env: Any = (
        env
        if env is not None
        else CorpsEnv(**(env_kwargs or {}))
    )
    try:
        generator = CorpsCOAGenerator(
            env=active_env,
            n_rollouts=n_rollouts,
            n_coas=n_coas,
            seed=seed,
            strategies=strategies,
        )
        coa_list = generator.generate(policy=policy)
        if explain:
            for coa in coa_list:
                coa.explanation = generator.explain_coa(coa)
        return coa_list
    finally:
        if owns_env:
            active_env.close()
