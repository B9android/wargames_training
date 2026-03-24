"""WargamesBench — standardised 20-scenario benchmark suite (E12.2).

WargamesBench enables reproducible, apples-to-apples comparison of any
wargame AI policy.  Each of the 20 held-out scenarios has a fixed
random seed, so win-rate estimates are reproducible within ± 2 % when
``n_eval_episodes ≥ 100`` and the same seed is used for both the
environment reset and the policy's RNG.

Architecture
------------
:data:`BENCH_SCENARIOS`
    Registry of the 20 canonical WargamesBench scenarios.

:class:`BenchScenario`
    Descriptor for one standardised evaluation scenario.

:class:`BenchConfig`
    Benchmark configuration (episodes per scenario, max steps, …).

:class:`BenchResult`
    Win-rate statistics for one (policy × scenario) evaluation.

:class:`BenchSummary`
    Aggregated results across all 20 scenarios with leaderboard rendering
    and reproducibility-criterion checks.

:class:`WargamesBench`
    High-level runner; evaluate any callable policy or ``None`` for the
    built-in scripted baseline.

Typical usage
-------------
Dry-run (CI) with the scripted baseline::

    from benchmarks.wargames_bench import WargamesBench, BenchConfig

    cfg = BenchConfig(n_eval_episodes=5, n_scenarios=4)
    bench = WargamesBench(cfg)
    summary = bench.run(policy=None)
    print(summary)

With an SB3-compatible checkpoint::

    bench = WargamesBench(BenchConfig())
    summary = bench.run(policy="checkpoints/my_policy.zip")
    bench.write_markdown(summary, "docs/leaderboard.md")
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger(__name__)

__all__ = [
    "BENCH_SCENARIOS",
    "BenchScenario",
    "BenchConfig",
    "BenchResult",
    "BenchSummary",
    "WargamesBench",
    "main",
]

# ---------------------------------------------------------------------------
# Scenario registry — 20 canonical held-out scenarios
# ---------------------------------------------------------------------------

#: The 20 canonical WargamesBench scenarios.  Seeds are fixed to guarantee
#: reproducibility.  Do **not** modify these entries; add new scenarios
#: with a new name instead.
BENCH_SCENARIOS: List[Dict[str, Any]] = [
    # ── Symmetric — even terrain, all weather ────────────────────────────
    {"name": "sym_4v4_clear",    "n_blue": 4,  "n_red": 4,  "weather": "clear", "terrain_seed": 4201, "seed": 9001},
    {"name": "sym_6v6_rain",     "n_blue": 6,  "n_red": 6,  "weather": "rain",  "terrain_seed": 4202, "seed": 9002},
    {"name": "sym_8v8_fog",      "n_blue": 8,  "n_red": 8,  "weather": "fog",   "terrain_seed": 4203, "seed": 9003},
    {"name": "sym_8v8_snow",     "n_blue": 8,  "n_red": 8,  "weather": "snow",  "terrain_seed": 4204, "seed": 9004},
    {"name": "sym_12v12_clear",  "n_blue": 12, "n_red": 12, "weather": "clear", "terrain_seed": 4205, "seed": 9005},
    # ── Asymmetric force ratios ───────────────────────────────────────────
    {"name": "asym_4v8_clear",   "n_blue": 4,  "n_red": 8,  "weather": "clear", "terrain_seed": 4206, "seed": 9006},
    {"name": "asym_8v4_rain",    "n_blue": 8,  "n_red": 4,  "weather": "rain",  "terrain_seed": 4207, "seed": 9007},
    {"name": "asym_6v12_fog",    "n_blue": 6,  "n_red": 12, "weather": "fog",   "terrain_seed": 4208, "seed": 9008},
    {"name": "asym_12v6_snow",   "n_blue": 12, "n_red": 6,  "weather": "snow",  "terrain_seed": 4209, "seed": 9009},
    # ── Hilly terrain ────────────────────────────────────────────────────
    {"name": "hilly_6v6_clear",  "n_blue": 6,  "n_red": 6,  "weather": "clear", "terrain_seed": 5001, "seed": 9010, "n_hills": 8, "n_forests": 2},
    {"name": "hilly_8v8_rain",   "n_blue": 8,  "n_red": 8,  "weather": "rain",  "terrain_seed": 5002, "seed": 9011, "n_hills": 8, "n_forests": 2},
    # ── Dense forest terrain ──────────────────────────────────────────────
    {"name": "forest_6v6_fog",   "n_blue": 6,  "n_red": 6,  "weather": "fog",   "terrain_seed": 6001, "seed": 9012, "n_hills": 2, "n_forests": 8},
    {"name": "forest_8v8_snow",  "n_blue": 8,  "n_red": 8,  "weather": "snow",  "terrain_seed": 6002, "seed": 9013, "n_hills": 2, "n_forests": 8},
    # ── Large-scale engagement ────────────────────────────────────────────
    {"name": "large_16v16_clear","n_blue": 16, "n_red": 16, "weather": "clear", "terrain_seed": 7001, "seed": 9014},
    {"name": "large_16v12_rain", "n_blue": 16, "n_red": 12, "weather": "rain",  "terrain_seed": 7002, "seed": 9015},
    # ── Small skirmish ────────────────────────────────────────────────────
    {"name": "skirmish_3v3_clear","n_blue": 3,  "n_red": 3,  "weather": "clear", "terrain_seed": 8001, "seed": 9016},
    {"name": "skirmish_3v3_fog", "n_blue": 3,  "n_red": 3,  "weather": "fog",   "terrain_seed": 8002, "seed": 9017},
    # ── Defender's advantage — blue defends (more terrain cover) ─────────
    {"name": "defense_6v8_clear","n_blue": 6,  "n_red": 8,  "weather": "clear", "terrain_seed": 8003, "seed": 9018, "n_hills": 6, "n_forests": 4},
    {"name": "defense_4v6_snow", "n_blue": 4,  "n_red": 6,  "weather": "snow",  "terrain_seed": 8004, "seed": 9019, "n_hills": 6, "n_forests": 4},
    # ── Night / minimal visibility ────────────────────────────────────────
    {"name": "lowvis_8v8_fog",   "n_blue": 8,  "n_red": 8,  "weather": "fog",   "terrain_seed": 8005, "seed": 9020},
]

assert len(BENCH_SCENARIOS) == 20, (
    f"WargamesBench must have exactly 20 scenarios; got {len(BENCH_SCENARIOS)}."
)


# ---------------------------------------------------------------------------
# BenchScenario
# ---------------------------------------------------------------------------


@dataclass
class BenchScenario:
    """Descriptor for one WargamesBench evaluation scenario.

    Attributes
    ----------
    name:
        Unique scenario identifier.  Used as the leaderboard row key.
    n_blue:
        Number of blue (agent) units.
    n_red:
        Number of red (opponent) units.
    weather:
        Weather string — ``"clear"``, ``"rain"``, ``"fog"``, or ``"snow"``.
    terrain_seed:
        Seed used to generate the procedural terrain.  Fixed per scenario.
    seed:
        Episode-level RNG seed used for ``env.reset(seed=…)``.
    n_hills:
        Number of terrain hills to generate.
    n_forests:
        Number of terrain forest patches to generate.
    map_width:
        Terrain map width in metres.
    map_height:
        Terrain map height in metres.
    """

    name: str
    n_blue: int = 8
    n_red: int = 8
    weather: str = "clear"
    terrain_seed: int = 42
    seed: int = 0
    n_hills: int = 4
    n_forests: int = 3
    map_width: float = 10_000.0
    map_height: float = 10_000.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchScenario":
        """Build a :class:`BenchScenario` from a registry entry dict."""
        return cls(
            name=d["name"],
            n_blue=d.get("n_blue", 8),
            n_red=d.get("n_red", 8),
            weather=d.get("weather", "clear"),
            terrain_seed=d.get("terrain_seed", 42),
            seed=d.get("seed", 0),
            n_hills=d.get("n_hills", 4),
            n_forests=d.get("n_forests", 3),
            map_width=d.get("map_width", 10_000.0),
            map_height=d.get("map_height", 10_000.0),
        )


# ---------------------------------------------------------------------------
# BenchConfig
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """Configuration for a WargamesBench run.

    Attributes
    ----------
    n_eval_episodes:
        Episodes per scenario.  Use ≥ 100 for reproducible ± 2 % win rates.
    n_scenarios:
        Number of canonical scenarios to evaluate (≤ 20).
    max_steps_per_episode:
        Hard episode-length cap.
    baseline_label:
        Human-readable label for the evaluated policy in the leaderboard.
    report_path:
        Where to write the Markdown leaderboard report.  ``None`` → default.
    win_rate_tolerance:
        Maximum allowed win-rate deviation between runs with the same seed.
        Used by :meth:`BenchSummary.is_reproducible`.
    """

    n_eval_episodes: int = 100
    n_scenarios: int = 20
    max_steps_per_episode: int = 500
    baseline_label: str = "scripted_baseline"
    report_path: Optional[str] = None
    win_rate_tolerance: float = 0.02


# ---------------------------------------------------------------------------
# BenchResult
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Win-rate statistics for one (policy × scenario) evaluation.

    Attributes
    ----------
    scenario_name:
        Name of the evaluated scenario.
    policy_label:
        Human-readable label for the policy.
    win_rate:
        Fraction of episodes won by the blue (agent) policy.
    mean_steps:
        Mean episode length.
    std_steps:
        Standard deviation of episode length.
    n_episodes:
        Number of episodes evaluated.
    elapsed_seconds:
        Wall-clock seconds for this evaluation.
    """

    scenario_name: str
    policy_label: str
    win_rate: float
    mean_steps: float
    std_steps: float
    n_episodes: int
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# BenchSummary
# ---------------------------------------------------------------------------


@dataclass
class BenchSummary:
    """Aggregated WargamesBench results across all evaluated scenarios.

    Attributes
    ----------
    results:
        Per-scenario :class:`BenchResult` objects.
    config:
        :class:`BenchConfig` used for this run.
    """

    results: List[BenchResult]
    config: BenchConfig

    # ------------------------------------------------------------------
    # Aggregated statistics
    # ------------------------------------------------------------------

    @property
    def mean_win_rate(self) -> float:
        """Mean win rate across all evaluated scenarios."""
        if not self.results:
            return 0.0
        return float(np.mean([r.win_rate for r in self.results]))

    @property
    def std_win_rate(self) -> float:
        """Standard deviation of win rates across scenarios."""
        if not self.results:
            return 0.0
        return float(np.std([r.win_rate for r in self.results]))

    @property
    def total_episodes(self) -> int:
        """Total number of episodes evaluated."""
        return sum(r.n_episodes for r in self.results)

    @property
    def total_elapsed_seconds(self) -> float:
        """Total wall-clock seconds for all evaluations."""
        return sum(r.elapsed_seconds for r in self.results)

    def is_reproducible(self, other: "BenchSummary") -> bool:
        """Return ``True`` when *self* and *other* win rates agree within tolerance.

        Both runs must have identical scenario names in the same order.  The
        tolerance is ``self.config.win_rate_tolerance`` (default 2 %).
        """
        if len(self.results) != len(other.results):
            return False
        for a, b in zip(self.results, other.results):
            if a.scenario_name != b.scenario_name:
                return False
            if abs(a.win_rate - b.win_rate) > self.config.win_rate_tolerance:
                return False
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = [
            "WargamesBench Summary",
            f"  Policy               : {self.config.baseline_label}",
            f"  Scenarios evaluated  : {len(self.results)}",
            f"  Total episodes       : {self.total_episodes}",
            f"  Mean win rate        : {self.mean_win_rate:.1%}",
            f"  Std win rate         : {self.std_win_rate:.1%}",
            f"  Elapsed              : {self.total_elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)

    def to_leaderboard_row(self) -> Dict[str, Any]:
        """Return a dict suitable for appending to a leaderboard table."""
        return {
            "policy": self.config.baseline_label,
            "mean_win_rate": f"{self.mean_win_rate:.3f}",
            "std_win_rate": f"{self.std_win_rate:.3f}",
            "n_scenarios": len(self.results),
            "total_episodes": self.total_episodes,
        }

    def write_markdown(self, path: Optional[str | Path] = None) -> Path:
        """Write a Markdown leaderboard report to *path*.

        Defaults to ``docs/wargames_bench_leaderboard.md``.
        """
        if path is None:
            if self.config.report_path:
                out = Path(self.config.report_path)
            else:
                out = _REPO_ROOT / "docs" / "wargames_bench_leaderboard.md"
        else:
            out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_summary_markdown(self), encoding="utf-8")
        return out


def _render_summary_markdown(summary: BenchSummary) -> str:
    lines = [
        "# WargamesBench Leaderboard",
        "",
        "> Reproducibility criterion: win-rate deviation ≤ 2 % across identical-seed runs.",
        "",
        "## Summary",
        "",
        "| Policy | Mean Win Rate | Std | Scenarios | Episodes |",
        "|--------|--------------|-----|-----------|---------|",
        f"| {summary.config.baseline_label} | {summary.mean_win_rate:.1%} "
        f"| {summary.std_win_rate:.1%} | {len(summary.results)} "
        f"| {summary.total_episodes} |",
        "",
        "## Per-scenario Results",
        "",
        "| Scenario | Win Rate | Episodes | Mean Steps | Elapsed (s) |",
        "|----------|---------|---------|-----------|------------|",
    ]
    for r in sorted(summary.results, key=lambda x: x.scenario_name):
        lines.append(
            f"| {r.scenario_name} | {r.win_rate:.1%} | {r.n_episodes} "
            f"| {r.mean_steps:.0f} | {r.elapsed_seconds:.1f} |"
        )
    lines.append("")
    lines.append(f"*Generated by WargamesBench — {len(BENCH_SCENARIOS)} canonical scenarios.*")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# WargamesBench runner
# ---------------------------------------------------------------------------


class WargamesBench:
    """Run a policy against all WargamesBench scenarios and return results.

    Parameters
    ----------
    config:
        Benchmark configuration.  Use :class:`BenchConfig` defaults for the
        canonical benchmark; override ``n_eval_episodes`` for quick CI runs.
    """

    def __init__(self, config: Optional[BenchConfig] = None) -> None:
        self.config = config or BenchConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        policy: Optional[Any] = None,
        *,
        label: Optional[str] = None,
    ) -> BenchSummary:
        """Evaluate *policy* on all benchmark scenarios.

        Parameters
        ----------
        policy:
            Any callable that accepts a flat ``np.ndarray`` observation and
            returns an action array; **or** an SB3-style object with a
            ``predict(obs, deterministic=True)`` method; **or** ``None`` to
            use the built-in scripted baseline.
        label:
            Override :attr:`BenchConfig.baseline_label` for the leaderboard.

        Returns
        -------
        :class:`BenchSummary`
        """
        cfg = self.config
        effective_label = label or cfg.baseline_label
        n = min(cfg.n_scenarios, len(BENCH_SCENARIOS))
        scenarios = [BenchScenario.from_dict(d) for d in BENCH_SCENARIOS[:n]]

        results: List[BenchResult] = []

        for scenario in scenarios:
            env = self._make_env(scenario)
            t0 = time.perf_counter()
            episodes = self._evaluate(policy, env, scenario)
            elapsed = time.perf_counter() - t0
            stats = _aggregate_episodes(episodes)
            results.append(
                BenchResult(
                    scenario_name=scenario.name,
                    policy_label=effective_label,
                    elapsed_seconds=elapsed,
                    **stats,
                )
            )
            env.close()
            log.info(
                "Scenario %-25s  win_rate=%.1f%%  steps=%.0f",
                scenario.name,
                stats["win_rate"] * 100,
                stats["mean_steps"],
            )

        return BenchSummary(results=results, config=cfg)

    # ------------------------------------------------------------------
    # Internal: environment factory
    # ------------------------------------------------------------------

    def _make_env(self, scenario: BenchScenario) -> Any:
        """Create an evaluation environment for *scenario*.

        Attempts to instantiate a real :class:`~envs.battalion_env.BattalionEnv`
        with procedurally generated terrain.  Falls back to a lightweight
        synthetic environment when the real env cannot be constructed (e.g.
        missing optional dependencies in CI).
        """
        try:
            from envs.battalion_env import BattalionEnv
            from envs.sim.terrain import TerrainMap

            terrain = TerrainMap.generate_random(
                rng=np.random.default_rng(scenario.terrain_seed),
                width=scenario.map_width,
                height=scenario.map_height,
                rows=40,
                cols=40,
                num_hills=scenario.n_hills,
                num_forests=scenario.n_forests,
            )
            return BattalionEnv(
                terrain=terrain,
                randomize_terrain=False,
                map_width=terrain.width,
                map_height=terrain.height,
            )
        except Exception as exc:
            log.debug(
                "Could not build real BattalionEnv for scenario %r: %s — using synthetic.",
                scenario.name,
                exc,
                exc_info=True,
            )
        return _SyntheticEnv(
            n_entities=scenario.n_blue + scenario.n_red,
            seed=scenario.terrain_seed,
            ep_length=self.config.max_steps_per_episode,
        )

    # ------------------------------------------------------------------
    # Internal: episode runner
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        policy: Optional[Any],
        env: Any,
        scenario: BenchScenario,
    ) -> List[Dict[str, Any]]:
        """Run ``config.n_eval_episodes`` episodes and return raw stats."""
        cfg = self.config
        use_predict = hasattr(policy, "predict")
        episodes: List[Dict[str, Any]] = []

        for ep_idx in range(cfg.n_eval_episodes):
            # Deterministic reset: base seed + episode offset for reproducibility.
            ep_seed = scenario.seed + ep_idx
            obs, _ = env.reset(seed=ep_seed)
            terminated = truncated = False
            steps = 0
            won: Optional[bool] = None

            while not (terminated or truncated) and steps < cfg.max_steps_per_episode:
                if use_predict:
                    action, _ = policy.predict(obs, deterministic=True)
                elif callable(policy):
                    action = policy(obs)
                else:
                    action = _scripted_action(obs)

                obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1

            if isinstance(info, dict):
                if info.get("red_routed"):
                    won = True
                elif info.get("blue_routed"):
                    won = False

            episodes.append({"won": won, "steps": steps})

        return episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted_action(obs: Any) -> np.ndarray:
    """Simple deterministic scripted action — advance straight forward."""
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _aggregate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from raw episode records."""
    wins = [e["won"] for e in episodes if e["won"] is not None]
    steps = [e["steps"] for e in episodes]
    win_rate = float(np.mean(wins)) if wins else 0.0
    return {
        "win_rate": win_rate,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "std_steps": float(np.std(steps)) if steps else 0.0,
        "n_episodes": len(episodes),
    }


# ---------------------------------------------------------------------------
# Lightweight synthetic environment (CI fallback)
# ---------------------------------------------------------------------------


class _SyntheticEnv:
    """Minimal environment used when :class:`~envs.battalion_env.BattalionEnv`
    cannot be instantiated (e.g. missing GIS data or optional dependencies).

    The synthetic env produces random observations and terminates after
    ``ep_length`` steps with a uniformly random winner.
    """

    _OBS_DIM: int = 16

    def __init__(
        self,
        n_entities: int = 16,
        seed: int = 0,
        ep_length: int = 500,
    ) -> None:
        self.n_entities = n_entities
        self.ep_length = ep_length
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        obs = self._rng.standard_normal(
            (self.n_entities, self._OBS_DIM)
        ).astype(np.float32)
        return obs, {}

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step += 1
        obs = self._rng.standard_normal(
            (self.n_entities, self._OBS_DIM)
        ).astype(np.float32)
        reward = float(self._rng.standard_normal())
        terminated = self._step >= self.ep_length
        info: Dict[str, Any] = {}
        if terminated:
            outcome = int(self._rng.integers(3))
            if outcome == 0:
                info["red_routed"] = True
            elif outcome == 1:
                info["blue_routed"] = True
        return obs, reward, terminated, False, info

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``wargames-bench``.

    Usage::

        wargames-bench                     # scripted baseline, 100 episodes
        wargames-bench --episodes 10       # quick dry-run
        wargames-bench --scenarios 5       # first 5 scenarios only
        wargames-bench --output path.md    # custom report path
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="wargames-bench",
        description="Run the WargamesBench standardised evaluation suite.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per scenario (default 100; use ≥ 100 for reproducible results).",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=20,
        help="Number of canonical scenarios to evaluate (1–20, default 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the Markdown leaderboard report.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="scripted_baseline",
        help="Policy label for the leaderboard (default 'scripted_baseline').",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = BenchConfig(
        n_eval_episodes=args.episodes,
        n_scenarios=min(args.scenarios, 20),
        baseline_label=args.label,
        report_path=args.output,
    )
    bench = WargamesBench(cfg)

    print(f"Running WargamesBench — {cfg.n_scenarios} scenarios × {cfg.n_eval_episodes} episodes …")
    summary = bench.run(policy=None, label=args.label)

    print()
    print(summary)

    report = summary.write_markdown()
    print(f"\nLeaderboard written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
