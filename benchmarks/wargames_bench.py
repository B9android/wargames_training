# SPDX-License-Identifier: MIT
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

With an SB3-compatible policy object::

    from stable_baselines3 import PPO
    from benchmarks.wargames_bench import WargamesBench, BenchConfig

    bench = WargamesBench(BenchConfig())
    model = PPO.load("checkpoints/my_policy.zip")
    summary = bench.run(policy=model, label="ppo_v1")
    summary.write_markdown("docs/leaderboard.md")
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

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
    force_synthetic_env: bool = False


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
    step_time_p50_ms: float = 0.0
    step_time_p95_ms: float = 0.0
    step_time_mean_ms: float = 0.0
    policy_time_p50_ms: float = 0.0
    policy_time_p95_ms: float = 0.0
    policy_time_mean_ms: float = 0.0


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

    @property
    def mean_step_time_p95_ms(self) -> float:
        """Mean scenario-level p95 of env.step latency in milliseconds."""
        if not self.results:
            return 0.0
        return float(np.mean([r.step_time_p95_ms for r in self.results]))

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
            f"  Mean step p95        : {self.mean_step_time_p95_ms:.3f} ms",
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

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable benchmark metrics for CI artifacts."""
        return {
            "policy": self.config.baseline_label,
            "n_scenarios": len(self.results),
            "total_episodes": self.total_episodes,
            "mean_win_rate": self.mean_win_rate,
            "std_win_rate": self.std_win_rate,
            "mean_step_time_p95_ms": self.mean_step_time_p95_ms,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "scenarios": [
                {
                    "scenario": r.scenario_name,
                    "n_episodes": r.n_episodes,
                    "win_rate": r.win_rate,
                    "mean_steps": r.mean_steps,
                    "elapsed_seconds": r.elapsed_seconds,
                    "step_time_ms": {
                        "p50": r.step_time_p50_ms,
                        "p95": r.step_time_p95_ms,
                        "mean": r.step_time_mean_ms,
                    },
                    "policy_time_ms": {
                        "p50": r.policy_time_p50_ms,
                        "p95": r.policy_time_p95_ms,
                        "mean": r.policy_time_mean_ms,
                    },
                }
                for r in sorted(self.results, key=lambda x: x.scenario_name)
            ],
        }

    def write_metrics_json(self, path: str | Path) -> Path:
        """Write benchmark latency metrics JSON for CI comparison."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_metrics_dict(), indent=2), encoding="utf-8")
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
        "| Scenario | Win Rate | Episodes | Mean Steps | Step p50 (ms) | Step p95 (ms) | Elapsed (s) |",
        "|----------|---------|---------|-----------|---------------|---------------|------------|",
    ]
    for r in sorted(summary.results, key=lambda x: x.scenario_name):
        lines.append(
            f"| {r.scenario_name} | {r.win_rate:.1%} | {r.n_episodes} "
            f"| {r.mean_steps:.0f} | {r.step_time_p50_ms:.3f} "
            f"| {r.step_time_p95_ms:.3f} | {r.elapsed_seconds:.1f} |"
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
        # Build a config copy that carries the effective label so all summary
        # rendering paths (str, write_markdown, to_leaderboard_row) are consistent.
        summary_cfg = dataclasses.replace(cfg, baseline_label=effective_label)
        n = min(cfg.n_scenarios, len(BENCH_SCENARIOS))
        scenarios = [BenchScenario.from_dict(d) for d in BENCH_SCENARIOS[:n]]

        results: List[BenchResult] = []

        for scenario in scenarios:
            env = self._make_env(scenario)
            t0 = time.perf_counter()
            episodes, step_times_ms, policy_times_ms = self._evaluate(policy, env, scenario)
            elapsed = time.perf_counter() - t0
            stats = _aggregate_episodes(episodes)
            step_stats = _aggregate_latencies_ms(step_times_ms)
            policy_stats = _aggregate_latencies_ms(policy_times_ms)
            results.append(
                BenchResult(
                    scenario_name=scenario.name,
                    policy_label=effective_label,
                    elapsed_seconds=elapsed,
                    step_time_p50_ms=step_stats["p50"],
                    step_time_p95_ms=step_stats["p95"],
                    step_time_mean_ms=step_stats["mean"],
                    policy_time_p50_ms=policy_stats["p50"],
                    policy_time_p95_ms=policy_stats["p95"],
                    policy_time_mean_ms=policy_stats["mean"],
                    **stats,
                )
            )
            env.close()
            log.info(
                "Scenario %-25s  win_rate=%.1f%%  steps=%.0f  step_p95=%.3fms",
                scenario.name,
                stats["win_rate"] * 100,
                stats["mean_steps"],
                step_stats["p95"],
            )

        return BenchSummary(results=results, config=summary_cfg)

    # ------------------------------------------------------------------
    # Internal: environment factory
    # ------------------------------------------------------------------

    def _make_env(self, scenario: BenchScenario) -> Any:
        """Create an evaluation environment for *scenario*.

        Attempts to instantiate a real :class:`~envs.battalion_env.BattalionEnv`
        with procedurally generated terrain and fixed weather (when available).
        Falls back to :class:`_SyntheticEnv` when the real env cannot be
        constructed (e.g. missing optional dependencies in CI).

        .. note::
            :class:`~envs.battalion_env.BattalionEnv` is a 1v1 environment.
            The ``n_blue`` / ``n_red`` fields in a :class:`BenchScenario` have
            no effect on the real env; they only influence the entity-count of
            the :class:`_SyntheticEnv` fallback.  Weather *is* applied to the
            real env via :class:`~envs.sim.weather.WeatherConfig`.
        """
        if self.config.force_synthetic_env:
            return _SyntheticEnv(
                n_entities=scenario.n_blue + scenario.n_red,
                seed=scenario.terrain_seed,
                ep_length=self.config.max_steps_per_episode,
            )

        try:
            from envs.battalion_env import BattalionEnv
            from envs.sim.terrain import TerrainMap
            from envs.sim.weather import WeatherCondition, WeatherConfig

            _WEATHER_MAP: Dict[str, WeatherCondition] = {
                "clear": WeatherCondition.CLEAR,
                "rain":  WeatherCondition.RAIN,
                "fog":   WeatherCondition.FOG,
                "snow":  WeatherCondition.SNOW,
            }
            fixed_condition = _WEATHER_MAP.get(scenario.weather.lower())
            weather_cfg = WeatherConfig(fixed_condition=fixed_condition)

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
                enable_weather=True,
                weather_config=weather_cfg,
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
    ) -> Tuple[List[Dict[str, Any]], List[float], List[float]]:
        """Run episodes and return win/length stats plus latency samples in ms."""
        cfg = self.config
        use_predict = hasattr(policy, "predict")
        episodes: List[Dict[str, Any]] = []
        step_times_ms: List[float] = []
        policy_times_ms: List[float] = []

        for ep_idx in range(cfg.n_eval_episodes):
            # Deterministic reset: base seed + episode offset for reproducibility.
            ep_seed = scenario.seed + ep_idx
            obs, _ = env.reset(seed=ep_seed)
            terminated = truncated = False
            steps = 0
            won: Optional[bool] = None

            while not (terminated or truncated) and steps < cfg.max_steps_per_episode:
                if use_predict:
                    predict_t0 = time.perf_counter_ns()
                    action, _ = policy.predict(obs, deterministic=True)
                    policy_times_ms.append((time.perf_counter_ns() - predict_t0) / 1_000_000.0)
                elif callable(policy):
                    policy_t0 = time.perf_counter_ns()
                    action = policy(obs)
                    policy_times_ms.append((time.perf_counter_ns() - policy_t0) / 1_000_000.0)
                else:
                    action = _scripted_action(obs)

                step_t0 = time.perf_counter_ns()
                obs, _reward, terminated, truncated, info = env.step(action)
                step_times_ms.append((time.perf_counter_ns() - step_t0) / 1_000_000.0)
                steps += 1

            if isinstance(info, dict):
                if info.get("red_routed"):
                    won = True
                elif info.get("blue_routed"):
                    won = False

            episodes.append({"won": won, "steps": steps})

        return episodes, step_times_ms, policy_times_ms


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


def _aggregate_latencies_ms(samples: List[float]) -> Dict[str, float]:
    """Compute p50/p95/mean latency in milliseconds from raw samples."""
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
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
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="Optional path to write JSON latency metrics for CI gating.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Use synthetic fallback env only (fast/stable for CI).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = BenchConfig(
        n_eval_episodes=args.episodes,
        n_scenarios=min(args.scenarios, 20),
        baseline_label=args.label,
        report_path=args.output,
        force_synthetic_env=args.synthetic_only,
    )
    bench = WargamesBench(cfg)

    print(f"Running WargamesBench — {cfg.n_scenarios} scenarios × {cfg.n_eval_episodes} episodes …")
    summary = bench.run(policy=None, label=args.label)

    print()
    print(summary)

    report = summary.write_markdown()
    print(f"\nLeaderboard written to {report}")

    if args.metrics_json:
        metrics_path = summary.write_metrics_json(args.metrics_json)
        print(f"Latency metrics written to {metrics_path}")
    return 0


if __name__ == "__main__":
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    raise SystemExit(main())
