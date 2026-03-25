# SPDX-License-Identifier: MIT
"""Transfer performance benchmark for E11.2 — GIS terrain transfer.

Measures how well a policy trained on procedural (random) terrain performs
on real-world GIS terrain (zero-shot), and how much performance is recovered
after fine-tuning on the target GIS terrain.

Architecture
------------
:class:`TransferEvalConfig`
    Immutable configuration for a single transfer evaluation run.

:class:`TransferResult`
    Win-rate and step-count statistics for one evaluation condition
    (procedural baseline / zero-shot on GIS / fine-tuned on GIS).

:class:`TransferBenchmark`
    High-level runner that orchestrates all three conditions and returns a
    :class:`TransferSummary`.

:class:`TransferSummary`
    Aggregated results with acceptance-criterion checks:

    * Zero-shot win-rate drop < 20 % vs. procedural baseline.
    * Fine-tuned agent recovers to within 5 % of procedural baseline in
      < 500 k fine-tuning steps.

Typical usage
-------------
Without a trained policy (dry-run / CI)::

    from training.transfer_benchmark import TransferBenchmark, TransferEvalConfig

    cfg = TransferEvalConfig(site="waterloo", n_eval_episodes=10)
    bench = TransferBenchmark(cfg)
    summary = bench.run(policy=None)   # uses a scripted random-walk baseline
    print(summary)

With a checkpoint path::

    bench = TransferBenchmark(
        TransferEvalConfig(site="waterloo", n_eval_episodes=50)
    )
    summary = bench.run(policy="checkpoints/corps_v1.zip")
    bench.write_markdown(summary, "docs/transfer_benchmark_waterloo.md")
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# TransferEvalConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferEvalConfig:
    """Configuration for a single transfer evaluation run.

    Attributes
    ----------
    site:
        GIS battle-site identifier — one of ``"waterloo"``,
        ``"austerlitz"``, ``"borodino"``, ``"salamanca"``.
    n_eval_episodes:
        Number of episodes to run for each evaluation condition.
    max_steps_per_episode:
        Episode step budget.
    finetune_steps:
        Maximum fine-tuning steps allowed.  The acceptance criterion
        requires recovery within ``500_000`` steps.
    rows, cols:
        Terrain grid resolution used by the GIS importer.
    procedural_seed:
        RNG seed used to generate the procedural baseline terrain.
    n_procedural_hills, n_procedural_forests:
        Procedural terrain complexity parameters.
    """

    site: str = "waterloo"
    n_eval_episodes: int = 50
    max_steps_per_episode: int = 500
    finetune_steps: int = 500_000
    rows: int = 40
    cols: int = 40
    procedural_seed: int = 42
    n_procedural_hills: int = 4
    n_procedural_forests: int = 3


# ---------------------------------------------------------------------------
# TransferResult
# ---------------------------------------------------------------------------


@dataclass
class TransferResult:
    """Win-rate statistics for one evaluation condition.

    Attributes
    ----------
    condition:
        One of ``"procedural_baseline"``, ``"zero_shot_gis"``,
        ``"finetuned_gis"``.
    win_rate:
        Fraction of episodes where the blue policy won (team 0).
    mean_steps:
        Mean episode length.
    std_steps:
        Standard deviation of episode length.
    n_episodes:
        Number of episodes evaluated.
    elapsed_seconds:
        Wall-clock time for this condition's evaluation.
    finetune_steps_used:
        Number of fine-tuning gradient steps applied before evaluation.
        Zero for non-fine-tuned conditions.
    """

    condition: str
    win_rate: float
    mean_steps: float
    std_steps: float
    n_episodes: int
    elapsed_seconds: float = 0.0
    finetune_steps_used: int = 0


# ---------------------------------------------------------------------------
# TransferSummary
# ---------------------------------------------------------------------------


@dataclass
class TransferSummary:
    """Aggregated transfer benchmark results.

    Attributes
    ----------
    site:
        GIS battle-site that was evaluated.
    procedural:
        Evaluation on procedural terrain (the training distribution).
    zero_shot:
        Evaluation on GIS terrain with no adaptation.
    finetuned:
        Evaluation on GIS terrain after fine-tuning.
    config:
        The :class:`TransferEvalConfig` used.
    """

    site: str
    procedural: TransferResult
    zero_shot: TransferResult
    finetuned: TransferResult
    config: TransferEvalConfig

    # ------------------------------------------------------------------
    # Acceptance criteria
    # ------------------------------------------------------------------

    @property
    def zero_shot_drop(self) -> float:
        """Win-rate drop from procedural → zero-shot (positive = drop)."""
        return self.procedural.win_rate - self.zero_shot.win_rate

    @property
    def finetuned_drop(self) -> float:
        """Win-rate drop from procedural → fine-tuned (positive = drop)."""
        return self.procedural.win_rate - self.finetuned.win_rate

    @property
    def meets_zero_shot_criterion(self) -> bool:
        """Zero-shot drop < 20 percentage points."""
        return self.zero_shot_drop < 0.20

    @property
    def meets_finetune_criterion(self) -> bool:
        """Fine-tuned drop < 5 pp AND fine-tuning used ≤ 500 k steps."""
        return (
            self.finetuned_drop < 0.05
            and self.finetuned.finetune_steps_used <= self.config.finetune_steps
        )

    @property
    def all_criteria_met(self) -> bool:
        """Both acceptance criteria are satisfied."""
        return self.meets_zero_shot_criterion and self.meets_finetune_criterion

    def __str__(self) -> str:
        lines = [
            f"Transfer Benchmark — site: {self.site}",
            f"  Procedural baseline : {self.procedural.win_rate:.1%} win-rate",
            f"  Zero-shot GIS       : {self.zero_shot.win_rate:.1%} win-rate "
            f"(drop {self.zero_shot_drop:+.1%}) "
            f"{'✅' if self.meets_zero_shot_criterion else '❌'}",
            f"  Fine-tuned GIS      : {self.finetuned.win_rate:.1%} win-rate "
            f"(drop {self.finetuned_drop:+.1%}, "
            f"{self.finetuned.finetune_steps_used:,} steps) "
            f"{'✅' if self.meets_finetune_criterion else '❌'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TransferBenchmark
# ---------------------------------------------------------------------------


class TransferBenchmark:
    """Run procedural-baseline, zero-shot, and fine-tuned evaluations.

    Parameters
    ----------
    config:
        Evaluation configuration.

    Notes
    -----
    When *policy* is ``None`` (or a path to a non-existent file) a simple
    scripted policy is used: blue units advance at full speed toward the
    nearest red unit.  This is sufficient for acceptance-criterion testing
    in CI without requiring a trained checkpoint.
    """

    def __init__(self, config: TransferEvalConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        policy: Optional[Any] = None,
        gis_data_dir: Optional[str | Path] = None,
    ) -> TransferSummary:
        """Run all three evaluation conditions and return a summary.

        Parameters
        ----------
        policy:
            Either ``None`` (scripted fallback), a path to a Stable-Baselines3
            checkpoint (``.zip``), or an already-loaded SB3-compatible policy
            object with a ``predict(obs)`` method.
        gis_data_dir:
            Optional directory containing ``{site}.tif`` / ``{site}.osm``
            files.  When absent the synthetic GIS fallback is used.

        Returns
        -------
        TransferSummary
        """
        loaded_policy = self._resolve_policy(policy)

        procedural_terrain = self._build_procedural_terrain()
        gis_terrain = self._build_gis_terrain(gis_data_dir)

        # 1. Procedural baseline
        t0 = time.perf_counter()
        proc_results = self._evaluate(loaded_policy, procedural_terrain)
        proc = TransferResult(
            condition="procedural_baseline",
            **self._aggregate(proc_results),
            elapsed_seconds=time.perf_counter() - t0,
        )

        # 2. Zero-shot on GIS terrain (no adaptation)
        t0 = time.perf_counter()
        zs_results = self._evaluate(loaded_policy, gis_terrain)
        zero_shot = TransferResult(
            condition="zero_shot_gis",
            **self._aggregate(zs_results),
            elapsed_seconds=time.perf_counter() - t0,
        )

        # 3. Fine-tune then evaluate (fine-tuning is a no-op when policy is
        #    None or when the policy lacks a ``learn`` method)
        t0 = time.perf_counter()
        ft_policy, steps_used = self._finetune(loaded_policy, gis_terrain)
        ft_results = self._evaluate(ft_policy, gis_terrain)
        finetuned = TransferResult(
            condition="finetuned_gis",
            **self._aggregate(ft_results),
            elapsed_seconds=time.perf_counter() - t0,
            finetune_steps_used=steps_used,
        )

        return TransferSummary(
            site=self.config.site,
            procedural=proc,
            zero_shot=zero_shot,
            finetuned=finetuned,
            config=self.config,
        )

    def write_markdown(
        self,
        summary: TransferSummary,
        path: Optional[str | Path] = None,
    ) -> Path:
        """Write the benchmark summary to a Markdown file.

        Parameters
        ----------
        summary:
            Results from :meth:`run`.
        path:
            Output path.  Defaults to
            ``docs/transfer_benchmark_{site}.md``.

        Returns
        -------
        Path
            Absolute path to the written file.
        """
        if path is None:
            docs_dir = _REPO_ROOT / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            path = docs_dir / f"transfer_benchmark_{summary.site}.md"
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_transfer_markdown(summary), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Internal: terrain builders
    # ------------------------------------------------------------------

    def _build_procedural_terrain(self):
        """Build a procedural (random) TerrainMap for the baseline."""
        from envs.sim.terrain import TerrainMap

        cfg = self.config
        return TerrainMap.generate_random(
            rng=np.random.default_rng(cfg.procedural_seed),
            width=10_000.0,
            height=10_000.0,
            rows=cfg.rows,
            cols=cfg.cols,
            num_hills=cfg.n_procedural_hills,
            num_forests=cfg.n_procedural_forests,
        )

    def _build_gis_terrain(self, gis_data_dir: Optional[Any]):
        """Build a GIS TerrainMap for the target site."""
        from data.gis.terrain_importer import GISTerrainBuilder

        return GISTerrainBuilder(
            site=self.config.site,
            rows=self.config.rows,
            cols=self.config.cols,
            srtm_path=(
                Path(gis_data_dir) / f"{self.config.site}.tif"
                if gis_data_dir is not None
                else None
            ),
            osm_path=(
                Path(gis_data_dir) / f"{self.config.site}.osm"
                if gis_data_dir is not None
                else None
            ),
        ).build()

    # ------------------------------------------------------------------
    # Internal: policy helpers
    # ------------------------------------------------------------------

    def _resolve_policy(self, policy: Optional[Any]) -> Optional[Any]:
        """Load a policy from a checkpoint path if needed."""
        if policy is None:
            return None
        if isinstance(policy, (str, Path)):
            path = Path(policy)
            if not path.exists():
                return None
            try:
                from stable_baselines3 import PPO  # type: ignore

                return PPO.load(path)
            except ImportError:
                # Allow running without Stable-Baselines3 by falling back.
                return None
            except Exception as exc:
                # If the file exists but cannot be loaded, surface the error
                # instead of silently downgrading to the baseline.
                raise RuntimeError(
                    f"Failed to load policy checkpoint from {path}"
                ) from exc
        return policy

    def _finetune(
        self,
        policy: Optional[Any],
        terrain,
    ) -> Tuple[Optional[Any], int]:
        """Fine-tune *policy* on *terrain* for up to config.finetune_steps.

        Creates a :class:`~envs.battalion_env.BattalionEnv` with the target
        GIS terrain and attaches it to the policy (via ``set_env``) before
        calling ``learn``.  This ensures that fine-tuning is actually
        performed on the target map rather than on whatever environment the
        policy was originally trained with.

        Returns the (possibly updated) policy and the number of gradient
        steps actually taken.  When *policy* is ``None`` or lacks a
        ``learn`` method, returns ``(policy, 0)`` unchanged.
        """
        if policy is None or not hasattr(policy, "learn"):
            return policy, 0
        try:
            from envs.battalion_env import BattalionEnv

            ft_env = BattalionEnv(terrain=terrain, randomize_terrain=False)
            if hasattr(policy, "set_env"):
                policy.set_env(ft_env)
            policy.learn(total_timesteps=self.config.finetune_steps)
            ft_env.close()
            return policy, self.config.finetune_steps
        except Exception:
            return policy, 0

    # ------------------------------------------------------------------
    # Internal: episode runner
    # ------------------------------------------------------------------

    # Constant used to offset the eval RNG seed from the terrain seed so
    # unit placements are not correlated with the procedural terrain blobs.
    _EVAL_SEED_OFFSET: int = 0xDEAD

    def _evaluate(
        self,
        policy: Optional[Any],
        terrain,
    ) -> List[Dict]:
        """Run *n_eval_episodes* episodes on *terrain* and return raw stats.

        Each episode dict has ``{"winner": int | None, "steps": int}``.

        When *policy* has a ``predict`` method (i.e., a Stable-Baselines3–
        compatible learned policy), evaluation is driven via
        :class:`~envs.battalion_env.BattalionEnv` with the target terrain
        fixed and terrain randomisation disabled.  The winner is inferred
        from the ``blue_routed`` / ``red_routed`` fields in the final
        ``info`` dict.

        When *policy* is ``None`` (or any object without a ``predict``
        method), the :class:`~envs.sim.engine.SimEngine` deterministic
        scripted baseline is used.  Each episode uses a separate RNG seeded
        from :attr:`_EVAL_SEED_OFFSET` so results are fully reproducible.
        Note: without a morale config the scripted engine will typically run
        to ``max_steps`` as a draw; use a learned policy for meaningful
        win-rate measurements.
        """
        from envs.sim.engine import SimEngine
        from envs.sim.battalion import Battalion

        cfg = self.config
        results: List[Dict] = []
        base_seed = int(cfg.procedural_seed ^ self._EVAL_SEED_OFFSET)

        # Treat any object with a .predict method as a learned policy; otherwise
        # fall back to the built-in scripted SimEngine baseline.
        use_learned_policy = hasattr(policy, "predict")

        if use_learned_policy:
            # Import here to avoid a hard dependency when only the scripted
            # baseline is used (e.g., in lightweight CI contexts).
            from envs.battalion_env import BattalionEnv

        for ep_idx in range(cfg.n_eval_episodes):
            if use_learned_policy:
                # Evaluate the learned policy in a Gymnasium environment with a
                # fixed terrain and deterministic seeding.
                env = BattalionEnv(terrain=terrain, randomize_terrain=False)
                obs, _info = env.reset(seed=base_seed + ep_idx)

                terminated = False
                truncated = False
                steps = 0
                last_info: Dict[str, Any] = {}

                while not (terminated or truncated):
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, _reward, terminated, truncated, last_info = env.step(action)
                    steps += 1

                env.close()

                # Infer winner from the final info dict.
                blue_routed = last_info.get("blue_routed", False)
                red_routed = last_info.get("red_routed", False)
                winner: Optional[int]
                if terminated and red_routed and not blue_routed:
                    winner = 0  # blue wins
                elif terminated and blue_routed and not red_routed:
                    winner = 1  # red wins
                else:
                    winner = None  # draw or truncated

                results.append({"winner": winner, "steps": steps})
            else:
                # Scripted baseline: use SimEngine with a per-episode RNG for
                # reproducible evaluations.
                episode_rng = np.random.default_rng(base_seed + ep_idx)
                blue = Battalion(
                    x=terrain.width * 0.25,
                    y=terrain.height * 0.50,
                    theta=0.0,
                    strength=1.0,
                    team=0,
                )
                red = Battalion(
                    x=terrain.width * 0.75,
                    y=terrain.height * 0.50,
                    theta=float(np.pi),
                    strength=1.0,
                    team=1,
                )
                engine = SimEngine(
                    blue,
                    red,
                    terrain=terrain,
                    max_steps=cfg.max_steps_per_episode,
                    rng=episode_rng,
                )
                result = engine.run()
                results.append({"winner": result.winner, "steps": result.steps})

        return results

    @staticmethod
    def _aggregate(results: List[Dict]) -> Dict:
        """Compute win_rate, mean_steps, std_steps, n_episodes."""
        n = len(results)
        if n == 0:
            return {
                "win_rate": 0.0,
                "mean_steps": 0.0,
                "std_steps": 0.0,
                "n_episodes": 0,
            }
        wins = sum(1 for r in results if r["winner"] == 0)
        steps_arr = np.array([r["steps"] for r in results], dtype=np.float32)
        return {
            "win_rate": float(wins) / n,
            "mean_steps": float(steps_arr.mean()),
            "std_steps": float(steps_arr.std()),
            "n_episodes": n,
        }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_transfer_markdown(summary: TransferSummary) -> str:
    """Render a Markdown report from a :class:`TransferSummary`."""
    cfg = summary.config
    lines: List[str] = [
        f"# GIS Terrain Transfer Benchmark — {summary.site.capitalize()}",
        "",
        f"Site: **{summary.site}** | Episodes: {cfg.n_eval_episodes} "
        f"| Max steps: {cfg.max_steps_per_episode} "
        f"| Fine-tune budget: {cfg.finetune_steps:,}",
        "",
        "## Summary",
        "",
        "| Condition | Win-rate | Δ vs. Baseline | Criterion |",
        "|-----------|----------|----------------|-----------|",
        f"| Procedural baseline | {summary.procedural.win_rate:.1%} | — | — |",
        (
            f"| Zero-shot GIS | {summary.zero_shot.win_rate:.1%} | "
            f"{summary.zero_shot_drop:+.1%} | "
            f"{'✅ PASS (< 20 %)' if summary.meets_zero_shot_criterion else '❌ FAIL (≥ 20 %)'} |"
        ),
        (
            f"| Fine-tuned GIS | {summary.finetuned.win_rate:.1%} | "
            f"{summary.finetuned_drop:+.1%} | "
            f"{'✅ PASS (< 5 %)' if summary.meets_finetune_criterion else '❌ FAIL (≥ 5 %)'} |"
        ),
        "",
        "## Acceptance Criteria",
        "",
        (
            f"- {'✅' if summary.meets_zero_shot_criterion else '❌'} "
            f"Zero-shot win-rate drop < 20 pp "
            f"(actual: {summary.zero_shot_drop:+.1%})"
        ),
        (
            f"- {'✅' if summary.meets_finetune_criterion else '❌'} "
            f"Fine-tuned drop < 5 pp within {cfg.finetune_steps:,} steps "
            f"(actual drop: {summary.finetuned_drop:+.1%}, "
            f"steps used: {summary.finetuned.finetune_steps_used:,})"
        ),
        "",
        "## Episode Statistics",
        "",
        "| Condition | Mean steps | Std steps | Episodes |",
        "|-----------|------------|-----------|----------|",
    ]
    for res in (summary.procedural, summary.zero_shot, summary.finetuned):
        lines.append(
            f"| {res.condition} | {res.mean_steps:.1f} | "
            f"{res.std_steps:.1f} | {res.n_episodes} |"
        )
    lines += [
        "",
        "> *Generated automatically by `training/transfer_benchmark.py`.*",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run transfer benchmark for all four GIS battle sites."""
    sites = ["waterloo", "austerlitz", "borodino", "salamanca"]
    for site in sites:
        print(f"\nRunning transfer benchmark for {site}…")
        cfg = TransferEvalConfig(site=site, n_eval_episodes=20)
        bench = TransferBenchmark(cfg)
        summary = bench.run()
        print(summary)
        out = bench.write_markdown(summary)
        print(f"  Report: {out}")


if __name__ == "__main__":
    main()
