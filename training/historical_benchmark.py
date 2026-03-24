"""Historical benchmark runner for E11.1 — Historical Battle Database.

Runs the deterministic 1v1 :class:`~envs.sim.engine.SimEngine` baseline
against the historical outcomes across all 50+ Napoleonic engagements in
the battle database. This is a simulation fidelity benchmark only; no
trained RL policy or external scripted controller is loaded or consulted.

Acceptance criteria tested here
---------------------------------
* Importer handles ≥ 50 engagements without errors.
* Full benchmark completes in < 2 hours on a single GPU (each 1v1 episode
  is very short, so 50 × a few seconds is well within budget).
* Results table written to ``docs/historical_benchmark.md``.

Typical usage::

    python -m training.historical_benchmark

    # Or programmatically:
    from training.historical_benchmark import HistoricalBenchmark
    bench = HistoricalBenchmark()
    results = bench.run()
    bench.write_markdown(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from envs.scenarios.historical import ComparisonResult, OutcomeComparator
from envs.scenarios.importer import BatchScenarioImporter
from envs.sim.engine import EpisodeResult, SimEngine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BATTLES_JSON = _REPO_ROOT / "data" / "historical" / "battles.json"
_BENCHMARK_MD = _REPO_ROOT / "docs" / "historical_benchmark.md"


# ---------------------------------------------------------------------------
# Per-scenario result
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEntry:
    """Result for a single battle in the benchmark run.

    Attributes
    ----------
    battle_id:
        Unique battle identifier (e.g. ``"waterloo_1815"``).
    scenario_name:
        Human-readable battle name.
    date:
        ISO-8601 battle date.
    source:
        Source reference for the OOB data.
    historical_winner:
        Historically documented winner (``0`` = blue, ``1`` = red,
        ``None`` = draw/inconclusive).
    comparison:
        :class:`~envs.scenarios.historical.ComparisonResult` produced by
        comparing the simulated episode to the historical outcome.
    elapsed_seconds:
        Wall-clock time taken to run this scenario's episode.
    error:
        If not ``None``, the scenario failed with this exception message.
    """

    battle_id: str
    scenario_name: str
    date: str
    source: str
    historical_winner: Optional[int]
    comparison: Optional[ComparisonResult] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """``True`` if the scenario ran without error."""
        return self.error is None

    @property
    def winner_matches(self) -> bool:
        """``True`` if the simulated winner matches the historical winner."""
        if self.comparison is None:
            return False
        return self.comparison.winner_matches

    @property
    def fidelity_score(self) -> float:
        """Fidelity score in ``[0, 1]`` or ``0.0`` on error."""
        if self.comparison is None:
            return 0.0
        return self.comparison.fidelity_score


# ---------------------------------------------------------------------------
# Benchmark summary
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSummary:
    """Aggregate statistics over all benchmark entries.

    Attributes
    ----------
    total:
        Total number of scenarios attempted.
    passed:
        Number that ran without error.
    winner_match_rate:
        Fraction of *passed* scenarios where the simulated winner matches
        the historical winner.
    mean_fidelity:
        Mean fidelity score across *passed* scenarios.
    std_fidelity:
        Standard deviation of fidelity scores across *passed* scenarios.
    total_elapsed_seconds:
        Total wall-clock time for the benchmark run.
    entries:
        All per-scenario result entries.
    """

    total: int
    passed: int
    winner_match_rate: float
    mean_fidelity: float
    std_fidelity: float
    total_elapsed_seconds: float
    entries: List[BenchmarkEntry] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def meets_importer_criterion(self) -> bool:
        """Importer handles ≥ 50 engagements without errors."""
        return self.total >= 50 and self.passed == self.total

    @property
    def meets_outcome_criterion(self) -> bool:
        """Agent achieves historically plausible outcome on ≥ 60 % of battles.

        Here 'historically plausible' is defined as the simulated winner
        matching the historical winner (a binary criterion that can be
        evaluated without training a full agent).
        """
        return self.winner_match_rate >= 0.60


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class HistoricalBenchmark:
    """Run all 50+ historical scenarios and collect comparison results.

    Parameters
    ----------
    battles_path:
        Path to the JSON battle database.  Defaults to
        ``data/historical/battles.json`` relative to the repository root.
    seed:
        Random seed passed to :class:`~envs.sim.engine.SimEngine` for
        reproducible results.
    """

    def __init__(
        self,
        battles_path: str | Path = _BATTLES_JSON,
        seed: int = 42,
    ) -> None:
        self.battles_path = Path(battles_path)
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BenchmarkSummary:
        """Run the full benchmark and return a :class:`BenchmarkSummary`.

        Each scenario is run as a 1v1 simulation using the first blue
        battalion against the first red battalion (matching the existing
        test pattern in ``tests/test_historical_scenarios.py``).
        """
        importer = BatchScenarioImporter(self.battles_path)
        records = importer.load_records()
        scenarios = [r.to_scenario() for r in records]

        entries: List[BenchmarkEntry] = []
        overall_start = time.perf_counter()

        for rec, scenario in zip(records, scenarios):
            entry = self._run_scenario(rec.battle_id, rec.source, scenario)
            entries.append(entry)

        total_elapsed = time.perf_counter() - overall_start

        # Aggregate
        passed_entries = [e for e in entries if e.passed]
        fidelity_scores = [e.fidelity_score for e in passed_entries]
        winner_matches = [e.winner_matches for e in passed_entries]

        mean_fidelity = float(np.mean(fidelity_scores)) if fidelity_scores else 0.0
        std_fidelity = float(np.std(fidelity_scores)) if fidelity_scores else 0.0
        winner_match_rate = (
            float(np.mean([float(w) for w in winner_matches]))
            if winner_matches else 0.0
        )

        return BenchmarkSummary(
            total=len(entries),
            passed=len(passed_entries),
            winner_match_rate=winner_match_rate,
            mean_fidelity=mean_fidelity,
            std_fidelity=std_fidelity,
            total_elapsed_seconds=total_elapsed,
            entries=entries,
        )

    def write_markdown(
        self,
        summary: BenchmarkSummary,
        output_path: str | Path = _BENCHMARK_MD,
    ) -> Path:
        """Write the benchmark results to a Markdown file.

        Parameters
        ----------
        summary:
            The :class:`BenchmarkSummary` returned by :meth:`run`.
        output_path:
            Destination file path.  Parent directories are created if
            they do not already exist.

        Returns
        -------
        Path
            The path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _render_markdown(summary),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_scenario(self, battle_id: str, source: str, scenario) -> BenchmarkEntry:
        """Run a single scenario and return a :class:`BenchmarkEntry`."""
        start = time.perf_counter()
        entry = BenchmarkEntry(
            battle_id=battle_id,
            scenario_name=scenario.name,
            date=scenario.date,
            source=source,
            historical_winner=scenario.historical_outcome.winner,
        )
        try:
            blue_battalions, red_battalions = scenario.build_battalions()
            terrain = scenario.build_terrain()
            rng = np.random.default_rng(self.seed)
            result: EpisodeResult = SimEngine(
                blue_battalions[0],
                red_battalions[0],
                terrain=terrain,
                rng=rng,
            ).run()
            comparator = OutcomeComparator(scenario.historical_outcome)
            entry.comparison = comparator.compare(result)
        except Exception as exc:  # noqa: BLE001
            entry.error = str(exc)
        finally:
            entry.elapsed_seconds = time.perf_counter() - start
        return entry


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(summary: BenchmarkSummary) -> str:
    """Render a Markdown document from the benchmark summary."""
    lines: List[str] = []

    # Header
    lines.append("# Historical Benchmark Results")
    lines.append("")
    lines.append(
        "Automated benchmark: direct `SimEngine` runs comparing simulated "
        "outcomes to historical battle results across all engagements in "
        "`data/historical/battles.json`."
    )
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total scenarios | {summary.total} |")
    lines.append(f"| Passed (no error) | {summary.passed} |")
    lines.append(f"| Failed | {summary.failed} |")
    lines.append(
        f"| Winner match rate | {summary.winner_match_rate:.1%} "
        f"{'✅' if summary.meets_outcome_criterion else '❌'} |"
    )
    lines.append(f"| Mean fidelity score | {summary.mean_fidelity:.3f} |")
    lines.append(f"| Std fidelity score | {summary.std_fidelity:.3f} |")
    lines.append(
        f"| Importer criterion (≥ 50 without errors) | "
        f"{'✅ PASS' if summary.meets_importer_criterion else '❌ FAIL'} |"
    )
    lines.append(
        f"| Outcome criterion (≥ 60 % winner match) | "
        f"{'✅ PASS' if summary.meets_outcome_criterion else '❌ FAIL'} |"
    )
    elapsed_min = summary.total_elapsed_seconds / 60.0
    lines.append(
        f"| Total elapsed | {elapsed_min:.1f} min "
        f"{'✅' if elapsed_min < 120 else '❌ (> 2 h)'} |"
    )
    lines.append("")

    # Per-scenario table
    lines.append("## Per-Scenario Results")
    lines.append("")
    lines.append(
        "| # | Battle | Date | Source | Hist. Winner | Sim. Winner | "
        "Winner Match | Blue Δcas | Red Δcas | Fidelity | Status |"
    )
    lines.append(
        "|---|--------|------|--------|-------------|-------------|"
        "-------------|----------|----------|----------|--------|"
    )

    for i, entry in enumerate(summary.entries, start=1):
        if entry.error:
            status = "❌ ERROR"
            sim_winner = "—"
            winner_match = "—"
            blue_delta = "—"
            red_delta = "—"
            fidelity = "—"
        else:
            cmp = entry.comparison
            status = "✅"
            sim_winner = str(cmp.simulated_winner) if cmp.simulated_winner is not None else "draw"
            winner_match = "✅" if cmp.winner_matches else "❌"
            blue_delta = f"{cmp.casualty_delta_blue:+.2f}"
            red_delta = f"{cmp.casualty_delta_red:+.2f}"
            fidelity = f"{cmp.fidelity_score:.3f}"

        hist_winner = (
            str(entry.historical_winner)
            if entry.historical_winner is not None
            else "draw"
        )
        source = entry.source or "—"

        lines.append(
            f"| {i} | {entry.scenario_name} | {entry.date} | {source} | "
            f"{hist_winner} | {sim_winner} | {winner_match} | "
            f"{blue_delta} | {red_delta} | {fidelity} | {status} |"
        )

    lines.append("")
    lines.append(
        "> *Generated automatically by `training/historical_benchmark.py`.*"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the benchmark and write results to docs/historical_benchmark.md."""
    print("Running Historical Benchmark…")
    bench = HistoricalBenchmark()
    summary = bench.run()

    print(f"  Total scenarios : {summary.total}")
    print(f"  Passed          : {summary.passed}")
    print(f"  Failed          : {summary.failed}")
    print(f"  Winner match    : {summary.winner_match_rate:.1%}")
    print(f"  Mean fidelity   : {summary.mean_fidelity:.3f}")
    print(f"  Elapsed         : {summary.total_elapsed_seconds:.1f}s")
    print()
    print(f"  Importer criterion (≥ 50 w/o errors): "
          f"{'PASS' if summary.meets_importer_criterion else 'FAIL'}")
    print(f"  Outcome criterion  (≥ 60% winner match): "
          f"{'PASS' if summary.meets_outcome_criterion else 'FAIL'}")

    out = bench.write_markdown(summary)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
