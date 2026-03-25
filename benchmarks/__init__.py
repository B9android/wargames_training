# SPDX-License-Identifier: MIT
"""WargamesBench — standardised evaluation benchmark suite (E12.2).

Public re-exports from :mod:`benchmarks.wargames_bench`.
"""

from benchmarks.wargames_bench import (
    BENCH_SCENARIOS,
    BenchScenario,
    BenchConfig,
    BenchResult,
    BenchSummary,
    WargamesBench,
    main,
)

__all__ = [
    "BENCH_SCENARIOS",
    "BenchScenario",
    "BenchConfig",
    "BenchResult",
    "BenchSummary",
    "WargamesBench",
    "main",
]
