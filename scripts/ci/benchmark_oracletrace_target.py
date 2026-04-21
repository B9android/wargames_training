#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""OracleTrace target runner for WargamesBench CI.

OracleTrace 2.0.0 CLI can reject forwarded script args depending on invocation
pattern. This wrapper keeps all benchmark arguments inside Python so the tracer
can execute a simple, argument-free target script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.wargames_bench import main as bench_main


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def main() -> int:
    artifacts_dir = Path(_env("BENCHMARK_ARTIFACTS_DIR", "benchmark-artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    episodes = _env("BENCH_EPISODES", "5")
    scenarios = _env("BENCH_SCENARIOS", "8")
    label = _env("BENCH_LABEL", "ci_scripted_baseline")

    argv = [
        "--episodes",
        episodes,
        "--scenarios",
        scenarios,
        "--synthetic-only",
        "--label",
        label,
        "--metrics-json",
        str(artifacts_dir / "oracletrace_metrics.json"),
        "--output",
        str(artifacts_dir / "oracletrace_leaderboard.md"),
    ]
    return bench_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
