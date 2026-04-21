#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark regression gate for WargamesBench latency artifacts.

Compares step-time p95 latency per scenario against a baseline artifact and
fails if any scenario regresses above the configured percentage threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_map(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scenarios = metrics.get("scenarios", [])
    out: Dict[str, Dict[str, Any]] = {}
    for entry in scenarios:
        name = str(entry.get("scenario", "")).strip()
        if name:
            out[name] = entry
    return out


def _pct_change(current: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def _build_markdown(
    *,
    threshold_pct: float,
    current_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any] | None,
    rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
    missing_baseline: List[str],
) -> str:
    lines: List[str] = []
    lines.append("## Wargames Benchmark Gate")
    lines.append("")
    lines.append(f"- Threshold: `{threshold_pct:.1f}%` p95 step-time regression")

    if baseline_metrics is None:
        lines.append("- Baseline: not found (gate skipped)")
        lines.append("")
        lines.append("No baseline artifact was available, so regression checks were skipped for this run.")
        return "\n".join(lines) + "\n"

    overall_current = float(current_metrics.get("mean_step_time_p95_ms", 0.0))
    overall_baseline = float(baseline_metrics.get("mean_step_time_p95_ms", 0.0))
    overall_delta = _pct_change(overall_current, overall_baseline)

    lines.append(f"- Mean scenario p95 (baseline): `{overall_baseline:.4f} ms`")
    lines.append(f"- Mean scenario p95 (current): `{overall_current:.4f} ms`")
    lines.append(f"- Mean scenario p95 delta: `{overall_delta:+.2f}%`")
    lines.append("")

    lines.append("| Scenario | Baseline p95 (ms) | Current p95 (ms) | Delta | Status |")
    lines.append("|---|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['baseline_p95_ms']:.4f} | {row['current_p95_ms']:.4f} "
            f"| {row['delta_pct']:+.2f}% | {row['status']} |"
        )

    if missing_baseline:
        lines.append("")
        lines.append("Scenarios missing from baseline:")
        for name in missing_baseline:
            lines.append(f"- {name}")

    lines.append("")
    if failed_rows:
        lines.append(f"Gate result: FAIL ({len(failed_rows)} scenario(s) exceeded threshold)")
    else:
        lines.append("Gate result: PASS")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Wargames benchmark metrics with a baseline")
    parser.add_argument("--current", required=True, help="Current run metrics JSON path")
    parser.add_argument("--baseline", required=False, help="Baseline metrics JSON path")
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=20.0,
        help="Fail if scenario p95 step-time regression exceeds this percentage",
    )
    parser.add_argument(
        "--summary-markdown",
        required=True,
        help="Path to write markdown summary",
    )
    parser.add_argument(
        "--result-json",
        required=True,
        help="Path to write machine-readable gate result JSON",
    )
    args = parser.parse_args()

    current_path = Path(args.current)
    baseline_path = Path(args.baseline) if args.baseline else None
    summary_path = Path(args.summary_markdown)
    result_path = Path(args.result_json)

    current_metrics = _load_json(current_path)
    baseline_metrics = None

    if baseline_path is not None and baseline_path.exists():
        baseline_metrics = _load_json(baseline_path)

    rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    missing_baseline: List[str] = []

    if baseline_metrics is not None:
        cur_map = _scenario_map(current_metrics)
        base_map = _scenario_map(baseline_metrics)

        for scenario_name in sorted(cur_map.keys()):
            cur_p95 = float(cur_map[scenario_name].get("step_time_ms", {}).get("p95", 0.0))
            base_entry = base_map.get(scenario_name)
            if base_entry is None:
                missing_baseline.append(scenario_name)
                continue
            base_p95 = float(base_entry.get("step_time_ms", {}).get("p95", 0.0))
            delta_pct = _pct_change(cur_p95, base_p95)
            failed = delta_pct > args.threshold_percent
            row = {
                "scenario": scenario_name,
                "baseline_p95_ms": base_p95,
                "current_p95_ms": cur_p95,
                "delta_pct": delta_pct,
                "status": "FAIL" if failed else "PASS",
            }
            rows.append(row)
            if failed:
                failed_rows.append(row)

    summary_md = _build_markdown(
        threshold_pct=args.threshold_percent,
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        rows=rows,
        failed_rows=failed_rows,
        missing_baseline=missing_baseline,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")

    result = {
        "threshold_percent": args.threshold_percent,
        "gate_skipped": baseline_metrics is None,
        "gate_passed": baseline_metrics is None or len(failed_rows) == 0,
        "failed_scenarios": failed_rows,
        "checked_scenarios": rows,
        "missing_baseline_scenarios": missing_baseline,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Do not fail when baseline is missing; first run establishes observability.
    if baseline_metrics is None:
        return 0
    return 1 if failed_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
