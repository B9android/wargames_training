# SPDX-License-Identifier: MIT
"""Platform entrypoint for deterministic static analysis.

Wraps static_analyzer.py with agent_platform runner semantics so this check can be
invoked from orchestration workflows and produce standard agent artifacts.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agent_platform.context import AgentContext
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event
from static_analyzer import (
    DEFAULT_SCAN_DIRS,
    analyze,
    apply_todos,
    build_module_index_with_parse_findings,
    write_json_report,
    write_text_summary,
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_scan_dirs() -> list[str]:
    raw = os.environ.get("STATIC_ANALYZER_PATHS", "").strip()
    if not raw:
        return list(DEFAULT_SCAN_DIRS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _normalize_changed_paths(lines: list[str]) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in lines
        if line.strip().endswith(".py")
    }


def _git_changed_python_files(repo_root: Path) -> set[str]:
    base_ref = os.environ.get("GITHUB_BASE_REF", "").strip()
    candidates: list[list[str]] = []
    if base_ref:
        candidates.append(["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"])
    candidates.append(["git", "diff", "--name-only", "HEAD~1", "HEAD"])

    for command in candidates:
        try:
            proc = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

        lines = proc.stdout.splitlines()
        changed = _normalize_changed_paths(lines)
        if changed:
            return changed

    return set()


def _filter_findings_to_changed_files(
    findings: list[dict[str, object]],
    repo_root: Path,
    changed_files: set[str],
) -> list[dict[str, object]]:
    if not changed_files:
        return findings

    filtered: list[dict[str, object]] = []
    for finding in findings:
        file_value = str(finding.get("file", ""))
        rel = _to_repo_relative(Path(file_value), repo_root)
        if rel in changed_files:
            filtered.append(finding)
    return filtered


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    del gql  # Analyzer is repository-local and does not need network calls.
    del rest

    repo_root = Path(os.environ.get("STATIC_ANALYZER_REPO_ROOT", ".")).resolve()
    scan_dirs = _env_scan_dirs()
    insert_todos = _env_bool("STATIC_ANALYZER_INSERT_TODOS", default=False)
    fail_on_high = _env_bool("STATIC_ANALYZER_FAIL_ON_HIGH", default=False)
    only_changed = _env_bool("STATIC_ANALYZER_ONLY_CHANGED", default=False)

    report_path = (repo_root / os.environ.get("STATIC_ANALYZER_REPORT_PATH", "logs/static_analysis_report.json")).resolve()
    summary_path = (repo_root / os.environ.get("STATIC_ANALYZER_SUMMARY_PATH", "logs/static_analysis_summary.txt")).resolve()

    summary.checkpoint("index", f"Analyzing deterministic scan paths: {scan_dirs}")
    module_index, parse_findings = build_module_index_with_parse_findings(repo_root, scan_dirs)
    findings = parse_findings + analyze(module_index)

    changed_files: set[str] = set()
    if only_changed:
        changed_files = _git_changed_python_files(repo_root)
        findings = _filter_findings_to_changed_files(findings, repo_root, changed_files)
        summary.checkpoint(
            "scope",
            f"Scoped to changed Python files: {len(changed_files)}",
        )

    write_json_report(report_path, findings)
    write_text_summary(summary_path, findings)

    todo_changed_files: list[Path] = []
    if insert_todos and not ctx.dry_run:
        todo_changed_files = apply_todos(findings)
    elif insert_todos and ctx.dry_run:
        summary.checkpoint("dry_run", "STATIC_ANALYZER_INSERT_TODOS requested but skipped because DRY_RUN=true")

    high_count = sum(1 for item in findings if item["confidence"] == "high")
    medium_count = sum(1 for item in findings if item["confidence"] == "medium")
    low_count = sum(1 for item in findings if item["confidence"] == "low")

    log_event(
        "static_analysis_completed",
        total=len(findings),
        high=high_count,
        medium=medium_count,
        low=low_count,
        report=str(report_path),
        summary=str(summary_path),
        changed_scope=sorted(changed_files),
        changed_files=[str(path) for path in todo_changed_files],
    )

    result = RunResult(agent="static_analysis", dry_run=ctx.dry_run)
    result.record_success(
        "scan_repository",
        str(repo_root),
        total_findings=len(findings),
        high=high_count,
        medium=medium_count,
        low=low_count,
        report=str(report_path),
        summary=str(summary_path),
        changed_scope=sorted(changed_files),
        changed_files=[str(path) for path in todo_changed_files],
    )

    if fail_on_high and high_count > 0:
        result.record_failure(
            "high_confidence_gate",
            str(repo_root),
            f"{high_count} high-confidence findings detected",
        )
        summary.decision(f"Failing run due to high-confidence findings: {high_count}")
    else:
        summary.decision(
            "Static analysis completed"
            f" (high={high_count}, medium={medium_count}, low={low_count})"
        )

    return result


if __name__ == "__main__":
    run_agent("static_analysis", _run)
