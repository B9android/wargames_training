# SPDX-License-Identifier: MIT
"""Milestone Health Check Agent.
Runs daily and detects at-risk milestones, stale issues, and unlabeled issues.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

yaml = importlib.import_module("yaml")

from agent_platform.context import AgentContext
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event
from common import agent_signature
from dependency_resolver import blocked_parent_rollup

AGENT_CREATED_LABEL = "status: agent-created"
PRIORITY_MEDIUM = "priority: medium"

_FALLBACK_RULE_MAP: dict[str, list[str]] = {
    "[BUG]": ["type: bug", "priority: high"],
    "[EXP]": ["type: experiment", PRIORITY_MEDIUM],
    "[RESEARCH]": ["type: research", "priority: low"],
    "[FEAT]": ["type: feature", PRIORITY_MEDIUM],
    "[FEATURE]": ["type: feature", PRIORITY_MEDIUM],
    "[EPIC]": ["type: epic", "priority: high"],
    "[WIP]": ["status: in-progress"],
}


def marker_for_milestone(milestone_number: int, due_date: str) -> str:
    return f"<!-- agent:milestone-risk:{milestone_number}:{due_date} -->"


def _load_rule_map() -> dict[str, list[str]]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if isinstance(raw, dict) and raw and all(isinstance(v, list) for v in raw.values()):
            return {str(marker).upper(): [str(label) for label in labels] for marker, labels in raw.items()}
        markers = raw.get("markers", {})
        if isinstance(markers, dict):
            loaded = {
                str(marker).upper(): [
                    str(label)
                    for label in (cfg.get("labels", []) if isinstance(cfg, dict) else [])
                ]
                for marker, cfg in markers.items()
            }
            if loaded:
                return loaded
    except Exception as exc:
        log_event("label_hygiene_rule_map_fallback", error=str(exc), config=str(config_path))
    return _FALLBACK_RULE_MAP


def load_rule_map() -> dict[str, list[str]]:
    return _load_rule_map()


def _dependency_blocker_lines(open_issues: list[object]) -> str:
    rollup = blocked_parent_rollup(open_issues)
    if not rollup:
        return "- None"
    lines: list[str] = []
    for entry in rollup:
        parent = f"- #{entry['parent_number']} {entry['parent_title']}"
        child_list = ", ".join(f"#{c['number']} {c['title']}" for c in entry["blocked_children"])
        lines.append(f"{parent} blocked by: {child_list}")
    return "\n".join(lines)


def dependency_blocker_lines(open_issues: list[object]) -> str:
    return _dependency_blocker_lines(open_issues)


def maybe_create_at_risk_issue(repo, milestone, now: datetime, dry_run: bool) -> int | None:
    if not milestone.due_on:
        return None
    due = milestone.due_on.replace(tzinfo=timezone.utc)
    days_left = (due - now).days
    total = milestone.open_issues + milestone.closed_issues
    pct = (milestone.closed_issues / total * 100) if total > 0 else 0
    at_risk = (days_left <= 14 and pct < 50) or (days_left <= 7 and pct < 80)
    if not at_risk:
        return None

    marker = marker_for_milestone(milestone.number, due.strftime("%Y-%m-%d"))
    existing = list(repo.get_issues(state="open", labels=[AGENT_CREATED_LABEL]))
    if any(marker in (e.body or "") for e in existing):
        return None

    title = f"⚠️ MILESTONE AT RISK: {milestone.title} ({pct:.0f}% complete, {days_left}d left)"
    related_open = list(repo.get_issues(state="open", milestone=milestone))
    body = f"""## Milestone At Risk

**Milestone:** [{milestone.title}]({milestone.url})
**Due:** {due.strftime('%Y-%m-%d')} ({days_left} days remaining)
**Progress:** {milestone.closed_issues}/{total} issues closed ({pct:.0f}%)

### Open Issues Remaining
{chr(10).join(f'- #{i.number} {i.title}' for i in related_open) or '- None'}

### Dependency Blockers
{_dependency_blocker_lines(related_open)}

{agent_signature("milestone_checker", context="milestone health check")}

{marker}
"""
    if dry_run:
        return None
    issue = repo.create_issue(
        title=title,
        body=body,
        labels=["priority: critical", "type: chore", AGENT_CREATED_LABEL],
    )
    return issue.number


def mark_stale_issues(repo, now: datetime, dry_run: bool) -> int:
    stale_cutoff = now - timedelta(days=14)
    stale_marked = 0
    for issue in repo.get_issues(state="open"):
        label_names = [label.name for label in issue.labels]
        if "status: stale" in label_names or "status: in-progress" in label_names:
            continue
        if issue.updated_at >= stale_cutoff:
            continue
        if not dry_run:
            issue.add_to_labels("status: stale")
            issue.create_comment(
                "This issue has had no activity in 14+ days and was marked `status: stale`.\n\n"
                + agent_signature("milestone_checker", context="stale issue sweep")
            )
        stale_marked += 1
    return stale_marked


def triage_unlabeled_issues(repo, unlabeled: list[object], dry_run: bool) -> list[int]:
    rule_map = _load_rule_map()
    repo_label_names = {label.name for label in repo.get_labels()}
    triaged: list[int] = []
    for issue in unlabeled:
        labels_to_apply: list[str] = []
        upper_title = issue.title.upper()
        for marker, mapped in rule_map.items():
            if marker in upper_title:
                labels_to_apply.extend(mapped)
        labels_to_apply = sorted(set(labels_to_apply)) or ["status: needs-manual-triage", PRIORITY_MEDIUM]
        valid = [lbl for lbl in labels_to_apply if lbl in repo_label_names]
        if not valid:
            continue
        if not dry_run:
            issue.add_to_labels(*valid)
            issue.create_comment(
                "Labels applied automatically: "
                + ", ".join(f"`{lbl}`" for lbl in valid)
                + "\n\n"
                + agent_signature("milestone_checker", context="label hygiene sweep")
            )
        triaged.append(issue.number)
    return triaged


STALE_REPORT_MARKER = "<!-- agent:stale-report -->"
STALE_APPROACHING_DAYS = 7


def generate_stale_report(repo, now: datetime, dry_run: bool) -> int | None:
    """Create (or skip if one exists) a GitHub issue listing stale and approaching-stale issues.

    Returns the issue number of the created report, or None if nothing was created.
    This is idempotent: if an open stale-report issue already exists it is skipped.
    """
    stale_cutoff = now - timedelta(days=14)
    approaching_cutoff = now - timedelta(days=STALE_APPROACHING_DAYS)

    stale: list[object] = []
    approaching: list[object] = []

    for issue in repo.get_issues(state="open"):
        label_names = [label.name for label in issue.labels]
        if "status: in-progress" in label_names:
            continue
        if "status: stale" in label_names:
            stale.append(issue)
        elif issue.updated_at < approaching_cutoff:
            approaching.append(issue)

    if not stale and not approaching:
        return None

    # Skip if an open stale-report issue already exists
    existing = list(repo.get_issues(state="open", labels=[AGENT_CREATED_LABEL]))
    if any(STALE_REPORT_MARKER in (e.body or "") for e in existing):
        return None

    stale_lines = "\n".join(
        f"- #{i.number} **{i.title}** — last updated {i.updated_at.strftime('%Y-%m-%d')}"
        for i in stale[:30]
    ) or "- None"
    approaching_lines = "\n".join(
        f"- #{i.number} **{i.title}** — last updated {i.updated_at.strftime('%Y-%m-%d')}"
        for i in approaching[:30]
    ) or "- None"

    title = f"📋 Stale Issue Report ({now.strftime('%Y-%m-%d')}): {len(stale)} stale, {len(approaching)} approaching"
    body = f"""\
## Stale Issue Report

Generated on {now.strftime('%Y-%m-%d')} by the milestone health-check agent.

**Standard practice:** Issues are automatically labelled `status: stale` after 14 days
of inactivity (excluding `status: in-progress`). Close or update them to remove the label.
Filter by [`status: stale`]({{}}) in the Issues tab for a live view.

### Already Stale (14+ days, {len(stale)} issues)

{stale_lines}

### Approaching Stale (7–13 days inactive, {len(approaching)} issues)

{approaching_lines}

{agent_signature("milestone_checker", context="stale issue report")}

{STALE_REPORT_MARKER}
"""
    if dry_run:
        return None
    created = repo.create_issue(
        title=title,
        body=body,
        labels=["type: chore", AGENT_CREATED_LABEL],
    )
    return created.number


def report_unlabeled_issues(repo, now: datetime, dry_run: bool) -> list[int]:
    unlabeled = [issue for issue in repo.get_issues(state="open") if not issue.labels]
    if len(unlabeled) < 5:
        return []
    title = f"⚠️ Label Hygiene Alert: {len(unlabeled)} issues need labels"
    body = (
        "## Unlabeled Issues Detected\n\n"
        f"Detected {len(unlabeled)} open issues without labels as of {now.strftime('%Y-%m-%d')}.\n\n"
        "### Sample\n"
        + "\n".join(f"- #{i.number} {i.title}" for i in unlabeled[:20])
        + "\n\n"
        + agent_signature("milestone_checker", context="label hygiene report")
    )
    if dry_run:
        return [0]
    created = repo.create_issue(title=title, body=body, labels=["type: chore", AGENT_CREATED_LABEL])
    return [created.number]


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    now = datetime.now(timezone.utc)
    results: list[ActionResult] = []

    summary.checkpoint("load_data", "Loading open milestones and issues")
    repo = rest._repo_obj
    open_milestones = list(repo.get_milestones(state="open"))

    # 1) At-risk milestone issue creation
    created_count = 0
    for milestone in open_milestones:
        created_number = maybe_create_at_risk_issue(repo, milestone, now, ctx.dry_run)
        if ctx.dry_run:
            if created_number is None:
                continue
            results.append(ActionResult("create_at_risk_issue", milestone.title, ActionStatus.DRY_RUN))
        elif created_number is not None:
            created_count += 1
            results.append(ActionResult("create_at_risk_issue", milestone.title, ActionStatus.SUCCESS, metadata={"number": created_number}))

    # 2) Mark stale issues
    stale_marked = mark_stale_issues(repo, now, ctx.dry_run)
    results.append(ActionResult("mark_stale", "open issues", ActionStatus.DRY_RUN if ctx.dry_run else ActionStatus.SUCCESS, metadata={"count": stale_marked}))

    # 3) Auto-label unlabeled issues
    unlabeled = [issue for issue in repo.get_issues(state="open") if not issue.labels]
    auto_labeled = len(triage_unlabeled_issues(repo, unlabeled, ctx.dry_run))
    results.append(ActionResult("auto_label_unlabeled", "open issues", ActionStatus.DRY_RUN if ctx.dry_run else ActionStatus.SUCCESS, metadata={"count": auto_labeled}))

    # 4) Generate stale issue report
    report_number = generate_stale_report(repo, now, ctx.dry_run)
    results.append(ActionResult(
        "stale_report",
        "open issues",
        ActionStatus.DRY_RUN if ctx.dry_run else ActionStatus.SUCCESS,
        metadata={"number": report_number},
    ))

    summary.decision(f"Milestone check complete: created={created_count}, stale_marked={stale_marked}, auto_labeled={auto_labeled}, stale_report={report_number}")
    log_event("milestone_check_complete", created=created_count, stale_marked=stale_marked, auto_labeled=auto_labeled, stale_report=report_number, dry_run=ctx.dry_run)
    return RunResult(agent="milestone_checker", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("milestone_checker", _run)
