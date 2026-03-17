"""
Milestone Health Check Agent.
Runs daily and detects at-risk milestones, stale issues, and unlabeled issues.
Automatically applies rule-based labels to unlabeled issues before reporting.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from github import Github

from common import agent_signature, log_event, require_env, retry
from dependency_resolver import blocked_parent_rollup

AGENT_CREATED_LABEL = "status: agent-created"

_FALLBACK_RULE_MAP: dict[str, list[str]] = {
    "[BUG]": ["type: bug", "priority: high"],
    "[EXP]": ["type: experiment", "priority: medium"],
    "[RESEARCH]": ["type: research", "priority: low"],
    "[FEAT]": ["type: feature", "priority: medium"],
    "[FEATURE]": ["type: feature", "priority: medium"],
    "[EPIC]": ["type: epic", "priority: high"],
    "[WIP]": ["status: in-progress"],
}


def load_rule_map() -> dict[str, list[str]]:
    """Load title-marker → label rules from orchestration.yaml, with a hardcoded fallback."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
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


def triage_unlabeled_issues(repo, unlabeled: list, *, dry_run: bool) -> list[int]:
    """Apply rule-based labels to each unlabeled issue.

    For issues whose title contains a known marker (e.g. ``[BUG]``, ``[WIP]``),
    the corresponding labels are applied automatically.  Issues with no
    recognizable marker receive ``status: needs-manual-triage`` and
    ``priority: medium`` so they are still surfaced in project views.

    Returns the list of issue numbers that were successfully labelled.
    """
    if not unlabeled:
        return []

    rule_map = load_rule_map()
    repo_label_names = {label.name for label in repo.get_labels()}
    triaged: list[int] = []

    for issue in unlabeled:
        upper_title = issue.title.upper()
        labels_to_apply: list[str] = []
        for marker, mapped_labels in rule_map.items():
            if marker in upper_title:
                labels_to_apply.extend(mapped_labels)
        labels_to_apply = sorted(set(labels_to_apply))

        if not labels_to_apply:
            labels_to_apply = ["status: needs-manual-triage", "priority: medium"]

        valid_labels = [lbl for lbl in labels_to_apply if lbl in repo_label_names]
        if not valid_labels:
            log_event("label_hygiene_no_valid_labels", issue=issue.number)
            continue

        if dry_run:
            print(f"[dry-run] Would label #{issue.number} '{issue.title}': {valid_labels}")
        else:
            retry(lambda issue=issue, labels=tuple(valid_labels): issue.add_to_labels(*labels))
            retry(
                lambda issue=issue, labels=valid_labels: issue.create_comment(
                    f"Labels applied automatically: "
                    f"{', '.join(f'`{lbl}`' for lbl in labels)}\n\n"
                    + agent_signature("milestone_checker", context="label hygiene sweep")
                )
            )

        triaged.append(issue.number)
        log_event(
            "label_hygiene_applied",
            issue=issue.number,
            labels=valid_labels,
            dry_run=dry_run,
        )

    return triaged


def marker_for_milestone(milestone_number: int, due_date: str) -> str:
    return f"<!-- agent:milestone-risk:{milestone_number}:{due_date} -->"


def dependency_blocker_lines(open_issues: list[object]) -> str:
    """Render parent issues blocked by blocked children within this milestone."""
    rollup = blocked_parent_rollup(open_issues)
    if not rollup:
        return "- None"

    lines: list[str] = []
    for entry in rollup:
        parent = f"- #{entry['parent_number']} {entry['parent_title']}"
        child_list = ", ".join(
            f"#{child['number']} {child['title']}" for child in entry["blocked_children"]
        )
        lines.append(f"{parent} blocked by: {child_list}")
    return "\n".join(lines)


def maybe_create_at_risk_issue(repo, milestone, now: datetime, *, dry_run: bool) -> int | None:
    if not milestone.due_on:
        return None

    due = milestone.due_on.replace(tzinfo=timezone.utc)
    days_left = (due - now).days
    total = milestone.open_issues + milestone.closed_issues
    pct = (milestone.closed_issues / total * 100) if total > 0 else 0
    at_risk = (days_left <= 14 and pct < 50) or (days_left <= 7 and pct < 80)
    if not at_risk:
        return None

    title = f"⚠️ MILESTONE AT RISK: {milestone.title} ({pct:.0f}% complete, {days_left}d left)"
    marker = marker_for_milestone(milestone.number, due.strftime("%Y-%m-%d"))
    existing = list(repo.get_issues(state="open", labels=[AGENT_CREATED_LABEL]))
    if any(marker in (existing_issue.body or "") for existing_issue in existing):
        return None

    open_issues = list(repo.get_issues(state="open", milestone=milestone))
    body = f"""## Milestone At Risk

**Milestone:** [{milestone.title}]({milestone.url})
**Due:** {due.strftime('%Y-%m-%d')} ({days_left} days remaining)
**Progress:** {milestone.closed_issues}/{total} issues closed ({pct:.0f}%)

### Open Issues Remaining
{chr(10).join(f'- #{i.number} {i.title}' for i in open_issues) or '- None'}

### Dependency Blockers
{dependency_blocker_lines(open_issues)}

### Recommended Actions
- Review and close or defer non-critical issues.
- Ensure all active work is tracked with `status: in-progress`.
- Adjust milestone due date if scope has changed.

{agent_signature("milestone_checker", context="milestone health check")}

{marker}
"""

    if dry_run:
        print(f"[dry-run] Would create at-risk milestone issue: {title}")
        return None

    created_issue = retry(
        lambda title=title, body=body: repo.create_issue(
            title=title,
            body=body,
            labels=["priority: critical", "type: chore", AGENT_CREATED_LABEL],
        )
    )
    return created_issue.number


def process_at_risk_milestones(repo, now: datetime, *, dry_run: bool) -> list[int]:
    created: list[int] = []
    for milestone in repo.get_milestones(state="open"):
        maybe_issue_number = maybe_create_at_risk_issue(repo, milestone, now, dry_run=dry_run)
        if maybe_issue_number is not None:
            created.append(maybe_issue_number)
    return created


def mark_stale_issues(repo, now: datetime, *, dry_run: bool) -> None:
    stale_cutoff = now - timedelta(days=14)
    for issue in repo.get_issues(state="open"):
        label_names = [label.name for label in issue.labels]
        if "status: stale" in label_names or "status: in-progress" in label_names:
            continue
        if issue.updated_at >= stale_cutoff:
            continue
        if dry_run:
            print(f"[dry-run] Would mark issue #{issue.number} as stale")
            continue

        retry(lambda issue=issue: issue.add_to_labels("status: stale"))
        retry(
            lambda issue=issue: issue.create_comment(
                "This issue has had no activity in 14+ days and was marked "
                "`status: stale`. Please update, close, or defer it.\n\n"
                + agent_signature("milestone_checker", context="stale issue sweep")
            )
        )


def report_unlabeled_issues(repo, now: datetime, *, dry_run: bool) -> list[int]:
    created: list[int] = []
    unlabeled = [issue for issue in repo.get_issues(state="open") if not issue.labels]
    if len(unlabeled) <= 3:
        return created

    marker = f"<!-- agent:unlabeled-count:{len(unlabeled)}:{now.strftime('%Y-%m-%d')} -->"
    existing_agent = list(repo.get_issues(state="open", labels=[AGENT_CREATED_LABEL]))
    if any(marker in (issue.body or "") for issue in existing_agent):
        return created

    body = f"""## Unlabeled Issues Detected

The following {len(unlabeled)} issues have no labels and may not be properly triaged:

{chr(10).join(f'- #{i.number} {i.title}' for i in unlabeled[:20])}

Please label these issues so they appear in the correct project views.

{agent_signature("milestone_checker", context="label hygiene sweep")}

{marker}
"""

    if dry_run:
        print(f"[dry-run] Would create unlabeled-issues report: {len(unlabeled)} issues")
        return created

    new_issue = retry(
        lambda body=body: repo.create_issue(
            title=f"🏷️ {len(unlabeled)} issues need labels",
            body=body,
            labels=["type: chore", "priority: low", AGENT_CREATED_LABEL],
        )
    )
    created.append(new_issue.number)
    return created


def main() -> None:
    env = require_env(["REPO_NAME", "GITHUB_TOKEN"])
    repo_name = env["REPO_NAME"]
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    now = datetime.now(timezone.utc)

    gh = Github(env["GITHUB_TOKEN"])
    repo = gh.get_repo(repo_name)
    issues_created = process_at_risk_milestones(repo, now, dry_run=dry_run)
    mark_stale_issues(repo, now, dry_run=dry_run)

    # Auto-label unlabeled issues, then report any that still have no labels.
    unlabeled = [issue for issue in repo.get_issues(state="open") if not issue.labels]
    triage_unlabeled_issues(repo, unlabeled, dry_run=dry_run)

    issues_created.extend(report_unlabeled_issues(repo, now, dry_run=dry_run))

    log_event("milestone_check_complete", created=issues_created, dry_run=dry_run)
    print(
        f"{'[dry-run] ' if dry_run else ''}✅ Milestone check complete. Created {len(issues_created)} issues: {issues_created}"
    )


if __name__ == "__main__":
    main()
