# SPDX-License-Identifier: MIT
"""Weekly Progress Report Agent: summarizes weekly issue activity and milestone progress."""

from __future__ import annotations

import json
import importlib
import os
import sys
import types
from datetime import datetime, timedelta, timezone

try:  # pragma: no cover - compatibility shim for test environments without openai installed
    _openai_module = importlib.import_module("openai")
except ModuleNotFoundError:  # pragma: no cover
    _openai_module = types.ModuleType("openai")
    _openai_module.OpenAI = None  # type: ignore[attr-defined]
    sys.modules["openai"] = _openai_module

from agent_platform.context import AgentContext
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event
from dependency_resolver import blocked_parent_rollup

REQUIRED_LABELS = ["type: docs", "status: agent-created"]


def _issue_lines(items: list[dict[str, object]], empty: str) -> str:
    if not items:
        return empty
    return "\n".join(f"- #{item['number']} {item['title']}" for item in items)


def _milestone_lines(items: list[dict[str, object]]) -> str:
    if not items:
        return "- No open milestones"
    return "\n".join(
        f"- {m['title']}: {m['closed']}/{m['closed'] + m['open']} closed ({m['pct']}%), due {m['due']}"
        for m in items
    )


def _dependency_blocker_lines(items: list[dict[str, object]]) -> str:
    if not items:
        return "- None"
    return "\n".join(
        f"- #{item['parent_number']} {item['parent_title']} blocked by "
        + ", ".join(f"#{child['number']} {child['title']}" for child in item["blocked_children"])
        for item in items[:15]
    )


SYSTEM_PROMPT = """
You are a project management assistant.
Write only a concise narrative section in GitHub markdown with these headings:
## Executive Summary
## Risks and Blockers
## Recommended Focus Next Week
Use ONLY the provided data. Do not repeat the deterministic tables verbatim.
"""


def _deterministic_report(context: dict[str, object]) -> str:
    return f"""## Weekly Snapshot (Deterministic Data)

- Week ending: {context['week_ending']}
- Closed this week: {len(context['closed_this_week'])}
- Opened this week: {len(context['opened_this_week'])}
- Currently blocked: {len(context['currently_blocked'])}
- In progress: {len(context['in_progress'])}
- Total open issues: {context['total_open_issues']}

### Closed This Week
{_issue_lines(context['closed_this_week'], '- None')}

### Opened This Week
{_issue_lines(context['opened_this_week'], '- None')}

### In Progress
{_issue_lines(context['in_progress'], '- None')}

### Blockers
{_issue_lines(context['currently_blocked'], '- None')}

### Milestone Progress
{_milestone_lines(context['milestones'])}

### Dependency Blockers
{_dependency_blocker_lines(context['dependency_blockers'])}
"""


# Backward-compatible helper names used by existing tests.
issue_lines = _issue_lines
milestone_lines = _milestone_lines
deterministic_report = _deterministic_report


def main() -> None:
    """Legacy test-friendly entrypoint preserved for compatibility."""
    repo_name = os.environ["REPO_NAME"]
    token = os.environ["GITHUB_TOKEN"]
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"

    github_module = importlib.import_module("github")
    gh = github_module.Github(auth=github_module.Auth.Token(token))
    repo = gh.get_repo(repo_name)

    now = datetime.now(timezone.utc)
    one_week_ago = now - timedelta(days=7)
    all_closed = list(repo.get_issues(state="closed"))
    all_open = list(repo.get_issues(state="open"))
    all_milestones = list(repo.get_milestones(state="open"))

    closed_this_week = [issue for issue in all_closed if issue.closed_at and issue.closed_at > one_week_ago]
    opened_this_week = [issue for issue in all_open if issue.created_at > one_week_ago]
    blocked = [issue for issue in all_open if any(label.name == "status: blocked" for label in issue.labels)]
    in_progress = [issue for issue in all_open if any(label.name == "status: in-progress" for label in issue.labels)]

    milestone_data = []
    for milestone in all_milestones:
        total = milestone.open_issues + milestone.closed_issues
        pct = round(milestone.closed_issues / total * 100) if total > 0 else 0
        milestone_data.append(
            {
                "title": milestone.title,
                "open": milestone.open_issues,
                "closed": milestone.closed_issues,
                "pct": pct,
                "due": str(milestone.due_on.date()) if milestone.due_on else "no due date",
            }
        )

    context = {
        "week_ending": now.strftime("%Y-%m-%d"),
        "closed_this_week": [{"number": issue.number, "title": issue.title} for issue in closed_this_week],
        "opened_this_week": [{"number": issue.number, "title": issue.title} for issue in opened_this_week],
        "currently_blocked": [{"number": issue.number, "title": issue.title} for issue in blocked],
        "in_progress": [{"number": issue.number, "title": issue.title} for issue in in_progress],
        "milestones": milestone_data,
        "total_open_issues": len(all_open),
        "dependency_blockers": blocked_parent_rollup(all_open),
    }

    ai_narrative = "## Executive Summary\nAI narrative unavailable for this run.\n"
    if openai_key:
        openai_cls = importlib.import_module("openai").OpenAI
        ai = openai_cls(api_key=openai_key)
        response = ai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a narrative from this data:\n{json.dumps(context, indent=2)}"},
            ],
            timeout=60,
        )
        ai_narrative = response.choices[0].message.content

    report_body = (
        _deterministic_report(context)
        + "\n\n"
        + ai_narrative
        + "\n\n---\n"
        + "<!-- agent:progress_reporter -->\n"
        + "> Source data: GitHub issues and milestones API."
    )

    title = f"📊 Weekly Progress Report — Week of {now.strftime('%Y-%m-%d')}"
    available_labels = [label.name for label in repo.get_labels()]
    labels_to_apply = [label for label in REQUIRED_LABELS if label in available_labels]

    if dry_run:
        return
    repo.create_issue(title=title, body=report_body, labels=labels_to_apply)


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    repo = rest._repo_obj
    now = datetime.now(timezone.utc)
    one_week_ago = now - timedelta(days=7)
    results: list[ActionResult] = []

    all_closed = list(repo.get_issues(state="closed"))
    all_open = list(repo.get_issues(state="open"))
    all_milestones = list(repo.get_milestones(state="open"))

    closed_this_week = [issue for issue in all_closed if issue.closed_at and issue.closed_at > one_week_ago]
    opened_this_week = [issue for issue in all_open if issue.created_at > one_week_ago]
    blocked = [issue for issue in all_open if any(label.name == "status: blocked" for label in issue.labels)]
    in_progress = [issue for issue in all_open if any(label.name == "status: in-progress" for label in issue.labels)]

    milestone_data = []
    for milestone in all_milestones:
        total = milestone.open_issues + milestone.closed_issues
        pct = round(milestone.closed_issues / total * 100) if total > 0 else 0
        milestone_data.append(
            {
                "title": milestone.title,
                "open": milestone.open_issues,
                "closed": milestone.closed_issues,
                "pct": pct,
                "due": str(milestone.due_on.date()) if milestone.due_on else "no due date",
            }
        )

    context = {
        "week_ending": now.strftime("%Y-%m-%d"),
        "closed_this_week": [{"number": issue.number, "title": issue.title} for issue in closed_this_week],
        "opened_this_week": [{"number": issue.number, "title": issue.title} for issue in opened_this_week],
        "currently_blocked": [{"number": issue.number, "title": issue.title} for issue in blocked],
        "in_progress": [{"number": issue.number, "title": issue.title} for issue in in_progress],
        "milestones": milestone_data,
        "total_open_issues": len(all_open),
        "dependency_blockers": blocked_parent_rollup(all_open),
    }

    ai_narrative = "## Executive Summary\nAI narrative unavailable for this run.\n"
    try:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            openai_cls = importlib.import_module("openai").OpenAI
            ai = openai_cls(api_key=key)
            response = ai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a narrative from this data:\n{json.dumps(context, indent=2)}"},
                ],
                timeout=60,
            )
            ai_narrative = response.choices[0].message.content
    except Exception as exc:
        log_event("progress_reporter_ai_failed", error=str(exc))

    report_body = (
        _deterministic_report(context)
        + "\n\n"
        + ai_narrative
        + "\n\n---\n"
        + "<!-- agent:progress_reporter -->\n"
        + "> Source data: GitHub issues and milestones API."
    )

    title = f"📊 Weekly Progress Report — Week of {now.strftime('%Y-%m-%d')}"
    available_labels = [label.name for label in repo.get_labels()]
    labels_to_apply = [label for label in REQUIRED_LABELS if label in available_labels]

    if ctx.dry_run:
        results.append(ActionResult("create_weekly_report", title, ActionStatus.DRY_RUN, metadata={"labels": labels_to_apply}))
    else:
        new_issue = repo.create_issue(title=title, body=report_body, labels=labels_to_apply)
        results.append(ActionResult("create_weekly_report", f"issue #{new_issue.number}", ActionStatus.SUCCESS))
        log_event("progress_report_created", issue_number=new_issue.number, week_ending=now.strftime("%Y-%m-%d"))

    summary.decision(f"Weekly report generated for {now.strftime('%Y-%m-%d')}")
    return RunResult(agent="progress_reporter", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("progress_reporter", _run)
